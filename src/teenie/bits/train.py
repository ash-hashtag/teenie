"""Meta-train a bit model over many relation families; held-out families never select checkpoints.

`serial` reads bit positions in order; `loop` sees each example's bits at once and loops on the answer;
`think` rewrites the example and question tokens themselves with one shared block per round;
`program` infers the weights of a small calculator from the examples and refines them by checking it.
"""
import argparse
import json
from collections.abc import Sequence
from typing import Literal
import time
from pathlib import Path

import torch

from ..train import select_device
from .checkpoint import AnyBitModel, load_bit_model, save_bit_model
from .data import BitEpisodes, BitSampler
from .eval import evaluate
from .bit_token_model import BitTokenModel
from .loop_model import LoopModel
from .model import BitModel
from .sets import TRAIN_SETS, test_groups
from .program_model import ProgramModel
from .think_model import ThinkModel
from .wide_model import WideBitModel


def step_loss(model: AnyBitModel, episode: BitEpisodes, width: int) -> torch.Tensor:
    """Forward pass and loss for one batch of episodes."""
    if isinstance(model, ProgramModel):
        return model.loss(model.forward(episode.context, episode.query, width), episode.context, episode.target, width)
    if isinstance(model, (LoopModel, BitTokenModel, WideBitModel)):
        prediction = model.forward(episode.context, episode.query, width)
        return model.loss(prediction, episode.context, episode.target, width, episode.steps, episode.step_counts)
    if isinstance(model, ThinkModel):
        return model.loss(model.forward(episode.context, episode.query, width), episode.context, episode.target, width)
    # Teacher forcing (true earlier c bits, training only) applies to the serial model.
    serial = model.forward(episode.context, episode.query, width, episode.target)
    return model.loss(serial, episode.context, episode.target, width)


def train(*, device: str = 'auto', updates: int = 3000, batch_size: int = 8, queries: int = 16,
          width: int = 8, hidden: int = 128, heads: int = 4, probes: int = 64, lr: float = 1e-3,
          seed: int = 0, log_every: int = 250, validation_episodes: int = 8,
          save: str | Path = 'checkpoints/bits.pt', log: str | Path | None = None,
          snapshots: str | Path | None = None,
          architecture: Literal['serial', 'loop', 'think', 'program', 'bittoken', 'wide'] = 'serial',
          layers: int = 2, steps: int = 8, train_set: str = 'base', channels: int = 16,
          informed: bool = True, multiplicative: bool = False, clock: bool = False,
          adaptive: bool = False, init: str | Path | None = None, resume: bool = False,
          scratch: int = 0, halting: bool = False, stepwise: bool = False, bf16: bool = False,
          recompute: bool = True, aligned: bool = False, hint_fade: int = 0,
          widths: Sequence[int] = (), recompute_above: int = 32, peers: bool = False,
          registers: int = 0, example_layers: int = 0, code_slots: int = 0, code_layers: int = 3,
          code_film: bool = False) -> AnyBitModel:
    """`init` starts from saved weights and trains `updates` more (fresh optimizer and schedule; update numbers
    continue from the checkpoint). `resume` continues an interrupted run exactly from `<save>.resume.pt`, which
    every log step writes: weights, optimizer, schedule, sampling weights and random state. `widths` (width-free
    model only) trains on several widths in turn — repeat a width to train on it more often; validation and
    checkpoints use the widest. Episodes wider than `recompute_above` bits always recompute activations to fit in
    memory."""
    if min(updates, batch_size, queries, log_every, validation_episodes) < 1 or not lr > 0:
        raise ValueError('counts and lr must be positive')
    torch.manual_seed(seed)
    selected = select_device(device)
    # TF32 matrix maths on Ampere GPUs: much faster float32 matmuls at slightly lower precision.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if train_set not in TRAIN_SETS:
        raise ValueError(f'unknown training set {train_set!r}; choose from {sorted(TRAIN_SETS)}')
    families = TRAIN_SETS[train_set]
    # Everything not trained on is only watched: never used to select checkpoints.
    watched = tuple(f for name, group in test_groups(train_set).items() if name != 'train' for f in group)
    model: AnyBitModel
    widths = tuple(widths) or (width,)
    if len(widths) > 1 and architecture != 'wide':
        raise ValueError('several widths need the width-free architecture')
    width = max(widths)
    if architecture == 'wide':
        model = WideBitModel(hidden, heads, layers, probes, steps, peers=peers, registers=registers,
                             example_layers=example_layers, code_slots=code_slots, code_layers=code_layers,
                             code_film=code_film).to(selected)
    elif architecture == 'bittoken':
        model = BitTokenModel(hidden, heads, layers, width, probes, steps, stepwise, aligned).to(selected)
    elif architecture == 'program':
        model = ProgramModel(hidden, channels, width, steps, informed).to(selected)
    elif architecture == 'think':
        model = ThinkModel(hidden, heads, layers, width, steps).to(selected)
    elif architecture == 'loop':
        model = LoopModel(hidden, heads, layers, width, probes, steps, multiplicative, clock, scratch,
                          halting, stepwise).to(selected)
    elif architecture == 'serial':
        model = BitModel(hidden, heads, probes=probes).to(selected)
    else:
        raise ValueError(f'unknown architecture {architecture!r}')
    if isinstance(model, (BitTokenModel, WideBitModel)):
        model.recompute = recompute
    path = Path(save)
    best_path = path.with_name(path.stem + '.best' + path.suffix)
    resume_path = path.with_name(path.stem + '.resume' + path.suffix)
    start = 0
    if init is not None and resume:
        raise ValueError('use either init or resume, not both')
    if init is not None:
        loaded, init_metadata = load_bit_model(init, selected)
        # Rounds reuse the same weights, so continuing with a different number of rounds is allowed.
        # Rounds reuse the same weights; a width-free model may also gain peer reading or registers.
        optional = {'steps', 'peers', 'registers', 'example_layers', 'code_slots', 'code_layers', 'code_film'}
        same = ({k: v for k, v in loaded.config.items() if k not in optional}
                == {k: v for k, v in model.config.items() if k not in optional})
        if type(loaded) is not type(model) or not same:
            raise ValueError(f'{init} has a different architecture or size: {loaded.config}')
        if isinstance(model, WideBitModel):
            model.load_extending(loaded.state_dict())
        else:
            model.load_state_dict(loaded.state_dict())
        start = int(str(init_metadata.get('update', 0)))
    end = start + updates
    samplers = [BitSampler(families, w, seed + i) for i, w in enumerate(widths)]
    sampler = max(samplers, key=lambda s: s.width)  # the widest; every sampler shares its sampling weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=updates, pct_start=.05)
    best = -1.
    if resume:
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['schedule'])
        sampler.weights = state['sampler_weights']
        generators = state['generator'] if isinstance(state['generator'], list) else [state['generator']]
        for each, generator_state in zip(samplers[::-1], generators[::-1]):
            each.generator.set_state(generator_state)
        torch.set_rng_state(state['torch_rng'])
        start, end, best = state['update'], state['end'], state['best']
    log_path = Path(log) if log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            log_path.write_text('')
    started = time.perf_counter()
    running = 0.
    snapshot_dir = Path(snapshots) if snapshots else None
    if snapshot_dir:
        save_bit_model(snapshot_dir / 'update-00000.pt', model,
                       {'architecture': architecture, 'train_set': train_set, 'width': width, 'update': 0, 'seed': seed})
    for update in range(start + 1, end + 1):
        model.train()
        if hint_fade and isinstance(model, BitTokenModel):
            model.hint = max(0., 1 - (update - start) / hint_fade)  # step hints fade out over `hint_fade` updates
        current = samplers[update % len(samplers)]
        current.weights = sampler.weights
        if isinstance(model, WideBitModel):
            model.recompute = recompute or current.width > recompute_above
        episode = current.sample(batch_size, queries, steps=stepwise).to(selected)
        optimizer.zero_grad(set_to_none=True)
        # bf16 autocast runs matmuls and attention in bfloat16 (weights and optimizer stay float32).
        with torch.autocast(selected.type, dtype=torch.bfloat16, enabled=bf16):
            loss = step_loss(model, episode, current.width)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss: training aborted before updating weights')
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's backward stub is untyped.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        schedule.step()
        running += float(loss.detach())
        if update % log_every == 0 or update == end:
            model.eval()
            options = {'width': width, 'episodes': validation_episodes, 'queries': queries,
                       'seed': seed + 100000, 'device': selected, 'baselines': False}
            known = evaluate(model, families, **options)
            heldout = evaluate(model, watched, **options)
            score = sum(float(r['exact'] or 0) for r in known.values()) / len(known)
            if adaptive:
                # Spend episodes where the model is still wrong; mastered families keep a small share.
                sampler.weights = torch.tensor([.05 + 1 - float(known[f.name]['exact'] or 0) for f in families])
            record = {'update': update, 'sampling': {f.name: round(float(w), 3) for f, w in
                                                     zip(families, sampler.weights / sampler.weights.sum())},
                      'seconds': round(time.perf_counter() - started, 1),
                      'loss': running / log_every, 'known_exact': score,
                      'exact': {name: r['exact'] for name, r in {**known, **heldout}.items()},
                      'heldout_exact': sum(float(r['exact'] or 0) for r in heldout.values()) / len(heldout)}
            running = 0.
            metadata: dict[str, object] = {
                'architecture': architecture, 'multiplicative': multiplicative, 'clock': clock, 'adaptive': adaptive,
                'scratch': scratch, 'halting': halting, 'stepwise': stepwise, 'bf16': bf16, 'aligned': aligned,
                'hint_fade': hint_fade, 'peers': peers, 'registers': registers,
                'example_layers': example_layers, 'code_slots': code_slots, 'code_layers': code_layers,
                'code_film': code_film,
                'train_set': train_set, 'width': width, 'widths': list(widths), 'update': update, 'seed': seed, 'known_exact': score}
            save_bit_model(path, model, metadata)
            if snapshot_dir:
                save_bit_model(snapshot_dir / f'update-{update:05d}.pt', model, metadata)
            if score > best:
                best = score
                save_bit_model(best_path, model, metadata)
            temporary = resume_path.with_suffix('.tmp')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'schedule': schedule.state_dict(), 'sampler_weights': sampler.weights,
                        'generator': [s.generator.get_state() for s in samplers], 'torch_rng': torch.get_rng_state(),
                        'update': update, 'end': end, 'best': best}, temporary)
            temporary.replace(resume_path)
            print(json.dumps(record), flush=True)
            if log_path:
                with log_path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the bit-serial 999-example relation model')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--probes', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=250)
    parser.add_argument('--save', default='checkpoints/bits.pt')
    parser.add_argument('--log')
    parser.add_argument('--widths', type=int, nargs='+', default=(),
                        help='wide: train on these widths in turn, repeats allowed (overrides --width)')
    parser.add_argument('--peers', action='store_true',
                        help='wide: questions also read the probes, worked through in depth, with their true answers')
    parser.add_argument('--code-slots', type=int, default=0,
                        help='wide: answer bits read only this many relation-code tokens per bit, not the examples')
    parser.add_argument('--code-layers', type=int, default=3, help='wide: layers that build the relation code')
    parser.add_argument('--code-film', action='store_true',
                        help='wide: the relation code also scales and shifts the answer bits after every layer')
    parser.add_argument('--example-layers', type=int, default=0,
                        help='wide: layers that work through every example before the questions read it')
    parser.add_argument('--registers', type=int, default=0,
                        help='wide: scratch numbers each question writes and reads back every round')
    parser.add_argument('--recompute-above', type=int, default=32,
                        help='wide: recompute activations for episodes wider than this, even with --no-recompute')
    parser.add_argument('--architecture', choices=['serial', 'loop', 'think', 'program', 'bittoken', 'wide'],
                        default='serial')
    parser.add_argument('--channels', type=int, default=16, help='program: calculator channels per kind')
    parser.add_argument('--multiplicative', action='store_true',
                        help='loop: bilinear inputs and gated feed-forward layers')
    parser.add_argument('--clock', action='store_true',
                        help='loop: tell each round its number and score mostly the last round')
    parser.add_argument('--adaptive', action='store_true',
                        help='after each validation, draw relations in proportion to how often they are still wrong')
    parser.add_argument('--scratch', type=int, default=0,
                        help='loop: per-question memory slots carried between rounds, never scored')
    parser.add_argument('--halting', action='store_true',
                        help='loop: learn when to stop; --steps becomes the maximum number of rounds')
    parser.add_argument('--stepwise', action='store_true',
                        help='loop/bittoken: train each round against the next intermediate step, plus a "done" signal')
    parser.add_argument('--bf16', action='store_true', help='train in bfloat16 mixed precision (faster on Ampere GPUs)')
    parser.add_argument('--no-recompute', action='store_true',
                        help='bittoken: keep activations instead of recomputing them in backward (faster, more memory)')
    parser.add_argument('--validation-episodes', type=int, default=8)
    parser.add_argument('--aligned', action='store_true', help='bittoken: question bit k reads the examples\' bit-k tokens')
    parser.add_argument('--hint-fade', type=int, default=0,
                        help='bittoken --stepwise: fade intermediate-step targets to zero over this many updates')
    parser.add_argument('--init', help='start from these weights and train --updates more (fresh optimizer)')
    parser.add_argument('--resume', action='store_true',
                        help='continue an interrupted run exactly from <save>.resume.pt')
    parser.add_argument('--blank', action='store_true',
                        help='program: start input scales and readout frequencies blank instead of at place values')
    parser.add_argument('--train-set', choices=sorted(TRAIN_SETS), default='base',
                        help='mul adds a*b; blocks adds a*const, a*(b&3), a*(b&15) but not a*b')
    parser.add_argument('--layers', type=int, default=2, help='loop/think: attention layers per round')
    parser.add_argument('--steps', type=int, default=8, help='loop/think: rounds')
    parser.add_argument('--snapshots', help='directory for a checkpoint at every log step (and update 0)')
    args = parser.parse_args()
    train(device=args.device, updates=args.updates, batch_size=args.batch_size, queries=args.queries,
          width=args.width, hidden=args.hidden, heads=args.heads, probes=args.probes, lr=args.lr,
          seed=args.seed, log_every=args.log_every, save=args.save, log=args.log,
          snapshots=args.snapshots, architecture=args.architecture, layers=args.layers, steps=args.steps,
          train_set=args.train_set, channels=args.channels, informed=not args.blank,
          multiplicative=args.multiplicative, clock=args.clock, adaptive=args.adaptive, init=args.init,
          resume=args.resume, scratch=args.scratch, halting=args.halting, stepwise=args.stepwise, bf16=args.bf16,
          recompute=not args.no_recompute, validation_episodes=args.validation_episodes, aligned=args.aligned,
          hint_fade=args.hint_fade, widths=args.widths, recompute_above=args.recompute_above,
          peers=args.peers, registers=args.registers, example_layers=args.example_layers,
          code_slots=args.code_slots, code_layers=args.code_layers, code_film=args.code_film)


if __name__ == '__main__':
    main()
