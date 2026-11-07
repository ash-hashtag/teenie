"""Train IEEE float32 recurrent regression from custom scalar functions."""
import argparse
from collections.abc import Sequence
import json
import time
from pathlib import Path

import torch

from .checkpoint import Metadata, save_model
from .config import CONTEXT_SIZE
from .data import EpisodeSampler, FunctionTask, load_tasks
from .families import bounded_basics
from .metrics import measure, training_loss
from .model import ContextModel


def select_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device(name)


def train(tasks: Sequence[FunctionTask] | None = None, *, device: str = 'auto', updates: int = 3000,
          batch_size: int = 8, knots: int = 16, steps: int = 256, queries: int = 16,
          lr: float = .0003, seed: int = 0, log_every: int = 500,
          save: str | Path = 'checkpoints/residual.pt', validation_batches: int = 16) -> ContextModel:
    selected = bounded_basics() if tasks is None else list(tasks)
    use_curriculum = tasks is None
    if min(updates, batch_size, log_every, validation_batches, queries) < 1 or not 0 < lr < float('inf'):
        raise ValueError('counts and finite lr must be positive')
    if not selected or len({t.name for t in selected}) != len(selected):
        raise ValueError('provide nonempty tasks with unique names')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    selected_device = select_device(device)
    model = ContextModel(knots, steps).to(selected_device)
    sampler = EpisodeSampler(selected, seed)
    validation = {task.name: [EpisodeSampler([task], seed + 100000 + j).sample(batch_size).to(selected_device)
                             for j in range(validation_batches)] for task in selected}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=lr * .1)
    maximum = 16.
    gate_streak = 0
    started = time.perf_counter()
    best = float('inf')
    path = Path(save)
    best_path = path.with_name(path.stem + '.best' + path.suffix)
    for update in range(1, updates + 1):
        model.train()
        if use_curriculum:
            sampler.tasks = bounded_basics(maximum)
        episode = sampler.sample_queries(batch_size, queries=queries).to(selected_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward(episode.context, episode.query, steps=steps)
        # Supervise only the endpoint: intermediate memory states need not already be answers.
        loss = training_loss(prediction, episode.target)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss: training aborted before updating weights')
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's backward stub is untyped.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if update == 1 or update % log_every == 0 or update == updates:
            model.eval()
            metrics = {name: measure(model, batches) for name, batches in validation.items()}
            with torch.no_grad():
                score = sum(float(training_loss(model.forward(e.context, e.query), e.target))
                            for batches in validation.values() for e in batches) / (len(selected) * validation_batches)
            gate: dict[str, float] = {}
            trained_maximum = maximum
            if use_curriculum:
                for task in bounded_basics(maximum):
                    batches = [EpisodeSampler([task], seed+200000+j).sample(batch_size).to(selected_device)
                               for j in range(validation_batches)]
                    gate[task.name] = float(measure(model, batches)['accuracy'] or 0.)
                gate_streak = gate_streak + 1 if min(gate.values()) >= .90 else 0
                if gate_streak >= 2 and maximum < 256:
                    maximum = 64. if maximum == 16. else 256.
                    gate_streak = 0
            record: Metadata = {'update': update, 'training_maximum': trained_maximum if use_curriculum else None, 'next_maximum': maximum if use_curriculum else None, 'curriculum_gate': gate, 'loss': loss.item(), 'validation_loss': score, 'elapsed_seconds': time.perf_counter()-started, 'val': metrics}
            print(json.dumps(record, allow_nan=False), flush=True)
            metadata: Metadata = {'training_functions': [t.name for t in selected],
                                  'ranges': {t.name: [list(t.a_range), list(t.b_range)] for t in selected},
                                  'distributions': {t.name: t.distribution for t in selected},
                                  'seed': seed, 'context_size': CONTEXT_SIZE,
                                  'curriculum': [16.,64.,256.] if use_curriculum else None,
                                  'families': [],
                                  'queries_per_context': queries, 'validation_batches': validation_batches, 'update_rule': 'designed conjugate gradient; learned features',
                                  'loop_budgets': [steps],
                                  'parameters': sum(p.numel() for p in model.parameters()),
                                  'training_contexts': update*batch_size, 'training_examples': update*batch_size*queries,
                                  'selection': 'training-family validation only; no held-out structures or deeper recurrences', 'numeric_contract': 'IEEE float32; nan=undefined', **record}
            save_model(path, model, metadata)
            if score < best:
                best = score
                save_model(best_path, model, metadata)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--functions', help='Trusted TASKS file; overrides the default bounded basic curriculum')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--knots', type=int, default=16)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=500)
    parser.add_argument('--validation-batches', type=int, default=16)
    parser.add_argument('--save', default='checkpoints/residual.pt')
    args = parser.parse_args()
    try:
        train(load_tasks(args.functions) if args.functions else None, device=args.device, updates=args.updates, batch_size=args.batch_size,
              knots=args.knots, steps=args.steps, queries=args.queries, lr=args.lr, seed=args.seed,
              log_every=args.log_every, save=args.save, validation_batches=args.validation_batches)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
