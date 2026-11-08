"""Cheap answer-time improvements, no training: voting across models, and an honest-confidence check.

uv run python -m teenie.bits.vote --models checkpoints/steps/bittoken-mix4.pt checkpoints/bt/bt-seed1.pt \
    --output experiments/bt/vote-eval.json

- Vote: average the bit probabilities of several models (and/or the same model at several round counts).
- Fit check: every model also re-predicts its probe examples (the first examples of the context, each hidden from
  itself). If the answers it gives there are often wrong, it has probably misread the relation, so it should not
  claim to be sure about the question either. "Sure" then needs the usual least-certain-bit >= 0.99 and a probe
  exact rate >= `--fit`.
"""
import argparse
import json
from pathlib import Path

import torch

from .bit_token_model import BitTokenModel
from .checkpoint import load_bit_model
from .data import BitSampler, from_bits, to_bits
from .eval import SURE, Report, summarise
from .loop_model import LoopModel, LoopPrediction
from .relations import Family
from .sets import test_groups

Voter = LoopModel | BitTokenModel


@torch.no_grad()
def answer(model: Voter, context: torch.Tensor, query: torch.Tensor, steps: int | None) -> tuple[torch.Tensor,
                                                                                                         torch.Tensor]:
    """Bit probabilities for the questions [B, Q, W] and the probe exact rate per episode [B]."""
    prediction: LoopPrediction = model.forward(context, query, 8, steps=steps)
    probes = prediction.logits.shape[2] - prediction.queries
    probe_bits = prediction.logits[-1, :, prediction.queries:] > 0
    fit = (probe_bits.float() == to_bits(context[:, :probes, 2], 8)).all(-1).float().mean(-1)
    return prediction.probabilities(), fit


def evaluate(members: list[tuple[Voter, int | None]], families: tuple[Family, ...], episodes: int, queries: int,
             seed: int, device: str, fit_threshold: float) -> dict[str, dict[str, Report]]:
    names = [f'member{i}' for i in range(len(members))] + ['vote', 'vote+fit']
    results: dict[str, dict[str, Report]] = {name: {} for name in names}
    for family in families:
        sampler = BitSampler([family], 8, seed)
        tallies: dict[str, dict[str, list[torch.Tensor]]] = {name: {} for name in names}
        for _ in range(episodes):
            episode = sampler.sample(1, queries).to(device)
            outputs = [answer(model, episode.context, episode.query, steps) for model, steps in members]
            probabilities = [p for p, _ in outputs] + [torch.stack([p for p, _ in outputs]).mean(0)]
            fits = [f for _, f in outputs] + [torch.stack([f for _, f in outputs]).mean(0)]
            for name, p, f in zip(names, probabilities + [probabilities[-1]], fits + [fits[-1]]):
                correct = from_bits(p > .5) == episode.target
                sure = torch.maximum(p, 1 - p).amin(-1) >= SURE
                if name == 'vote+fit':
                    sure = sure & (f[:, None] >= fit_threshold)
                tallies[name].setdefault('exact', []).append(correct.flatten())
                tallies[name].setdefault('sure', []).append(sure.flatten())
                tallies[name].setdefault('sure_correct', []).append(correct[sure])
        for name in names:
            values = {k: torch.cat(v).float() for k, v in tallies[name].items()}
            results[name][family.name] = {
                'exact': float(values['exact'].mean()), 'sure_coverage': float(values['sure'].mean()),
                'sure_accuracy': float(values['sure_correct'].mean()) if len(values['sure_correct']) else None}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Voting and an honest-confidence check, without training')
    parser.add_argument('--models', nargs='+', required=True, help='checkpoints; add @N to run one with N rounds')
    parser.add_argument('--tests', default='mix4')
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--seed', type=int, default=60000)
    parser.add_argument('--fit', type=float, default=.9, help='probe exact rate needed to claim "sure"')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    members: list[tuple[Voter, int | None]] = []
    for spec in args.models:
        path, _, rounds = spec.partition('@')
        model, _ = load_bit_model(path, args.device)
        if not isinstance(model, (LoopModel, BitTokenModel)):
            raise SystemExit(f'{path}: voting needs loop or bit-token models')
        members.append((model, int(rounds) if rounds else None))
    groups = {g: f for g, f in test_groups(args.tests).items() if g != 'train'}
    report = {group: evaluate(members, families, args.episodes, args.queries, args.seed, args.device, args.fit)
              for group, families in groups.items()}
    summary = {name: {group: summarise(report[group][name]) for group in groups} for name in report[next(iter(groups))]}
    if args.output:
        Path(args.output).write_text(json.dumps({'models': args.models, 'fit': args.fit, 'summary': summary,
                                                 'results': report}, indent=1))
    for name, groups_summary in summary.items():
        print(name, ' '.join(f'{g}={100 * float(s["exact"] or 0):.1f}%/sure-right '
                             f'{"-" if s["sure_accuracy"] is None else f"{100 * float(s["sure_accuracy"]):.1f}%"}'
                             for g, s in groups_summary.items()))


if __name__ == '__main__':
    main()
