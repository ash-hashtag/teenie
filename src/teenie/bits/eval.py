"""Exact-match evaluation of a bit checkpoint and non-learned controls on identical episodes."""
import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import torch

from .baselines import nearest, window_lookup
from .checkpoint import AnyBitModel, load_bit_model
from .data import BitSampler, to_bits
from .bit_token_model import BitTokenModel
from .loop_model import LoopModel
from .relations import Family
from .sets import ALL_NAMED, test_groups
from .program_model import ProgramModel
from .think_model import ThinkModel

Report = dict[str, float | None]
SURE = .99


@torch.no_grad()
def evaluate_family(model: AnyBitModel | None, family: Family, *, width: int = 8, episodes: int = 32,
                    queries: int = 16, seed: int = 50000, device: str | torch.device = 'cpu',
                    baselines: bool = True, steps: int | None = None) -> Report:
    """Fresh episodes for one family; `seed` fixes contexts and queries for every method.
    `steps` overrides the number of answer-refinement rounds of a loop model."""
    sampler = BitSampler([family], width, seed)
    report: dict[str, list[torch.Tensor]] = {}
    for _ in range(episodes):
        episode = sampler.sample(1, queries).to(device)
        if model is not None:
            prediction = (model.forward(episode.context, episode.query, width, steps=steps)
                          if isinstance(model, (LoopModel, ThinkModel, ProgramModel, BitTokenModel)) else model.forward(episode.context, episode.query, width))
            correct = prediction.values() == episode.target
            sure = prediction.confidence() >= SURE
            bits = (prediction.probabilities() > .5).float() == to_bits(episode.target, width)
            report.setdefault('exact', []).append(correct.flatten())
            report.setdefault('bit', []).append(bits.flatten())
            report.setdefault('sure', []).append(sure.flatten())
            report.setdefault('sure_correct', []).append(correct[sure])
        if baselines:
            report.setdefault('nearest', []).append((nearest(episode) == episode.target).flatten())
            for window in (1, 2):
                for high_first in (False, True):
                    key = f'window{window}{"H" if high_first else "L"}'
                    found = window_lookup(episode, window, high_first)
                    report.setdefault(key, []).append((found == episode.target).flatten())
    result: Report = {}
    for key, parts in report.items():
        values = torch.cat(parts).float()
        name = {'sure': 'sure_coverage', 'sure_correct': 'sure_accuracy'}.get(key, key)
        result[name] = float(values.mean()) if len(values) else None
    return result


def evaluate(model: AnyBitModel | None, families: Sequence[Family], **options: object) -> dict[str, Report]:
    return {f.name: evaluate_family(model, f, **options) for f in families}  # type: ignore[arg-type]


def summarise(reports: dict[str, Report]) -> dict[str, float | bool | None]:
    """Mean exact match of a group, and how often its sure answers were right (pooled by coverage).
    A tier counts as won at >= 50% exact with sure answers right >= 95% of the time."""
    exact = [r['exact'] for r in reports.values() if r.get('exact') is not None]
    covered = [(r['sure_coverage'] or 0., r['sure_accuracy']) for r in reports.values()
               if r.get('sure_coverage') is not None]
    weight = sum(c for c, _ in covered)
    sure = sum(c * (a or 0.) for c, a in covered) / weight if weight else None
    mean = sum(e for e in exact if e is not None) / len(exact) if exact else None
    return {'exact': mean, 'sure_accuracy': sure,
            'won': mean is not None and mean >= .5 and (sure is None or sure >= .95)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Exact-match accuracy of a bit-relation checkpoint')
    parser.add_argument('--load', help='checkpoint; omitted = baselines only')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--families', nargs='*', default=None, help=f'names from {sorted(ALL_NAMED)}')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--seed', type=int, default=50000)
    parser.add_argument('--no-baselines', action='store_true')
    parser.add_argument('--tests', help='use the test groups of this training set (default: the checkpoint\'s own)')
    parser.add_argument('--steps', type=int, help='refinement rounds for a loop model (default: as trained)')
    parser.add_argument('--output')
    args = parser.parse_args()
    model, metadata = load_bit_model(args.load, args.device) if args.load else (None, {})
    width = int(metadata.get('width', 8))  # type: ignore[call-overload]
    groups = test_groups(args.tests or str(metadata.get('train_set', 'base')))
    if args.families:
        groups = {'selected': tuple(ALL_NAMED[n] for n in args.families)}
    results = {group: evaluate(model, families, width=width, episodes=args.episodes, queries=args.queries,
                               seed=args.seed, device=args.device, baselines=not args.no_baselines,
                               steps=args.steps)
               for group, families in groups.items()}
    summary = {group: summarise(reports) for group, reports in results.items()} if model is not None else {}
    output = {'checkpoint': args.load, 'train_set': metadata.get('train_set', 'base'), 'tests': args.tests,
              'width': width, 'episodes': args.episodes, 'queries': args.queries, 'seed': args.seed,
              'sure_threshold': SURE, 'steps': args.steps, 'summary': summary, 'results': results}
    text = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + '\n')
    for group, reports in results.items():
        print(f'[{group}]')
        for name, report in reports.items():
            print(f'  {name:12s} ' + ' '.join(f'{k}={100 * v:5.1f}' for k, v in report.items() if v is not None))
    for group, line in summary.items():
        exact, sure = line['exact'], line['sure_accuracy']
        print(f'{group:8s} exact={100 * float(exact or 0):5.1f}%  sure-right='
              f'{"   -" if sure is None else f"{100 * float(sure):5.1f}%"}  {"WON" if line["won"] else ""}')


if __name__ == '__main__':
    main()
