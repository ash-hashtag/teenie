"""Exact-match evaluation of a bit checkpoint and non-learned controls on identical episodes."""
import argparse
import os
from collections.abc import Sequence
import json
from pathlib import Path

import torch

from .baselines import nearest, window_lookup
from .checkpoint import AnyBitModel, load_bit_model
from .data import BitSampler, to_bits
from .bit_token_model import BitTokenModel
from .loop_model import LoopModel, LoopPrediction
from .relations import Family
from .sets import ALL_NAMED, test_groups
from .program_model import ProgramModel
from .think_model import ThinkModel
from .wide_model import WideBitModel

Report = dict[str, float | None]
SURE = .99
IN_BITS = ('rounds', 'recovered', 'recovered_median', 'lookup_recovered', 'advantage')  # reported as is, not as a percentage


FLOOR = 2. ** -20  # a confidently wrong bit costs at most 20 bits, so one saturated mistake stays finite


def recovered_bits(probabilities: torch.Tensor, target: torch.Tensor, width: int) -> torch.Tensor:
    """Per question: how much the model's guess beats chance, in bits — width + log2 of the probability its per-bit
    confidences give the true answer. 0 is pure guessing, `width` is certain and right; confident mistakes go
    negative. Unlike exact match, it counts a guess that gets most bits right with honest confidence."""
    p = probabilities.double().clamp(FLOOR, 1 - FLOOR)
    truth = to_bits(target, width).double()
    return (width + (truth * p.log2() + (1 - truth) * (1 - p).log2()).sum(-1)).flatten().float()


def baseline_recovered(answers: list[torch.Tensor], targets: list[torch.Tensor], width: int) -> float:
    """Bits recovered by a method that gives one answer without confidence, scored as generously as possible: each
    bit gets the probability of being right that this method has at that bit position on these very questions
    (calibrated after the fact). So a lookup that gets a bit right half the time earns nothing for it."""
    right = (to_bits(torch.cat(answers), width) == to_bits(torch.cat(targets), width)).double()
    # Rates measured on one half of the questions score the other half (and vice versa): rates measured on the
    # questions they score would credit chance agreement, about a bit on a hash with a few hundred questions.
    scores = []
    for fit, score in ((right[0::2], right[1::2]), (right[1::2], right[0::2])):
        rate = fit.mean(0, keepdim=True).clamp(FLOOR, 1 - FLOOR)
        scores.append(width + (score * rate.log2() + (1 - score) * (1 - rate).log2()).sum(-1))
    return float(torch.cat(scores).mean())


@torch.no_grad()
def evaluate_family(model: AnyBitModel | None, family: Family, *, width: int = 8, episodes: int = 32,
                    queries: int = 16, seed: int = 50000, device: str | torch.device = 'cpu',
                    baselines: bool = True, steps: int | None = None) -> Report:
    """Fresh episodes for one family; `seed` fixes contexts and queries for every method.
    `steps` overrides the number of answer-refinement rounds of a loop model. With a model and baselines, also
    `beyond_lookup`: the model's exact rate on the questions that every lookup baseline gets wrong, and
    `lookup_fails`: how many questions those are."""
    sampler = BitSampler([family], width, seed)
    report: dict[str, list[torch.Tensor]] = {}
    answers: dict[str, list[torch.Tensor]] = {}
    targets: list[torch.Tensor] = []
    for _ in range(episodes):
        episode = sampler.sample(1, queries).to(device)
        if model is not None:
            prediction = (model.forward(episode.context, episode.query, width, steps=steps)
                          if isinstance(model, (LoopModel, ThinkModel, ProgramModel, BitTokenModel, WideBitModel)) else model.forward(episode.context, episode.query, width))
            correct = prediction.values() == episode.target
            sure = prediction.confidence() >= SURE
            bits = (prediction.probabilities() > .5).float() == to_bits(episode.target, width)
            report.setdefault('exact', []).append(correct.flatten())
            report.setdefault('bit', []).append(bits.flatten())
            report.setdefault('recovered', []).append(recovered_bits(prediction.probabilities(), episode.target, width))
            report.setdefault('sure', []).append(sure.flatten())
            report.setdefault('sure_correct', []).append(correct[sure])
            if isinstance(prediction, LoopPrediction) and (prediction.halts is not None or prediction.done is not None):
                report.setdefault('rounds', []).append(prediction.stop().flatten().float() + 1)
        if baselines:
            targets.append(episode.target.flatten())
            found = nearest(episode)
            answers.setdefault('nearest', []).append(found.flatten())
            lookup = [(found == episode.target).flatten()]
            report.setdefault('nearest', []).append(lookup[0])
            for window in (1, 2):
                for high_first in (False, True):
                    key = f'window{window}{"H" if high_first else "L"}'
                    found = window_lookup(episode, window, high_first)
                    answers.setdefault(key, []).append(found.flatten())
                    lookup.append((found == episode.target).flatten())
                    report.setdefault(key, []).append(lookup[-1])
            if model is not None:
                # Questions no lookup method answers: what the model gets there cannot come from copying examples.
                hard = ~torch.stack(lookup).any(0)
                report.setdefault('lookup_fails', []).append(hard)
                report.setdefault('beyond_lookup', []).append(correct.flatten()[hard])
    result: Report = {}
    for key, parts in report.items():
        values = torch.cat(parts).float()
        name = {'sure': 'sure_coverage', 'sure_correct': 'sure_accuracy'}.get(key, key)
        result[name] = float(values.mean()) if len(values) else None
        if key == 'recovered' and len(values):
            result['recovered_median'] = float(values.median())
    if answers:
        # The best lookup's bits recovered, and how far the model's guess goes beyond it.
        result['lookup_recovered'] = max(baseline_recovered(parts, targets, width) for parts in answers.values())
        if result.get('recovered') is not None:
            result['advantage'] = result['recovered'] - result['lookup_recovered']
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

    def average(key: str) -> float | None:
        values = [r[key] for r in reports.values() if r.get(key) is not None]
        return sum(values) / len(values) if values else None
    return {'exact': mean, 'sure_accuracy': sure, 'recovered': average('recovered'),
            'lookup_recovered': average('lookup_recovered'), 'advantage': average('advantage'),
            'won': mean is not None and mean >= .5 and (sure is None or sure >= .95)}


def be_gentle_on_cpu() -> None:
    """CPU evaluations run beside training and other work: use a few threads (TEENIE_CPU_THREADS, default 4) at
    lower priority instead of every core."""
    torch.set_num_threads(int(os.environ.get('TEENIE_CPU_THREADS', '4')))
    os.nice(10)


def main() -> None:
    parser = argparse.ArgumentParser(description='Exact-match accuracy of a bit-relation checkpoint')
    parser.add_argument('--load', help='checkpoint; omitted = baselines only')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--families', nargs='*', default=None, help='names from ' + str(sorted(ALL_NAMED)).replace('%', '%%'))
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--seed', type=int, default=50000)
    parser.add_argument('--no-baselines', action='store_true')
    parser.add_argument('--tests', help='use the test groups of this training set (default: the checkpoint\'s own)')
    parser.add_argument('--steps', type=int, help='refinement rounds for a loop model (default: as trained)')
    parser.add_argument('--width', type=int, help='bits per number for a width-free model (default: as trained)')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.device == 'cpu':
        be_gentle_on_cpu()
    model, metadata = load_bit_model(args.load, args.device) if args.load else (None, {})
    width = args.width or int(metadata.get('width', 8))  # type: ignore[call-overload]
    if args.width and model is not None and not isinstance(model, WideBitModel) and args.width != int(metadata.get('width', 8)):  # type: ignore[call-overload]
        raise SystemExit('--width other than the trained width needs a width-free model')
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
            print(f'  {name:12s} ' + ' '.join(f'{k}={v:5.2f}' if k in IN_BITS else f'{k}={100 * v:5.1f}'
                                              for k, v in report.items() if v is not None))
    for group, line in summary.items():
        exact, sure = line['exact'], line['sure_accuracy']
        bits = '' if line.get('recovered') is None else f'  recovered={line["recovered"]:5.2f} bits' + (
            '' if line.get('advantage') is None else f' (lookup {line["lookup_recovered"]:5.2f}, advantage {line["advantage"]:+5.2f})')
        print(f'{group:8s} exact={100 * float(exact or 0):5.1f}%  sure-right='
              f'{"   -" if sure is None else f"{100 * float(sure):5.1f}%"}{bits}  {"WON" if line["won"] else ""}')


if __name__ == '__main__':
    main()
