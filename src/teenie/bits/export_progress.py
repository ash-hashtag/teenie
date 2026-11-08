"""Collect every experiment's evaluation and training log into the docs site's data.

uv run python -m teenie.bits.export_progress     # writes site/src/data/progress.json

Add a line to EXPERIMENTS when a new run finishes; files that do not exist yet are skipped, so
runs in progress simply do not appear.
"""
import argparse
import json
from pathlib import Path

from torch import nn

from .bit_token_model import BitTokenModel
from .checkpoint import load_bit_model
from .eval import Report, summarise
from .sets import TIERS, regroup

# (id, chapter, title, evaluation file, training logs in order)
EXPERIMENTS: list[tuple[str, str, str, str, tuple[str, ...]]] = [
    ('serial', 'Bit by bit', 'Bit-serial model (reads one bit position at a time)',
     'experiments/bits-seed0-eval.json', ('experiments/bits-seed0-train.jsonl',)),
    ('loop', 'All at once', 'All bits at once, loop on the answer',
     'experiments/bits-loop-seed0-eval.json', ('experiments/bits-loop-seed0-train.jsonl',)),
    ('mul-loop', 'Multiplication', 'a*b in training, plain loop',
     'experiments/mul/mul-loop-eval.json', ('experiments/mul/mul-loop-train.jsonl',)),
    ('mulonly-loop', 'Multiplication', 'a*b only, plain loop (capacity check)',
     'experiments/mul/mulonly-loop-eval.json', ('experiments/mul/mulonly-loop-train.jsonl',)),
    ('mulonly-mloop', 'Multiplication', 'a*b only, multiplicative units (capacity check)',
     'experiments/mul/mulonly-mloop-eval.json', ('experiments/mul/mulonly-mloop-train.jsonl',)),
    ('mul-mloop', 'Multiplication', 'a*b in training, multiplicative units',
     'experiments/mul/mul-mloop-eval.json', ('experiments/mul/mul-mloop-train.jsonl',)),
    ('mul-mloop-clock8', 'Multiplication', 'multiplicative units + round counter',
     'experiments/mul/mul-mloop-clock8-eval.json', ('experiments/mul/mul-mloop-clock8-train.jsonl',)),
    ('blocks-mloop', 'Multiplication', 'only pieces of multiplication trained (a*b unseen)',
     'experiments/mul/blocks-mloop-eval.json', ('experiments/mul/blocks-mloop-train.jsonl',)),
    ('mix', 'Expressions', 'random 1-3 operation expressions + adaptive sampling',
     'experiments/mix/mix-mloop-eval.json',
     ('experiments/mix/mix-mloop-train.jsonl', 'experiments/mix/mix-mloop-cont-train.jsonl')),
    ('mix2', 'Expressions', 'more steps: 1-5 operations, more products',
     'experiments/mix2/mix2-mloop-eval.json', ('experiments/mix2/mix2-mloop-train.jsonl',)),
    ('mix2-clock12', 'Expressions', 'more steps + round counter, 12 rounds',
     'experiments/mix2/mix2-mloop-clock12-eval.json', ('experiments/mix2/mix2-mloop-clock12-train.jsonl',)),
    ('nearonly', 'Expressions', '"product, then another step" only (capacity check)',
     'experiments/mix2/nearonly-mloop-eval.json', ('experiments/mix2/nearonly-mloop-train.jsonl',)),
    ('ladder-baseline', 'Loops and the ladder', 'mix2 model on the ladder (never saw a loop)',
     'experiments/mix3/baseline-mix2-mloop-eval.json', ()),
    ('mix3', 'Loops and the ladder', 'trained with loops and held-out combinations',
     'experiments/mix3/mix3-mloop-eval.json', ('experiments/mix3/mix3-mloop-train.jsonl',)),
    ('mix3-scratch4', 'Loops and the ladder', '+ 4 scratch slots per question',
     'experiments/mix3/mix3-mloop-scratch4-eval.json', ('experiments/mix3/mix3-mloop-scratch4-train.jsonl',)),
    ('nearonly-scratch4', 'Bit tokens', '"product, then a step" only, loop + scratch slots (capacity check)',
     'experiments/arch/nearonly-scratch4-eval.json', ('experiments/arch/nearonly-scratch4-train.jsonl',)),
    ('nearonly-bittoken', 'Bit tokens', '"product, then a step" only, bit tokens (capacity check)',
     'experiments/arch/nearonly-bittoken-eval.json', ('experiments/arch/nearonly-bittoken-train.jsonl',)),
    ('loop-steps', 'Step by step', 'loop, step-by-step training, width 96 (mix4)',
     'experiments/steps/loop-steps-eval.json', ('experiments/steps/loop-steps-train.jsonl',)),
    ('bittoken-steps', 'Step by step', 'bit tokens, step-by-step training, width 96 (mix4)',
     'experiments/steps/bittoken-steps-eval.json', ('experiments/steps/bittoken-steps-train.jsonl',)),
    ('bittoken-mix4', 'Bit tokens', 'bit tokens, final answers only (mix4) — current model',
     'experiments/steps/bittoken-mix4-eval.json', ('experiments/steps/bittoken-mix4-train.jsonl',)),
    ('bt-seed1', 'Bit tokens', 'bit tokens, second seed',
     'experiments/bt/bt-seed1-eval.json', ('experiments/bt/bt-seed1-train.jsonl',)),
    ('bt-continue', 'Bit tokens', 'bit tokens, trained 10,000 updates longer',
     'experiments/bt/bt-continue-eval.json', ('experiments/bt/bt-continue-train.jsonl',)),
    ('bt-rounds16', 'Bit tokens', 'bit tokens, fine-tuned with 16 rounds',
     'experiments/bt/bt-rounds16-eval.json', ('experiments/bt/bt-rounds16-train.jsonl',)),
]

# Architectures shown on the site besides the current model: (id, title, checkpoint).
ARCHITECTURES = [('bittoken', 'Bit tokens', 'checkpoints/steps/bittoken-mix4.pt')]

LADDER = ('T1', 'T2', 'T3', 'T4', 'T5a', 'T5b', 'hidden', 'near')
HIGHLIGHTS = ('a*b', 'a*b+a', 'min', '3a', 'transducer')


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _tier_names() -> dict[str, list[str]]:
    names = {tier: [f.name for f in families] for tier, families in TIERS.items()}
    names['hidden'] = ['a*b+a', 'a*a', '(a+1)*(b+1)', '(a^b)*3', '|a-b|', '(a+b)>>1', '(a*b)^b', '(a<<1)+(b>>1)']
    names['near'] = ['a*b+b', 'a*b+5', 'a*b-a', '(a*b)^a']
    return names


def experiment(identifier: str, chapter: str, title: str, evaluation: Path, logs: list[Path]) -> dict[str, object]:
    data = json.loads(evaluation.read_text())
    results: dict[str, dict[str, Report]] = regroup(data['results'])
    groups = {group: summarise(reports) for group, reports in results.items()}
    flat = {name: report for reports in results.values() for name, report in reports.items()}
    highlights = {name: flat[name].get('exact') for name in HIGHLIGHTS if name in flat}
    curve: list[dict[str, object]] = []
    tiers = _tier_names()
    for log in logs:
        for line in log.read_text().splitlines():
            record = json.loads(line)
            exact: dict[str, float] = record['exact']
            point: dict[str, object] = {'update': record['update'], 'known': record['known_exact']}
            for tier, members in tiers.items():
                values = [exact[m] for m in members if m in exact and exact[m] is not None]
                if values:
                    point[tier] = _mean(values)
            curve.append(point)
    return {'id': identifier, 'chapter': chapter, 'title': title, 'train_set': data.get('train_set'),
            'tests': data.get('tests'), 'groups': groups, 'highlights': highlights,
            'families': {name: {k: report.get(k) for k in ('exact', 'bit', 'sure_coverage', 'sure_accuracy')}
                         for name, report in flat.items()},
            'curve': curve}


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def bit_token_parts(model: BitTokenModel) -> dict[str, object]:
    config = model.config
    layers = list(model.layers)
    parts = [
        {'part': 'Example encoder', 'detail': '24 bits → bilinear layer → GELU → linear → norm (run once)',
         'parameters': count(model.example)},
        {'part': 'Example keys and values', 'detail': f'one projection pair per layer ({config["layers"]})',
         'parameters': count(model.keys) + count(model.values)},
        {'part': 'Question input', 'detail': '(a, b, current guess) → bilinear layer, plus 8 bit-position vectors',
         'parameters': count(model.ask) + model.position.numel()},
        *[{'part': f'Bit layer {i + 1}', 'detail': 'bit tokens attend to each other (bias per bit distance), '
           f'read the examples ({config["heads"]} heads), gated feed-forward', 'parameters': count(layer)}
          for i, layer in enumerate(layers)],
        {'part': 'Memory', 'detail': 'GRU cell per bit token, carried to the next round', 'parameters': count(model.cell)},
        {'part': 'Readout', 'detail': 'each bit token → its own bit', 'parameters': count(model.readout)},
    ]
    if config.get('stepwise', False):
        parts.append({'part': 'Done signal', 'detail': '"the steps are finished"', 'parameters': count(model.done)})
    return {'hidden': config['hidden'], 'heads': config['heads'], 'layers': config['layers'], 'rounds': config['steps'],
            'width': config['width'], 'parameters': count(model), 'parts': parts}


def main() -> None:
    parser = argparse.ArgumentParser(description='Export experiment progress for the docs site')
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default='site/src/data/progress.json')
    args = parser.parse_args()
    root = Path(args.root)
    found = []
    for identifier, chapter, title, evaluation, logs in EXPERIMENTS:
        path = root / evaluation
        if path.exists():
            found.append(experiment(identifier, chapter, title, path, [root / log for log in logs if (root / log).exists()]))
    tiers = {tier: [f.name for f in families] for tier, families in TIERS.items()}
    architectures = {}
    for identifier, title, checkpoint in ARCHITECTURES:
        if (root / checkpoint).exists():
            model, _ = load_bit_model(root / checkpoint)
            if isinstance(model, BitTokenModel):
                architectures[identifier] = {'title': title, 'checkpoint': checkpoint, **bit_token_parts(model)}
    Path(args.out).write_text(json.dumps({'ladder': list(LADDER), 'tiers': tiers, 'experiments': found,
                                          'architectures': architectures}))
    print(f'exported {len(found)} experiments to {args.out}')


if __name__ == '__main__':
    main()
