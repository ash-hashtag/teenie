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
from .wide_model import WideBitModel
from .checkpoint import load_bit_model
from .eval import Report, summarise
from .sets import TARGET, TIERS, regroup

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
    ('bt-aligned', 'Bit tokens', 'bit tokens, each bit reads its own bit of the examples',
     'experiments/bt/bt-aligned-eval.json', ('experiments/bt/bt-aligned-train.jsonl',)),
    ('bt-pairs', 'Bit tokens', 'bit tokens + every operation pair equally often (mix5)',
     'experiments/bt/bt-pairs-eval.json', ('experiments/bt/bt-pairs-train.jsonl',)),
    ('bt-loops31', 'Bit tokens', 'bit tokens, training loops up to 31 iterations (mix6)',
     'experiments/bt/bt-loops31-eval.json', ('experiments/bt/bt-loops31-train.jsonl',)),
    ('bt-width12', 'Bit tokens', 'bit tokens, trained on 12-bit numbers',
     'experiments/bt/bt-width12-eval.json', ('experiments/bt/bt-width12-train.jsonl',)),
    ('bt-hintfade', 'Bit tokens', 'bit tokens, step-by-step hints faded out',
     'experiments/bt/bt-hintfade-eval.json', ('experiments/bt/bt-hintfade-train.jsonl',)),
    ('bt-mix7', 'More concepts', 'bit tokens + division, remainder, comparisons, variable shifts, squares, bit counts',
     'experiments/bt/bt-mix7-eval.json', ('experiments/bt/bt-mix7-train.jsonl',)),
    ('wide-mix', 'Wide numbers', 'width-free bit tokens, 8/12/16 bits — scored at 16 bits',
     'experiments/wide/wide-mix-w16-eval.json', ('experiments/wide/wide-mix-train.jsonl',)),
    ('wide-32', 'Wide numbers', '+ 24- and 32-bit episodes — scored at 32 bits',
     'experiments/wide/wide-32-w32-eval.json', ('experiments/wide/wide-32-train.jsonl',)),
    ('wide-ps16', 'Composition at 16 bits', 'product-then-step only, from the 32-bit model',
     'experiments/wide/wide-ps16-eval.json', ('experiments/wide/wide-ps16-train.jsonl',)),
    ('wide-ps16-peers', 'Composition at 16 bits', 'same, questions also read deep probes',
     'experiments/wide/wide-ps16-peers-eval.json', ('experiments/wide/wide-ps16-peers-train.jsonl',)),
    ('wide-near16', 'Composition at 16 bits', 'four fixed product-then-step relations only (diagnostic)',
     'experiments/wide/wide-near16-eval.json', ('experiments/wide/wide-near16-train.jsonl',)),
    ('wide-pool16', 'Composition at 16 bits', 'a pool of 16 product-then-step relations',
     'experiments/wide/wide-pool16-eval.json', ('experiments/wide/wide-pool16-train.jsonl',)),
    ('wide-grid', 'Composition at 16 bits', 'composition grid: 52 combinations trained, 12 held out',
     'experiments/wide/wide-grid-eval.json', ('experiments/wide/wide-grid-train.jsonl',)),
    ('wide-grid-code', 'Composition at 16 bits', 'same grid, answers read only a relation code',
     'experiments/wide/wide-grid-code-eval.json', ('experiments/wide/wide-grid-code-train.jsonl',)),
]

# Architectures shown on the site besides the current model: (id, title, checkpoint).
ARCHITECTURES = [('bittoken', 'Bit tokens', 'checkpoints/steps/bittoken-mix4.pt')]

# The width-free model at 16 and 32 bits, next to the best of five lookup baselines on the same episodes (seed 60000,
# sealed 61000): (width, model evaluation, model sealed evaluation, baseline files).
WIDE_CHECKPOINT = 'checkpoints/wide/wide-32.pt'
WIDE = [(16, 'experiments/wide/wide-32-w16-eval.json', 'experiments/sealed/wide-32-w16-sealed.json',
         ('experiments/sealed/baselines-w16-sealed.json', 'experiments/wide/baselines-w16-target.json',
          'experiments/wide/baselines-w16-T4.json', 'experiments/wide/baselines-w16-T2T3.json')),
        (32, 'experiments/wide/wide-32-w32-eval.json', 'experiments/sealed/wide-32-w32-sealed.json',
         ('experiments/sealed/baselines-w32-sealed.json', 'experiments/wide/baselines-w32-target.json',
          'experiments/wide/baselines-w32-T4.json', 'experiments/wide/baselines-w32-near.json',
          'experiments/wide/baselines-w32-T2T3.json'))]

# The current model: the width-free model the Model page describes (parameters from the checkpoint).
CURRENT_WIDE = 'checkpoints/wide/wide-mix7-chains.pt'

# The middle band (opaque.py) at 32 bits, scored by bits recovered on the same episodes (seed 80000, first 8): the
# release model, the older general model, the best lookup and models fitted to each task's 999 examples.
# Every test group at 32 bits: the release model (with bits recovered) and the older general model (exact only).
GROUPS_EVAL = {'current': 'experiments/wide/wide-mix7-chains-w32-eval.json', 'older': 'experiments/wide/wide-32-w32-eval.json'}
GROUP_ORDER = [
    ('heldout', 'Held-out single operations', 'Simple relations it never trained on.'),
    ('near', 'A product, then a step', 'A product of a and b followed by one more operation, never trained.'),
    ('hidden', 'Hidden compositions', 'Combinations of trained operations that training never produces.'),
    ('T4', 'T4 · new operation types', 'Operations absent from training: division, remainder, rotations, and more.'),
    ('T1', 'T1 · 2 operations', 'Random 2-operation expressions containing a pair of operations never seen together.'),
    ('T2', 'T2 · 3-4 operations', 'The same with 3 or 4 operations.'),
    ('T3', 'T3 · 5-7 operations', 'The same with 5 to 7 operations.'),
    ('T5a', 'T5a · unseen loop bodies', 'Loops whose step it never trained on; n = b & 15.'),
    ('T5b', 'T5b · longer loops', 'Loops longer than any it trained on: n = b & 63 or all of b.'),
    ('target', 'Target · unseen multi-step expressions', 'The expression the project aims at, and random 4-6 '
                                                         'operation expressions over every concept.'),
]


def groups(root: Path) -> list[dict[str, object]] | None:
    """Each test group with its scores and, per relation, a few example equations: the front page's expandable table."""
    import torch
    from .sets import test_groups
    if not all((root / path).exists() for path in GROUPS_EVAL.values()):
        return None
    current = json.loads((root / GROUPS_EVAL['current']).read_text())
    older = json.loads((root / GROUPS_EVAL['older']).read_text())
    families = test_groups('mix7')
    rows = []
    for key, title, what in GROUP_ORDER:
        summary = current['summary'].get(key)
        if summary is None:
            continue
        relations = []
        for family in families.get(key, ()):
            report = current['results'][key].get(family.name, {})
            drawn: list[str] = []
            for seed in range(30):
                if len(drawn) == 5:
                    break
                name = _pretty(family.make(torch.Generator().manual_seed(2000 + seed), 32).name)
                if name not in drawn:
                    drawn.append(name)
            relations.append({'name': _pretty(family.name), 'examples': drawn, 'exact': report.get('exact'),
                              'recovered': report.get('recovered'), 'lookup': report.get('lookup_recovered')})
        rows.append({'key': key, 'title': title, 'what': what, 'exact': summary.get('exact'),
                     'recovered': summary.get('recovered'), 'lookup': summary.get('lookup_recovered'),
                     'sure_accuracy': summary.get('sure_accuracy'),
                     'older': older['summary'].get(key, {}).get('exact'), 'relations': relations})
    return rows


RELEASE_CHECK = 'experiments/wide/checks-wide-mix7-chains/check-85000.json'  # products and targets at 32 bits

MIDDLE = {
    'model': 'experiments/wide/checks-wide-mix7-chains/opaque-85000.json',
    'older': 'experiments/wide/opaque-wide-32-w32.json',
    'learners': 'experiments/baselines/learners-opaque-w32.json',
}


BASICS = {'add': 'a + b', 'sub': 'a - b', 'b-a': 'b - a', 'xor': 'a ^ b', 'and': 'a & b', 'or': 'a | b',
          'a+2b': 'a + 2b', '2a+b': '2a + b', 'max': 'max(a, b)', 'a<<1': 'a << 1', 'a+const': 'a + K (a random K)',
          'b^const': 'b ^ K (a random K)', 'transducer': 'a random carry machine (like a + b with a twist)',
          'a*(b&63)': 'a * (b & 63)', 'a*b': 'a * b', 'a+b+1': 'a + b + 1', 'a-2b': 'a - 2b', 'min': 'min(a, b)',
          'nand': '~(a & b)', '3a': '3a', 'mul': 'a * b'}


LOOPS_READ = {'sum a..a+n-1': 'a + (a+1) + … + (a+n-1)', 'a^n': 'a to the power n', 'fib(a,a)':
              'Fibonacci-like from a, a: x, y = x + y, x, n times', 'x->3x+1': 'x = 3x + 1, n times, from x = a',
              'acc=(acc<<1)^a': 'acc = (acc << 1) ^ a, n times', 'popcount(a|b)': 'number of 1 bits in a | b',
              'reverse(a^b)': 'the bits of a ^ b in reverse order', 'popcount(a)*b': '(number of 1 bits in a) * b',
              'accumulate': 'acc = step(acc, a, i), n times', 'repeat': 'x = step(x), n times, from x = a',
              'recurrence': 'x, y = step(x, y), x, n times'}
COUNTS = {'n15': 'n = b & 15', 'n31': 'n = b & 31', 'n63': 'n = b & 63', 'n255': 'n = b'}


def _pretty(name: str) -> str:
    """Big constants in hex (0x94AD9A8E reads better than 2494371982); named loops read out."""
    import re
    if name in BASICS:
        return BASICS[name]
    words = name.rsplit(' ', 1)
    if len(words) == 2 and words[1] in COUNTS and words[0] in LOOPS_READ:
        return f'{LOOPS_READ[words[0]]}  ({COUNTS[words[1]]})'
    if name in LOOPS_READ:
        return f'{name}: {LOOPS_READ[name]}'
    return re.sub(r'\b\d{5,}\b', lambda m: f'0x{int(m.group()):X}', name)


def examples() -> list[dict[str, object]]:
    """What each training and test group contains, as equations: fixed relations by name, generated families by a few
    draws (fixed seeds, 32 bits). For the site's 'what it is tested on' explorer."""
    import torch
    from .expressions import HIDDEN_FAMILIES, NEAR_FAMILIES
    from .opaque import OPAQUE
    from .relations import HELDOUT_FAMILIES, MULTIPLY, TRAIN_FAMILIES
    from .sets import (CHAINS, CHAINS_UNSEEN, CHAINS_UNSEEN2, CONCEPT_STEPS, LOOPS, PRODUCT_CHAINS,
                       PRODUCT_CHAINS_UNSEEN, PRODUCT_STEP, STEPS3, TARGET, TIERS)

    def draws(family, count: int) -> list[str]:
        found: list[str] = []
        for seed in range(40):
            if len(found) == count:
                break
            name = _pretty(family.make(torch.Generator().manual_seed(1000 + seed), 32).name)
            if name not in found:
                found.append(name)
        return found

    def section(key: str, title: str, kind: str, what: str, families, count: int = 3) -> dict[str, object]:
        return {'id': key, 'title': title, 'kind': kind, 'what': what,
                'families': [{'name': _pretty(f.name), 'examples': draws(f, count)} for f in families]}

    loops_note = 'n is read from b: n = b & 15 for loops it trained on'
    sections = [
        section('basics', 'Single operations', 'trained', 'One operation of a and b, the building blocks.',
                TRAIN_FAMILIES + (MULTIPLY,), 1),
        section('steps', 'Random expressions, 1-5 operations', 'trained',
                'A fresh random expression every episode over + - * ^ & | min max << >> ~ and constants.', STEPS3),
        section('concepts', 'Expressions with more concepts', 'trained',
                'The same, adding division, remainder, comparisons, shifts by a variable, squares and bit counts.',
                CONCEPT_STEPS),
        section('chains', 'Chains', 'trained', 'A start (a two-number operation) followed by one or two steps, with fresh '
                'random constants; product chains start from a product.', (CHAINS, PRODUCT_CHAINS, PRODUCT_STEP)),
        section('loops', 'Loops', 'trained', f'A small step repeated n times; {loops_note}.', LOOPS),
        section('heldout', 'Held-out single operations', 'tested', 'Simple relations it never trained on.',
                HELDOUT_FAMILIES, 1),
        section('T1', 'T1-T3: unseen expressions', 'tested', 'Random expressions of 2 (T1), 3-4 (T2) and 5-7 (T3) '
                'operations, each containing a pair of operations never seen together in training.',
                TIERS['T1'] + TIERS['T2'] + TIERS['T3']),
        section('T4', 'T4: new operation types', 'tested', 'Operations absent from training.', TIERS['T4'], 1),
        section('T5', 'T5: unseen loops', 'tested', 'Loop bodies never trained (T5a, n = b & 15) and longer loops than '
                'trained (T5b, n = b & 63 or all of b).', TIERS['T5a'] + TIERS['T5b'], 1),
        section('hidden', 'Hidden compositions', 'tested', 'Combinations of trained operations that training never '
                'produces, such as a product followed by a step.', HIDDEN_FAMILIES + NEAR_FAMILIES, 1),
        section('unseen-chains', 'Chains with an unseen first step', 'tested', 'A start and a first step never paired '
                'in training.', (CHAINS_UNSEEN, CHAINS_UNSEEN2, PRODUCT_CHAINS_UNSEEN)),
        section('target', 'The target', 'tested', 'The expression the project aims at, and random 4-6 operation '
                'expressions over every concept.', TARGET),
    ]
    descriptions = {
        'arx ladder': 'Add-rotate-xor rounds with random round keys each episode: x = rotl(x ^ k, 7) + y; y = rotl(y, 3) ^ x.',
        'random': 'A fresh random expression of 6-10 operations each episode, longer than anything trained.',
        'sbox': 'A random 4-bit substitution table applied to every nibble of a ^ b, then added to a.',
        'data-dependent': 'Steps chosen by the data: Collatz-like loops and branches on a bit of a.',
        'automaton': 'Cellular automaton rules 30 and 110 run for 2 or 4 steps on the bits of a, then xor b.',
        'mixes': 'Rotations (never a training operation), xor and multiplication by odd constants.',
        'deep': 'Expressions of 4-8 operations with fixed constants.',
        'floor: products': 'Squares and products of derived values (scored, but not the headline).',
        'loops': 'Loops it never trained on.',
        'computed products': 'Products of values computed from a and b: its known weak spot.',
        'control: hash': 'Hash functions and many mixing rounds: nothing should beat chance here.',
    }
    for group, families in OPAQUE.items():
        sections.append(section(f'middle-{group}', f'Middle band: {group}', 'middle band', descriptions.get(group, ''),
                                families, 2 if group in ('random',) else 1))
    return sections


def release(root: Path) -> dict[str, object] | None:
    path = root / RELEASE_CHECK
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {'width': data['width'], 'episodes': data['episodes'],
            'results': {name: {'exact': r.get('exact'), 'recovered': r.get('recovered')}
                        for name, r in data['results']['selected'].items()}}


def middle(root: Path) -> dict[str, object] | None:
    if not all((root / path).exists() for path in MIDDLE.values()):
        return None
    model = json.loads((root / MIDDLE['model']).read_text())
    older = json.loads((root / MIDDLE['older']).read_text())
    learners = json.loads((root / MIDDLE['learners']).read_text())['results']
    groups = []
    for group, families in model['results'].items():
        if group not in learners:
            continue
        rows = []
        for name, report in families.items():
            fitted = learners[group].get(name, {})
            rows.append({'name': name, 'model': report.get('recovered'), 'exact': report.get('exact'),
                         'older': older['results'].get(group, {}).get(name, {}).get('recovered'),
                         'lookup': report.get('lookup_recovered'),
                         'trees': fitted.get('lgbm', {}).get('recovered'), 'mlp': fitted.get('mlp', {}).get('recovered')})
        average = lambda key: _mean([r[key] for r in rows if r[key] is not None])
        groups.append({'group': group, 'rows': rows, **{key: average(key) for key in
                                                          ('model', 'older', 'lookup', 'trees', 'mlp')}})
    return {'width': model['width'], 'episodes': model['episodes'], 'checkpoint': model['checkpoint'], 'groups': groups}

# Product chains at 16 bits as training progressed (held-out start-step pairs; chain_split.py), oldest first.
PRODUCT_STAGES = [
    ('General model, no chain training', 'experiments/wide/chains-split-wide-32.json'),
    ('Trained on chains', 'experiments/wide/chains-split-wide-chains.json'),
    ('Product-heavy chains', 'experiments/wide/chains-split-wide-chains-products-w16.json'),
    ('Scratch registers', 'experiments/wide/chains-split-wide-chains-registers-w16.json'),
    ('Scratch registers, trained longer', 'experiments/wide/chains-split-wide-chains-registers-long-w16.json'),
    ('32-bit episodes, 12 rounds', 'experiments/wide/chains-split-wide-chains-rounds12-w16.json'),
]

LADDER = ('target', 'T1', 'T2', 'T3', 'T4', 'T5a', 'T5b', 'hidden', 'near')
HIGHLIGHTS = ('a*b', 'a*b+a', 'min', '3a', 'transducer')


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _tier_names() -> dict[str, list[str]]:
    names = {tier: [f.name for f in families] for tier, families in TIERS.items()}
    names['hidden'] = ['a*b+a', 'a*a', '(a+1)*(b+1)', '(a^b)*3', '|a-b|', '(a+b)>>1', '(a*b)^b', '(a<<1)+(b>>1)']
    names['near'] = ['a*b+b', 'a*b+5', 'a*b-a', '(a*b)^a']
    names['target'] = [f.name for f in TARGET]
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


def _best_lookup(report: Report) -> float:
    return max(float(v) for k, v in report.items() if v is not None and k in
               ('nearest', 'window1L', 'window1H', 'window2L', 'window2H'))


def wide(root: Path) -> dict[str, object]:
    """Exact match per relation and per group at each width, with the best lookup baseline where one was run."""
    widths: dict[str, object] = {}
    for width, evaluation, sealed, baselines in WIDE:
        if not (root / evaluation).exists():
            continue
        results = regroup(json.loads((root / evaluation).read_text())['results'])
        lookup: dict[str, float] = {}
        for path in baselines:
            if (root / path).exists():
                for reports in json.loads((root / path).read_text())['results'].values():
                    lookup |= {name: _best_lookup(r) for name, r in reports.items()}
        entry: dict[str, object] = {
            'groups': {group: summarise(reports) for group, reports in results.items()},
            'families': {name: r.get('exact') for reports in results.values() for name, r in reports.items()},
            'lookup': lookup}
        chains = root / f'experiments/wide/chains-split-wide-mix7-chains-w{width}.json'  # the release model
        if chains.exists():
            entry['chains'] = json.loads(chains.read_text())['summary']
        if (root / sealed).exists():
            reports = json.loads((root / sealed).read_text())['results']['sealed']
            entry['sealed'] = summarise(reports)
            entry['sealed_lookup'] = (sum(lookup[n] for n in reports if n in lookup) / len(reports)
                                      if all(n in lookup for n in reports) else None)
        widths[str(width)] = entry
    parameters = None
    if (root / WIDE_CHECKPOINT).exists():
        model, _ = load_bit_model(root / WIDE_CHECKPOINT)
        parameters = count(model)
    return {'checkpoint': WIDE_CHECKPOINT, 'parameters': parameters, 'widths': widths}


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def wide_parts(model: WideBitModel, checkpoint: str, update: object) -> dict[str, object]:
    config = model.config
    registers, peers = config.get('registers', 0), config.get('peers', False)
    parts = [
        {'part': 'Example reader', 'detail': 'each example bit seen relative to its position and from the lowest bit '
         '(a, b, c); bilinear layer computed as a convolution, then mixing', 'parameters': count(model.example)},
        {'part': 'Question reader', 'detail': f'a, b, the current guess{f" and {registers} scratch registers" if registers else ""}, '
         'seen the same way', 'parameters': count(model.ask)},
        *[{'part': f'Bit layer {i + 1}', 'detail': 'answer bits attend to each other (bias per bit distance), read the '
           f'same bit of all 999 examples ({config["heads"]} heads), gated feed-forward', 'parameters': count(layer)}
          for i, layer in enumerate(model.layers)],
        {'part': 'Example keys and values', 'detail': f'one pair per bit layer', 'parameters': count(model.keys) + count(model.values)},
    ]
    if peers:
        parts.append({'part': 'Worked-example reading', 'detail': f'answer bits read {config["probes"]} worked examples '
                      'and their true answers, after each bit layer', 'parameters': count(model.peers)})
    parts += [
        {'part': 'Memory', 'detail': 'GRU cell per answer bit, carried to the next round', 'parameters': count(model.cell)},
        {'part': 'Readout', 'detail': f'each answer bit → its answer bit{f" and {registers} register bits" if registers else ""}',
         'parameters': count(model.readout)},
    ]
    return {'checkpoint': checkpoint, 'update': update, 'hidden': config['hidden'], 'heads': config['heads'],
            'layers': config['layers'], 'rounds': config['steps'], 'span': config['span'], 'registers': registers,
            'worked_examples': config['probes'] if peers else 0, 'parameters': count(model), 'parts': parts}


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
    current = None
    if (root / CURRENT_WIDE).exists():
        model, metadata = load_bit_model(root / CURRENT_WIDE)
        if isinstance(model, WideBitModel):
            current = wide_parts(model, CURRENT_WIDE, metadata.get('update'))
    products = [{'stage': stage, 'summary': json.loads((root / path).read_text())['summary']}
                for stage, path in PRODUCT_STAGES if (root / path).exists()]
    Path(args.out).write_text(json.dumps({'ladder': list(LADDER), 'tiers': tiers, 'experiments': found,
                                          'architectures': architectures, 'wide': wide(root),
                                          'products': products, 'current': current, 'middle': middle(root),
                                          'examples': examples(), 'release': release(root),
                                          'groups': groups(root)}))
    print(f'exported {len(found)} experiments to {args.out}')


if __name__ == '__main__':
    main()
