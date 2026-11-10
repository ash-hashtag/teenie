"""Per-task learners on the same 999 examples, scored by bits recovered: the fair comparison for Teenie.

Teenie reads the examples in one forward pass with frozen weights. These baselines instead fit a fresh model to each
episode's 999 examples (64 input bits: a and b) and predict each answer bit's probability for the same questions:

- lgbm: gradient-boosted trees, one classifier per answer bit
- mlp: a two-layer neural network predicting all answer bits at once

Episodes match `teenie-bits-eval` exactly (same family, width, seed, episodes, queries), so the numbers sit next to the
model's. Fitted probabilities can be overconfident, which bits recovered punishes, so each learner is calibrated on
its own examples: fit on 800 of the 999, choose per answer bit how far to pull its probabilities toward 1/2
(p' = 1/2 + s (p - 1/2), s in 0..1) to score best on the other 199, then refit on all 999 and apply that pull. s = 0
means "this bit is a coin toss", which scores exactly chance, so a calibrated learner never does worse than chance on
average: generous to the baselines.

uv run --extra baselines python -m teenie.bits.learners --tests opaque --width 32 --episodes 8 --seed 80000 \
    --output experiments/baselines/learners-opaque-w32.json
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from .data import BitSampler, to_bits
from .eval import FLOOR, be_gentle_on_cpu, recovered_bits
from .sets import ALL_NAMED, test_groups


def _features(pairs: torch.Tensor, width: int) -> np.ndarray:
    """[N, 2] (a, b) -> [N, 2 * width] bits."""
    return torch.cat([to_bits(pairs[:, 0], width), to_bits(pairs[:, 1], width)], -1).float().numpy()


def _constant(column: np.ndarray, rows: int) -> np.ndarray:
    """A bit that never varies in the examples: predict its value, softened as (count + 1/2) / (n + 1)."""
    return np.full(rows, (column.sum() + .5) / (len(column) + 1))


def fit_lgbm(x: np.ndarray, y: np.ndarray, queries: np.ndarray) -> np.ndarray:
    from lightgbm import LGBMClassifier
    out = np.empty((len(queries), y.shape[1]))
    for k in range(y.shape[1]):
        if y[:, k].min() == y[:, k].max():
            out[:, k] = _constant(y[:, k], len(queries))
            continue
        model = LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=.1, min_child_samples=10,
                               n_jobs=4, verbose=-1)
        model.fit(x, y[:, k])
        out[:, k] = model.predict_proba(queries)[:, 1]
    return out


def fit_mlp(x: np.ndarray, y: np.ndarray, queries: np.ndarray) -> np.ndarray:
    from sklearn.neural_network import MLPClassifier
    out = np.empty((len(queries), y.shape[1]))
    varying = [k for k in range(y.shape[1]) if y[:, k].min() != y[:, k].max()]
    for k in set(range(y.shape[1])) - set(varying):
        out[:, k] = _constant(y[:, k], len(queries))
    if varying:
        model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=400, early_stopping=True, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(x, y[:, varying])
        p = model.predict_proba(queries)
        out[:, varying] = p if p.ndim == 2 else p[:, None]
    return out


def fit_tabpfn(x: np.ndarray, y: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """TabPFN (Hollmann et al., Nature 2025): a transformer pretrained on synthetic tables that predicts from the
    examples in one pass, like Teenie, but one binary target at a time, so once per answer bit. GPU only; needs
    TABPFN_TOKEN (a free Prior Labs account with the license accepted)."""
    from tabpfn import TabPFNClassifier
    out = np.empty((len(queries), y.shape[1]))
    for k in range(y.shape[1]):
        if y[:, k].min() == y[:, k].max():
            out[:, k] = _constant(y[:, k], len(queries))
            continue
        model = TabPFNClassifier(device='cuda', n_estimators=4, random_state=0)
        model.fit(x, y[:, k])
        out[:, k] = model.predict_proba(queries)[:, list(model.classes_).index(1)]
    return out


LEARNERS = {'lgbm': fit_lgbm, 'mlp': fit_mlp, 'tabpfn': fit_tabpfn}
PULLS = np.linspace(0, 1, 11)


def _bit_scores(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[len(PULLS), bits] mean log2-likelihood gain over chance of each pulled probability."""
    q = .5 + PULLS[:, None, None] * (p[None] - .5)
    q = np.clip(q, FLOOR, 1 - FLOOR)
    return (1 + y[None] * np.log2(q) + (1 - y[None]) * np.log2(1 - q)).mean(1)


def calibrated(fit, x: np.ndarray, y: np.ndarray, queries: np.ndarray) -> np.ndarray:
    held = _bit_scores(fit(x[:800], y[:800], x[800:]), y[800:])
    pull = PULLS[held.argmax(0)]
    return .5 + pull[None] * (fit(x, y, queries) - .5)


def score(probabilities: np.ndarray, target: torch.Tensor, width: int) -> dict[str, float]:
    p = torch.from_numpy(probabilities)
    recovered = recovered_bits(p.clamp(FLOOR, 1 - FLOOR), target, width)
    bits = ((p > .5) == to_bits(target, width)).float()
    return {'recovered': float(recovered.mean()), 'bit': float(bits.mean()), 'exact': float(bits.all(-1).float().mean())}


def evaluate_family(family, width: int, episodes: int, queries: int, seed: int,
                    learners: tuple[str, ...] = ('lgbm', 'mlp')) -> dict[str, dict[str, float]]:
    sampler = BitSampler([family], width, seed)  # the same draws as teenie-bits-eval with this seed
    parts: dict[str, list[dict[str, float]]] = {name: [] for name in learners}
    for _ in range(episodes):
        e = sampler.sample(1, queries)
        x = _features(e.context[0, :, :2], width)
        y = to_bits(e.context[0, :, 2], width).float().numpy()
        q = _features(e.query[0], width)
        for name in learners:
            fit = LEARNERS[name]
            parts[name].append(score(calibrated(fit, x, y, q), e.target[0], width))
    result = {}
    for name, rows in parts.items():
        result[name] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Per-task learners scored by bits recovered')
    parser.add_argument('--tests', default='opaque')
    parser.add_argument('--families', nargs='*')
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--seed', type=int, default=80000)
    parser.add_argument('--learners', nargs='+', default=['lgbm', 'mlp'], choices=sorted(LEARNERS))
    parser.add_argument('--output')
    args = parser.parse_args()
    be_gentle_on_cpu()
    groups = ({'selected': tuple(ALL_NAMED[n] for n in args.families)} if args.families else test_groups(args.tests))
    results = {}
    for group, families in groups.items():
        results[group] = {}
        for family in families:
            results[group][family.name] = evaluate_family(family, args.width, args.episodes, args.queries, args.seed,
                                                          tuple(args.learners))
            line = '  '.join(f'{n} {r["recovered"]:6.2f}' for n, r in results[group][family.name].items())
            print(f'{group[:14]:15s} {family.name[:40]:41s} {line}', flush=True)
    output = {'learners': args.learners, 'tests': args.tests, 'width': args.width, 'episodes': args.episodes, 'queries': args.queries,
              'seed': args.seed, 'results': results}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
