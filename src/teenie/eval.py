"""Frozen held-out callable evaluation with value/status metrics and loop controls."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model
from .config import CONTEXT_SIZE
from .data import Episode, EpisodeSampler, FunctionTask, function_value, load_tasks
from .families import heldout_tasks, validation_tasks
from .metrics import Metrics, measure
from .model import ContextModel
from .numeric import F32_MIN
from .train import select_device


def composite(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a * a + b / a)


COMPOSITE = FunctionTask('square_plus_ratio', composite)


def boundary_episodes(task: FunctionTask, seed: int = 20000) -> list[Episode]:
    """Explicit edge probes; random float sampling can miss undefined queries."""
    a_values = sorted({v for v in (*task.a_range, F32_MIN, 1.) if task.a_range[0] <= v <= task.a_range[1]})
    b_values = sorted({v for v in (*task.b_range, F32_MIN, 1.) if task.b_range[0] <= v <= task.b_range[1]})
    queries = [(a, b) for a in a_values for b in b_values]
    sampler = EpisodeSampler([task], seed)
    pool: dict[tuple[float, float], float] = {}
    for _ in range(20):
        for a, b, c in sampler.sample(1).context[0].tolist():
            pool[(a, b)] = c
        if len(pool) >= CONTEXT_SIZE + 1:
            break
    if len(pool) < CONTEXT_SIZE + 1:
        raise ValueError('not enough distinct pairs for boundary probes')
    contexts = [[[a, b, c] for (a, b), c in pool.items() if (a, b) != query][:CONTEXT_SIZE]
                for query in queries]
    return [Episode(torch.tensor(contexts, dtype=torch.float32), torch.tensor(queries, dtype=torch.float32),
                    torch.tensor([function_value(task, *q) for q in queries], dtype=torch.float32))]


def evaluate_function(model: ContextModel, task: FunctionTask, *, count: int = 512,
                      seed: int = 10000, batch_size: int = 32,
                      rtol: float = .01, atol: float = 0., controls: bool = True) -> dict[str, Metrics]:
    if (min(count, batch_size) < 1 or min(rtol, atol) < 0
            or not all(math.isfinite(v) for v in (rtol, atol))):
        raise ValueError('counts must be positive and tolerances finite/nonnegative')
    sampler = EpisodeSampler([task], seed)
    device = next(model.parameters()).device
    batches = [sampler.sample(min(batch_size, count - start)).to(device)
               for start in range(0, count, batch_size)]
    model.eval()
    if not controls:
        return {'correct_context': measure(model, batches, rtol=rtol, atol=atol)}
    generator = torch.Generator().manual_seed(seed + 1)
    shuffled: list[Episode] = []
    zero: list[Episode] = []
    for batch in batches:
        context = batch.context.clone()
        order = torch.rand(context.shape[:2], generator=generator).argsort(-1).to(device)
        context[..., 2] = context[..., 2].gather(1, order)
        shuffled.append(Episode(context, batch.query, batch.target))
        context = batch.context.clone()
        context[..., 2] = 0
        zero.append(Episode(context, batch.query, batch.target))
    report = {'correct_context': measure(model, batches, rtol=rtol, atol=atol),
              'shuffled_values': measure(model, shuffled, rtol=rtol, atol=atol),
              'zero_values': measure(model, zero, rtol=rtol, atol=atol),
              'nearest_context': measure(model, batches, rtol=rtol, atol=atol, baseline='nearest'),
              'context_mean': measure(model, batches, rtol=rtol, atol=atol, baseline='mean')}
    for budget in (1, 8, 64, 128, 256):
        report[f'steps_{budget}'] = measure(model, batches, steps=budget, rtol=rtol, atol=atol)
    return report



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', required=True)
    parser.add_argument('--functions', help='Trusted TASKS file; default bounded known/held-out structures and recursion depths')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--stress', action='store_true', help='Separate full-float32 stress test instead of bounded tasks')
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--rtol', type=float, default=.01)
    parser.add_argument('--atol', type=float, default=0.)
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        model, metadata = load_model(args.load, select_device(args.device))
        model.requires_grad_(False)
        tasks = load_tasks(args.functions) if args.functions else ([*load_tasks(), COMPOSITE] if args.stress
                                                                else validation_tasks(seed=55555) + heldout_tasks())
        torch.manual_seed(12345)
        baseline = ContextModel(**model.config).to(next(model.parameters()).device).eval()
        results: dict[str, dict[str, Metrics]] = {}
        untrained: dict[str, Metrics] = {}
        boundaries: dict[str, Metrics] = {}
        for task in tasks:
            results[task.name] = evaluate_function(model, task, count=args.count,
                                                  seed=args.seed, rtol=args.rtol, atol=args.atol)
            untrained[task.name] = evaluate_function(baseline, task, count=args.count,
                                                     seed=args.seed, rtol=args.rtol, atol=args.atol, controls=False)['correct_context']
            probes = [e.to(next(model.parameters()).device) for e in boundary_episodes(task)]
            boundaries[task.name] = measure(model, probes, rtol=args.rtol, atol=args.atol)
        report = {'checkpoint': args.load, 'training_functions': metadata['training_functions'],
                  'seed': args.seed, 'context_size': CONTEXT_SIZE, 'domain': 'full-float32-stress' if args.stress else 'bounded-or-custom',
                  'loop_budgets_trained': metadata.get('loop_budgets'), 'results': results,
                  'untrained_same_architecture': untrained, 'boundary_probes': boundaries}
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
