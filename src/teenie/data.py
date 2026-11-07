"""Typed scalar callables generate IEEE float32 context episodes."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import random
import runpy
from typing import Literal, SupportsFloat, cast

import numpy as np
import torch
from torch import Tensor

from .config import CONTEXT_SIZE
from .numeric import F32_MAX, F32_MIN

ScalarFunction = Callable[[np.float32, np.float32], SupportsFloat]
Bounds = tuple[float, float]
Distribution = Literal['mixed', 'uniform']


@dataclass(frozen=True)
class FunctionTask:
    name: str
    f: ScalarFunction
    a_range: Bounds = (0., F32_MAX)
    b_range: Bounds = (0., F32_MAX)
    distribution: Distribution = 'mixed'

    def __post_init__(self) -> None:
        if not self.name or not callable(self.f):
            raise ValueError('task needs a name and callable f(a,b)')
        for low, high in (self.a_range, self.b_range):
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= F32_MAX):
                raise ValueError('input ranges must satisfy 0 <= low < high <= max float32')
        if self.distribution not in ('mixed', 'uniform'):
            raise ValueError('unknown distribution')


def add(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a + b)


def sub(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a - b)


def mul(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a * b)


def div(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a / b)


DEFAULT_TASKS: tuple[FunctionTask, ...] = tuple(FunctionTask(name, f) for name, f in
                                               [('add', add), ('sub', sub), ('mul', mul), ('div', div)])


def function_value(task: FunctionTask, a: float, b: float) -> float:
    # NumPy scalar arithmetic implements float32 overflow and zero-division.
    # Exceptions from custom Python code are NOT silently converted to NaN.
    with np.errstate(all='ignore'):
        return float(np.float32(task.f(np.float32(a), np.float32(b))))


def load_tasks(path: str | Path | None = None) -> list[FunctionTask]:
    tasks: object = DEFAULT_TASKS if path is None else runpy.run_path(str(path)).get('TASKS')
    if not isinstance(tasks, (list, tuple)) or not tasks or not all(isinstance(t, FunctionTask) for t in tasks):
        raise ValueError('task file must define a nonempty TASKS list of FunctionTask objects')
    result = list(cast(Sequence[FunctionTask], tasks))
    if len({t.name for t in result}) != len(result):
        raise ValueError('task names must be unique')
    return result


@dataclass
class Episode:
    context: Tensor  # [B,K,3]; a,b finite; c may be inf/-inf/nan
    query: Tensor    # [B,2] or [B,Q,2]; no target
    target: Tensor   # [B] or [B,Q]; never passed to model

    def to(self, device: str | torch.device) -> 'Episode':
        return Episode(self.context.to(device), self.query.to(device), self.target.to(device))


class EpisodeSampler:
    def __init__(self, tasks: Sequence[FunctionTask] | None = None, seed: int = 0,
                 task_factory: Callable[[random.Random], FunctionTask] | None = None) -> None:
        self.tasks = list(DEFAULT_TASKS if tasks is None else tasks)
        if not self.tasks:
            raise ValueError('at least one task required')
        self.rng = random.Random(seed)
        self.task_factory = task_factory

    def _operand(self, bounds: Bounds, distribution: Distribution) -> float:
        low, high = bounds
        if distribution == 'uniform':
            return float(np.float32(self.rng.uniform(low, high)))
        p = self.rng.random()
        if p < .10:
            candidates = [low, high, F32_MIN, 0., 1., 2., 256.]
            return float(np.float32(self.rng.choice([v for v in candidates if low <= v <= high])))
        if p < .40 and low < min(high, 256.):
            return float(np.float32(self.rng.uniform(low, min(high, 256.))))
        return float(np.float32(math.exp(self.rng.uniform(math.log(max(low, F32_MIN)), math.log(high)))))

    def sample(self, batch_size: int = 16) -> Episode:
        episode = self.sample_queries(batch_size, queries=1)
        return Episode(episode.context, episode.query[:, 0], episode.target[:, 0])

    def sample_queries(self, batch_size: int = 16, *, queries: int = 16) -> Episode:
        if queries < 1:
            raise ValueError("queries must be positive")
        if batch_size < 1:
            raise ValueError('batch size must be positive')
        contexts: list[list[list[float]]] = []
        query_rows: list[list[list[float]]] = []
        targets: list[list[float]] = []
        for _ in range(batch_size):
            task = self.task_factory(self.rng) if self.task_factory else self.rng.choice(self.tasks)
            rows: list[list[float]] = []
            seen: set[tuple[float, float]] = set()
            for _ in range((CONTEXT_SIZE + queries) * 20):
                a = self._operand(task.a_range, task.distribution)
                b = self._operand(task.b_range, task.distribution)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                rows.append([a, b, function_value(task, a, b)])
                if len(rows) == CONTEXT_SIZE + queries:
                    break
            if len(rows) != CONTEXT_SIZE + queries:
                raise ValueError('range has too few distinct float32 pairs')
            # Randomly select query so boundary atoms have the same opportunity as supports.
            held_out = [rows.pop(self.rng.randrange(len(rows))) for _ in range(queries)]
            contexts.append(rows)
            query_rows.append([row[:2] for row in held_out])
            targets.append([row[2] for row in held_out])
        return Episode(torch.tensor(contexts, dtype=torch.float32), torch.tensor(query_rows, dtype=torch.float32),
                       torch.tensor(targets, dtype=torch.float32))
