"""Data-only relation families; no expressions or operation IDs enter the model."""
from collections.abc import Callable
import random
from typing import Literal

import numpy as np

from .data import FunctionTask, ScalarFunction, add, sub, mul, div

Family = Literal['affine', 'quadratic', 'product', 'ratio', 'recurrence']
TRAIN_FAMILIES: tuple[Family, ...] = ('affine', 'quadratic', 'product', 'ratio', 'recurrence')
TaskFactory = Callable[[random.Random], FunctionTask]


def bounded_basics(maximum: float = 256.) -> list[FunctionTask]:
    return [FunctionTask(name, f, (0., maximum), (1., maximum) if name == 'div' else (0., maximum), 'uniform')
            for name, f in [('add', add), ('sub', sub), ('mul', mul), ('div', div)]]


def make_family(rng: random.Random, family: Family, maximum: float = 256., *, depth: int | None = None) -> FunctionTask:
    p, q, r = (np.float32(rng.uniform(.25, 2.)) for _ in range(3))
    if rng.random() < .5:
        q = -q
    f: ScalarFunction
    if family == 'affine':
        def f(a: np.float32, b: np.float32) -> np.float32:
            return np.float32(p*a + q*b + r)
    elif family == 'quadratic':
        def f(a: np.float32, b: np.float32) -> np.float32:
            return np.float32(p*a*a + q*b*b + r)
    elif family == 'product':
        def f(a: np.float32, b: np.float32) -> np.float32:
            return np.float32((a+p)*(b+q))
    elif family == 'ratio':
        def f(a: np.float32, b: np.float32) -> np.float32:
            return np.float32((p*a+r)/(b+abs(q)))
    elif family == 'recurrence':
        n = rng.randint(1, 4) if depth is None else depth
        if n < 1:
            raise ValueError('recursion depth must be positive')
        gain = np.float32(rng.uniform(.3, .9))
        def f(a: np.float32, b: np.float32) -> np.float32:
            state = a
            for _ in range(n):
                state = np.float32(gain*state + p*b/(np.float32(1.) + abs(state)))
            return state
    else:
        raise ValueError('unknown training family')
    return FunctionTask(family if depth is None else f'{family}_depth{depth}', f,
                        (0., maximum), (0., maximum), 'uniform')


def family_factory(maximum: float = 256.) -> TaskFactory:
    def factory(rng: random.Random) -> FunctionTask:
        # Half familiar primitives, half newly parameterized relations each episode.
        if rng.random() < .5:
            return rng.choice(bounded_basics(maximum))
        return make_family(rng, rng.choice(TRAIN_FAMILIES), maximum)
    return factory


def curriculum_max(update: int, updates: int) -> float:
    fraction = update / updates
    return 16. if fraction <= 1/3 else 64. if fraction <= 2/3 else 256.


def validation_tasks(seed: int = 98765) -> list[FunctionTask]:
    rng = random.Random(seed)
    return bounded_basics() + [make_family(rng, family) for family in TRAIN_FAMILIES]


def heldout_tasks() -> list[FunctionTask]:
    def composite(a: np.float32, b: np.float32) -> np.float32:
        return np.float32(a*a + b/a)
    def cross_ratio(a: np.float32, b: np.float32) -> np.float32:
        return np.float32((a*b + np.float32(1.))/(a+b+np.float32(1.)))
    # Exact expression structures absent from TRAIN_FAMILIES, not merely fresh coefficients.
    result = [FunctionTask('square_plus_ratio', composite, (1.,256.), (0.,256.), 'uniform'),
              FunctionTask('cross_ratio', cross_ratio, (0.,256.), (0.,256.), 'uniform')]
    # Same coefficients at different depths to isolate recursive-depth generalization.
    result += [make_family(random.Random(7788), 'recurrence', depth=n) for n in (4, 8, 16)]
    return result
