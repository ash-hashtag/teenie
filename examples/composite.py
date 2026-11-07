"""Held-out evaluation: uv run python -m teenie.eval --functions examples/composite.py ..."""
import numpy as np
from teenie import FunctionTask


def f(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a * a + b / a)


TASKS = [FunctionTask('square_plus_ratio', f, (1.,256.), (0.,256.), 'uniform')]
