"""Integer-valued inputs; class indices are an internal model/loss detail."""

from enum import IntEnum
from itertools import product

import torch
from torch.utils.data import TensorDataset, random_split


class Op(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2


def result_bound(num_range: int) -> int:
    if num_range < 1:
        raise ValueError("num_range must be positive")
    return max(2 * num_range, num_range**2)


def make_dataset(num_range: int = 10, ops=(Op.ADD,)) -> TensorDataset:
    """Enumerate (a, b, op, result), with signed integer results."""
    result_bound(num_range)
    ops = tuple(Op(op) for op in ops)
    if not ops or len(set(ops)) != len(ops):
        raise ValueError("ops must be nonempty and unique")
    rows = []
    for a, b, op in product(range(-num_range, num_range + 1),
                            range(-num_range, num_range + 1), ops):
        result = {Op.ADD: a + b, Op.SUB: a - b, Op.MUL: a * b}[op]
        rows.append((a, b, int(op), result))
    return TensorDataset(*torch.tensor(rows, dtype=torch.long).unbind(1))


def split_dataset(dataset, train_fraction: float = 0.8, seed: int = 0):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    n = int(len(dataset) * train_fraction)
    if n == 0 or n == len(dataset):
        raise ValueError("both splits must be nonempty")
    return random_split(dataset, [n, len(dataset) - n],
                        generator=torch.Generator().manual_seed(seed))
