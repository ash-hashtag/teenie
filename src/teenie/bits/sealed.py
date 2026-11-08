"""A sealed test set of everyday functions, written once and never used to make decisions.

Every other test in this project has been used to choose architectures and data, which slowly tunes choices to
those particular tests. These functions were written independently of all of that and are only evaluated at the
end of a round of experiments, to check that the model is general rather than fitted to our benchmarks. Several
use operations or shapes the training generator never produces (>> by more than 1, squares of sums, GCD).

uv run teenie-bits-eval --load <checkpoint> --tests sealed --device cuda --seed 61000
"""
from collections.abc import Callable

import torch
from torch import Tensor

from .loops import popcount
from .relations import Family, Relation

Function = Callable[[Tensor, Tensor, int], Tensor]


def _family(name: str, f: Function) -> Family:
    relation = Relation(name, f)
    return Family(name, lambda generator, width: relation)


def _gcd(a: Tensor, b: Tensor, w: int) -> Tensor:
    x, y = a.clone(), b.clone()
    for _ in range(2 * w + 2):  # Euclid on 8-bit numbers finishes well within this
        nonzero = y != 0
        x, y = torch.where(nonzero, y, x), torch.where(nonzero, x % torch.clamp(y, min=1), y)
    return x


def _sum_of_squares(a: Tensor, b: Tensor, w: int) -> Tensor:
    n = b & 7
    return sum(((i * i) * (i < n).long() for i in range(8)), torch.zeros_like(a)) + a


def _count_down(a: Tensor, b: Tensor, w: int) -> Tensor:
    """Subtract b from a until it would go below b; how many times (a // b, by repetition)."""
    steps, x = torch.zeros_like(a), a.clone()
    d = torch.clamp(b, min=1)
    for _ in range(1 << w):
        go = x >= d
        if not bool(go.any()):
            break
        x, steps = torch.where(go, x - d, x), steps + go.long()
    return steps


SEALED_FAMILIES: tuple[Family, ...] = (
    _family('(a+b)*(a-b)', lambda a, b, w: (a + b) * (a - b)),
    _family('(a*b)>>4', lambda a, b, w: (a * b) >> 4),
    _family('(a+b)^2', lambda a, b, w: (a + b) * (a + b)),
    _family('a*b-a-b', lambda a, b, w: a * b - a - b),
    _family('7a+3b', lambda a, b, w: 7 * a + 3 * b),
    _family('a*(a+b)', lambda a, b, w: a * (a + b)),
    _family('(a-b)^2', lambda a, b, w: (a - b) * (a - b)),
    _family('2max-min', lambda a, b, w: 2 * torch.maximum(a, b) - torch.minimum(a, b)),
    _family('((a+b)>>1)^b', lambda a, b, w: ((a + b) >> 1) ^ b),
    _family('(a&b)*2+(a^b)', lambda a, b, w: (a & b) * 2 + (a ^ b)),
    _family('a>>(b&7)', lambda a, b, w: a >> (b & 7)),
    _family('a<<(b&3)', lambda a, b, w: a << (b & 3)),
    _family('a if a odd else b', lambda a, b, w: torch.where((a & 1) == 1, a, b)),
    _family('a>b ? a*2 : b*3', lambda a, b, w: torch.where(a > b, a * 2, b * 3)),
    _family('popcount(a)+popcount(b)', lambda a, b, w: popcount(a, w) + popcount(b, w)),
    _family('popcount(a*b)', lambda a, b, w: popcount((a * b) & ((1 << w) - 1), w)),
    _family('a % (b|1)', lambda a, b, w: a % (b | 1)),
    _family('a // (b|1) + b', lambda a, b, w: a // (b | 1) + b),
    _family('gcd(a,b)', _gcd),
    _family('a + sum i^2, i<b&7', _sum_of_squares),
    _family('a // b by repeated subtraction', _count_down),
    _family('a*b*b', lambda a, b, w: a * b * b),
    _family('(a|1)*(b|1)', lambda a, b, w: (a | 1) * (b | 1)),
    _family('~(a+b)', lambda a, b, w: ~(a + b)),
)
