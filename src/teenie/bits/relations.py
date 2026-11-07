"""Relations c = f(a, b) on unsigned `width`-bit integers, wrapping modulo 2**width.

A `Family` draws a concrete `Relation` per episode, so constants and random
finite-state transducers differ between episodes. The model never sees names.
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

IntFunction = Callable[[Tensor, Tensor, int], Tensor]
# Intermediate values of a relation, for step-by-step training: (values [S, N], steps taken [N]). Each column holds
# the value after step 1, 2, ... and then repeats its final answer; steps taken is at least 1.
StepFunction = Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class Relation:
    name: str
    f: IntFunction
    steps: StepFunction | None = None  # None: the relation counts as a single step

    def __call__(self, a: Tensor, b: Tensor, width: int) -> Tensor:
        return self.f(a, b, width) & ((1 << width) - 1)


@dataclass(frozen=True)
class Family:
    name: str
    make: Callable[[torch.Generator, int], Relation]


def fixed(name: str, f: IntFunction) -> Family:
    relation = Relation(name, f)
    return Family(name, lambda generator, width: relation)


def _constant(generator: torch.Generator, width: int) -> int:
    return int(torch.randint(0, 1 << width, (), generator=generator))


def add_constant(generator: torch.Generator, width: int) -> Relation:
    k = _constant(generator, width)
    return Relation(f'a+{k}', lambda a, b, w: a + k)


def xor_constant(generator: torch.Generator, width: int) -> Relation:
    k = _constant(generator, width)
    return Relation(f'b^{k}', lambda a, b, w: b ^ k)


def mul_constant(generator: torch.Generator, width: int) -> Relation:
    k = _constant(generator, width)
    return Relation(f'a*{k}', lambda a, b, w: a * k)


def transducer(generator: torch.Generator, width: int) -> Relation:
    """Random Mealy machine reading (a_k, b_k) one bit position at a time."""
    states = int(torch.randint(1, 5, (), generator=generator))
    output = torch.randint(0, 2, (states, 4), generator=generator)
    following = torch.randint(0, states, (states, 4), generator=generator)
    high_first = bool(torch.randint(0, 2, (), generator=generator))

    def run(a: Tensor, b: Tensor, w: int) -> Tensor:
        state = torch.zeros_like(a)
        c = torch.zeros_like(a)
        for k in (range(w - 1, -1, -1) if high_first else range(w)):
            symbol = ((a >> k) & 1) * 2 + ((b >> k) & 1)
            c = c | (output.to(a.device)[state, symbol] << k)
            state = following.to(a.device)[state, symbol]
        return c
    return Relation(f'transducer{states}{"H" if high_first else "L"}', run)


TRAIN_FAMILIES: tuple[Family, ...] = (
    fixed('add', lambda a, b, w: a + b),
    fixed('sub', lambda a, b, w: a - b),
    fixed('b-a', lambda a, b, w: b - a),
    fixed('xor', lambda a, b, w: a ^ b),
    fixed('and', lambda a, b, w: a & b),
    fixed('or', lambda a, b, w: a | b),
    fixed('a+2b', lambda a, b, w: a + 2 * b),
    fixed('2a+b', lambda a, b, w: 2 * a + b),
    fixed('max', lambda a, b, w: torch.maximum(a, b)),
    fixed('a<<1', lambda a, b, w: a << 1),
    Family('a+const', add_constant),
    Family('b^const', xor_constant),
    Family('transducer', transducer),
)

# Never sampled during training; each is a near neighbour of a trained idea.
HELDOUT_FAMILIES: tuple[Family, ...] = (
    fixed('a+b+1', lambda a, b, w: a + b + 1),
    fixed('a-2b', lambda a, b, w: a - 2 * b),
    fixed('min', lambda a, b, w: torch.minimum(a, b)),
    fixed('nand', lambda a, b, w: ~(a & b)),
    fixed('3a', lambda a, b, w: 3 * a),
)

MULTIPLY = fixed('a*b', lambda a, b, w: a * b)

# Pieces of shift-and-add multiplication: fixed multipliers and short multipliers (only b's low bits count).
BLOCK_FAMILIES: tuple[Family, ...] = (
    Family('a*const', mul_constant),
    fixed('a*(b&3)', lambda a, b, w: a * (b & 3)),
    fixed('a*(b&15)', lambda a, b, w: a * (b & 15)),
)

# Multiplication has no small bit-serial state; reported, not targeted by the default training set.
STRESS_FAMILIES: tuple[Family, ...] = (
    fixed('a*(b&63)', lambda a, b, w: a * (b & 63)),
    MULTIPLY,
)

NAMED: dict[str, Family] = {f.name: f for f in TRAIN_FAMILIES + HELDOUT_FAMILIES + STRESS_FAMILIES + BLOCK_FAMILIES}
