"""Procedurally generated relations: random expression trees over a, b and constants, modulo 2**width.

Operations (every intermediate result wraps to `width` bits):
binary  + - * xor and or min max      unary  <<1  >>1  not

`expression_family(ops)` draws a fresh tree with exactly `ops` operations per episode. Trees that
are (nearly) constant, or that behave like any hidden test relation, are redrawn: equivalence is
checked on fixed probe inputs, so `a + a*b` cannot leak a hidden `a*b + a`.
"""
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Union

import torch
from torch import Tensor

from .relations import Family, Relation

# A tree is a leaf ('a' | 'b' | constant int) or (op, child) / (op, left, right).
Tree = Union[str, int, tuple[str, 'Tree'], tuple[str, 'Tree', 'Tree']]

BINARY: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    '+': torch.add, '-': torch.sub, '*': torch.mul, '^': torch.bitwise_xor, '&': torch.bitwise_and,
    '|': torch.bitwise_or, 'min': torch.minimum, 'max': torch.maximum,
}
UNARY: dict[str, Callable[[Tensor], Tensor]] = {
    '<<1': lambda x: x << 1, '>>1': lambda x: x >> 1, '~': torch.bitwise_not,
}


def _parts(tree: Tree) -> tuple[str, list[Tree]]:
    """An operation node as (op, children)."""
    assert isinstance(tree, tuple)
    node: tuple[Any, ...] = tree
    return node[0], list(node[1:])


def evaluate(tree: Tree, a: Tensor, b: Tensor, width: int) -> Tensor:
    mask = (1 << width) - 1
    if tree == 'a':
        return a & mask
    if tree == 'b':
        return b & mask
    if isinstance(tree, int):
        return torch.full_like(a, tree & mask)
    op, children = _parts(tree)
    values = [evaluate(child, a, b, width) for child in children]
    return (UNARY[op](values[0]) if len(values) == 1 else BINARY[op](values[0], values[1])) & mask


def show(tree: Tree) -> str:
    if not isinstance(tree, tuple):
        return str(tree)
    op, children = _parts(tree)
    shown = [show(child) for child in children]
    if len(shown) == 1:
        return f'~{shown[0]}' if op == '~' else f'({shown[0]}){op}'
    if op in ('min', 'max'):
        return f'{op}({shown[0]}, {shown[1]})'
    return f'({shown[0]}{op}{shown[1]})'


def random_tree(ops: int, generator: torch.Generator, width: int) -> Tree:
    def pick(options: list[str]) -> str:
        return options[int(torch.randint(0, len(options), (), generator=generator))]

    if ops == 0:
        roll = float(torch.rand((), generator=generator))
        if roll < .42:
            return 'a'
        if roll < .84:
            return 'b'
        return int(torch.randint(1, 1 << width, (), generator=generator))
    if float(torch.rand((), generator=generator)) < .2:
        return (pick(list(UNARY)), random_tree(ops - 1, generator, width))
    left = int(torch.randint(0, ops, (), generator=generator))
    return (pick(list(BINARY)), random_tree(left, generator, width), random_tree(ops - 1 - left, generator, width))


@lru_cache(maxsize=8)
def _probe(width: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(4242)
    return (torch.randint(0, 1 << width, (128,), generator=generator),
            torch.randint(0, 1 << width, (128,), generator=generator))


def fingerprint(f: Callable[[Tensor, Tensor, int], Tensor], width: int) -> tuple[int, ...]:
    a, b = _probe(width)
    return tuple((f(a, b, width) & ((1 << width) - 1)).tolist())


def tree_relation(name: str, tree: Tree) -> Relation:
    return Relation(name, lambda a, b, w: evaluate(tree, a, b, w))


def named(name: str, tree: Tree) -> Family:
    relation = tree_relation(name, tree)
    return Family(name, lambda generator, width: relation)


# New operations never generated for training: compositions of the trained building blocks.
HIDDEN_FAMILIES: tuple[Family, ...] = (
    named('a*b+a', ('+', ('*', 'a', 'b'), 'a')),
    named('a*a', ('*', 'a', 'a')),
    named('(a+1)*(b+1)', ('*', ('+', 'a', 1), ('+', 'b', 1))),
    named('(a^b)*3', ('*', ('^', 'a', 'b'), 3)),
    named('|a-b|', ('-', ('max', 'a', 'b'), ('min', 'a', 'b'))),
    named('(a+b)>>1', ('>>1', ('+', 'a', 'b'))),
    named('(a*b)^b', ('^', ('*', 'a', 'b'), 'b')),
    named('(a<<1)+(b>>1)', ('+', ('<<1', 'a'), ('>>1', 'b'))),
)


def expression_family(ops: int, forbidden: tuple[Family, ...], name: str | None = None) -> Family:
    """A fresh random expression with exactly `ops` operations per episode, never equivalent to `forbidden`."""
    def make(generator: torch.Generator, width: int) -> Relation:
        banned = _banned(forbidden, width)
        for _ in range(200):
            tree = random_tree(ops, generator, width)
            relation = tree_relation(show(tree), tree)
            outputs = fingerprint(relation, width)
            if len(set(outputs)) >= 8 and outputs not in banned:
                return relation
        raise RuntimeError(f'could not draw a usable {ops}-operation expression')
    return Family(name or f'expr{ops}', make)


_banned_cache: dict[tuple[int, int], frozenset[tuple[int, ...]]] = {}


def _banned(forbidden: tuple[Family, ...], width: int) -> frozenset[tuple[int, ...]]:
    key = (id(forbidden), width)
    if key not in _banned_cache:
        _banned_cache[key] = frozenset(fingerprint(f.make(torch.Generator().manual_seed(0), width), width)
                                       for f in forbidden)
    return _banned_cache[key]
