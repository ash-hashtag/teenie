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


def random_tree(ops: int, generator: torch.Generator, width: int, product_weight: float = 1.) -> Tree:
    """`product_weight` > 1 draws `*` more often than each other binary operation."""
    def pick(options: list[str]) -> str:
        return options[int(torch.randint(0, len(options), (), generator=generator))]

    def pick_binary() -> str:
        if product_weight == 1.:  # keeps the original draws for the `mix` set
            return pick(list(BINARY))
        weights = torch.tensor([product_weight if op == '*' else 1. for op in BINARY])
        return list(BINARY)[int(torch.multinomial(weights, 1, generator=generator))]

    if ops == 0:
        roll = float(torch.rand((), generator=generator))
        if roll < .42:
            return 'a'
        if roll < .84:
            return 'b'
        return int(torch.randint(1, 1 << width, (), generator=generator))
    if float(torch.rand((), generator=generator)) < .2:
        return (pick(list(UNARY)), random_tree(ops - 1, generator, width, product_weight))
    left = int(torch.randint(0, ops, (), generator=generator))
    return (pick_binary(), random_tree(left, generator, width, product_weight),
            random_tree(ops - 1 - left, generator, width, product_weight))


def has_variable(tree: Tree) -> bool:
    if not isinstance(tree, tuple):
        return tree in ('a', 'b')
    return any(has_variable(child) for child in _parts(tree)[1])


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


# Close relatives of the hidden set that training may produce: tracks "a product, then another step".
NEAR_FAMILIES: tuple[Family, ...] = (
    named('a*b+b', ('+', ('*', 'a', 'b'), 'b')),
    named('a*b+5', ('+', ('*', 'a', 'b'), 5)),
    named('a*b-a', ('-', ('*', 'a', 'b'), 'a')),
    named('(a*b)^a', ('^', ('*', 'a', 'b'), 'a')),
)

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


def _draw(draw: Callable[[torch.Generator, int], Tree], forbidden: tuple[Family, ...],
          generator: torch.Generator, width: int) -> Relation:
    """Redraw until the expression varies enough and behaves unlike every forbidden relation."""
    banned = _banned(forbidden, width)
    for _ in range(200):
        tree = draw(generator, width)
        relation = tree_relation(show(tree), tree)
        outputs = fingerprint(relation, width)
        if len(set(outputs)) >= 8 and outputs not in banned:
            return relation
    raise RuntimeError('could not draw a usable expression')


def expression_family(ops: int, forbidden: tuple[Family, ...], name: str | None = None,
                      product_weight: float = 1.) -> Family:
    """A fresh random expression with exactly `ops` operations per episode, never equivalent to `forbidden`."""
    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw(lambda g, w: random_tree(ops, g, w, product_weight), forbidden, generator, width)
    return Family(name or f'expr{ops}', make)


def composed_family(forbidden: tuple[Family, ...], product_weight: float = 2.) -> Family:
    """One binary operation joining two multi-step pieces that both use a or b, e.g. (a*b)+(a^b)."""
    def piece(generator: torch.Generator, width: int) -> Tree:
        while True:
            tree = random_tree(int(torch.randint(1, 3, (), generator=generator)), generator, width, product_weight)
            if has_variable(tree):
                return tree

    def draw(generator: torch.Generator, width: int) -> Tree:
        op = list(BINARY)[int(torch.randint(0, len(BINARY), (), generator=generator))]
        return (op, piece(generator, width), piece(generator, width))

    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw(draw, forbidden, generator, width)
    return Family('composed', make)


_banned_cache: dict[tuple[int, int], frozenset[tuple[int, ...]]] = {}


def _banned(forbidden: tuple[Family, ...], width: int) -> frozenset[tuple[int, ...]]:
    key = (id(forbidden), width)
    if key not in _banned_cache:
        _banned_cache[key] = frozenset(fingerprint(f.make(torch.Generator().manual_seed(0), width), width)
                                       for f in forbidden)
    return _banned_cache[key]
