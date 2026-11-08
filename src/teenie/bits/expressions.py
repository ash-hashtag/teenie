"""Procedurally generated relations: random expression trees over a, b and constants, modulo 2**width.

Operations (every intermediate result wraps to `width` bits):
binary  + - * xor and or min max      unary  <<1  >>1  not
More concepts (only drawn when asked for, see `CONCEPTS`):
binary  // %  < >  <<v >>v (shift by y & 7)      unary  square  popcount

`expression_family(ops)` draws a fresh tree with exactly `ops` operations per episode. Trees that
are (nearly) constant, or that behave like any hidden test relation, are redrawn: equivalence is
checked on fixed probe inputs, so `a + a*b` cannot leak a hidden `a*b + a`. Whole combinations can be
held out too: `banned_pairs` such as ('^', '*') reject any tree where a xor is applied to a product.
Leaves are 'a' and 'b' by default; loops (`loops.py`) reuse the same trees over other variables.
"""
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Union

import torch
from torch import Tensor

from .relations import Family, Relation

# A tree is a leaf (a variable name such as 'a' | constant int) or (op, child) / (op, left, right).
Pair = tuple[str, str]  # (parent operation, child operation)
Tree = Union[str, int, tuple[str, 'Tree'], tuple[str, 'Tree', 'Tree']]

BINARY: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    '+': torch.add, '-': torch.sub, '*': torch.mul, '^': torch.bitwise_xor, '&': torch.bitwise_and,
    '|': torch.bitwise_or, 'min': torch.minimum, 'max': torch.maximum,
}
UNARY: dict[str, Callable[[Tensor], Tensor]] = {
    '<<1': lambda x: x << 1, '>>1': lambda x: x >> 1, '~': torch.bitwise_not,
}
# The original operations, in draw order; existing families draw from exactly these.
BASE: tuple[tuple[str, ...], tuple[str, ...]] = (tuple(BINARY), tuple(UNARY))


def _popcount(x: Tensor) -> Tensor:
    return sum(((x >> i) & 1 for i in range(31)), torch.zeros_like(x))


# Values reaching an operation are already wrapped to the width, so these need no width of their own.
# Division by zero gives all ones (masked to the width) and x % 0 gives x, as in the Try it page.
BINARY |= {
    '//': lambda x, y: torch.where(y == 0, -1, x // torch.clamp(y, min=1)),
    '%': lambda x, y: torch.where(y == 0, x, x % torch.clamp(y, min=1)),
    '<': lambda x, y: (x < y).long(), '>': lambda x, y: (x > y).long(),
    '<<v': lambda x, y: x << (y & 7), '>>v': lambda x, y: x >> (y & 7),
}
UNARY |= {'sq': lambda x: x * x, 'pop': _popcount}
# The operations plus the more concepts: division, remainder, comparisons, variable shifts, squares, bit counts.
CONCEPTS: tuple[tuple[str, ...], tuple[str, ...]] = (tuple(BINARY), tuple(UNARY))


def _parts(tree: Tree) -> tuple[str, list[Tree]]:
    """An operation node as (op, children)."""
    assert isinstance(tree, tuple)
    node: tuple[Any, ...] = tree
    return node[0], list(node[1:])


def evaluate_with(tree: Tree, variables: dict[str, Tensor], width: int) -> Tensor:
    mask = (1 << width) - 1
    if isinstance(tree, str):
        return variables[tree] & mask
    if isinstance(tree, int):
        return torch.full_like(next(iter(variables.values())), tree & mask)
    op, children = _parts(tree)
    values = [evaluate_with(child, variables, width) for child in children]
    return (UNARY[op](values[0]) if len(values) == 1 else BINARY[op](values[0], values[1])) & mask


def evaluate(tree: Tree, a: Tensor, b: Tensor, width: int) -> Tensor:
    return evaluate_with(tree, {'a': a, 'b': b}, width)


def pairs(tree: Tree) -> set[Pair]:
    """Every (parent operation, child operation) link in the tree."""
    if not isinstance(tree, tuple):
        return set()
    op, children = _parts(tree)
    found: set[Pair] = set()
    for child in children:
        if isinstance(child, tuple):
            found.add((op, _parts(child)[0]))
        found |= pairs(child)
    return found


def show(tree: Tree) -> str:
    if not isinstance(tree, tuple):
        return str(tree)
    op, children = _parts(tree)
    shown = [show(child) for child in children]
    if len(shown) == 1:
        return {'~': f'~{shown[0]}', 'sq': f'({shown[0]})²', 'pop': f'popcount({shown[0]})'}.get(op, f'({shown[0]}){op}')
    if op in ('<<v', '>>v'):
        return f'({shown[0]}{op[:2]}({shown[1]}&7))'
    if op in ('min', 'max'):
        return f'{op}({shown[0]}, {shown[1]})'
    return f'({shown[0]}{op}{shown[1]})'


def random_tree(ops: int, generator: torch.Generator, width: int, product_weight: float = 1.,
                leaves: tuple[str, ...] = ('a', 'b'), vocabulary: tuple[tuple[str, ...], tuple[str, ...]] = BASE) -> Tree:
    """`product_weight` > 1 draws `*` more often than each other binary operation. Leaves are a variable
    (84%, evenly among `leaves`) or a constant. `vocabulary` is the (binary, unary) operations to draw from."""
    binary, unary = vocabulary

    def pick(options: list[str]) -> str:
        return options[int(torch.randint(0, len(options), (), generator=generator))]

    def pick_binary() -> str:
        if product_weight == 1.:  # keeps the original draws for the `mix` set
            return pick(list(binary))
        weights = torch.tensor([product_weight if op == '*' else 1. for op in binary])
        return binary[int(torch.multinomial(weights, 1, generator=generator))]

    if ops == 0:
        roll = float(torch.rand((), generator=generator))
        if roll < .84:  # same draws as before for the default ('a', 'b'): below .42 'a', else 'b'
            return leaves[min(int(roll / .84 * len(leaves)), len(leaves) - 1)]
        return int(torch.randint(1, 1 << width, (), generator=generator))
    if float(torch.rand((), generator=generator)) < .2:
        return (pick(list(unary)), random_tree(ops - 1, generator, width, product_weight, leaves, vocabulary))
    left = int(torch.randint(0, ops, (), generator=generator))
    return (pick_binary(), random_tree(left, generator, width, product_weight, leaves, vocabulary),
            random_tree(ops - 1 - left, generator, width, product_weight, leaves, vocabulary))


def has_variable(tree: Tree, names: tuple[str, ...] = ('a', 'b')) -> bool:
    if not isinstance(tree, tuple):
        return tree in names
    return any(has_variable(child, names) for child in _parts(tree)[1])


@lru_cache(maxsize=8)
def _probe(width: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(4242)
    return (torch.randint(0, 1 << width, (128,), generator=generator),
            torch.randint(0, 1 << width, (128,), generator=generator))


def fingerprint(f: Callable[[Tensor, Tensor, int], Tensor], width: int) -> tuple[int, ...]:
    a, b = _probe(width)
    return tuple((f(a, b, width) & ((1 << width) - 1)).tolist())


def tree_steps(tree: Tree) -> Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor]]:
    """Every operation's value in evaluation order (children before parents): step k is the k-th operation."""
    def run(a: Tensor, b: Tensor, width: int) -> tuple[Tensor, Tensor]:
        mask = (1 << width) - 1
        values: list[Tensor] = []

        def walk(node: Tree) -> Tensor:
            if not isinstance(node, tuple):
                return evaluate_with(node, {'a': a, 'b': b}, width)
            op, children = _parts(node)
            inputs = [walk(child) for child in children]
            value = (UNARY[op](inputs[0]) if len(inputs) == 1 else BINARY[op](inputs[0], inputs[1])) & mask
            values.append(value)
            return value

        final = walk(tree)
        trace = torch.stack(values) if values else final[None]
        return trace, torch.full_like(a, len(trace))
    return run


def tree_relation(name: str, tree: Tree) -> Relation:
    return Relation(name, lambda a, b, w: evaluate(tree, a, b, w), tree_steps(tree))


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
          generator: torch.Generator, width: int, banned_pairs: frozenset[Pair] = frozenset(),
          require_pair: bool = False) -> Relation:
    """Redraw until the expression varies enough and behaves unlike every forbidden relation. With
    `require_pair` the tree must contain one of `banned_pairs` (a test of an unseen combination); otherwise
    it must contain none of them."""
    banned = _banned(forbidden, width)
    for _ in range(2000 if require_pair else 200):
        tree = draw(generator, width)
        if bool(pairs(tree) & banned_pairs) != require_pair:
            continue
        relation = tree_relation(show(tree), tree)
        outputs = fingerprint(relation, width)
        if len(set(outputs)) >= 8 and outputs not in banned:
            return relation
    raise RuntimeError('could not draw a usable expression')


def expression_family(ops: int | tuple[int, int], forbidden: tuple[Family, ...], name: str | None = None,
                      product_weight: float = 1., banned_pairs: frozenset[Pair] = frozenset(),
                      require_pair: bool = False,
                      vocabulary: tuple[tuple[str, ...], tuple[str, ...]] = BASE) -> Family:
    """A fresh random expression per episode with `ops` operations (or a number drawn from an inclusive
    range), never equivalent to `forbidden`; see `_draw` for `banned_pairs` / `require_pair`."""
    def draw(generator: torch.Generator, width: int) -> Tree:
        count = ops if isinstance(ops, int) else int(torch.randint(ops[0], ops[1] + 1, (), generator=generator))
        return random_tree(count, generator, width, product_weight, vocabulary=vocabulary)

    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw(draw, forbidden, generator, width, banned_pairs, require_pair)
    return Family(name or f'expr{ops}', make)


def composed_family(forbidden: tuple[Family, ...], product_weight: float = 2.,
                    banned_pairs: frozenset[Pair] = frozenset(), name: str = 'composed') -> Family:
    """One binary operation joining two multi-step pieces that both use a or b, e.g. (a*b)+(a^b)."""
    def piece(generator: torch.Generator, width: int) -> Tree:
        while True:
            tree = random_tree(int(torch.randint(1, 3, (), generator=generator)), generator, width, product_weight)
            if has_variable(tree):
                return tree

    def draw(generator: torch.Generator, width: int) -> Tree:
        op = BASE[0][int(torch.randint(0, len(BASE[0]), (), generator=generator))]
        return (op, piece(generator, width), piece(generator, width))

    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw(draw, forbidden, generator, width, banned_pairs)
    return Family(name, make)


_banned_cache: dict[tuple[int, int], frozenset[tuple[int, ...]]] = {}


def _banned(forbidden: tuple[Family, ...], width: int) -> frozenset[tuple[int, ...]]:
    key = (id(forbidden), width)
    if key not in _banned_cache:
        _banned_cache[key] = frozenset(fingerprint(f.make(torch.Generator().manual_seed(0), width), width)
                                       for f in forbidden)
    return _banned_cache[key]


def product_step_family(forbidden: tuple[Family, ...], banned_pairs: frozenset[Pair] = frozenset(),
                        name: str = 'product-step') -> Family:
    """A product of two parts that use a or b, followed by one more step: (x*y) op z, z op (x*y), or a shift
    or not of the product — the shape the model finds hardest. Hidden relations and held-out combinations
    stay excluded."""
    def piece(generator: torch.Generator, width: int, variable: bool) -> Tree:
        while True:
            tree = random_tree(int(torch.randint(0, 2, (), generator=generator)), generator, width, 1.)
            if not variable or has_variable(tree):
                return tree

    def draw(generator: torch.Generator, width: int) -> Tree:
        product = ('*', piece(generator, width, True), piece(generator, width, True))
        roll = int(torch.randint(0, 5, (), generator=generator))
        if roll == 4:
            return (BASE[1][int(torch.randint(0, len(BASE[1]), (), generator=generator))], product)
        op = BASE[0][int(torch.randint(0, len(BASE[0]), (), generator=generator))]
        other = piece(generator, width, False)
        return (op, product, other) if roll < 2 else (op, other, product)

    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw(draw, forbidden, generator, width, banned_pairs)
    return Family(name, make)


def pair_family(forbidden: tuple[Family, ...], banned_pairs: frozenset[Pair], name: str = 'pairs') -> Family:
    """Every allowed (parent, child) operation pair equally often: pick a pair uniformly, put the child under the
    parent, and fill the remaining slots with leaves or one-operation pieces."""
    operations = list(BASE[0]) + list(BASE[1])
    allowed = [(p, c) for p in operations for c in operations if (p, c) not in banned_pairs]

    def node(op: str, generator: torch.Generator, width: int, child: Tree | None = None) -> Tree:
        def piece() -> Tree:
            return random_tree(int(torch.randint(0, 2, (), generator=generator)), generator, width)
        if op in BASE[1]:
            return (op, child if child is not None else piece())
        first, second = (child, piece()) if child is not None else (piece(), piece())
        if child is not None and bool(torch.randint(0, 2, (), generator=generator)):
            first, second = second, first
        return (op, first, second)

    def draw(generator: torch.Generator, width: int) -> Tree:
        parent, child = allowed[int(torch.randint(0, len(allowed), (), generator=generator))]
        return node(parent, generator, width, node(child, generator, width))

    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw(draw, forbidden, generator, width, banned_pairs)
    return Family(name, make)
