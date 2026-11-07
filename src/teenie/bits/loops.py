"""Relations defined by loops: the answer comes from repeating a small step n times, with n read from b.

`count_bits` sets how much of b is the repeat count: 4 -> n = b & 15 (training), 6 -> b & 63 and
8 -> b (longer loops than trained, for testing). Steps are random expression trees over the loop's own
variables, drawn fresh per episode and behaviour-checked like the plain expressions.

- accumulate:  acc = init; repeat n times: acc = body(acc, a, i)       (sums of sequences, powers, ...)
- repeat:      x = a; repeat n times: x = step(x)                      (f applied n times)
- recurrence:  x, y = a, init; repeat n times: x, y = step(x, y), x    (Fibonacci-like; answer = y)
- bits:        loops over the bit positions (popcount, reverse, parity) — they do not use n

Some have short closed forms (sum a..a+n-1 = a*n + n*(n-1)/2) and some do not; either way of getting the
answers right counts.
"""
from collections.abc import Callable

import torch
from torch import Tensor

from .expressions import Tree, _banned, _probe, evaluate_with, fingerprint, has_variable, random_tree, show
from .relations import Family, Relation

LENGTHS = {4: 'n15', 6: 'n63', 8: 'n255'}


def _count(b: Tensor, count_bits: int) -> Tensor:
    return b & ((1 << count_bits) - 1)


def accumulate(init: Tree, body: Tree, count_bits: int) -> Callable[[Tensor, Tensor, int], Tensor]:
    def run(a: Tensor, b: Tensor, width: int) -> Tensor:
        n = _count(b, count_bits)
        acc = evaluate_with(init, {'a': a}, width)
        for i in range(int(n.max()) if n.numel() else 0):
            step = evaluate_with(body, {'acc': acc, 'a': a, 'i': torch.full_like(a, i)}, width)
            acc = torch.where(i < n, step, acc)
        return acc
    return run


def repeat(step: Tree, count_bits: int) -> Callable[[Tensor, Tensor, int], Tensor]:
    def run(a: Tensor, b: Tensor, width: int) -> Tensor:
        n = _count(b, count_bits)
        x = a & ((1 << width) - 1)
        for i in range(int(n.max()) if n.numel() else 0):
            x = torch.where(i < n, evaluate_with(step, {'x': x}, width), x)
        return x
    return run


def recurrence(init: Tree, step: Tree, count_bits: int) -> Callable[[Tensor, Tensor, int], Tensor]:
    def run(a: Tensor, b: Tensor, width: int) -> Tensor:
        n = _count(b, count_bits)
        y = a & ((1 << width) - 1)  # term 0
        x = evaluate_with(init, {'a': a}, width)  # term 1
        for i in range(int(n.max()) if n.numel() else 0):
            following = evaluate_with(step, {'x': x, 'y': y}, width)
            x, y = torch.where(i < n, following, x), torch.where(i < n, x, y)
        return y
    return run


def popcount(x: Tensor, width: int) -> Tensor:
    return sum(((x >> k) & 1 for k in range(width)), torch.zeros_like(x))


def reverse(x: Tensor, width: int) -> Tensor:
    return sum((((x >> k) & 1) << (width - 1 - k) for k in range(width)), torch.zeros_like(x))


BIT_LOOPS: dict[str, Callable[[Tensor, Tensor, int], Tensor]] = {
    'popcount(a)': lambda a, b, w: popcount(a, w),
    'popcount(b)': lambda a, b, w: popcount(b, w),
    'popcount(a&b)': lambda a, b, w: popcount(a & b, w),
    'popcount(a^b)': lambda a, b, w: popcount(a ^ b, w),
    'reverse(a)': lambda a, b, w: reverse(a, w),
    'reverse(b)': lambda a, b, w: reverse(b, w),
    'reverse(a)^b': lambda a, b, w: reverse(a, w) ^ b,
    'parity(a^b)': lambda a, b, w: popcount(a ^ b, w) & 1,
    'popcount(a)+b': lambda a, b, w: popcount(a, w) + b,
}
HIDDEN_BIT_LOOPS: dict[str, Callable[[Tensor, Tensor, int], Tensor]] = {
    'popcount(a|b)': lambda a, b, w: popcount(a | b, w),
    'reverse(a^b)': lambda a, b, w: reverse(a ^ b, w),
    'popcount(a)*b': lambda a, b, w: popcount(a, w) * b,
}


def _fixed(name: str, f: Callable[[Tensor, Tensor, int], Tensor]) -> Family:
    relation = Relation(name, f)
    return Family(name, lambda generator, width: relation)


def hidden_loops(count_bits: int) -> tuple[Family, ...]:
    """Loop bodies never generated for training (checked by behaviour at the training length too)."""
    suffix = LENGTHS[count_bits]
    return (
        _fixed(f'sum a..a+n-1 {suffix}', accumulate(0, ('+', 'acc', ('+', 'a', 'i')), count_bits)),
        _fixed(f'a^n {suffix}', accumulate(1, ('*', 'acc', 'a'), count_bits)),
        _fixed(f'fib(a,a) {suffix}', recurrence('a', ('+', 'x', 'y'), count_bits)),
        _fixed(f'x->3x+1 {suffix}', repeat(('+', ('*', 'x', 3), 1), count_bits)),
        _fixed(f'acc=(acc<<1)^a {suffix}', accumulate('a', ('^', ('<<1', 'acc'), 'a'), count_bits)),
    )


def _draw_loop(build: Callable[[torch.Generator, int], tuple[str, Callable[[Tensor, Tensor, int], Tensor]]],
               forbidden: tuple[Family, ...], generator: torch.Generator, width: int) -> Relation:
    banned = _banned(forbidden, width)
    for _ in range(300):
        name, run = build(generator, width)
        relation = Relation(name, run)
        outputs = fingerprint(relation, width)
        # The repeat count must matter: compare with the same loop run zero times.
        a, b = _probe(width)
        unrolled = relation(a, torch.zeros_like(b), width)
        changed = float((relation(a, b, width) != unrolled).float().mean())
        if len(set(outputs)) >= 8 and outputs not in banned and changed >= .25:
            return relation
    raise RuntimeError('could not draw a usable loop')


def _tree(generator: torch.Generator, width: int, leaves: tuple[str, ...], need: tuple[str, ...]) -> Tree:
    """A 1-2 operation tree over `leaves` that uses every variable in `need`."""
    while True:
        tree = random_tree(int(torch.randint(1, 3, (), generator=generator)), generator, width, 2., leaves)
        if all(has_variable(tree, (name,)) for name in need):
            return tree


def _start(generator: torch.Generator, width: int) -> Tree:
    roll = int(torch.randint(0, 3, (), generator=generator))
    return 0 if roll == 0 else 'a' if roll == 1 else int(torch.randint(1, 1 << width, (), generator=generator))


def loop_family(kind: str, count_bits: int, forbidden: tuple[Family, ...]) -> Family:
    """Fresh random loops of one kind per episode; `forbidden` relations are redrawn."""
    suffix = LENGTHS[count_bits]

    def build(generator: torch.Generator, width: int) -> tuple[str, Callable[[Tensor, Tensor, int], Tensor]]:
        if kind == 'accumulate':
            init, body = _start(generator, width), _tree(generator, width, ('acc', 'a', 'i'), ('acc',))
            return f'acc={show(init)}; n: acc={show(body)}', accumulate(init, body, count_bits)
        if kind == 'repeat':
            step = _tree(generator, width, ('x',), ('x',))
            return f'x=a; n: x={show(step)}', repeat(step, count_bits)
        if kind == 'recurrence':
            init, step = _start(generator, width), _tree(generator, width, ('x', 'y'), ('x', 'y'))
            return f'y,x=a,{show(init)}; n: x={show(step)}', recurrence(init, step, count_bits)
        raise ValueError(kind)

    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw_loop(build, forbidden, generator, width)
    return Family(f'{kind} {suffix}', make)


def bit_loop_family() -> Family:
    names = list(BIT_LOOPS)

    def make(generator: torch.Generator, width: int) -> Relation:
        name = names[int(torch.randint(0, len(names), (), generator=generator))]
        return Relation(name, BIT_LOOPS[name])
    return Family('bit loops', make)


def hidden_bit_loops() -> tuple[Family, ...]:
    return tuple(_fixed(name, f) for name, f in HIDDEN_BIT_LOOPS.items())


# T4: new operation types made from known pieces, never trained.
NEW_OPERATIONS: tuple[Family, ...] = (
    _fixed('a//b', lambda a, b, w: a // torch.clamp(b, min=1)),
    _fixed('a%b', lambda a, b, w: torch.where(b == 0, a, a % torch.clamp(b, min=1))),
    _fixed('rotl(a,1)', lambda a, b, w: (a << 1) | (a >> (w - 1))),
    _fixed('rotl(a,b&7)', lambda a, b, w: (a << (b & 7)) | (a >> ((w - (b & 7)) % w))),
    _fixed('a>b?a-b:a+b', lambda a, b, w: torch.where(a > b, a - b, a + b)),
    _fixed('a>b?a*b:b', lambda a, b, w: torch.where(a > b, a * b, b)),
    _fixed('a<b?b^a:a&b', lambda a, b, w: torch.where(a < b, b ^ a, a & b)),
)
