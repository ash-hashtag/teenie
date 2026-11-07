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


Steps = Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor]]


def _iterate(start: Callable[[Tensor, Tensor, int], list[Tensor]],
             update: Callable[[list[Tensor], Tensor, int, int], list[Tensor]], count_bits: int) -> Steps:
    """Run a loop n = b & (2**count_bits - 1) times; each iteration is one step. The loop's answer is its first
    state variable. Returns its value after every iteration (questions with fewer iterations keep their final
    value) and the number of iterations (at least 1)."""
    def run(a: Tensor, b: Tensor, width: int) -> tuple[Tensor, Tensor]:
        n = _count(b, count_bits)
        state = start(a, b, width)
        values = []
        for i in range(int(n.max()) if n.numel() else 0):
            following = update(state, a, i, width)
            state = [torch.where(i < n, new, old) for new, old in zip(following, state)]
            values.append(state[0])
        trace = torch.stack(values) if values else state[0][None]
        return trace, n.clamp(min=1)
    return run


def accumulate_steps(init: Tree, body: Tree, count_bits: int) -> Steps:
    return _iterate(lambda a, b, w: [evaluate_with(init, {'a': a}, w)],
                    lambda s, a, i, w: [evaluate_with(body, {'acc': s[0], 'a': a, 'i': torch.full_like(a, i)}, w)],
                    count_bits)


def repeat_steps(step: Tree, count_bits: int) -> Steps:
    return _iterate(lambda a, b, w: [a & ((1 << w) - 1)],
                    lambda s, a, i, w: [evaluate_with(step, {'x': s[0]}, w)], count_bits)


def recurrence_steps(init: Tree, step: Tree, count_bits: int) -> Steps:
    """Term 0 is a, term 1 is init; each iteration moves one term on. The answer (state[0]) is term n."""
    return _iterate(lambda a, b, w: [a & ((1 << w) - 1), evaluate_with(init, {'a': a}, w)],
                    lambda s, a, i, w: [s[1], evaluate_with(step, {'x': s[1], 'y': s[0]}, w)], count_bits)


def _final(steps: Steps) -> Callable[[Tensor, Tensor, int], Tensor]:
    return lambda a, b, w: steps(a, b, w)[0][-1]


def accumulate(init: Tree, body: Tree, count_bits: int) -> Callable[[Tensor, Tensor, int], Tensor]:
    return _final(accumulate_steps(init, body, count_bits))


def repeat(step: Tree, count_bits: int) -> Callable[[Tensor, Tensor, int], Tensor]:
    return _final(repeat_steps(step, count_bits))


def recurrence(init: Tree, step: Tree, count_bits: int) -> Callable[[Tensor, Tensor, int], Tensor]:
    return _final(recurrence_steps(init, step, count_bits))


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


def _fixed(name: str, f: Callable[[Tensor, Tensor, int], Tensor], steps: Steps | None = None) -> Family:
    relation = Relation(name, f, steps)
    return Family(name, lambda generator, width: relation)


def _fixed_loop(name: str, steps: Steps) -> Family:
    return _fixed(name, _final(steps), steps)


def hidden_loops(count_bits: int) -> tuple[Family, ...]:
    """Loop bodies never generated for training (checked by behaviour at the training length too)."""
    suffix = LENGTHS[count_bits]
    return (
        _fixed_loop(f'sum a..a+n-1 {suffix}', accumulate_steps(0, ('+', 'acc', ('+', 'a', 'i')), count_bits)),
        _fixed_loop(f'a^n {suffix}', accumulate_steps(1, ('*', 'acc', 'a'), count_bits)),
        _fixed_loop(f'fib(a,a) {suffix}', recurrence_steps('a', ('+', 'x', 'y'), count_bits)),
        _fixed_loop(f'x->3x+1 {suffix}', repeat_steps(('+', ('*', 'x', 3), 1), count_bits)),
        _fixed_loop(f'acc=(acc<<1)^a {suffix}', accumulate_steps('a', ('^', ('<<1', 'acc'), 'a'), count_bits)),
    )


def _draw_loop(build: Callable[[torch.Generator, int], tuple[str, Steps]],
               forbidden: tuple[Family, ...], generator: torch.Generator, width: int) -> Relation:
    banned = _banned(forbidden, width)
    for _ in range(300):
        name, steps = build(generator, width)
        relation = Relation(name, _final(steps), steps)
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

    def build(generator: torch.Generator, width: int) -> tuple[str, Steps]:
        if kind == 'accumulate':
            init, body = _start(generator, width), _tree(generator, width, ('acc', 'a', 'i'), ('acc',))
            return f'acc={show(init)}; n: acc={show(body)}', accumulate_steps(init, body, count_bits)
        if kind == 'repeat':
            step = _tree(generator, width, ('x',), ('x',))
            return f'x=a; n: x={show(step)}', repeat_steps(step, count_bits)
        if kind == 'recurrence':
            init, step = _start(generator, width), _tree(generator, width, ('x', 'y'), ('x', 'y'))
            return f'y,x=a,{show(init)}; n: x={show(step)}', recurrence_steps(init, step, count_bits)
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
