"""Relations a person would struggle to see from examples, all held out of every training set.

Scored by bits recovered (eval.recovered_bits): how far the model's guess beats chance, next to the same score for
the lookup baselines. The core is a difficulty ladder: add-rotate-xor rounds with fresh random constants per episode,
1 to 8 rounds. Few rounds hide the structure from a person but still leak it statistically; at 8 rounds the output is
indistinguishable from random, a control where nothing should beat chance. Around it:

- random: fresh 6 to 10 operation expressions per episode, longer than anything trained
- sbox: a random 4-bit substitution table inside arithmetic, as in block ciphers
- data-dependent: steps chosen by the data (Collatz-style, branches inside loops)
- automaton: a few steps of cellular automata (rules 30 and 110) on the bits: fully defined, no known shortcut

The target is the middle band: a definite rule with no textbook way to solve it from examples. Groups named
'floor:' (relations with a direct solution) only catch regressions, and 'control:' ones (hashes, 6 and 8 mixing
rounds) must score at chance, a check that the scoring is honest.

- mixes: rotations (never a training operation), xor and multiplication by odd constants
- deep: 4 to 8 operations with constants
- products: squares and products of derived numbers, which the model handles worst so far
- loops: a step repeated a fixed or data-dependent number of times
- hash: deliberately scrambling functions, a control where nothing should beat chance

Constants are 32-bit and cut to the width, so the same relation is defined at every width.
"""
import torch
from torch import Tensor

from .expressions import CONCEPTS, expression_family
from .relations import Family, Relation, fixed


def _mask(w: int) -> int:
    return (1 << w) - 1


def _rotl(x: Tensor, r: int, w: int) -> Tensor:
    x = x & _mask(w)
    r %= w
    return ((x << r) | (x >> (w - r))) & _mask(w)


def _rotr(x: Tensor, r: int, w: int) -> Tensor:
    return _rotl(x, w - r % w, w)


def _k(value: int, w: int) -> int:
    return value & _mask(w)


def _repeat_rotate_xor(a: Tensor, b: Tensor, w: int) -> Tensor:
    x, count = a & _mask(w), b & 7
    for i in range(7):
        x = (_rotl(x, 1, w) ^ b).where(count > i, x)
    return x


def _times3_xor(a: Tensor, b: Tensor, w: int) -> Tensor:
    x = a
    for _ in range(8):
        x = ((x * 3) ^ b) & _mask(w)
    return x


def _halve_add_and(a: Tensor, b: Tensor, w: int) -> Tensor:
    x = a
    for _ in range(4):
        x = ((x >> 1) + (x & b)) & _mask(w)
    return x


def _finalizer(a: Tensor, b: Tensor, w: int) -> Tensor:
    """A MurmurHash3-style finalizer of a^b: every output bit depends on every input bit."""
    h, m = (a ^ b) & _mask(w), _mask(w)
    h = ((h ^ (h >> (w // 2))) * _k(0x85EBCA6B, w)) & m
    h = ((h ^ (h >> (w // 2 - w // 8))) * _k(0xC2B2AE35, w)) & m
    return h ^ (h >> (w // 2))


def _xorshift(a: Tensor, b: Tensor, w: int) -> Tensor:
    x, m = (a + b * _k(0x9E3779B1, w)) & _mask(w), _mask(w)
    for _ in range(3):
        x ^= (x << (w // 4 + 1)) & m
        x ^= x >> (w // 4 - 1)
        x ^= (x << (w // 8 + 1)) & m
    return x


def _arx(rounds: int) -> Family:
    """`rounds` add-rotate-xor rounds on (a, b) with round constants drawn per episode."""
    def make(generator: torch.Generator, width: int) -> Relation:
        keys = [int(torch.randint(0, 1 << width, (), generator=generator)) for _ in range(rounds)]

        def f(a: Tensor, b: Tensor, w: int) -> Tensor:
            x, y, m = a & _mask(w), b & _mask(w), _mask(w)
            for key in keys:
                x = (_rotl(x ^ (key & m), 7, w) + y) & m
                y = _rotl(y, 3, w) ^ x
            return x
        return Relation(f'arx{rounds}', f)
    return Family(f'arx {rounds} round{"s" if rounds > 1 else ""}', make)


def _sbox_family() -> Family:
    """A random 4-bit substitution table applied to every nibble of a^b, then added to a."""
    def make(generator: torch.Generator, width: int) -> Relation:
        table = torch.randperm(16, generator=generator)

        def f(a: Tensor, b: Tensor, w: int) -> Tensor:
            x, out = (a ^ b) & _mask(w), torch.zeros_like(a)
            for shift in range(0, w, 4):
                out |= table.to(a.device)[(x >> shift) & 15] << shift
            return out + a
        return Relation('sbox', f)
    return Family('sbox(a^b)+a', make)


def _collatz(a: Tensor, b: Tensor, w: int) -> Tensor:
    x, m = a & _mask(w), _mask(w)
    for _ in range(6):
        x = torch.where((x & 1) == 1, (3 * x + b) & m, x >> 1)
    return x


def _branchy_loop(a: Tensor, b: Tensor, w: int) -> Tensor:
    x, m = a & _mask(w), _mask(w)
    for _ in range(6):
        x = torch.where((x & 1) == 1, (x >> 1) ^ b, (x + b) & m)
    return x


def _bit_select(a: Tensor, b: Tensor, w: int) -> Tensor:
    return torch.where(((a >> (b & 15)) & 1) == 1, a + b, a ^ (b << 1))


def _automaton(rule: int, steps: int):
    def f(a: Tensor, b: Tensor, w: int) -> Tensor:
        x = a & _mask(w)
        for _ in range(steps):
            left, right = _rotr(x, 1, w), _rotl(x, 1, w)  # bit i's neighbours: bit i+1 and bit i-1, wrapping
            x = (left ^ (x | right)) if rule == 30 else ((x ^ right) | (x & ~left))
            x &= _mask(w)
        return x ^ b
    return f


OPAQUE: dict[str, tuple[Family, ...]] = {
    'arx ladder': tuple(_arx(r) for r in (1, 2, 3, 4)),
    'random': (expression_family((6, 10), (), 'random 6-10 ops', vocabulary=CONCEPTS),),
    'sbox': (_sbox_family(),),
    'data-dependent': (
        fixed('x=a; 6 times: odd ? 3x+b : x>>1', _collatz),
        fixed('x=a; 6 times: odd ? (x>>1)^b : x+b', _branchy_loop),
        fixed('bit (b&15) of a ? a+b : a^(b<<1)', _bit_select),
    ),
    'automaton': (
        fixed('rule30 x2 on a, ^b', _automaton(30, 2)),
        fixed('rule30 x4 on a, ^b', _automaton(30, 4)),
        fixed('rule110 x2 on a, ^b', _automaton(110, 2)),
        fixed('rule110 x4 on a, ^b', _automaton(110, 4)),
    ),
    'mixes': (
        fixed('rotl(a,7)^b', lambda a, b, w: _rotl(a, 7, w) ^ b),
        fixed('rotl(a^b,13)+a', lambda a, b, w: _rotl(a ^ b, 13, w) + a),
        fixed('rotl(a*b,5)^a', lambda a, b, w: _rotl((a * b) & _mask(w), 5, w) ^ a),
        fixed('(a+rotl(b,9))*3', lambda a, b, w: (a + _rotl(b, 9, w)) * 3),
        fixed('rotr(a,3)^(b*5)', lambda a, b, w: _rotr(a, 3, w) ^ (b * 5)),
        fixed('(a*0x9E3779B1)>>half ^ b', lambda a, b, w: (((a * _k(0x9E3779B1, w)) & _mask(w)) >> (w // 2)) ^ b),
    ),
    'deep': (
        fixed('(((a^0x5A5A5A5A)+b)>>1)^(a&b)', lambda a, b, w: ((((a ^ _k(0x5A5A5A5A, w)) + b) & _mask(w)) >> 1) ^ (a & b)),
        fixed('((((a*3)^b)+(a>>2))*5)-b', lambda a, b, w: (((((a * 3) ^ b) + (a >> 2)) & _mask(w)) * 5) - b),
        fixed('(((a|b)*7)^(a&b))+(a>>4)-11', lambda a, b, w: ((((a | b) * 7) & _mask(w)) ^ (a & b)) + (a >> 4) - 11),
        fixed('((((a+b)*(a^b))>>3)^(b<<2)', lambda a, b, w: ((((a + b) * (a ^ b)) & _mask(w)) >> 3) ^ (b << 2)),
        fixed('max(a*3,b^a)-min(a,b)*2', lambda a, b, w: (a * 3 & _mask(w)).maximum(b ^ a) - a.minimum(b) * 2),
        fixed('((((((a^b)*3)+a)>>1)^b)*7)', lambda a, b, w: ((((((a ^ b) * 3) + a) & _mask(w)) >> 1) ^ b) * 7),
    ),
    'floor: products': (
        fixed('sq(a)', lambda a, b, w: a * a),
        fixed('3*(a^b)', lambda a, b, w: (a ^ b) * 3),
        fixed('a*a+b*b', lambda a, b, w: a * a + b * b),
        fixed('(a*b)^(a*a)', lambda a, b, w: ((a * b) ^ (a * a))),
        fixed('a*a-b', lambda a, b, w: a * a - b),
    ),
    'loops': (
        fixed('x=a; b&7 times: x=rotl(x,1)^b', _repeat_rotate_xor),
        fixed('x=a; 8 times: x=(x*3)^b', _times3_xor),
        fixed('x=a; 4 times: x=(x>>1)+(x&b)', _halve_add_and),
    ),
    'control: hash': (
        fixed('murmur finalizer of a^b', _finalizer),
        fixed('xorshift of a+b*K', _xorshift),
        _arx(6),
        _arx(8),
    ),
}
