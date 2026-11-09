"""Named training sets and the test groups that go with them."""
import torch
Report = dict[str, float | None]
from .expressions import (CONCEPTS, HIDDEN_FAMILIES, NEAR_FAMILIES, Pair, Tree, _draw, composed_family,
                          expression_family, named, pair_family, product_step_family)
from .loops import NEW_OPERATIONS, bit_loop_family, hidden_bit_loops, hidden_loops, loop_family
from .sealed import SEALED_FAMILIES
from .relations import (BLOCK_FAMILIES, HELDOUT_FAMILIES, MULTIPLY, NAMED, STRESS_FAMILIES, TRAIN_FAMILIES,
                        Family, Relation)

# Random expressions must never behave like any relation we test as unseen.
FORBIDDEN: tuple[Family, ...] = HELDOUT_FAMILIES + HIDDEN_FAMILIES
EXPRESSIONS: tuple[Family, ...] = tuple(expression_family(n, FORBIDDEN) for n in (1, 2, 3))
DEEPER: tuple[Family, ...] = (expression_family(5, FORBIDDEN),)
# `mix2`: more steps (1-5 operations), products twice as likely, and compositions of multi-step pieces.
STEPS: tuple[Family, ...] = tuple(expression_family(n, FORBIDDEN, f'steps{n}', product_weight=2.)
                                  for n in range(1, 6)) + (composed_family(FORBIDDEN),)
DEEPER2: tuple[Family, ...] = (expression_family(7, FORBIDDEN, 'expr7', product_weight=2.),)

# `mix3`: mix2 plus loops, with whole operation combinations held out (never e.g. a xor applied to a product).
HELD_PAIRS: frozenset[Pair] = frozenset({('^', '*'), ('*', '-'), ('min', '+'), ('>>1', '^')})
LOOP_FORBIDDEN: tuple[Family, ...] = FORBIDDEN + hidden_loops(4)
STEPS3: tuple[Family, ...] = tuple(expression_family(n, FORBIDDEN, f'steps{n}', 2., HELD_PAIRS)
                                   for n in range(1, 6)) + (composed_family(FORBIDDEN, 2., HELD_PAIRS),)
LOOPS: tuple[Family, ...] = tuple(loop_family(kind, 4, LOOP_FORBIDDEN)
                                  for kind in ('accumulate', 'repeat', 'recurrence')) + (bit_loop_family(),)
# `mix4`: mix3 plus a family aimed at the weakest shape, a product followed by another step.
PRODUCT_STEP: Family = product_step_family(FORBIDDEN, HELD_PAIRS)
# `mix5`: mix4 plus every allowed operation pair equally often. `mix6`: mix4 with training loops of up to 31
# iterations instead of 15 (63 and 255 stay longer than trained).
PAIRS: Family = pair_family(FORBIDDEN, HELD_PAIRS)
LOOPS31: tuple[Family, ...] = tuple(loop_family(kind, 5, LOOP_FORBIDDEN + hidden_loops(5))
                                    for kind in ('accumulate', 'repeat', 'recurrence')) + (bit_loop_family(),)
# `mix7`: mix3 plus expressions that also use more concepts — division, remainder, comparisons, shifts by a
# variable amount, squares and bit counts. Rotations and conditionals stay unseen (T4). It keeps mix4's
# product-then-a-step family: a basic shape the model should recognise. Every unseen test relation — including the T4
# operations and the sealed set — is forbidden by behaviour, so e.g. a//b itself is never drawn, only expressions
# that use //.
# The target (from 2026-11-08): unseen expressions of several operations, at up to 32 bits, where a right answer
# cannot be luck. One written by the user, and random 4-6 operation expressions over every concept, always with a
# held-out operation pair and at least 100 different answers on 128 random inputs (so nearly constant expressions,
# which lookup can answer, are left out). Both are excluded from training by behaviour.
TARGET: tuple[Family, ...] = (
    named('((a*b)^(a>>3))|(b+9)', ('|', ('^', ('*', 'a', 'b'), ('>>v', 'a', 3)), ('+', 'b', 9))),
)
CONCEPT_FORBIDDEN: tuple[Family, ...] = FORBIDDEN + NEW_OPERATIONS + SEALED_FAMILIES + TARGET
TARGET += (expression_family((4, 6), CONCEPT_FORBIDDEN, 'T6 4-6 ops, all concepts', 2., HELD_PAIRS,
                             require_pair=True, vocabulary=CONCEPTS, distinct=100),)
CONCEPT_STEPS: tuple[Family, ...] = tuple(expression_family(n, CONCEPT_FORBIDDEN, f'concepts{n}', 2., HELD_PAIRS,
                                                            vocabulary=CONCEPTS) for n in range(1, 6))
# `compose`: compositions only — expressions of 2-5 operations (with and without the extra concepts), products
# followed by a step, and loops. No relation is a single operation: primitives are only practised inside
# compositions, and are watched as a test group so forgetting them shows.
COMPOSE: tuple[Family, ...] = STEPS3[1:] + CONCEPT_STEPS[1:] + (PRODUCT_STEP,) + LOOPS
PRIMITIVES: tuple[Family, ...] = TRAIN_FAMILIES + (MULTIPLY, STEPS3[0], CONCEPT_STEPS[0])
def pool_family(base: Family, size: int, seed: int = 123, name: str | None = None) -> Family:
    """A fixed pool of `size` relations drawn once from `base` (per width), sampled uniformly: how many different
    relations training covers, with every one of them seen often."""
    pools: dict[int, list[Relation]] = {}

    def make(generator: torch.Generator, width: int) -> Relation:
        if width not in pools:
            drawing = torch.Generator().manual_seed(seed)
            pools[width] = [base.make(drawing, width) for _ in range(size)]
        return pools[width][int(torch.randint(0, size, (), generator=generator))]
    return Family(name or f'{base.name} pool{size}', make)


# The diversity ladder (section 25): pools of 16, 64 and 256 product-then-step relations.
POOLS: dict[int, Family] = {size: pool_family(PRODUCT_STEP, size) for size in (16, 64, 256)}
# Composition grid (diagnostic only, section 26): 8 product-like first steps × 8 second steps. 52 cells are
# trained, each seen often; 12 are held out so that every first step and every second step is still trained in
# other combinations. Held-out cells test combining known parts in an unseen way. Real relations are not limited to
# a grid like this; it only isolates that one question.
GRID_FIRST: tuple[tuple[str, Tree], ...] = (
    ('a*b', ('*', 'a', 'b')), ('a*a', ('*', 'a', 'a')), ('b*b', ('*', 'b', 'b')),
    ('(a+1)*b', ('*', ('+', 'a', 1), 'b')), ('a*(b>>1)', ('*', 'a', ('>>1', 'b'))),
    ('(a^b)*a', ('*', ('^', 'a', 'b'), 'a')), ('(a|b)*b', ('*', ('|', 'a', 'b'), 'b')),
    ('(a-b)*(a+b)', ('*', ('-', 'a', 'b'), ('+', 'a', 'b'))),
)
GRID_SECOND: tuple[tuple[str, str, Tree | None], ...] = (
    ('+a', '+', 'a'), ('+b', '+', 'b'), ('^a', '^', 'a'), ('^b', '^', 'b'), ('-a', '-', 'a'), ('-b', '-', 'b'),
    ('|b', '|', 'b'), ('>>1', '>>1', None),
)
GRID_HELD: frozenset[tuple[int, int]] = frozenset({(i, i) for i in range(8)} | {(i, (i + 4) % 8) for i in range(4)})


def _grid_cell(i: int, j: int) -> Family:
    (first_name, first), (second_name, op, operand) = GRID_FIRST[i], GRID_SECOND[j]
    tree: Tree = (op, first) if operand is None else (op, first, operand)
    return named(f'({first_name}){second_name}', tree)


GRID_TRAIN: tuple[Family, ...] = tuple(_grid_cell(i, j) for i in range(8) for j in range(8) if (i, j) not in GRID_HELD)
GRID_UNSEEN: tuple[Family, ...] = tuple(_grid_cell(i, j) for i, j in sorted(GRID_HELD))


# Chains (section 27): a start followed by 1-2 steps, each step one of 21 types — + - ^ | & * with a, b or a fresh
# random constant, or >>1, <<1, not. Thousands of combinations, each rare, while every part is frequent: too many to
# memorise one by one. About one in seven (start, first step) pairs never appear together in training — including
# a*b followed by +a — and are the test.
CHAIN_STARTS: tuple[tuple[str, Tree], ...] = (
    ('a*b', ('*', 'a', 'b')), ('a+b', ('+', 'a', 'b')), ('a-b', ('-', 'a', 'b')), ('a^b', ('^', 'a', 'b')),
    ('a&b', ('&', 'a', 'b')), ('a|b', ('|', 'a', 'b')), ('a*a', ('*', 'a', 'a')), ('b*b', ('*', 'b', 'b')),
    ('min(a,b)', ('min', 'a', 'b')), ('max(a,b)', ('max', 'a', 'b')),
)
CHAIN_STEPS: tuple[tuple[str, str | None], ...] = tuple(
    (op, operand) for op in ('+', '-', '^', '|', '&', '*') for operand in ('a', 'b', 'k')) + (
    ('>>1', None), ('<<1', None), ('~', None))
CHAIN_HELD: frozenset[tuple[int, int]] = frozenset(
    {(i, j) for i in range(len(CHAIN_STARTS)) for j in range(len(CHAIN_STEPS)) if (3 * i + 5 * j) % 7 == 0}
    | {(0, 0)})  # (a*b, +a)


# Product starts (section 29): products of simple pieces, so that half the training can go through a product.
PRODUCT_STARTS: tuple[tuple[str, Tree], ...] = (
    ('a*b', ('*', 'a', 'b')), ('a*a', ('*', 'a', 'a')), ('b*b', ('*', 'b', 'b')),
    ('(a+1)*b', ('*', ('+', 'a', 1), 'b')), ('a*(b+1)', ('*', 'a', ('+', 'b', 1))),
    ('(a^b)*b', ('*', ('^', 'a', 'b'), 'b')), ('(a|b)*a', ('*', ('|', 'a', 'b'), 'a')),
    ('(a-b)*b', ('*', ('-', 'a', 'b'), 'b')), ('(a+b)*(a-b)', ('*', ('+', 'a', 'b'), ('-', 'a', 'b'))),
    ('(a>>1)*b', ('*', ('>>1', 'a'), 'b')),
)


def chain_held(start: str, step: int) -> bool:
    """Whether a (start, first step) pair is held out of all chain training — the same rule for every chain
    family, so a pair held out of one is never trained in another."""
    names = [name for name, _ in CHAIN_STARTS]
    if start in names:
        return (names.index(start), step) in CHAIN_HELD
    return (3 * sum(map(ord, start)) + 5 * step) % 7 == 0


def chain_family(held: bool, steps: tuple[int, int] = (1, 2), name: str | None = None,
                 starts: tuple[tuple[str, Tree], ...] = CHAIN_STARTS) -> Family:
    """A start then `steps` steps. `held`: the (start, first step) pair is held out (test); otherwise never."""
    def draw(generator: torch.Generator, width: int) -> Tree:
        while True:
            start = int(torch.randint(0, len(starts), (), generator=generator))
            first = int(torch.randint(0, len(CHAIN_STEPS), (), generator=generator))
            if chain_held(starts[start][0], first) == held:
                break
        tree: Tree = starts[start][1]
        count = int(torch.randint(steps[0], steps[1] + 1, (), generator=generator))
        for index in range(count):
            step = first if index == 0 else int(torch.randint(0, len(CHAIN_STEPS), (), generator=generator))
            op, operand = CHAIN_STEPS[step]
            if operand is None:
                tree = (op, tree)
            else:
                leaf: Tree = (int(torch.randint(1, 1 << width, (), generator=generator)) if operand == 'k'
                              else operand)
                tree = (op, tree, leaf)
        return tree

    def make(generator: torch.Generator, width: int) -> Relation:
        return _draw(draw, () if held else FORBIDDEN, generator, width)
    return Family(name or ('chains, unseen pair' if held else 'chains'), make)


CHAINS: Family = chain_family(False)
CHAINS_UNSEEN: Family = chain_family(True, (1, 1), 'chains, unseen pair, 1 step')
CHAINS_UNSEEN2: Family = chain_family(True, (2, 2), 'chains, unseen pair, 2 steps')
PRODUCT_CHAINS: Family = chain_family(False, name='product chains', starts=PRODUCT_STARTS)
PRODUCT_CHAINS_UNSEEN: Family = chain_family(True, (1, 1), 'product chains, unseen pair, 1 step', PRODUCT_STARTS)
PRODUCT_CHAINS_UNSEEN2: Family = chain_family(True, (2, 2), 'product chains, unseen pair, 2 steps', PRODUCT_STARTS)
# The difficulty ladder (see docs/findings.md): T1-T3 always contain a held-out combination.
TIERS: dict[str, tuple[Family, ...]] = {
    'T1': (expression_family(2, FORBIDDEN, 'T1 2 ops', 2., HELD_PAIRS, require_pair=True),),
    'T2': (expression_family((3, 4), FORBIDDEN, 'T2 3-4 ops', 2., HELD_PAIRS, require_pair=True),),
    'T3': (expression_family((5, 7), FORBIDDEN, 'T3 5-7 ops', 2., HELD_PAIRS, require_pair=True),),
    'T4': NEW_OPERATIONS,
    # T5a: loop bodies never generated, at the trained length (up to 15 iterations), and unseen bit loops —
    # recognising a new loop's pattern. T5b: loops run longer than any trained (63 and 255 iterations) —
    # repeating a step further than trained, a research stretch rather than a pass/fail target.
    'T5a': hidden_loops(4) + hidden_bit_loops(),
    'T5b': hidden_loops(6) + hidden_loops(8)
           + tuple(loop_family(kind, bits, LOOP_FORBIDDEN + hidden_loops(bits))
                   for bits in (6, 8) for kind in ('accumulate', 'repeat', 'recurrence')),
}


def regroup(results: dict[str, dict[str, Report]]) -> dict[str, dict[str, Report]]:
    """Re-sort an evaluation's relations into the current ladder tiers by name, so evaluations written before a
    tier was split (e.g. one 'T5') report the current tiers."""
    tier_of = {f.name: tier for tier, families in TIERS.items() for f in families}
    regrouped: dict[str, dict[str, Report]] = {}
    for group, reports in results.items():
        for name, report in reports.items():
            regrouped.setdefault(tier_of.get(name, group), {})[name] = report
    return regrouped

# `mul` shows a*b, `blocks` only its pieces, `mix` adds random 1-3 operation expressions to `mul`.
TRAIN_SETS: dict[str, tuple[Family, ...]] = {
    'base': TRAIN_FAMILIES,
    'mul': TRAIN_FAMILIES + (MULTIPLY,),
    'mul-only': (MULTIPLY,),  # diagnostic: can the architecture compute a*b at all?
    'blocks': TRAIN_FAMILIES + BLOCK_FAMILIES,
    'mix': TRAIN_FAMILIES + (MULTIPLY,) + EXPRESSIONS,
    'mix2': TRAIN_FAMILIES + (MULTIPLY,) + STEPS,
    'mix3': TRAIN_FAMILIES + (MULTIPLY,) + STEPS3 + LOOPS,
    'mix4': TRAIN_FAMILIES + (MULTIPLY,) + STEPS3 + LOOPS + (PRODUCT_STEP,),
    'mix5': TRAIN_FAMILIES + (MULTIPLY,) + STEPS3 + LOOPS + (PRODUCT_STEP, PAIRS),
    'mix6': TRAIN_FAMILIES + (MULTIPLY,) + STEPS3 + LOOPS31 + (PRODUCT_STEP,),
    'mix7': TRAIN_FAMILIES + (MULTIPLY,) + STEPS3 + LOOPS + (PRODUCT_STEP,) + CONCEPT_STEPS,
    'compose': COMPOSE,
    'grid': GRID_TRAIN,
    'chains': (CHAINS,),
    'chains-products': (CHAINS, PRODUCT_CHAINS),  # half the episodes through a product (adaptive sampling)
    **{f'ps-pool{size}': (family,) for size, family in POOLS.items()},
    'near-only': NEAR_FAMILIES,  # diagnostic: can the model compute "a product, then another step" at all?
    # diagnostic: random products-then-a-step only; tested on fresh ones (fit) and on hidden compositions such as
    # a*b+a or (a*b)^b that the generator never produces (generalization).
    'product-step-only': (PRODUCT_STEP,),
}

ALL_NAMED: dict[str, Family] = {**NAMED, **{f.name: f for f in HIDDEN_FAMILIES + NEAR_FAMILIES + EXPRESSIONS + DEEPER
                                              + STEPS + DEEPER2 + LOOPS + (PRODUCT_STEP, PAIRS) + LOOPS31 + CONCEPT_STEPS + TARGET + GRID_TRAIN + GRID_UNSEEN
                                              + (CHAINS, CHAINS_UNSEEN, CHAINS_UNSEEN2, PRODUCT_CHAINS,
                                                 PRODUCT_CHAINS_UNSEEN, PRODUCT_CHAINS_UNSEEN2)
                                              + SEALED_FAMILIES
                                              + tuple(f for tier in TIERS.values() for f in tier)}}


def test_groups(train_set: str) -> dict[str, tuple[Family, ...]]:
    """Trained families, then everything held out from that set, split by kind. 'sealed' is the sealed set of
    everyday functions (see sealed.py), only to be read at the end of a round of experiments."""
    if train_set == 'sealed':
        return {'sealed': SEALED_FAMILIES}
    trained = TRAIN_SETS[train_set]
    groups = {'train': trained, 'heldout': HELDOUT_FAMILIES,
              'stress': tuple(f for f in STRESS_FAMILIES if f not in trained)}
    if train_set == 'mix':
        groups |= {'hidden': HIDDEN_FAMILIES, 'deeper': DEEPER}
    if train_set == 'mix2':
        groups |= {'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES, 'deeper': DEEPER2}
    if train_set in ('chains', 'chains-products'):
        groups |= {'unseen': (CHAINS_UNSEEN, CHAINS_UNSEEN2, PRODUCT_CHAINS_UNSEEN, PRODUCT_CHAINS_UNSEEN2),
                   'hidden': HIDDEN_FAMILIES}
    if train_set == 'grid':
        groups |= {'unseen': GRID_UNSEEN, 'hidden': HIDDEN_FAMILIES}
    if train_set.startswith('ps-pool'):
        # fresh draws from the whole family (mostly outside the pool), and relations it never produces
        groups |= {'family': (PRODUCT_STEP,), 'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES}
    if train_set == 'product-step-only':
        groups |= {'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES}
    if train_set in ('mix3', 'mix4', 'mix5', 'mix6', 'mix7', 'compose'):
        groups |= {'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES} | TIERS | {'target': TARGET}
    if train_set == 'compose':
        groups['primitives'] = PRIMITIVES  # watched only
    return groups
