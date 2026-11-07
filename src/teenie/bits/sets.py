"""Named training sets and the test groups that go with them."""
from .expressions import (HIDDEN_FAMILIES, NEAR_FAMILIES, Pair, composed_family, expression_family,
                          product_step_family)
from .loops import NEW_OPERATIONS, bit_loop_family, hidden_bit_loops, hidden_loops, loop_family
from .relations import (BLOCK_FAMILIES, HELDOUT_FAMILIES, MULTIPLY, NAMED, STRESS_FAMILIES, TRAIN_FAMILIES,
                        Family)

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
# The difficulty ladder (see docs/findings.md): T1-T3 always contain a held-out combination.
TIERS: dict[str, tuple[Family, ...]] = {
    'T1': (expression_family(2, FORBIDDEN, 'T1 2 ops', 2., HELD_PAIRS, require_pair=True),),
    'T2': (expression_family((3, 4), FORBIDDEN, 'T2 3-4 ops', 2., HELD_PAIRS, require_pair=True),),
    'T3': (expression_family((5, 7), FORBIDDEN, 'T3 5-7 ops', 2., HELD_PAIRS, require_pair=True),),
    'T4': NEW_OPERATIONS,
    'T5': hidden_loops(4) + hidden_loops(6) + hidden_loops(8)
          + tuple(loop_family(kind, bits, LOOP_FORBIDDEN + hidden_loops(bits))
                  for bits in (6, 8) for kind in ('accumulate', 'repeat', 'recurrence'))
          + hidden_bit_loops(),
}

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
    'near-only': NEAR_FAMILIES,  # diagnostic: can the model compute "a product, then another step" at all?
}

ALL_NAMED: dict[str, Family] = {**NAMED, **{f.name: f for f in HIDDEN_FAMILIES + NEAR_FAMILIES + EXPRESSIONS + DEEPER
                                              + STEPS + DEEPER2 + LOOPS + (PRODUCT_STEP,)
                                              + tuple(f for tier in TIERS.values() for f in tier)}}


def test_groups(train_set: str) -> dict[str, tuple[Family, ...]]:
    """Trained families, then everything held out from that set, split by kind."""
    trained = TRAIN_SETS[train_set]
    groups = {'train': trained, 'heldout': HELDOUT_FAMILIES,
              'stress': tuple(f for f in STRESS_FAMILIES if f not in trained)}
    if train_set == 'mix':
        groups |= {'hidden': HIDDEN_FAMILIES, 'deeper': DEEPER}
    if train_set == 'mix2':
        groups |= {'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES, 'deeper': DEEPER2}
    if train_set in ('mix3', 'mix4'):
        groups |= {'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES} | TIERS
    return groups
