"""Named training sets and the test groups that go with them."""
Report = dict[str, float | None]
from .expressions import (CONCEPTS, HIDDEN_FAMILIES, NEAR_FAMILIES, Pair, composed_family, expression_family,
                          pair_family, product_step_family)
from .loops import NEW_OPERATIONS, bit_loop_family, hidden_bit_loops, hidden_loops, loop_family
from .sealed import SEALED_FAMILIES
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
# `mix5`: mix4 plus every allowed operation pair equally often. `mix6`: mix4 with training loops of up to 31
# iterations instead of 15 (63 and 255 stay longer than trained).
PAIRS: Family = pair_family(FORBIDDEN, HELD_PAIRS)
LOOPS31: tuple[Family, ...] = tuple(loop_family(kind, 5, LOOP_FORBIDDEN + hidden_loops(5))
                                    for kind in ('accumulate', 'repeat', 'recurrence')) + (bit_loop_family(),)
# `mix7`: mix3 plus expressions that also use more concepts — division, remainder, comparisons, shifts by a
# variable amount, squares and bit counts. Rotations and conditionals stay unseen (T4). No family is aimed at a
# particular test, so the product-step family of mix4 is left out. Every unseen test relation — including the T4
# operations and the sealed set — is forbidden by behaviour, so e.g. a//b itself is never drawn, only expressions
# that use //.
CONCEPT_FORBIDDEN: tuple[Family, ...] = FORBIDDEN + NEW_OPERATIONS + SEALED_FAMILIES
CONCEPT_STEPS: tuple[Family, ...] = tuple(expression_family(n, CONCEPT_FORBIDDEN, f'concepts{n}', 2., HELD_PAIRS,
                                                            vocabulary=CONCEPTS) for n in range(1, 6))
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
    'mix7': TRAIN_FAMILIES + (MULTIPLY,) + STEPS3 + LOOPS + CONCEPT_STEPS,
    'near-only': NEAR_FAMILIES,  # diagnostic: can the model compute "a product, then another step" at all?
}

ALL_NAMED: dict[str, Family] = {**NAMED, **{f.name: f for f in HIDDEN_FAMILIES + NEAR_FAMILIES + EXPRESSIONS + DEEPER
                                              + STEPS + DEEPER2 + LOOPS + (PRODUCT_STEP, PAIRS) + LOOPS31 + CONCEPT_STEPS
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
    if train_set in ('mix3', 'mix4', 'mix5', 'mix6', 'mix7'):
        groups |= {'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES} | TIERS
    return groups
