"""Named training sets and the test groups that go with them."""
from .expressions import HIDDEN_FAMILIES, NEAR_FAMILIES, composed_family, expression_family
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

# `mul` shows a*b, `blocks` only its pieces, `mix` adds random 1-3 operation expressions to `mul`.
TRAIN_SETS: dict[str, tuple[Family, ...]] = {
    'base': TRAIN_FAMILIES,
    'mul': TRAIN_FAMILIES + (MULTIPLY,),
    'mul-only': (MULTIPLY,),  # diagnostic: can the architecture compute a*b at all?
    'blocks': TRAIN_FAMILIES + BLOCK_FAMILIES,
    'mix': TRAIN_FAMILIES + (MULTIPLY,) + EXPRESSIONS,
    'mix2': TRAIN_FAMILIES + (MULTIPLY,) + STEPS,
    'near-only': NEAR_FAMILIES,  # diagnostic: can the model compute "a product, then another step" at all?
}

ALL_NAMED: dict[str, Family] = {**NAMED, **{f.name: f for f in HIDDEN_FAMILIES + NEAR_FAMILIES + EXPRESSIONS + DEEPER
                                              + STEPS + DEEPER2}}


def test_groups(train_set: str) -> dict[str, tuple[Family, ...]]:
    """Trained families, then everything held out from that set, split by kind."""
    trained = TRAIN_SETS[train_set]
    groups = {'train': trained, 'heldout': HELDOUT_FAMILIES,
              'stress': tuple(f for f in STRESS_FAMILIES if f not in trained)}
    if train_set == 'mix':
        groups |= {'hidden': HIDDEN_FAMILIES, 'deeper': DEEPER}
    if train_set == 'mix2':
        groups |= {'near': NEAR_FAMILIES, 'hidden': HIDDEN_FAMILIES, 'deeper': DEEPER2}
    return groups
