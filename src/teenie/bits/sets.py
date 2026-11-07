"""Named training sets and the test groups that go with them."""
from .expressions import HIDDEN_FAMILIES, expression_family
from .relations import (BLOCK_FAMILIES, HELDOUT_FAMILIES, MULTIPLY, NAMED, STRESS_FAMILIES, TRAIN_FAMILIES,
                        Family)

# Random expressions must never behave like any relation we test as unseen.
FORBIDDEN: tuple[Family, ...] = HELDOUT_FAMILIES + HIDDEN_FAMILIES
EXPRESSIONS: tuple[Family, ...] = tuple(expression_family(n, FORBIDDEN) for n in (1, 2, 3))
DEEPER: tuple[Family, ...] = (expression_family(5, FORBIDDEN),)

# `mul` shows a*b, `blocks` only its pieces, `mix` adds random 1-3 operation expressions to `mul`.
TRAIN_SETS: dict[str, tuple[Family, ...]] = {
    'base': TRAIN_FAMILIES,
    'mul': TRAIN_FAMILIES + (MULTIPLY,),
    'mul-only': (MULTIPLY,),  # diagnostic: can the architecture compute a*b at all?
    'blocks': TRAIN_FAMILIES + BLOCK_FAMILIES,
    'mix': TRAIN_FAMILIES + (MULTIPLY,) + EXPRESSIONS,
}

ALL_NAMED: dict[str, Family] = {**NAMED, **{f.name: f for f in HIDDEN_FAMILIES + EXPRESSIONS + DEEPER}}


def test_groups(train_set: str) -> dict[str, tuple[Family, ...]]:
    """Trained families, then everything held out from that set, split by kind."""
    trained = TRAIN_SETS[train_set]
    groups = {'train': trained, 'heldout': HELDOUT_FAMILIES,
              'stress': tuple(f for f in STRESS_FAMILIES if f not in trained)}
    if train_set == 'mix':
        groups |= {'hidden': HIDDEN_FAMILIES, 'deeper': DEEPER}
    return groups
