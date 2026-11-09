"""Exact and beyond-lookup accuracy on held-out chain pairs, split by whether the chain contains a product.

uv run python -m teenie.bits.chain_split checkpoints/wide/wide-32.pt experiments/wide/chains-split-wide-32.json [width] [products]

With `products`, also the product-start chains (section 29). Also how often the model is sure (least certain bit >= 0.99) and how often its sure answers are right: a low score
is still useful if the model knows which of its answers are the right ones.
"""
import json
import sys

import torch
from teenie.bits.checkpoint import load_bit_model
from teenie.bits.data import BitSampler
from teenie.bits.eval import be_gentle_on_cpu
from teenie.bits.baselines import nearest, window_lookup
from teenie.bits.sets import CHAINS_UNSEEN, CHAINS_UNSEEN2, PRODUCT_CHAINS_UNSEEN, PRODUCT_CHAINS_UNSEEN2

be_gentle_on_cpu()
model, metadata = load_bit_model(sys.argv[1], 'cpu')
width = int(sys.argv[3]) if len(sys.argv) > 3 else 16
stats: dict[str, list[int]] = {}
with torch.no_grad():
    families = (CHAINS_UNSEEN, CHAINS_UNSEEN2) + ((PRODUCT_CHAINS_UNSEEN, PRODUCT_CHAINS_UNSEEN2)
                                                  if len(sys.argv) > 4 and sys.argv[4] == 'products' else ())
    for family in families:
        sampler = BitSampler([family], width, 60000)
        for _ in range(48):
            e = sampler.sample(1, 16)
            steps = ' 1 step' if family in (CHAINS_UNSEEN, PRODUCT_CHAINS_UNSEEN) else ' 2 steps'
            if family in (PRODUCT_CHAINS_UNSEEN, PRODUCT_CHAINS_UNSEEN2):
                kind = 'product start' + steps
            else:
                kind = ('product' if '*' in e.names[0] else 'no product') + steps
            prediction = model(e.context, e.query, width)
            right = (prediction.values() == e.target).flatten()
            sure = (prediction.confidence() >= .99).flatten()
            lookup = [(nearest(e) == e.target).flatten()] + [(window_lookup(e, w, h) == e.target).flatten()
                                                              for w in (1, 2) for h in (False, True)]
            hard = ~torch.stack(lookup).any(0)
            s = stats.setdefault(kind, [0, 0, 0, 0, 0, 0])
            s[0] += int(right.sum()); s[1] += len(right); s[2] += int(right[hard].sum()); s[3] += int(hard.sum())
            s[4] += int(sure.sum()); s[5] += int(right[sure].sum())
summary = {kind: {'exact': r / n, 'beyond_lookup': rh / max(nh, 1), 'questions': n, 'lookup_fails': nh,
                  'sure_coverage': ns / n, 'sure_accuracy': rs / ns if ns else None}
           for kind, (r, n, rh, nh, ns, rs) in sorted(stats.items())}
for kind, (r, n, rh, nh, ns, rs) in sorted(stats.items()):
    right_when_sure = f'{100 * rs / ns:.1f}%' if ns else '-'
    print(f'{kind:20s} exact {100 * r / n:5.1f}%  beyond lookup {100 * rh / max(nh, 1):5.1f}% of {nh} hard questions'
          f'  sure {100 * ns / n:5.1f}% (right {right_when_sure})')
if len(sys.argv) > 2:
    with open(sys.argv[2], 'w') as handle:
        json.dump({'checkpoint': sys.argv[1], 'width': width, 'seed': 60000, 'episodes': 48, 'summary': summary}, handle,
                  indent=1)
