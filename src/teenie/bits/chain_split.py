"""Exact and beyond-lookup accuracy on held-out chain pairs, split by whether the chain contains a product.

uv run python -m teenie.bits.chain_split checkpoints/wide/wide-32.pt experiments/wide/chains-split-wide-32.json
"""
import json
import sys

import torch
from teenie.bits.checkpoint import load_bit_model
from teenie.bits.data import BitSampler
from teenie.bits.baselines import nearest, window_lookup
from teenie.bits.sets import CHAINS_UNSEEN, CHAINS_UNSEEN2

model, metadata = load_bit_model(sys.argv[1], 'cpu')
stats: dict[str, list[int]] = {}
with torch.no_grad():
    for family in (CHAINS_UNSEEN, CHAINS_UNSEEN2):
        sampler = BitSampler([family], 16, 60000)
        for _ in range(48):
            e = sampler.sample(1, 16)
            kind = ('product' if '*' in e.names[0] else 'no product') + (' 1 step' if family is CHAINS_UNSEEN else ' 2 steps')
            right = (model(e.context, e.query, 16).values() == e.target).flatten()
            lookup = [(nearest(e) == e.target).flatten()] + [(window_lookup(e, w, h) == e.target).flatten()
                                                              for w in (1, 2) for h in (False, True)]
            hard = ~torch.stack(lookup).any(0)
            s = stats.setdefault(kind, [0, 0, 0, 0])
            s[0] += int(right.sum()); s[1] += len(right); s[2] += int(right[hard].sum()); s[3] += int(hard.sum())
summary = {kind: {'exact': r / n, 'beyond_lookup': rh / max(nh, 1), 'questions': n, 'lookup_fails': nh}
           for kind, (r, n, rh, nh) in sorted(stats.items())}
for kind, row in summary.items():
    print(f"{kind:20s} exact {100 * row['exact']:5.1f}%  beyond lookup {100 * row['beyond_lookup']:5.1f}% "
          f"of {row['lookup_fails']} hard questions")
if len(sys.argv) > 2:
    with open(sys.argv[2], 'w') as handle:
        json.dump({'checkpoint': sys.argv[1], 'width': 16, 'seed': 60000, 'episodes': 48, 'summary': summary}, handle,
                  indent=1)
