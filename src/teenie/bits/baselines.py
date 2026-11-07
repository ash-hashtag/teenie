"""Non-learned controls on identical episodes.

`window_lookup` is the hand-written version of the bit-serial idea: at bit k it
keeps context examples whose (a, b) bits at k and whose (a, b, c) bits at the
previous `window` positions equal the query's (its own predicted c bits), and
copies their majority c_k. Not finding any match counts as a wrong answer.
"""
import torch
from torch import Tensor

from .data import BitEpisodes, from_bits, to_bits


def nearest(episodes: BitEpisodes) -> Tensor:
    """c of the context example with the smallest Hamming distance on (a, b)."""
    w = episodes.width
    context = torch.cat([to_bits(episodes.context[..., 0], w), to_bits(episodes.context[..., 1], w)], -1)
    query = torch.cat([to_bits(episodes.query[..., 0], w), to_bits(episodes.query[..., 1], w)], -1)
    distance = torch.cdist(query, context, p=1)
    return episodes.context[..., 2].gather(1, distance.argmin(-1))


def window_lookup(episodes: BitEpisodes, window: int, high_first: bool = False) -> Tensor:
    """Returns predictions [B, Q]; -1 marks queries with an unmatched key."""
    w = episodes.width
    a, b, c = (to_bits(episodes.context[..., i], w) for i in range(3))  # [B, N, W]
    qa, qb = to_bits(episodes.query[..., 0], w), to_bits(episodes.query[..., 1], w)  # [B, Q, W]
    qc = torch.zeros_like(qa)
    failed = torch.zeros(qa.shape[:2], dtype=torch.bool, device=qa.device)
    order = list(range(w - 1, -1, -1) if high_first else range(w))
    for i, k in enumerate(order):
        previous = order[max(0, i - window):i]
        context_key = torch.stack([a[..., k], b[..., k]] + [x[..., p] for p in previous for x in (a, b, c)], -1)
        query_key = torch.stack([qa[..., k], qb[..., k]] + [x[..., p] for p in previous for x in (qa, qb, qc)], -1)
        match = (query_key[:, :, None] == context_key[:, None]).all(-1).float()  # [B, Q, N]
        count = match.sum(-1)
        failed |= count == 0
        qc[..., k] = ((match * c[:, None, :, k]).sum(-1) / count.clamp(min=1) > .5).float()
    return torch.where(failed, -1, from_bits(qc))
