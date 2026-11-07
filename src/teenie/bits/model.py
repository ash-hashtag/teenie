"""Bit-serial in-context model: a learned recurrence over bit positions plus retrieval.

Every example (context and query) is read one bit position at a time by a shared
GRU. At position k the query describes its situation — its own (a_k, b_k) and a
learned summary of the positions already read, including its predicted c bits —
and attends to the context examples' descriptions at the same position, whose
values carry their known c_k. The GRU must learn what to remember (a carry, a
borrow, "a is already larger") so that matching situations share c_k.

One pass runs least-significant-bit first, another most-significant first. The
first `probes` context examples are re-predicted with themselves masked out; how
well each pass explains them decides how much that pass is trusted on the query.
Weights never change at inference.
"""
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TypedDict, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import CONTEXT_SIZE
from .data import from_bits, to_bits


class BitConfig(TypedDict):
    hidden: int
    heads: int
    max_width: int
    probes: int


@dataclass
class BitPrediction:
    logits: Tensor  # [D, B, Q + P, W] per pass, bit index order
    weights: Tensor  # [B, D] trust in each pass from its probe fit
    queries: int

    def probabilities(self) -> Tensor:
        """Pass-weighted probability that each query bit is 1, [B, Q, W]."""
        p = torch.sigmoid(self.logits[:, :, :self.queries])
        return torch.einsum('dbqw,bd->bqw', p, self.weights)

    def values(self) -> Tensor:
        return from_bits(self.probabilities() > .5)

    def confidence(self) -> Tensor:
        """Certainty of the least certain bit, in [0.5, 1]."""
        p = self.probabilities()
        return torch.maximum(p, 1 - p).amin(-1)


class _Pass(nn.Module):
    def __init__(self, hidden: int, heads: int, max_width: int) -> None:
        super().__init__()
        self.heads = heads
        self.position = nn.Embedding(max_width, hidden)
        self.read = nn.Linear(3, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.describe = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden + 1, hidden)
        self.readout = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # When set to a dict, forward stores head-averaged attention [B, Q + P, N] per bit position (inspection only).
        self.attention: dict[int, Tensor] | None = None

    def _split(self, x: Tensor) -> Tensor:
        batch, items, hidden = x.shape
        return x.view(batch, items, self.heads, hidden // self.heads).transpose(1, 2)

    def _advance(self, state: Tensor, a: Tensor, b: Tensor, c: Tensor, position: Tensor) -> Tensor:
        batch, items, hidden = state.shape
        step = self.read(torch.stack([a, b, c], -1) * 2 - 1) + position
        updated: Tensor = self.cell(step.reshape(-1, hidden), state.reshape(-1, hidden))
        return updated.view(batch, items, hidden)

    def forward(self, a: Tensor, b: Tensor, c: Tensor, qa: Tensor, qb: Tensor, teacher: Tensor | None,
                probes: int, high_first: bool) -> Tensor:
        batch, count, width = a.shape
        queries = qa.shape[1]
        hidden = self.position.embedding_dim
        state = a.new_zeros(batch, count, hidden)
        query_state = a.new_zeros(batch, queries, hidden)
        allowed = torch.ones(queries + probes, count, dtype=torch.bool, device=a.device)
        allowed[queries + torch.arange(probes), torch.arange(probes)] = False  # a probe never sees itself
        order = list(range(width - 1, -1, -1) if high_first else range(width))
        outputs: dict[int, Tensor] = {}
        for k in order:
            position = self.position.weight[k]
            described = self.describe(torch.cat([state + position, torch.stack([a[..., k], b[..., k]], -1) * 2 - 1], -1))
            asking = torch.cat([self.describe(torch.cat(
                [query_state + position, torch.stack([qa[..., k], qb[..., k]], -1) * 2 - 1], -1)),
                described[:, :probes]], 1)
            values = self.value(torch.cat([described, c[..., k, None] * 2 - 1], -1))
            q, key = self._split(self.query(asking)), self._split(self.key(described))
            found = F.scaled_dot_product_attention(q, key, self._split(values), attn_mask=allowed)
            if self.attention is not None:
                scores = (q @ key.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~allowed, -math.inf)
                self.attention[k] = scores.softmax(-1).mean(1)
            found = found.transpose(1, 2).reshape(batch, queries + probes, hidden)
            outputs[k] = self.readout(torch.cat([asking, found], -1)).squeeze(-1)
            query_c = teacher[..., k] if teacher is not None else (outputs[k][:, :queries] > 0).float()
            state = self._advance(state, a[..., k], b[..., k], c[..., k], position)
            query_state = self._advance(query_state, qa[..., k], qb[..., k], query_c, position)
        return torch.stack([outputs[k] for k in range(width)], -1)


class BitModel(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, max_width: int = 64, probes: int = 64) -> None:
        super().__init__()
        if hidden % heads or not 0 < probes <= CONTEXT_SIZE:
            raise ValueError('hidden must divide into heads and 0 < probes <= 999')
        self.config: BitConfig = {'hidden': hidden, 'heads': heads, 'max_width': max_width, 'probes': probes}
        self.passes = nn.ModuleList([_Pass(hidden, heads, max_width) for _ in range(2)])
        self.log_trust = nn.Parameter(torch.tensor(1.))

    def forward(self, context: Tensor, query: Tensor, width: int, target: Tensor | None = None) -> BitPrediction:
        """context [B, 999, 3] and query [B, Q, 2] integers; `target` [B, Q] enables teacher forcing."""
        if context.shape[1] != CONTEXT_SIZE or width > self.config['max_width']:
            raise ValueError(f'context must contain exactly {CONTEXT_SIZE} examples and width <= max_width')
        a, b, c = (to_bits(context[..., i], width) for i in range(3))
        qa, qb = to_bits(query[..., 0], width), to_bits(query[..., 1], width)
        teacher = None if target is None else to_bits(target, width)
        probes = self.config['probes']
        logits = torch.stack([p(a, b, c, qa, qb, teacher, probes, high_first)
                              for p, high_first in zip(self.passes, (False, True))])
        misfit = F.binary_cross_entropy_with_logits(
            logits[:, :, query.shape[1]:], c[None, :, :probes].expand(2, -1, -1, -1), reduction='none').sum(-1).mean(-1)
        weights = torch.softmax(-misfit.detach().T * self.log_trust.exp(), -1)
        return BitPrediction(logits, weights, query.shape[1])

    def loss(self, prediction: BitPrediction, context: Tensor, target: Tensor, width: int) -> Tensor:
        return bit_loss(prediction, context, target, width)

    def record_attention(self, enabled: bool = True) -> list[dict[int, Tensor]]:
        """Toggle attention recording; returns the per-pass dicts the next forward fills (bit -> [B, Q + P, N])."""
        stores: list[dict[int, Tensor]] = [{} for _ in self.passes]
        for p, store in zip(self.passes, stores):
            cast(_Pass, p).attention = store if enabled else None
        return stores

    @torch.no_grad()
    def predict(self, context: Sequence[tuple[int, int, int]], query: tuple[int, int],
                width: int = 8) -> tuple[int, float]:
        """Returns (c, confidence) for one query; confidence is the least certain bit's probability."""
        device = self.log_trust.device
        ctx = torch.tensor([list(t) for t in context], dtype=torch.long, device=device)[None]
        q = torch.tensor([list(query)], dtype=torch.long, device=device)[None]
        if ((ctx[..., :2] == q[:, :, None]).all(-1)).any():
            raise ValueError('query already appears in the context')
        prediction = self.forward(ctx, q, width)
        return int(prediction.values()[0, 0]), float(prediction.confidence()[0, 0])


def bit_loss(prediction: BitPrediction, context: Tensor, target: Tensor, width: int) -> Tensor:
    """Per-bit cross-entropy: weighted combination on queries, plus each pass on queries and probes."""
    queries = to_bits(target, width)
    combined = F.binary_cross_entropy(prediction.probabilities().clamp(1e-6, 1 - 1e-6), queries)
    probes = to_bits(context[:, :prediction.logits.shape[2] - prediction.queries, 2], width)
    everything = torch.cat([queries, probes], 1).expand_as(prediction.logits)
    return combined + F.binary_cross_entropy_with_logits(prediction.logits, everything)
