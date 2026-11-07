"""All-at-once context with a refinement loop on the answer only.

Each of the 999 examples becomes one vector from all of its bits (a, b and c together); that
context is computed once and stays fixed for the whole run. Nothing reads bit positions in
order. The question keeps a guess for every bit of c, looks at the whole context with that
guess, and corrects it — the same weights every round. Whatever order bits settle in is up
to the model.

The first `probes` context examples are also answered (with themselves masked out) for extra
training signal; weights never change at inference.
"""
from dataclasses import dataclass
import math
from typing import NotRequired, TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import CONTEXT_SIZE
from .data import from_bits, to_bits


class LoopConfig(TypedDict):
    hidden: int
    heads: int
    layers: int
    width: int
    probes: int
    steps: int
    multiplicative: NotRequired[bool]
    clock: NotRequired[bool]


@dataclass
class LoopPrediction:
    logits: Tensor  # [T, B, Q + P, W] after each round, bit index order
    queries: int

    def probabilities(self, round: int = -1) -> Tensor:
        """Probability that each query bit is 1 after a round (default: the last), [B, Q, W]."""
        return torch.sigmoid(self.logits[round, :, :self.queries])

    def values(self) -> Tensor:
        return from_bits(self.probabilities() > .5)

    def confidence(self) -> Tensor:
        """Certainty of the least certain bit, in [0.5, 1]."""
        p = self.probabilities()
        return torch.maximum(p, 1 - p).amin(-1)


def round_signal(step: int, hidden: int, device: torch.device) -> Tensor:
    """Parameter-free sinusoidal encoding of the round number."""
    frequencies = torch.exp(torch.arange(0, hidden, 2, device=device) * (-math.log(1000.) / hidden))
    angles = (step + 1) * frequencies
    return torch.stack([angles.sin(), angles.cos()], -1).flatten()[:hidden]


class _Gated(nn.Module):
    """SwiGLU feed-forward: the product of two projections lets the layer multiply its own signals."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, 4 * hidden)
        self.down = nn.Linear(2 * hidden, hidden)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.up(self.norm(x)).chunk(2, -1)
        result: Tensor = self.down(value * F.silu(gate))
        return result


class _Bilinear(nn.Module):
    """Linear term plus the elementwise product of two projections of the same input (e.g. all a_i * b_j)."""

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(inputs, hidden)
        self.left = nn.Linear(inputs, hidden)
        self.right = nn.Linear(inputs, hidden)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.linear(x) + self.left(x) * self.right(x)
        return result


class _Look(nn.Module):
    """Cross-attention from question tokens to the fixed context, then a feed-forward layer."""

    def __init__(self, hidden: int, heads: int, multiplicative: bool = False) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.feed: nn.Module = _Gated(hidden) if multiplicative else nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def _split(self, x: Tensor) -> Tensor:
        batch, items, hidden = x.shape
        return x.view(batch, items, self.heads, hidden // self.heads).transpose(1, 2)

    def forward(self, x: Tensor, context: Tensor, allowed: Tensor) -> Tensor:
        batch, items, hidden = x.shape
        found = F.scaled_dot_product_attention(self._split(self.query(self.norm(x))), self._split(self.key(context)),
                                               self._split(self.value(context)), attn_mask=allowed)
        x = x + self.out(found.transpose(1, 2).reshape(batch, items, hidden))
        result: Tensor = x + self.feed(x)
        return result


class LoopModel(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 2, width: int = 8, probes: int = 64,
                 steps: int = 8, multiplicative: bool = False, clock: bool = False) -> None:
        super().__init__()
        if hidden % heads or not 0 < probes <= CONTEXT_SIZE or min(layers, width, steps) < 1:
            raise ValueError('hidden must divide into heads; probes in 1..999; layers, width, steps >= 1')
        self.config: LoopConfig = {'hidden': hidden, 'heads': heads, 'layers': layers, 'width': width,
                                   'probes': probes, 'steps': steps, 'multiplicative': multiplicative,
                                   'clock': clock}
        # `clock` tells every round which round it is (a fixed sinusoid, no parameters) and scores mostly the last
        # round, so earlier rounds are free to hold partial work instead of a finished answer.
        # `multiplicative` adds bilinear inputs and gated feed-forward layers: generic units that can multiply
        # signals (bit by bit products, gates) instead of only summing them.
        first: nn.Module = _Bilinear(3 * width, hidden) if multiplicative else nn.Linear(3 * width, hidden)
        self.example = nn.Sequential(first, nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        # (a, b, current guess of c)
        self.ask: nn.Module = _Bilinear(3 * width, hidden) if multiplicative else nn.Linear(3 * width, hidden)
        self.looks = nn.ModuleList([_Look(hidden, heads, multiplicative) for _ in range(layers)])
        self.cell = nn.GRUCell(hidden, hidden)
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width))

    def forward(self, context: Tensor, query: Tensor, width: int, target: Tensor | None = None,
                steps: int | None = None) -> LoopPrediction:
        """context [B, 999, 3], query [B, Q, 2] integers. `target` is unused (no teacher forcing needed)."""
        if context.shape[1] != CONTEXT_SIZE or width != self.config['width']:
            raise ValueError(f'context must contain exactly {CONTEXT_SIZE} examples; width must be {self.config["width"]}')
        rounds = self.config['steps'] if steps is None else steps
        probes = self.config['probes']
        bits = torch.cat([to_bits(context[..., i], width) for i in range(3)], -1) * 2 - 1  # [B, N, 3W]
        examples = self.example(bits)  # computed once, fixed for every round
        queries = query.shape[1]
        asked = torch.cat([torch.cat([to_bits(query[..., i], width) for i in range(2)], -1) * 2 - 1,
                           bits[:, :probes, :2 * width]], 1)  # [B, Q + P, 2W]
        allowed = torch.ones(queries + probes, CONTEXT_SIZE, dtype=torch.bool, device=context.device)
        allowed[queries + torch.arange(probes), torch.arange(probes)] = False  # a probe never sees itself
        batch, items = asked.shape[:2]
        hidden = self.config['hidden']
        state = asked.new_zeros(batch * items, hidden)
        guess = asked.new_zeros(batch, items, width)  # in [-1, 1]; 0 = undecided
        outputs = []
        for step in range(rounds):
            x = self.ask(torch.cat([asked, guess], -1)) + state.view(batch, items, hidden)
            if self.config.get('clock', False):
                x = x + round_signal(step, hidden, context.device)
            for look in self.looks:
                x = look(x, examples, allowed)
            state = self.cell(x.reshape(-1, hidden), state)
            logits = self.readout(state).view(batch, items, width)
            outputs.append(logits)
            guess = torch.tanh(logits / 2)  # = 2 * sigmoid(logits) - 1
        return LoopPrediction(torch.stack(outputs), queries)

    def loss(self, prediction: LoopPrediction, context: Tensor, target: Tensor, width: int) -> Tensor:
        """Per-bit cross-entropy after every round, later rounds weighted more; queries and probes."""
        probes = to_bits(context[:, :prediction.logits.shape[2] - prediction.queries, 2], width)
        truth = torch.cat([to_bits(target, width), probes], 1).expand_as(prediction.logits)
        per_round = F.binary_cross_entropy_with_logits(prediction.logits, truth, reduction='none').mean((1, 2, 3))
        if self.config.get('clock', False) and len(per_round) > 1:
            return per_round[-1] + .1 * per_round[:-1].mean()
        weights = torch.arange(1, len(per_round) + 1, device=per_round.device, dtype=per_round.dtype)
        return (per_round * weights).sum() / weights.sum()
