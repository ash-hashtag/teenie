"""Bit tokens: each question is 8 tokens, one per answer bit, refined round after round.

Same idea as the loop model — read the 999 examples once, hold a guess, look and correct with the same weights
every round — but the question is no longer one vector. Each round, its bit tokens

1. attend to each other (with a learned bias per distance between bit positions), so carries and partial
   products can move between positions,
2. each read the 999 examples (cross-attention), so bit k can find examples in the same situation,
3. pass through a gated (SwiGLU) layer, which can multiply signals.

A GRU cell per token carries its state; a shared readout turns each token into its own bit's logit. Rounds
give depth: one step's carries can settle before the next step uses them. Probe questions (the first examples,
masked from themselves) are answered too, for extra training signal, exactly as in the loop model.
"""
from typing import NotRequired, TypedDict, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import CONTEXT_SIZE
from .data import to_bits
from .loop_model import LoopPrediction, _Bilinear, _Gated, step_loss


class BitTokenConfig(TypedDict):
    hidden: int
    heads: int
    layers: int
    width: int
    probes: int
    steps: int
    stepwise: NotRequired[bool]


class _BitLayer(nn.Module):
    """Self-attention among a question's bit tokens, cross-attention to the examples, gated feed-forward."""

    def __init__(self, hidden: int, heads: int, width: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_self = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out_self = nn.Linear(hidden, hidden)
        self.distance = nn.Parameter(torch.zeros(heads, 2 * width - 1))  # bias per head and bit-position offset
        self.norm_cross = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out_cross = nn.Linear(hidden, hidden)
        self.feed = _Gated(hidden)
        self.offsets: Tensor  # [W, W] index into `distance` for each pair of bit positions
        self.register_buffer('offsets', torch.arange(width)[:, None] - torch.arange(width)[None, :] + width - 1,
                             persistent=False)
        # When set to a list, forward appends attention to the examples [B, Q + P, N], averaged over heads and over
        # the question's bit tokens (inspection only).
        self.attention: list[Tensor] | None = None

    def forward(self, x: Tensor, keys: Tensor, values: Tensor, allowed: Tensor) -> Tensor:
        batch, items, width, hidden = x.shape
        d = hidden // self.heads
        # 1. Bit tokens of one question attend to each other.
        q, k, v = self.qkv(self.norm_self(x)).view(batch * items, width, 3, self.heads, d).permute(2, 0, 3, 1, 4)
        bias = self.distance[:, self.offsets]  # [heads, W, W]
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out_self(mixed.transpose(1, 2).reshape(batch, items, width, hidden))
        # 2. Every bit token reads the examples.
        q = self.query(self.norm_cross(x)).view(batch, items * width, self.heads, d).transpose(1, 2)
        found = F.scaled_dot_product_attention(q, keys, values, attn_mask=allowed)
        if self.attention is not None:
            scores = (q @ keys.transpose(-1, -2) / d ** .5).masked_fill(~allowed, float('-inf'))
            weights = scores.softmax(-1).mean(1)  # [B, items * W, N]
            self.attention.append(weights.view(batch, items, width, -1).mean(2))
        x = x + self.out_cross(found.transpose(1, 2).reshape(batch, items, width, hidden))
        # 3. Gated feed-forward.
        result: Tensor = x + self.feed(x)
        return result


class BitTokenModel(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 2, width: int = 8, probes: int = 64,
                 steps: int = 8, stepwise: bool = False) -> None:
        super().__init__()
        if hidden % heads or not 0 < probes <= CONTEXT_SIZE or min(layers, width, steps) < 1:
            raise ValueError('hidden must divide into heads; probes in 1..999; layers, width, steps >= 1')
        self.config: BitTokenConfig = {'hidden': hidden, 'heads': heads, 'layers': layers, 'width': width,
                                       'probes': probes, 'steps': steps, 'stepwise': stepwise}
        self.example = nn.Sequential(_Bilinear(3 * width, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.LayerNorm(hidden))
        self.ask = _Bilinear(3 * width, hidden)  # (a, b, current guess), shared by the question's bit tokens
        self.position = nn.Parameter(torch.randn(width, hidden) * .02)
        self.layers = nn.ModuleList([_BitLayer(hidden, heads, width) for _ in range(layers)])
        self.keys = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.values = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.cell = nn.GRUCell(hidden, hidden)
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        if stepwise:  # "the steps are finished", read from the mean of the question's bit tokens
            self.done = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        # Recompute each round in the backward pass instead of storing its activations: less memory, ~30% slower.
        self.recompute = True

    def forward(self, context: Tensor, query: Tensor, width: int, target: Tensor | None = None,
                steps: int | None = None) -> LoopPrediction:
        """context [B, 999, 3], query [B, Q, 2] integers; `target` is unused. Logits after every round."""
        if context.shape[1] != CONTEXT_SIZE or width != self.config['width']:
            raise ValueError(f'context must contain exactly {CONTEXT_SIZE} examples; width must be {self.config["width"]}')
        rounds = self.config['steps'] if steps is None else steps
        bits = torch.cat([to_bits(context[..., i], width) for i in range(3)], -1) * 2 - 1
        asked = torch.cat([to_bits(query[..., i], width) for i in range(2)], -1) * 2 - 1
        self._done_logits: list[Tensor] = []
        logits = self.forward_bits(bits, asked, rounds)
        done = torch.stack(self._done_logits) if self._done_logits else None
        return LoopPrediction(logits, query.shape[1], done=done)

    def forward_bits(self, bits: Tensor, asked: Tensor, rounds: int) -> Tensor:
        """Examples [B, 999, 3W] and questions [B, Q, 2W] in {-1, 1} -> logits [T, B, Q + P, W]."""
        width, probes, hidden, heads = (self.config['width'], self.config['probes'], self.config['hidden'],
                                        self.config['heads'])
        examples = self.example(bits)
        d = hidden // heads
        keys = [self._split(k(examples), heads, d) for k in self.keys]
        values = [self._split(v(examples), heads, d) for v in self.values]
        queries = asked.shape[1]
        asked = torch.cat([asked, bits[:, :probes, :2 * width]], 1)
        batch, items = asked.shape[:2]
        rows = torch.arange(queries + probes, device=bits.device)[:, None] - queries
        allowed = (rows != torch.arange(CONTEXT_SIZE, device=bits.device)[None, :]).repeat_interleave(width, 0)
        state = asked.new_zeros(batch * items * width, hidden)
        guess = asked.new_zeros(batch, items, width)
        outputs = []
        for _ in range(rounds):
            if self.training and self.recompute:
                # Recompute activations in backward: memory stays flat in the number of rounds.
                state, logits = checkpoint(self._round, asked, guess, state, keys, values, allowed,
                                           use_reentrant=False)
            else:
                state, logits = self._round(asked, guess, state, keys, values, allowed)
            outputs.append(logits)
            if self.config.get('stepwise', False):
                self._done_logits.append(self.done(state.view(batch, items, width, hidden).mean(2)).squeeze(-1))
            guess = torch.tanh(logits / 2)
        return torch.stack(outputs)

    @staticmethod
    def _split(x: Tensor, heads: int, d: int) -> Tensor:
        batch, items, _ = x.shape
        return x.view(batch, items, heads, d).transpose(1, 2)

    def _round(self, asked: Tensor, guess: Tensor, state: Tensor, keys: list[Tensor], values: list[Tensor],
               allowed: Tensor) -> tuple[Tensor, Tensor]:
        batch, items, width = guess.shape
        hidden = self.config['hidden']
        x = (self.ask(torch.cat([asked, guess], -1))[:, :, None] + self.position
             + state.view(batch, items, width, hidden))
        for layer, k, v in zip(self.layers, keys, values):
            x = layer(x, k, v, allowed)
        state = self.cell(x.reshape(-1, hidden), state)
        logits = self.readout(state).view(batch, items, width)
        return state, logits

    def record_attention(self, enabled: bool = True) -> list[list[Tensor]]:
        """Toggle attention recording; returns per-layer lists the next forward fills, one entry per round."""
        stores: list[list[Tensor]] = [[] for _ in self.layers]
        for layer, store in zip(self.layers, stores):
            cast(_BitLayer, layer).attention = store if enabled else None
        return stores

    def loss(self, prediction: LoopPrediction, context: Tensor, target: Tensor, width: int,
             steps: Tensor | None = None, step_counts: Tensor | None = None) -> Tensor:
        """Per-bit cross-entropy after every round, later rounds weighted more; queries and probes. With `steps`,
        step-by-step training instead (see `step_loss`)."""
        if steps is not None and step_counts is not None and prediction.done is not None:
            return step_loss(prediction, steps, step_counts, width)
        probes = to_bits(context[:, :prediction.logits.shape[2] - prediction.queries, 2], width)
        truth = torch.cat([to_bits(target, width), probes], 1).expand_as(prediction.logits)
        per_round = F.binary_cross_entropy_with_logits(prediction.logits, truth, reduction='none').mean((1, 2, 3))
        weights = torch.arange(1, len(per_round) + 1, device=per_round.device, dtype=per_round.dtype)
        return (per_round * weights).sum() / weights.sum()
