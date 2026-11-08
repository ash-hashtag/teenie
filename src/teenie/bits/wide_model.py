"""Width-free bit tokens: the same weights answer 8-, 16-, 24- or 32-bit relations.

The bit-token model (bit_token_model.py) reads whole numbers through layers sized for one width, so it has to be
trained again for every width. Here nothing depends on the width: every answer bit k, and every example's bit k,
sees the numbers through two fixed views,

- relative to k: bits k-31 .. k+31 of a, b and c (or the current guess), plus whether each position exists — the
  same view at every k and every width, and enough to locate k between the lowest and highest bit;
- from the lowest bit: bits 0 .. 31.

A bilinear layer over both views can pair bit k-i of one number with bit i of another, which is what the column of
a product needs at every k (the alignment idea of Abacus embeddings and position coupling, applied to bits). Answer
bit k reads only the examples' bit-k tokens (bit-aligned reading, findings section 15); bit tokens attend to each
other with a bias per bit distance; rounds, memory and readout are as in the bit-token model. Widths up to `span`.
"""
from typing import TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import CONTEXT_SIZE
from .bit_token_model import BitTokenModel, _BitLayer, answer_loss
from .data import to_bits
from .loop_model import LoopPrediction


class WideConfig(TypedDict):
    hidden: int
    heads: int
    layers: int
    probes: int
    steps: int
    span: int


def views(numbers: Tensor, span: int) -> Tensor:
    """Bits [..., 3, W] in {-1, 1} -> per-bit views [..., W, 4 * (3 * span - 1)]: for each of the three numbers and
    an "exists" channel, bits k-span+1 .. k+span-1 (relative to k) and bits 0 .. span-1 (from the lowest bit).
    Reference for `_Seen`, which computes a bilinear layer over these views without building them."""
    width = numbers.shape[-1]
    channels = torch.cat([numbers, torch.ones_like(numbers[..., :1, :])], -2)  # [..., 4, W]; 0 = no such bit
    padded = F.pad(channels, (span - 1, span - 1))
    relative = padded.unfold(-1, 2 * span - 1, 1)  # [..., 4, W, 2 * span - 1]
    low = F.pad(channels, (0, span - width))[..., None, :].expand(*channels.shape[:-1], width, span)
    both = torch.cat([relative, low], -1)  # [..., 4, W, 3 * span - 1]
    return both.transpose(-3, -2).flatten(-2)


class _Seen(nn.Module):
    """A bilinear layer (linear term + product of two projections) over every bit's views, computed cheaply: the
    relative view's projection is a 1-D convolution along the bits, and the from-lowest-bit view's projection is
    the same for every bit, so it is computed once per number."""

    def __init__(self, span: int, hidden: int) -> None:
        super().__init__()
        self.span, self.hidden = span, hidden
        self.relative = nn.Conv1d(4, 3 * hidden, 2 * span - 1, padding=span - 1)
        self.low = nn.Linear(4 * span, 3 * hidden, bias=False)

    def forward(self, numbers: Tensor) -> Tensor:
        """[..., 3, W] in {-1, 1} -> [..., W, hidden]."""
        lead, width = numbers.shape[:-2], numbers.shape[-1]
        channels = torch.cat([numbers, torch.ones_like(numbers[..., :1, :])], -2).reshape(-1, 4, width)
        projected = self.relative(channels).transpose(1, 2)  # [M, W, 3H]
        low = self.low(F.pad(channels, (0, self.span - width)).flatten(1))[:, None]  # [M, 1, 3H]
        linear, left, right = (projected + low).chunk(3, -1)
        result: Tensor = (linear + left * right).view(*lead, width, self.hidden)
        return result


class WideBitModel(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 2, probes: int = 16, steps: int = 8,
                 span: int = 32) -> None:
        super().__init__()
        if hidden % heads or not 0 < probes <= CONTEXT_SIZE or min(layers, steps, span) < 1:
            raise ValueError('hidden must divide into heads; probes in 1..999; layers, steps, span >= 1')
        self.config: WideConfig = {'hidden': hidden, 'heads': heads, 'layers': layers, 'probes': probes,
                                   'steps': steps, 'span': span}
        self.example = nn.Sequential(_Seen(span, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.ask = _Seen(span, hidden)  # (a, b, current guess) as seen from each answer bit
        self.layers = nn.ModuleList([_BitLayer(hidden, heads, span) for _ in range(layers)])
        self.keys = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.values = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.cell = nn.GRUCell(hidden, hidden)
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.recompute = True
        self.hint = 1.

    def loss(self, prediction: LoopPrediction, context: Tensor, target: Tensor, width: int,
             steps: Tensor | None = None, step_counts: Tensor | None = None) -> Tensor:
        return answer_loss(prediction, context, target, width)

    def forward(self, context: Tensor, query: Tensor, width: int, target: Tensor | None = None,
                steps: int | None = None) -> LoopPrediction:
        """context [B, 999, 3], query [B, Q, 2] integers of `width` bits; `target` is unused."""
        if context.shape[1] != CONTEXT_SIZE or not 1 <= width <= self.config['span']:
            raise ValueError(f'context must contain exactly {CONTEXT_SIZE} examples; width at most '
                             f'{self.config["span"]}')
        rounds = self.config['steps'] if steps is None else steps
        bits = torch.stack([to_bits(context[..., i], width) for i in range(3)], -2) * 2 - 1  # [B, N, 3, W]
        asked = torch.stack([to_bits(query[..., i], width) for i in range(2)], -2) * 2 - 1  # [B, Q, 2, W]
        return LoopPrediction(self.forward_bits(bits, asked, rounds), query.shape[1])

    def forward_bits(self, bits: Tensor, asked: Tensor, rounds: int) -> Tensor:
        """Examples [B, 999, 3, W] and questions [B, Q, 2, W] in {-1, 1} -> logits [T, B, Q + P, W]."""
        probes, hidden, heads, span = (self.config['probes'], self.config['hidden'], self.config['heads'],
                                       self.config['span'])
        batch, count, _, width = bits.shape
        tokens = self.example(bits)  # [B, N, W, H]: example bit k as seen from k
        flat = tokens.transpose(1, 2).reshape(batch * width, count, hidden)
        d = hidden // heads
        keys = [BitTokenModel._split(k(flat), heads, d) for k in self.keys]
        values = [BitTokenModel._split(v(flat), heads, d) for v in self.values]
        queries = asked.shape[1]
        asked = torch.cat([asked, bits[:, :probes, :2]], 1)
        items = asked.shape[1]
        rows = torch.arange(items, device=bits.device)[:, None] - queries
        allowed = (rows != torch.arange(CONTEXT_SIZE, device=bits.device)[None, :]).repeat_interleave(width, 0)
        state = asked.new_zeros(batch * items * width, hidden)
        guess = asked.new_zeros(batch, items, width)
        outputs = []
        for _ in range(rounds):
            if self.training and self.recompute:
                state, logits = checkpoint(self._round, asked, guess, state, keys, values, allowed,
                                           use_reentrant=False)
            else:
                state, logits = self._round(asked, guess, state, keys, values, allowed)
            outputs.append(logits)
            guess = torch.tanh(logits / 2)
        return torch.stack(outputs)

    def _round(self, asked: Tensor, guess: Tensor, state: Tensor, keys: list[Tensor], values: list[Tensor],
               allowed: Tensor) -> tuple[Tensor, Tensor]:
        batch, items, width = guess.shape
        hidden = self.config['hidden']
        x = self.ask(torch.cat([asked, guess[:, :, None]], 2)) + state.view(batch, items, width, hidden)
        for layer, k, v in zip(self.layers, keys, values):
            x = layer(x, k, v, allowed)
        state = self.cell(x.reshape(-1, hidden), state)
        logits = self.readout(state).view(batch, items, width)
        return state, logits
