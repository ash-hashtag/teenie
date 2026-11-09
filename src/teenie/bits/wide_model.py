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

`peers`: the examples are read through one shallow layer, so a relation that can only be checked by computing
something deep on an example (is c = a*b + b? needs a*b) cannot be recognised from them. The probe examples, though,
are worked through with full depth every round, like the questions. With `peers`, every round each question's bit k
also reads the probes' bit-k tokens together with their true answer bit — examples worked out in depth next to
what they should have given — and no probe sees its own answer.

`registers`: besides its guess at the answer, each question keeps this many scratch numbers, one bit per position,
written every round and read back the next through the same views as a, b and the guess — room to hold an
intermediate value such as a*b while the next step is applied. Only the answer is trained; what the registers
hold is up to the model.

`example_layers`: the examples are otherwise read through one shallow layer per bit, enough to see local patterns
but not to work anything out — whether c = a*b + b holds for an example needs that example's a*b. With example
layers, each example's bit tokens first attend to each other (bias per bit distance) and pass through a gated layer,
this many times, so an example can be worked through before the questions read it. Every example is processed
(999 × width tokens), so these layers recompute their activations in training to fit in memory.

`code_slots`: a bottleneck between guessing the relation and applying it. A few code tokens per bit position read
the examples (cross-attention to that bit of every example), share across bit positions and among themselves, for
`code_layers` layers; the answer bits then read only these code tokens, never the examples. With no direct lookup of
similar examples, an answer can only come from what the code holds about the relation — the separation of task
inference from execution that lets in-context models combine known parts (Kobayashi et al., 2024). The code is free
vectors, not a choice among fixed operations. Probes are not trained with a code (the code may hold their answers).
"""
from typing import NotRequired, TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import CONTEXT_SIZE
from .bit_token_model import BitTokenModel, _BitLayer, answer_loss
from .loop_model import _Gated
from .data import to_bits
from .loop_model import LoopPrediction


class WideConfig(TypedDict):
    hidden: int
    heads: int
    layers: int
    probes: int
    steps: int
    span: int
    peers: NotRequired[bool]
    registers: NotRequired[int]
    example_layers: NotRequired[int]
    code_slots: NotRequired[int]
    code_layers: NotRequired[int]


class _CodeLayer(nn.Module):
    """Code tokens [B, W, K, H] read their bit of every example, then attend across bit positions (bias per
    distance) and among the slots at one position, then a gated layer."""

    def __init__(self, hidden: int, heads: int, span: int) -> None:
        super().__init__()
        self.heads, self.span = heads, span
        self.norm_read = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out_read = nn.Linear(hidden, hidden)
        self.norm_bits = nn.LayerNorm(hidden)
        self.qkv_bits = nn.Linear(hidden, 3 * hidden)
        self.out_bits = nn.Linear(hidden, hidden)
        self.distance = nn.Parameter(torch.zeros(heads, 2 * span - 1))
        self.norm_slots = nn.LayerNorm(hidden)
        self.qkv_slots = nn.Linear(hidden, 3 * hidden)
        self.out_slots = nn.Linear(hidden, hidden)
        self.feed = _Gated(hidden)

    def _attend(self, x: Tensor, qkv: nn.Linear, out: nn.Linear, bias: Tensor | None = None) -> Tensor:
        count, length, hidden = x.shape
        q, k, v = qkv(x).view(count, length, 3, self.heads, hidden // self.heads).permute(2, 0, 3, 1, 4)
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        result: Tensor = out(mixed.transpose(1, 2).reshape(count, length, hidden))
        return result

    def forward(self, code: Tensor, examples: Tensor) -> Tensor:
        """code [B, W, K, H]; examples [B * W, N, H] (bit k of every example)."""
        batch, width, slots, hidden = code.shape
        d = hidden // self.heads
        flat = code.reshape(batch * width, slots, hidden)
        q = self.query(self.norm_read(flat)).view(batch * width, slots, self.heads, d).transpose(1, 2)
        k = self.key(examples).view(batch * width, -1, self.heads, d).transpose(1, 2)
        v = self.value(examples).view(batch * width, -1, self.heads, d).transpose(1, 2)
        found = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch * width, slots, hidden)
        flat = flat + self.out_read(found)
        across = flat.view(batch, width, slots, hidden).transpose(1, 2).reshape(batch * slots, width, hidden)
        position = torch.arange(width, device=code.device)
        bias = self.distance[:, (position[:, None] - position[None, :]).clamp(1 - self.span, self.span - 1)
                             + self.span - 1].to(across.dtype)
        across = across + self._attend(self.norm_bits(across), self.qkv_bits, self.out_bits, bias)
        flat = across.view(batch, slots, width, hidden).transpose(1, 2).reshape(batch * width, slots, hidden)
        flat = flat + self._attend(self.norm_slots(flat), self.qkv_slots, self.out_slots)
        flat = flat + self.feed(flat)
        result: Tensor = flat.view(batch, width, slots, hidden)
        return result


class _ExampleLayer(nn.Module):
    """Within one example: its bit tokens attend to each other, then a gated layer. Outputs start at zero, so a
    trained model extended with these layers answers exactly as before until they are trained."""

    def __init__(self, hidden: int, heads: int, span: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.distance = nn.Parameter(torch.zeros(heads, 2 * span - 1))
        self.feed = _Gated(hidden)
        for layer in (self.out, self.feed.down):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """[M, W, H] -> [M, W, H]: M examples' bit tokens."""
        count, width, hidden = x.shape
        span = (self.distance.shape[1] + 1) // 2
        position = torch.arange(width, device=x.device)
        bias = self.distance[:, (position[:, None] - position[None, :]).clamp(1 - span, span - 1) + span - 1]
        q, k, v = self.qkv(self.norm(x)).view(count, width, 3, self.heads, hidden // self.heads).permute(2, 0, 3, 1, 4)
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.out(mixed.transpose(1, 2).reshape(count, width, hidden))
        result: Tensor = x + self.feed(x)
        return result


class _Peers(nn.Module):
    """Bit k of every question and probe attends to bit k of the probes, each with its true answer bit; a probe
    never attends to itself."""

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(hidden)
        self.answer = nn.Linear(1, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x: Tensor, known: Tensor) -> Tensor:
        """x [B, items, W, H] (questions first, then probes); known [B, P, W] true probe bits in {-1, 1}."""
        batch, items, width, hidden = x.shape
        probes, d = known.shape[1], hidden // self.heads
        normed = self.norm(x)
        peers = normed[:, items - probes:] + self.answer(known[..., None])  # [B, P, W, H]

        def split(t: Tensor) -> Tensor:  # [B, N, W, H] -> [B * W, heads, N, d]
            return t.permute(0, 2, 1, 3).reshape(batch * width, -1, self.heads, d).transpose(1, 2)

        rows = torch.arange(items, device=x.device)[:, None] - (items - probes)
        allowed = rows != torch.arange(probes, device=x.device)[None, :]  # [items, P]
        found = F.scaled_dot_product_attention(split(self.query(normed)), split(self.key(peers)),
                                               split(self.value(peers)), attn_mask=allowed)
        found = found.transpose(1, 2).reshape(batch, width, items, hidden).transpose(1, 2)
        result: Tensor = x + self.out(found)
        return result


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

    def __init__(self, span: int, hidden: int, extra: int = 0) -> None:
        super().__init__()
        self.span, self.hidden, self.channels = span, hidden, 4 + extra
        self.relative = nn.Conv1d(self.channels, 3 * hidden, 2 * span - 1, padding=span - 1)
        self.low = nn.Linear(self.channels * span, 3 * hidden, bias=False)

    def forward(self, numbers: Tensor) -> Tensor:
        """[..., 3 + extra, W] in {-1, 1} -> [..., W, hidden]. Channels: the first three numbers, "exists", then any
        extra numbers (so a model with extra numbers extends one without)."""
        lead, width = numbers.shape[:-2], numbers.shape[-1]
        exists = torch.ones_like(numbers[..., :1, :])
        channels = torch.cat([numbers[..., :3, :], exists, numbers[..., 3:, :]], -2).reshape(-1, self.channels, width)
        projected = self.relative(channels).transpose(1, 2)  # [M, W, 3H]
        low = self.low(F.pad(channels, (0, self.span - width)).flatten(1))[:, None]  # [M, 1, 3H]
        linear, left, right = (projected + low).chunk(3, -1)
        result: Tensor = (linear + left * right).view(*lead, width, self.hidden)
        return result


class WideBitModel(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 2, probes: int = 16, steps: int = 8,
                 span: int = 32, peers: bool = False, registers: int = 0, example_layers: int = 0,
                 code_slots: int = 0, code_layers: int = 3) -> None:
        super().__init__()
        if hidden % heads or not 0 < probes <= CONTEXT_SIZE or min(layers, steps, span) < 1:
            raise ValueError('hidden must divide into heads; probes in 1..999; layers, steps, span >= 1')
        self.config: WideConfig = {'hidden': hidden, 'heads': heads, 'layers': layers, 'probes': probes,
                                   'steps': steps, 'span': span, 'peers': peers, 'registers': registers,
                                   'example_layers': example_layers, 'code_slots': code_slots,
                                   'code_layers': code_layers}
        self.example = nn.Sequential(_Seen(span, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.ask = _Seen(span, hidden, registers)  # (a, b, current guess, registers) as seen from each answer bit
        self.layers = nn.ModuleList([_BitLayer(hidden, heads, span) for _ in range(layers)])
        self.keys = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.values = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.cell = nn.GRUCell(hidden, hidden)
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1 + registers))  # answer, registers
        if peers:
            self.peers = nn.ModuleList([_Peers(hidden, heads) for _ in range(layers)])
        self.thinking = nn.ModuleList([_ExampleLayer(hidden, heads, span) for _ in range(example_layers)])
        if code_slots:
            self.code = nn.Parameter(torch.randn(code_slots, hidden) * .02)
            self.coder = nn.ModuleList([_CodeLayer(hidden, heads, span) for _ in range(code_layers)])
        self.recompute = True
        self.hint = 1.

    def loss(self, prediction: LoopPrediction, context: Tensor, target: Tensor, width: int,
             steps: Tensor | None = None, step_counts: Tensor | None = None) -> Tensor:
        if self.config.get('peers', False) or self.config.get('code_slots', 0):
            # Through two hops a probe can see its own answer (it reads a peer that has read it), and a code read from
            # every example may hold a probe's answer, so then only the questions — whose answers are never in the
            # context — are trained.
            prediction = LoopPrediction(prediction.logits[:, :, :prediction.queries], prediction.queries)
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
        if len(self.thinking):
            flat_tokens = tokens.reshape(batch * count, width, hidden)
            for layer in self.thinking:
                flat_tokens = (checkpoint(layer, flat_tokens, use_reentrant=False) if self.training
                               else layer(flat_tokens))
            tokens = flat_tokens.view(batch, count, width, hidden)
        flat = tokens.transpose(1, 2).reshape(batch * width, count, hidden)
        d = hidden // heads
        slots = self.config.get('code_slots', 0)
        if slots:
            code = self.code.expand(batch, width, slots, hidden)
            for coder in self.coder:
                code = checkpoint(coder, code, flat, use_reentrant=False) if self.training else coder(code, flat)
            flat = code.reshape(batch * width, slots, hidden)  # the answer bits read only the code
        keys = [BitTokenModel._split(k(flat), heads, d) for k in self.keys]
        values = [BitTokenModel._split(v(flat), heads, d) for v in self.values]
        queries = asked.shape[1]
        asked = torch.cat([asked, bits[:, :probes, :2]], 1)
        known = bits[:, :probes, 2]  # [B, P, W]
        items = asked.shape[1]
        rows = torch.arange(items, device=bits.device)[:, None] - queries
        allowed = (rows != torch.arange(CONTEXT_SIZE, device=bits.device)[None, :]).repeat_interleave(width, 0)
        if slots:
            allowed = torch.ones(items * width, slots, dtype=torch.bool, device=bits.device)
        state = asked.new_zeros(batch * items * width, hidden)
        guess = asked.new_zeros(batch, items, 1 + self.config.get('registers', 0), width)  # answer, registers
        outputs = []
        for _ in range(rounds):
            if self.training and self.recompute:
                state, written = checkpoint(self._round, asked, guess, state, keys, values, allowed, known,
                                            use_reentrant=False)
            else:
                state, written = self._round(asked, guess, state, keys, values, allowed, known)
            outputs.append(written[:, :, 0])
            guess = torch.tanh(written / 2)
        return torch.stack(outputs)

    def _round(self, asked: Tensor, guess: Tensor, state: Tensor, keys: list[Tensor], values: list[Tensor],
               allowed: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        batch, items, _, width = guess.shape
        hidden = self.config['hidden']
        x = self.ask(torch.cat([asked, guess], 2)) + state.view(batch, items, width, hidden)
        for i, (layer, k, v) in enumerate(zip(self.layers, keys, values)):
            x = layer(x, k, v, allowed)
            if self.config.get('peers', False):
                x = self.peers[i](x, known)
        state = self.cell(x.reshape(-1, hidden), state)
        written = self.readout(state).view(batch, items, width, -1).transpose(2, 3)  # [B, items, 1 + R, W]
        return state, written

    def load_extending(self, state_dict: dict[str, Tensor]) -> None:
        """Load a model with fewer registers, no peers or no example layers: matching weights are copied, the new
        registers' input weights start at zero and new example layers output zero (so the loaded model behaves as
        before), and new peer layers start fresh."""
        own = self.state_dict()
        for key, value in state_dict.items():
            target = own[key]
            if target.shape == value.shape:
                target.copy_(value)
                continue
            if key in ('ask.relative.weight', 'ask.low.weight'):
                target.zero_()
                if key == 'ask.relative.weight':
                    target[:, :value.shape[1]] = value
                else:
                    span = self.config['span']
                    target[:, :value.shape[1]] = value  # channel-major flattening: old channels come first
                    assert value.shape[1] % span == 0
            elif key.startswith('readout.1.'):
                target[:value.shape[0]] = value
            else:
                raise ValueError(f'cannot extend {key}: {tuple(value.shape)} -> {tuple(target.shape)}')
