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
from typing import NotRequired, TypedDict, cast

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
    scratch: NotRequired[int]
    halting: NotRequired[bool]
    stepwise: NotRequired[bool]


@dataclass
class LoopPrediction:
    logits: Tensor  # [T, B, Q + P, W] after each round, bit index order
    queries: int
    halts: Tensor | None = None  # [T, B, Q + P] probability of stopping at each round, if the model halts itself
    done: Tensor | None = None  # [T, B, Q + P] logits of "the steps are finished", for step-trained models

    def stop(self) -> Tensor:
        """Round each question answers at, [B, Q]: for halting models the median of the stopping distribution; for
        step-trained models the first round that says it is done; otherwise the last round."""
        last = len(self.logits) - 1
        if self.halts is not None:
            reached = self.halts[:, :, :self.queries].cumsum(0) >= .5
            return reached.float().argmax(0)
        if self.done is not None:
            finished = self.done[:, :, :self.queries] > 0
            finished[-1] = True  # the cap
            return finished.float().argmax(0)
        return torch.full(self.logits.shape[1:2] + (self.queries,), last, device=self.logits.device, dtype=torch.long)

    def probabilities(self, round: int | None = None) -> Tensor:
        """Probability that each query bit is 1, [B, Q, W]: after a given round, or by default at the round each
        question stops (the last round for models without halting)."""
        if round is not None:
            return torch.sigmoid(self.logits[round, :, :self.queries])
        chosen = self.stop()[None, :, :, None].expand(1, -1, -1, self.logits.shape[-1])
        return torch.sigmoid(self.logits[:, :, :self.queries].gather(0, chosen)[0])

    def values(self) -> Tensor:
        return from_bits(self.probabilities() > .5)

    def confidence(self) -> Tensor:
        """Certainty of the least certain bit, in [0.5, 1]."""
        p = self.probabilities()
        return torch.maximum(p, 1 - p).amin(-1)


def step_loss(prediction: LoopPrediction, steps: Tensor, step_counts: Tensor, width: int) -> Tensor:
    """Step-by-step training: round t is scored against intermediate step t (then the final answer, also at the
    last round), and `done` against whether the steps are finished. Shared by the loop and bit-token models."""
    assert prediction.done is not None
    rounds, queries = len(prediction.logits), prediction.queries
    probes = prediction.logits.shape[2] - queries
    # Items are ordered like the models': the questions (after the 999 examples), then the probe examples.
    values = torch.cat([steps[:, :, CONTEXT_SIZE:], steps[:, :, :probes]], 2)  # [B, S, I]
    counts = torch.cat([step_counts[:, CONTEXT_SIZE:], step_counts[:, :probes]], 1)  # [B, I]
    available = values.shape[1]
    index = torch.arange(rounds, device=values.device).clamp(max=available - 1)
    index[-1] = available - 1
    targets = to_bits(values[:, index].transpose(0, 1), width)  # [T, B, I, W]
    answer = F.binary_cross_entropy_with_logits(prediction.logits, targets)
    finished = (torch.arange(rounds, device=values.device)[:, None, None] >= counts[None] - 1).float()
    finished[-1] = 1.
    return answer + .5 * F.binary_cross_entropy_with_logits(prediction.done, finished)


HALT_COST = .005  # per round, in bits of cross-entropy: a round has to buy a real improvement to be worth it


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
        # When set to a list, forward appends head-averaged attention [B, Q + P, N] (inspection only).
        self.attention: list[Tensor] | None = None

    def _split(self, x: Tensor) -> Tensor:
        batch, items, hidden = x.shape
        return x.view(batch, items, self.heads, hidden // self.heads).transpose(1, 2)

    def forward(self, x: Tensor, context: Tensor, allowed: Tensor) -> Tensor:
        batch, items, hidden = x.shape
        q, k = self._split(self.query(self.norm(x))), self._split(self.key(context))
        found = F.scaled_dot_product_attention(q, k, self._split(self.value(context)), attn_mask=allowed)
        if self.attention is not None:
            scores = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~allowed, -math.inf)
            self.attention.append(scores.softmax(-1).mean(1))
        x = x + self.out(found.transpose(1, 2).reshape(batch, items, hidden))
        result: Tensor = x + self.feed(x)
        return result


class _Within(nn.Module):
    """Self-attention among one question's own tokens (its main token and its scratch slots)."""

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.feed = _Gated(hidden)

    def forward(self, x: Tensor) -> Tensor:
        groups, tokens, hidden = x.shape
        q, k, v = self.qkv(self.norm(x)).view(groups, tokens, 3, self.heads, hidden // self.heads).permute(2, 0, 3, 1, 4)
        found = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(found.transpose(1, 2).reshape(groups, tokens, hidden))
        result: Tensor = x + self.feed(x)
        return result


class LoopModel(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 2, width: int = 8, probes: int = 64,
                 steps: int = 8, multiplicative: bool = False, clock: bool = False, scratch: int = 0,
                 halting: bool = False, stepwise: bool = False) -> None:
        super().__init__()
        if hidden % heads or not 0 < probes <= CONTEXT_SIZE or min(layers, width, steps) < 1:
            raise ValueError('hidden must divide into heads; probes in 1..999; layers, width, steps >= 1')
        self.config: LoopConfig = {'hidden': hidden, 'heads': heads, 'layers': layers, 'width': width,
                                   'probes': probes, 'steps': steps, 'multiplicative': multiplicative,
                                   'clock': clock, 'scratch': scratch, 'halting': halting, 'stepwise': stepwise}
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
        # `scratch` slots: per-question memory carried between rounds and never scored, for intermediate results
        # (a partial product, a running total, a counter). Only the main token reads the examples; the slots
        # exchange information with it through `within`. Each slot has a learned identity added every round.
        # `halting`: every round also outputs the probability that the answer is done (PonderNet-style). Training
        # minimises the expected error over the resulting stopping distribution plus a small cost per round, so it
        # stops early when one pass is enough; `steps` is then only the cap.
        if halting:
            halt_out = nn.Linear(hidden, 1)
            nn.init.constant_(halt_out.bias, -2.)  # start out preferring to keep going
            self.halt = nn.Sequential(nn.LayerNorm(hidden), halt_out)
        # `stepwise`: trained on the relation's intermediate values — round t's guess is scored against step t (then
        # the final answer) and a `done` output learns when the steps are finished. One round = one step, so a
        # loop of n iterations takes n rounds; answers are read at the first round that says it is done.
        if stepwise:
            self.done = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        if scratch:
            self.slot_identity = nn.Parameter(torch.randn(scratch, hidden) * .02)
            self.within = _Within(hidden, heads)
            self.slot_cell = nn.GRUCell(hidden, hidden)

    def forward(self, context: Tensor, query: Tensor, width: int, target: Tensor | None = None,
                steps: int | None = None) -> LoopPrediction:
        """context [B, 999, 3], query [B, Q, 2] integers. `target` is unused (no teacher forcing needed)."""
        if context.shape[1] != CONTEXT_SIZE or width != self.config['width']:
            raise ValueError(f'context must contain exactly {CONTEXT_SIZE} examples; width must be {self.config["width"]}')
        rounds = self.config['steps'] if steps is None else steps
        bits = torch.cat([to_bits(context[..., i], width) for i in range(3)], -1) * 2 - 1  # [B, N, 3W]
        asked = torch.cat([to_bits(query[..., i], width) for i in range(2)], -1) * 2 - 1  # [B, Q, 2W]
        self._halt_logits: list[Tensor] = []
        self._done_logits: list[Tensor] = []
        logits = self.forward_bits(bits, asked, rounds)
        halts = None
        if self.config.get('halting', False):
            # Stopping distribution: halt at round t with probability h_t * prod_{s<t} (1 - h_s); the last round
            # takes whatever probability is left, so every question stops by the cap.
            h = torch.sigmoid(torch.stack(self._halt_logits))  # [T, B, Q + P]
            h = torch.cat([h[:-1], torch.ones_like(h[-1:])])
            survive = torch.cumprod(torch.cat([torch.ones_like(h[:1]), 1 - h[:-1]]), 0)
            halts = h * survive
        done = torch.stack(self._done_logits) if self._done_logits else None
        return LoopPrediction(logits, query.shape[1], halts, done)

    def forward_bits(self, bits: Tensor, asked: Tensor, rounds: int) -> Tensor:
        """The model on bits in {-1, 1}: examples [B, 999, 3W] and questions [B, Q, 2W]. Returns logits after every
        round for the questions and the probes, [T, B, Q + P, W]. Used directly by the ONNX export."""
        width = self.config['width']
        probes = self.config['probes']
        examples = self.example(bits)  # computed once, fixed for every round
        queries = asked.shape[1]
        asked = torch.cat([asked, bits[:, :probes, :2 * width]], 1)  # [B, Q + P, 2W]
        # A probe never sees itself: probe row Q + j may not attend to example j.
        rows = torch.arange(queries + probes, device=bits.device)[:, None] - queries
        allowed = rows != torch.arange(CONTEXT_SIZE, device=bits.device)[None, :]
        batch, items = asked.shape[:2]
        hidden = self.config['hidden']
        state = asked.new_zeros(batch * items, hidden)
        guess = asked.new_zeros(batch, items, width)  # in [-1, 1]; 0 = undecided
        slots = self.config.get('scratch', 0)
        slot_state = asked.new_zeros(batch * items * slots, hidden)
        outputs = []
        for step in range(rounds):
            x = self.ask(torch.cat([asked, guess], -1)) + state.view(batch, items, hidden)
            if self.config.get('clock', False):
                x = x + round_signal(step, hidden, bits.device)
            for look in self.looks:
                x = look(x, examples, allowed)
            if slots:
                memory = slot_state.view(batch * items, slots, hidden) + self.slot_identity
                mixed = self.within(torch.cat([x.reshape(batch * items, 1, hidden), memory], 1))
                x = mixed[:, 0].view(batch, items, hidden)
                slot_state = self.slot_cell(mixed[:, 1:].reshape(-1, hidden), slot_state)
            state = self.cell(x.reshape(-1, hidden), state)
            logits = self.readout(state).view(batch, items, width)
            outputs.append(logits)
            if self.config.get('halting', False):
                self._halt_logits.append(self.halt(state).view(batch, items))
            if self.config.get('stepwise', False):
                self._done_logits.append(self.done(state).view(batch, items))
            guess = torch.tanh(logits / 2)  # = 2 * sigmoid(logits) - 1
        return torch.stack(outputs)

    def record_attention(self, enabled: bool = True) -> list[list[Tensor]]:
        """Toggle attention recording; returns per-layer lists the next forward fills, one entry per round."""
        stores: list[list[Tensor]] = [[] for _ in self.looks]
        for look, store in zip(self.looks, stores):
            cast(_Look, look).attention = store if enabled else None
        return stores

    def loss(self, prediction: LoopPrediction, context: Tensor, target: Tensor, width: int,
             steps: Tensor | None = None, step_counts: Tensor | None = None) -> Tensor:
        """Per-bit cross-entropy after every round, later rounds weighted more; queries and probes. With `steps`
        (step-trained models), round t is scored against step t instead and `done` against the step count."""
        if steps is not None and step_counts is not None and prediction.done is not None:
            return step_loss(prediction, steps, step_counts, width)
        probes = to_bits(context[:, :prediction.logits.shape[2] - prediction.queries, 2], width)
        truth = torch.cat([to_bits(target, width), probes], 1).expand_as(prediction.logits)
        if prediction.halts is not None:
            # Expected per-question error over the stopping distribution, plus a small cost per round used.
            error = F.binary_cross_entropy_with_logits(prediction.logits, truth, reduction='none').mean(-1)  # [T,B,I]
            rounds = torch.arange(1, len(error) + 1, device=error.device, dtype=error.dtype)[:, None, None]
            return ((prediction.halts * error).sum(0) + HALT_COST * (prediction.halts * rounds).sum(0)).mean()
        per_round = F.binary_cross_entropy_with_logits(prediction.logits, truth, reduction='none').mean((1, 2, 3))
        if self.config.get('clock', False) and len(per_round) > 1:
            return per_round[-1] + .1 * per_round[:-1].mean()
        weights = torch.arange(1, len(per_round) + 1, device=per_round.device, dtype=per_round.dtype)
        return (per_round * weights).sum() / weights.sum()
