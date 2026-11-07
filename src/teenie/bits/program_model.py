"""Infer a small numeric program from the 999 examples, then refine it by checking it against them.

The examples are summarised into a relation state z. z sets every weight of a tiny calculator that
runs on the bits of (a, b) — the calculator's weights are activations of this episode, not trained
parameters. Its generic parts:

- weighted sums of the input bits (can form a, b, a + 2b, a constant, ...),
- products of two weighted sums (can form a * b, or a_i * b_i for bitwise relations),
- rectified weighted sums (can form max / min via relu(a - b)),
- a scalar s mixing all of these, read out as bits through a periodic (cosine) unit per output bit,
  which gives carries and the wraparound modulo 2^width in one step,
- plus a direct per-bit linear path.

The loop: every round the calculator runs on all 999 examples; each example reports which of its
bits came out wrong, those reports are pooled, and a GRU updates z. A fixed sinusoidal signal tells
the planner which round it is. Questions are answered by the calculator after each round. How many
examples the final program reproduces exactly is returned as `fit`, a self-check.

`informed=True` starts the input scales at binary place values (1, 2, ..., 2^(w-1)) and the readout
frequencies at the matching bit periods; `informed=False` starts both blank (all ones / a random
spread) so the model has to discover place value itself. Both are trainable either way.
"""
from dataclasses import dataclass
import math
from typing import TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import CONTEXT_SIZE
from .data import to_bits
from .loop_model import LoopPrediction, round_signal


class ProgramConfig(TypedDict):
    hidden: int
    channels: int
    width: int
    steps: int
    informed: bool


@dataclass
class ProgramPrediction(LoopPrediction):
    context_logits: Tensor | None = None  # [T, B, 999, W] the program run on the examples after each round
    fit: Tensor | None = None  # [B] share of the 999 examples the final program reproduces exactly


class ProgramModel(nn.Module):
    def __init__(self, hidden: int = 256, channels: int = 16, width: int = 8, steps: int = 6,
                 informed: bool = True) -> None:
        super().__init__()
        if min(hidden, channels, width, steps) < 1:
            raise ValueError('hidden, channels, width and steps must be positive')
        self.config: ProgramConfig = {'hidden': hidden, 'channels': channels, 'width': width, 'steps': steps,
                                      'informed': informed}
        inputs = 2 * width
        # Calculator weight shapes, in the order `plan` emits them.
        self.shapes: list[tuple[int, ...]] = [
            (channels, inputs), (channels,),  # sums
            (channels, inputs), (channels,), (channels, inputs), (channels,),  # product factors
            (channels, inputs), (channels,),  # rectified sums
            (3 * channels,), (1,),  # mix into s
            (width, inputs), (width,),  # direct per-bit path
        ]
        self.example = nn.Sequential(nn.Linear(3 * width, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.key = nn.Linear(hidden, hidden)
        self.first = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))  # what to look for before any check
        self.ask = nn.Linear(hidden, hidden)
        self.plan = nn.Linear(hidden, sum(math.prod(s) for s in self.shapes))
        nn.init.normal_(self.plan.weight, std=.02)
        nn.init.zeros_(self.plan.bias)
        self.report = nn.Sequential(nn.Linear(hidden + 2 * width, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.cell = nn.GRUCell(hidden, hidden)
        place = torch.arange(width, dtype=torch.float32)
        if informed:
            log_scale = torch.cat([place, place]) * math.log(2)
            frequency = math.pi / 2 ** place
            centre = 1.5 * 2 ** place - .5  # middle of the run of s where bit k is 1
        else:
            log_scale = torch.zeros(inputs)
            frequency = torch.rand(width) * math.pi
            centre = torch.zeros(width)
        self.log_scale = nn.Parameter(log_scale)
        self.log_frequency = nn.Parameter(torch.log(frequency))
        self.centre = nn.Parameter(centre)
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(6.)))

    def _pool(self, values: Tensor, keys: Tensor, query: Tensor) -> Tensor:
        weights = torch.softmax(torch.einsum('bnh,bh->bn', keys, query) / math.sqrt(query.shape[-1]), -1)
        return torch.einsum('bn,bnh->bh', weights, values)

    def _unpack(self, z: Tensor) -> list[Tensor]:
        flat = self.plan(z)
        pieces, start = [], 0
        for shape in self.shapes:
            size = math.prod(shape)
            pieces.append(flat[:, start:start + size].view(-1, *shape))
            start += size
        return pieces

    def _run(self, program: list[Tensor], x: Tensor) -> Tensor:
        """Calculator: x [B, n, 2W] input bits in {0, 1} -> logits [B, n, W]."""
        sw, sb, pw1, pb1, pw2, pb2, rw, rb, mix, bias, dw, db = program
        scaled = x * self.log_scale.exp()

        def apply(weight: Tensor, offset: Tensor) -> Tensor:
            return torch.einsum('bmi,bni->bnm', weight, scaled) + offset[:, None]

        sums = apply(sw, sb)
        products = apply(pw1, pb1) * apply(pw2, pb2)
        rectified = F.relu(apply(rw, rb))
        s = (torch.cat([sums, products, rectified], -1) * mix[:, None]).sum(-1) + bias  # [B, n]
        periodic = self.log_sharpness.exp() * torch.cos(self.log_frequency.exp() * (s[..., None] - self.centre))
        direct = torch.einsum('bwi,bni->bnw', dw, x * 2 - 1) + db[:, None]
        return periodic + direct

    def forward(self, context: Tensor, query: Tensor, width: int, target: Tensor | None = None,
                steps: int | None = None) -> ProgramPrediction:
        """context [B, 999, 3], query [B, Q, 2] integers; `target` is unused. Logits after every round."""
        if context.shape[1] != CONTEXT_SIZE or width != self.config['width']:
            raise ValueError(f'context must contain exactly {CONTEXT_SIZE} examples; width must be {self.config["width"]}')
        rounds = self.config['steps'] if steps is None else steps
        a, b, c = (to_bits(context[..., i], width) for i in range(3))
        examples = self.example(torch.cat([a, b, c], -1) * 2 - 1)
        keys = self.key(examples)
        known = torch.cat([a, b], -1)
        asked = torch.cat([to_bits(query[..., 0], width), to_bits(query[..., 1], width)], -1)
        hidden = self.config['hidden']
        z = self._pool(examples, keys, self.first.expand(len(context), -1))
        outputs, context_outputs = [], []
        for step in range(rounds):
            program = self._unpack(z + round_signal(step, hidden, context.device))
            outputs.append(self._run(program, asked))
            fitted = self._run(program, known)
            context_outputs.append(fitted)
            # Each example reports its own wrong bits; the pooled reports update the relation state.
            error = torch.sigmoid(fitted) - c
            reports = self.report(torch.cat([examples, error, c * 2 - 1], -1))
            z = self.cell(self._pool(reports, keys, self.ask(z)), z)
        fit = ((context_outputs[-1] > 0).float() == c).all(-1).float().mean(-1)
        return ProgramPrediction(torch.stack(outputs), query.shape[1], torch.stack(context_outputs), fit)

    def loss(self, prediction: ProgramPrediction, context: Tensor, target: Tensor, width: int) -> Tensor:
        """Per-bit cross-entropy on the questions after every round (later rounds weigh more), plus the same on
        the examples themselves — the program should reproduce what it was built from."""
        weights = torch.arange(1, len(prediction.logits) + 1, device=target.device, dtype=torch.float32)
        weights = weights / weights.sum()
        truth = to_bits(target, width).expand_as(prediction.logits)
        asked = F.binary_cross_entropy_with_logits(prediction.logits, truth, reduction='none').mean((1, 2, 3))
        assert prediction.context_logits is not None
        known = to_bits(context[..., 2], width).expand_as(prediction.context_logits)
        fitted = F.binary_cross_entropy_with_logits(prediction.context_logits, known, reduction='none').mean((1, 2, 3))
        return ((asked + .5 * fitted) * weights).sum()
