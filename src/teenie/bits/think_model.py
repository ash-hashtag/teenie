"""The context is the working memory: one shared block rewrites every token, round after round.

Every example and every question becomes one token through the same encoder (a question's unknown
c bits are 0). Each round, one weight-shared block updates all tokens in place: examples attend to
examples, a question attends to the examples and to itself. Nothing else is stored between rounds,
so any intermediate result (a partial sum, a relation summary) has to live in those slots. Each round
the token's original encoding is added back (input injection) together with a fixed sinusoidal
signal saying which round it is. The answer is read from each question token after the last round.
Training draws a random number of rounds per batch (between a quarter of `steps` and `steps`), so
short runs give early learning signal and the block cannot depend on one fixed count.

Examples never see the questions and questions never see each other, so each answer depends only
on the 999 examples and its own (a, b). Weights never change at inference.
"""
from typing import TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import CONTEXT_SIZE
from .data import to_bits
from .loop_model import LoopPrediction, round_signal


class ThinkConfig(TypedDict):
    hidden: int
    heads: int
    layers: int
    width: int
    steps: int


class _Block(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.feed = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                  nn.Linear(2 * hidden, hidden))

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        batch, items, hidden = x.shape
        q, k, v = self.qkv(self.norm(x)).view(batch, items, 3, self.heads, hidden // self.heads).permute(2, 0, 3, 1, 4)
        found = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + self.out(found.transpose(1, 2).reshape(batch, items, hidden))
        result: Tensor = x + self.feed(x)
        return result


class ThinkModel(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 1, width: int = 8, steps: int = 16) -> None:
        super().__init__()
        if hidden % heads or hidden % 2 or min(layers, width, steps) < 1:
            raise ValueError('hidden must be even and divide into heads; layers, width, steps >= 1')
        self.config: ThinkConfig = {'hidden': hidden, 'heads': heads, 'layers': layers, 'width': width,
                                    'steps': steps}
        self.encode = nn.Sequential(nn.Linear(3 * width, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([_Block(hidden, heads) for _ in range(layers)])
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width))

    def _round(self, x: Tensor, start: Tensor, allowed: Tensor, signal: Tensor) -> Tensor:
        x = x + start + signal
        for block in self.blocks:
            x = block(x, allowed)
        return x

    def forward(self, context: Tensor, query: Tensor, width: int, target: Tensor | None = None,
                steps: int | None = None) -> LoopPrediction:
        """context [B, 999, 3], query [B, Q, 2] integers; `target` is unused. Logits after every round."""
        if context.shape[1] != CONTEXT_SIZE or width != self.config['width']:
            raise ValueError(f'context must contain exactly {CONTEXT_SIZE} examples; width must be {self.config["width"]}')
        if steps is not None:
            rounds = steps
        elif self.training:
            low = max(1, self.config['steps'] // 4)
            rounds = int(torch.randint(low, self.config['steps'] + 1, ()))
        else:
            rounds = self.config['steps']
        queries = query.shape[1]
        examples = torch.cat([to_bits(context[..., i], width) for i in range(3)], -1) * 2 - 1
        asked = torch.cat([to_bits(query[..., 0], width) * 2 - 1, to_bits(query[..., 1], width) * 2 - 1,
                           query.new_zeros(*query.shape[:2], width, dtype=examples.dtype)], -1)  # c unknown = 0
        start = self.encode(torch.cat([examples, asked], 1))
        x = torch.zeros_like(start)
        total = CONTEXT_SIZE + queries
        allowed = torch.zeros(total, total, dtype=torch.bool, device=context.device)
        allowed[:, :CONTEXT_SIZE] = True  # everyone reads the examples
        allowed[CONTEXT_SIZE:, CONTEXT_SIZE:] = torch.eye(queries, dtype=torch.bool, device=context.device)
        outputs = []
        for step in range(rounds):
            signal = round_signal(step, self.config['hidden'], context.device)
            if self.training:
                # Recompute activations in backward: memory stays flat in the number of rounds.
                x = checkpoint(self._round, x, start, allowed, signal, use_reentrant=False)
            else:
                x = self._round(x, start, allowed, signal)
            outputs.append(self.readout(x[:, CONTEXT_SIZE:]))
        return LoopPrediction(torch.stack(outputs), queries)

    def loss(self, prediction: LoopPrediction, context: Tensor, target: Tensor, width: int) -> Tensor:
        """Mostly the final round, so earlier rounds are free to hold intermediate results; a little on each
        earlier round keeps training stable."""
        truth = to_bits(target, width).expand_as(prediction.logits)
        per_round = F.binary_cross_entropy_with_logits(prediction.logits, truth, reduction='none').mean((1, 2, 3))
        return per_round[-1] + .1 * per_round[:-1].mean() if len(per_round) > 1 else per_round[-1]
