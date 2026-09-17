"""Shared recurrent cell with per-example inference halting."""

import torch
from torch import nn

from .data import result_bound


class RecursiveArithModel(nn.Module):
    def __init__(self, num_range=10, hidden=128, max_steps=4, representation="embedding"):
        super().__init__()
        if hidden < 1 or max_steps < 1:
            raise ValueError("hidden and max_steps must be positive")
        self.result_bound = result_bound(num_range)
        if representation not in ("embedding", "numeric"):
            raise ValueError("representation must be embedding or numeric")
        self.representation = representation
        self.num_range, self.max_steps = num_range, max_steps
        self.hidden = hidden
        self.embed_a = (nn.Embedding(2 * num_range + 1, hidden) if representation == "embedding"
                        else nn.Linear(1, hidden))
        self.embed_b = (nn.Embedding(2 * num_range + 1, hidden) if representation == "embedding"
                        else nn.Linear(1, hidden))
        self.embed_op = nn.Embedding(3, hidden)
        if representation == "numeric":
            self.cell = nn.Sequential(nn.Linear(3 * hidden + 1, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh())
        else:
            self.cell = nn.Sequential(nn.Linear(3 * hidden + 1, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden))
        self.halt_head = nn.Linear(hidden, 1)
        self.result_head = nn.Linear(hidden, 1 if representation == "numeric" else 2 * self.result_bound + 1)

    def _initial(self, a, b, op):
        if a.ndim != 1 or a.shape != b.shape or a.shape != op.shape or not a.numel():
            raise ValueError("inputs must be matching nonempty vectors")
        if any(x.dtype != torch.long for x in (a, b, op)):
            raise ValueError("inputs must use torch.long")
        if ((a.abs() > self.num_range) | (b.abs() > self.num_range)).any():
            raise ValueError("operand outside configured range")
        if ((op < 0) | (op > 2)).any():
            raise ValueError("unknown operation")
        if self.representation == "numeric":
            ea = self.embed_a(a.float()[:, None] / self.num_range)
            eb = self.embed_b(b.float()[:, None] / self.num_range)
        else:
            ea, eb = self.embed_a(a + self.num_range), self.embed_b(b + self.num_range)
        return ea, eb, ea + eb + self.embed_op(op)

    def _step(self, h, ea, eb, t):
        step = h.new_full((h.size(0), 1), t)
        h = self.cell(torch.cat((h, ea, eb, step), dim=-1))
        return h, self.result_head(h), self.halt_head(h).squeeze(-1).sigmoid()

    def forward(self, a, b, op):
        """Logits [B,T,C] or normalized numeric predictions [B,T,1], plus halts [B,T]."""
        ea, eb, h = self._initial(a, b, op)
        outputs, halts = [], []
        for t in range(self.max_steps):
            h, logits, halt = self._step(h, ea, eb, t)
            outputs.append(logits)
            halts.append(halt)
        return torch.stack(outputs, 1), torch.stack(halts, 1)

    def decode(self, output):
        if self.representation == "numeric":
            return (output.squeeze(-1) * self.result_bound).round().clamp(
                -self.result_bound, self.result_bound).long()
        return output.argmax(-1) - self.result_bound

    @torch.no_grad()
    def predict(self, a, b, op, adaptive=True):
        """Return signed results and steps; remove halted examples from the loop."""
        ea, eb, h = self._initial(a, b, op)
        active = torch.arange(a.size(0), device=a.device)
        results, steps = torch.empty_like(a), torch.empty_like(a)
        for t in range(self.max_steps):
            h, logits, halt = self._step(h, ea, eb, t)
            stop = (halt > 0.5) if adaptive else torch.zeros_like(halt, dtype=torch.bool)
            if t == self.max_steps - 1:
                stop = torch.ones_like(stop)
            results[active[stop]] = self.decode(logits[stop])
            steps[active[stop]] = t + 1
            active = active[~stop]
            if not active.numel():
                break
            h, ea, eb = h[~stop], ea[~stop], eb[~stop]
        return results, steps
