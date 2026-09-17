"""Permutation-invariant demonstration encoder with shared query refinement."""

import torch
from torch import nn

from .data import result_bound


class ContextArithModel(nn.Module):
    def __init__(self, num_range=10, hidden=64, heads=4, steps=4):
        super().__init__()
        if hidden < 1 or heads < 1 or hidden % heads or steps < 1:
            raise ValueError("hidden must be divisible by heads; all dimensions must be positive")
        self.config = dict(num_range=num_range, hidden=hidden, heads=heads, steps=steps)
        self.num_range = num_range
        self.result_bound = result_bound(num_range)
        self.steps = steps
        self.support_encoder = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.query_encoder = nn.Sequential(nn.Linear(2, hidden), nn.Tanh())
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        self.result_head = nn.Linear(hidden, 1)

    def forward(self, support_pairs, support_ops, support_results, query_pairs, query_ops):
        """Predict normalized query result; there is no query-target argument.

        Operation one-hots are categorical matching keys, not semantic learned
        embeddings. A new label uses the same computation as all existing ones.
        """
        batch = query_pairs.size(0)
        count = support_pairs.size(1)
        if (query_pairs.shape != (batch, 2) or query_ops.shape != (batch, 4)
                or support_pairs.shape != (batch, count, 2)
                or support_ops.shape != (batch, count, 4)
                or support_results.shape != (batch, count) or batch == 0):
            raise ValueError("invalid episode tensor shapes")
        for labels in (support_ops, query_ops):
            if not (((labels == 0) | (labels == 1)).all() and (labels.sum(-1) == 1).all()):
                raise ValueError("operation labels must be four-wide one-hot vectors")
        for pairs in (support_pairs, query_pairs):
            if pairs.dtype != torch.long or (pairs.abs() > self.num_range).any():
                raise ValueError("operands must be integer tensors in the configured range")
        if not torch.isfinite(support_results).all():
            raise ValueError("support results must be finite")
        h = self.query_encoder(query_pairs.float() / self.num_range)
        if count:
            mismatch = (support_ops != query_ops[:, None, :]).any(-1)
            if mismatch.all(-1).any():
                raise ValueError("each query needs a demonstration with the same operation label")
            values = torch.cat((support_pairs.float() / self.num_range,
                                support_results.float().unsqueeze(-1) / self.result_bound), -1)
            encoded = self.support_encoder(values)
        for _ in range(self.steps):
            if count:
                read, _ = self.attention(h[:, None, :], encoded, encoded,
                                         key_padding_mask=mismatch, need_weights=False)
                read = read.squeeze(1)
            else:
                read = torch.zeros_like(h)
            h = self.cell(read, h)
        return self.result_head(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, *inputs):
        return (self(*inputs) * self.result_bound).round().clamp(
            -self.result_bound, self.result_bound).long()
