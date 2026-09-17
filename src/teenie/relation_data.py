"""Anonymous relation episodes on 0..N; training targets are only add/sub/mul."""

from dataclasses import dataclass, fields

import torch

from .context_data import sample_support


@dataclass
class RelationBatch:
    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    targets: torch.Tensor

    def inputs(self):
        return self.context_x, self.context_y, self.query_x

    def to(self, device):
        return RelationBatch(*(getattr(self, f.name).to(device) for f in fields(self)))

    def slice(self, start, stop):
        return RelationBatch(*(getattr(self, f.name)[start:stop] for f in fields(self)))


def known_values(pairs, relation):
    if ((relation < 0) | (relation > 2)).any():
        raise ValueError("training is restricted to add/sub/mul")
    a, b = pairs.unbind(-1)
    outputs = torch.stack((a + b, a - b, a * b), -1)
    return outputs.gather(-1, relation.expand_as(a).unsqueeze(-1)).squeeze(-1)


class RelationSampler:
    def __init__(self, operand_max=20, context_size=99, split_seed=0, sample_seed=0):
        if operand_max < 1:
            raise ValueError("operand_max must be positive")
        self.pairs = torch.cartesian_prod(torch.arange(operand_max + 1), torch.arange(operand_max + 1))
        order = torch.randperm(len(self.pairs), generator=torch.Generator().manual_seed(split_seed))
        n = int(0.8 * len(order))
        self.train, self.val = order[:n], order[n:]
        if not 1 <= context_size < len(self.train):
            raise ValueError("context must leave a training query available")
        self.context_size = context_size
        self.generator = torch.Generator().manual_seed(sample_seed)

    def _build(self, queries, relations):
        support = sample_support(self.pairs, queries, self.train, self.context_size, self.generator)
        x, q = self.pairs[support], self.pairs[queries]
        return RelationBatch(x, known_values(x, relations[:, None]), q, known_values(q, relations))

    def batch(self, count):
        queries = self.train[torch.randint(len(self.train), (count,), generator=self.generator)]
        return self._build(queries, torch.randint(3, (count,), generator=self.generator))

    def validation(self):
        return self._build(self.val.repeat(3), torch.arange(3).repeat_interleave(len(self.val)))
