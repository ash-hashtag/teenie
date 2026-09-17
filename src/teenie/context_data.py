"""Add/sub/mul-only episodic training data. Division targets live in evaluation."""

from dataclasses import dataclass, fields

import torch
from torch.nn.functional import one_hot

from .data import result_bound


@dataclass
class EpisodeBatch:
    support_pairs: torch.Tensor
    support_ops: torch.Tensor
    support_results: torch.Tensor
    query_pairs: torch.Tensor
    query_ops: torch.Tensor
    targets: torch.Tensor

    def to(self, device):
        return EpisodeBatch(*(getattr(self, f.name).to(device) for f in fields(self)))

    def inputs(self):
        # Query answers are deliberately excluded from the model's interface.
        return (self.support_pairs, self.support_ops, self.support_results,
                self.query_pairs, self.query_ops)

    def slice(self, start, stop):
        return EpisodeBatch(*(getattr(self, f.name)[start:stop] for f in fields(self)))


def operand_grid(num_range):
    result_bound(num_range)
    values = torch.arange(-num_range, num_range + 1)
    return torch.cartesian_prod(values, values)


def pair_split(num_range=10, seed=0):
    pairs = operand_grid(num_range)
    indices = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed))
    n = int(0.8 * len(pairs))
    return pairs, indices[:n], indices[n:]


def known_results(pairs, ops):
    """No fourth operation is accepted or calculated on this training path."""
    if ((ops < 0) | (ops > 2)).any():
        raise ValueError("training targets only support add/sub/mul")
    a, b = pairs.unbind(-1)
    values = torch.stack((a + b, a - b, a * b), -1)
    return values.gather(-1, ops.expand_as(a).unsqueeze(-1)).squeeze(-1)


def sample_support(pairs, query_indices, pool, context_size, generator):
    if context_size < 1:
        raise ValueError("context_size must be positive")
    eligible = pool[None, :] != query_indices[:, None]
    if (eligible.sum(-1) < context_size).any():
        raise ValueError("not enough distinct support pairs after excluding the query")
    scores = torch.rand((len(query_indices), len(pool)), generator=generator)
    scores.masked_fill_(~eligible, -1)
    return pool[scores.topk(context_size, dim=-1).indices]


def operation_vectors(labels, context_size):
    query_ops = one_hot(labels, num_classes=4).float()
    return query_ops[:, None, :].expand(-1, context_size, -1), query_ops


class KnownEpisodeSampler:
    def __init__(self, num_range=10, context_size=99, split_seed=0, sample_seed=0):
        self.pairs, self.train_indices, self.val_indices = pair_split(num_range, split_seed)
        if not 1 <= context_size < len(self.train_indices):
            raise ValueError("context_size must leave a training pair available for the query")
        self.context_size = context_size
        self.generator = torch.Generator().manual_seed(sample_seed)

    def batch(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        query_indices = self.train_indices[torch.randint(len(self.train_indices), (batch_size,),
                                                        generator=self.generator)]
        operations = torch.randint(3, (batch_size,), generator=self.generator)
        return self._build(query_indices, operations)

    def validation(self):
        # Every held-out pair is queried for each known operation. All support
        # answers come from the training pool, not other validation pairs.
        queries = self.val_indices.repeat(3)
        operations = torch.arange(3).repeat_interleave(len(self.val_indices))
        return self._build(queries, operations)

    def _build(self, query_indices, operations):
        support = sample_support(self.pairs, query_indices, self.train_indices,
                                 self.context_size, self.generator)
        # Label identity is randomized independently of the actual operation.
        # The reserved fourth slot is NEVER active in training/known validation.
        labels = torch.randint(3, (len(query_indices),), generator=self.generator)
        support_ops, query_ops = operation_vectors(labels, self.context_size)
        return EpisodeBatch(
            self.pairs[support], support_ops,
            known_results(self.pairs[support], operations[:, None]),
            self.pairs[query_indices], query_ops,
            known_results(self.pairs[query_indices], operations),
        )
