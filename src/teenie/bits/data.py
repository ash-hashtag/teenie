"""Episodes of 999 distinct (a, b, c) examples plus held-out query pairs."""
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import CONTEXT_SIZE
from .relations import Family, TRAIN_FAMILIES


def to_bits(x: Tensor, width: int) -> Tensor:
    """Integers [...] -> float bits [..., width], least significant first."""
    return ((x[..., None] >> torch.arange(width, device=x.device)) & 1).float()


def from_bits(bits: Tensor) -> Tensor:
    """Bits [..., width] (0/1, least significant first) -> int64 integers."""
    weights = 1 << torch.arange(bits.shape[-1], device=bits.device)
    return (bits.long() * weights).sum(-1)


@dataclass
class BitEpisodes:
    context: Tensor  # [B, 999, 3] int64 (a, b, c)
    query: Tensor  # [B, Q, 2] int64, never present in the context
    target: Tensor  # [B, Q] int64
    names: list[str]
    width: int

    def to(self, device: torch.device | str) -> 'BitEpisodes':
        return BitEpisodes(self.context.to(device), self.query.to(device), self.target.to(device),
                           self.names, self.width)


class BitSampler:
    def __init__(self, families: Sequence[Family] = TRAIN_FAMILIES, width: int = 8, seed: int = 0) -> None:
        if not families or not 1 <= width <= 31:
            raise ValueError('need families and 1 <= width <= 31')
        self.families = list(families)
        # Relative chance of drawing each family; uniform unless training reweights it.
        self.weights = torch.ones(len(self.families))
        self.width = width
        self.generator = torch.Generator().manual_seed(seed)

    def pairs(self, count: int) -> tuple[Tensor, Tensor]:
        space = 1 << (2 * self.width)
        if count > space:
            raise ValueError(f'{count} distinct pairs do not exist at width {self.width}')
        if space <= 1 << 20:
            chosen = torch.randperm(space, generator=self.generator)[:count]
        else:
            chosen = torch.empty(0, dtype=torch.long)
            while len(chosen) < count:
                drawn = torch.randint(0, space, (2 * count,), generator=self.generator)
                chosen = torch.unique(torch.cat([chosen, drawn]))
            chosen = chosen[torch.randperm(len(chosen), generator=self.generator)[:count]]
        return chosen >> self.width, chosen & ((1 << self.width) - 1)

    def sample(self, batch_size: int, queries: int = 1) -> BitEpisodes:
        contexts, query_pairs, targets, names = [], [], [], []
        for _ in range(batch_size):
            if bool((self.weights == self.weights[0]).all()):  # uniform: same draws as before weights existed
                index = int(torch.randint(0, len(self.families), (), generator=self.generator))
            else:
                index = int(torch.multinomial(self.weights, 1, generator=self.generator))
            family = self.families[index]
            relation = family.make(self.generator, self.width)
            a, b = self.pairs(CONTEXT_SIZE + queries)
            c = relation(a, b, self.width)
            contexts.append(torch.stack([a, b, c], -1)[:CONTEXT_SIZE])
            query_pairs.append(torch.stack([a, b], -1)[CONTEXT_SIZE:])
            targets.append(c[CONTEXT_SIZE:])
            names.append(relation.name)
        return BitEpisodes(torch.stack(contexts), torch.stack(query_pairs), torch.stack(targets), names, self.width)
