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


def _first_rows(rows: Tensor) -> tuple[Tensor, Tensor]:
    """Distinct rows and, in their original order, the index of each one's first occurrence."""
    unique, inverse = torch.unique(rows, dim=0, return_inverse=True)
    first = torch.full((len(unique),), len(rows), dtype=torch.long).scatter_reduce(
        0, inverse, torch.arange(len(rows)), 'amin')
    return unique, first.sort().values


@dataclass
class BitEpisodes:
    context: Tensor  # [B, 999, 3] int64 (a, b, c)
    query: Tensor  # [B, Q, 2] int64, never present in the context
    target: Tensor  # [B, Q] int64
    names: list[str]
    width: int
    # With `sample(..., steps=True)`: intermediate values for every example and question, [B, S, 999 + Q]
    # (padded with the final answer), and how many steps each one takes, [B, 999 + Q].
    steps: Tensor | None = None
    step_counts: Tensor | None = None

    def to(self, device: torch.device | str) -> 'BitEpisodes':
        return BitEpisodes(self.context.to(device), self.query.to(device), self.target.to(device), self.names,
                           self.width, None if self.steps is None else self.steps.to(device),
                           None if self.step_counts is None else self.step_counts.to(device))


class BitSampler:
    def __init__(self, families: Sequence[Family] = TRAIN_FAMILIES, width: int = 8, seed: int = 0,
                 small: float = 0.) -> None:
        """`small`: the share of episodes whose inputs a and b are drawn below 2**L, with L between max(width/4, 6)
        and width/2 bits (chosen per episode), so that products, squares and sums keep their true size instead of
        wrapping at the width: the size of the answer is then a hint, as it is for a person. Answers keep the full
        width. Other episodes draw inputs over the whole width, as before."""
        if not families or not 1 <= width <= 32 or not 0 <= small <= 1:
            raise ValueError('need families, 1 <= width <= 32 and 0 <= small <= 1')
        self.small = small
        self.families = list(families)
        # Relative chance of drawing each family; uniform unless training reweights it.
        self.weights = torch.ones(len(self.families))
        self.width = width
        self.generator = torch.Generator().manual_seed(seed)

    def input_bits(self) -> int:
        """How many bits this episode's inputs use: the full width, or for a `small` episode fewer."""
        low = max(self.width // 4, 6)
        if self.small and low <= self.width // 2 and float(torch.rand((), generator=self.generator)) < self.small:
            return int(torch.randint(low, self.width // 2 + 1, (), generator=self.generator))
        return self.width

    def pairs(self, count: int, bits: int | None = None) -> tuple[Tensor, Tensor]:
        bits = self.width if bits is None else bits
        if bits > 15:  # draw a and b separately; a repeated pair is all but impossible, but still removed
            chosen = torch.empty(0, 2, dtype=torch.long)
            while len(chosen) < count:
                drawn = torch.randint(0, 1 << bits, (count, 2), generator=self.generator)
                chosen = torch.cat([chosen, drawn])
                unique, first = _first_rows(chosen)
                chosen = chosen[first]
            return chosen[:count, 0], chosen[:count, 1]
        space = 1 << (2 * bits)
        if count > space:
            raise ValueError(f'{count} distinct pairs do not exist at {bits} bits')
        if space <= 1 << 20:
            chosen = torch.randperm(space, generator=self.generator)[:count]
        else:
            chosen = torch.empty(0, dtype=torch.long)
            while len(chosen) < count:
                drawn = torch.randint(0, space, (2 * count,), generator=self.generator)
                chosen = torch.unique(torch.cat([chosen, drawn]))
            chosen = chosen[torch.randperm(len(chosen), generator=self.generator)[:count]]
        return chosen >> bits, chosen & ((1 << bits) - 1)

    def sample(self, batch_size: int, queries: int = 1, steps: bool = False) -> BitEpisodes:
        contexts, query_pairs, targets, names = [], [], [], []
        values: list[Tensor] = []
        counts: list[Tensor] = []
        for _ in range(batch_size):
            if bool((self.weights == self.weights[0]).all()):  # uniform: same draws as before weights existed
                index = int(torch.randint(0, len(self.families), (), generator=self.generator))
            else:
                index = int(torch.multinomial(self.weights, 1, generator=self.generator))
            family = self.families[index]
            relation = family.make(self.generator, self.width)
            a, b = self.pairs(CONTEXT_SIZE + queries, self.input_bits())
            c = relation(a, b, self.width)
            if steps:
                if relation.steps is None:
                    trace, count = c[None], torch.ones_like(c)
                else:
                    trace, count = relation.steps(a, b, self.width)
                values.append(trace & ((1 << self.width) - 1))
                counts.append(count)
            contexts.append(torch.stack([a, b, c], -1)[:CONTEXT_SIZE])
            query_pairs.append(torch.stack([a, b], -1)[CONTEXT_SIZE:])
            targets.append(c[CONTEXT_SIZE:])
            names.append(relation.name)
        episodes = BitEpisodes(torch.stack(contexts), torch.stack(query_pairs), torch.stack(targets), names, self.width)
        if steps:
            longest = max(len(v) for v in values)  # pad every trace by repeating its final answer
            episodes.steps = torch.stack([torch.cat([v, v[-1:].expand(longest - len(v), -1)]) for v in values])
            episodes.step_counts = torch.stack(counts)
        return episodes
