"""Differentiable halting objective (expected task loss + expected steps)."""

import torch
from torch.nn import functional as F


def halt_distribution(halts):
    # Final step absorbs all surviving probability, guaranteeing total mass one.
    survival = torch.cat((torch.ones_like(halts[:, :1]),
                          (1 - halts[:, :-1]).cumprod(dim=1)), dim=1)
    conditional = torch.cat((halts[:, :-1], torch.ones_like(halts[:, -1:])), dim=1)
    return survival * conditional


def compute_loss(logits, halts, target, result_bound, ponder_weight=0.01,
                 representation="embedding"):
    if ponder_weight < 0:
        raise ValueError("ponder_weight must be nonnegative")
    weights = halt_distribution(halts)
    if representation == "numeric":
        ce = (logits.squeeze(-1) - target[:, None].float() / result_bound).square()
    elif representation == "embedding":
        labels = (target + result_bound)[:, None].expand(-1, logits.size(1))
        ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    else:
        raise ValueError("unknown representation")
    steps = torch.arange(1, logits.size(1) + 1, device=logits.device)
    return (weights * (ce + ponder_weight * steps)).sum(1).mean()
