"""Trainable generic features with an explicit conjugate-gradient fitting loop.

The numerical update is designed, not a learned/discovered arithmetic algorithm.
Only temporary coefficients adapt to support labels at inference.
"""
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import CONTEXT_SIZE
from .numeric import F32_MAX, decode, encode, kinds


class ModelConfig(TypedDict):
    knots: int
    steps: int


@dataclass
class Prediction:
    encoded: Tensor
    kind_logits: Tensor
    finite_values: Tensor | None = None
    output_scale: Tensor | None = None
    support_residual: Tensor | None = None

    def values(self) -> Tensor:
        kind = self.kind_logits.argmax(-1)
        if self.finite_values is None:
            return decode(self.encoded, kind)
        result = self.finite_values.clamp(-F32_MAX, F32_MAX).float()
        result = torch.where(kind == 1, float('inf'), result)
        result = torch.where(kind == 2, -float('inf'), result)
        return torch.where(kind == 3, float('nan'), result)


class ContextModel(nn.Module):
    def __init__(self, knots: int = 16, steps: int = 256) -> None:
        super().__init__()
        if knots < 4 or steps < 1:
            raise ValueError('knots>=4 and steps>=1 required')
        self.config: ModelConfig = {'knots': knots, 'steps': steps}
        self.centers = nn.Parameter(torch.linspace(-.1, 1.1, knots, dtype=torch.float64))
        self.log_width = nn.Parameter(torch.full((knots,), math.log(.09), dtype=torch.float64))
        self.log_ridge = nn.Parameter(torch.tensor(math.log(1e-8), dtype=torch.float64))
        self.kind_head = nn.Linear(6, 4, dtype=torch.float64)
        nn.init.zeros_(self.kind_head.weight)
        with torch.no_grad():
            self.kind_head.bias.copy_(torch.tensor([5., 0., 0., 0.]))

    def features(self, x: Tensor) -> Tensor:
        u, v = x.unbind(-1)
        width = self.log_width.clamp(-5., 0.).exp()
        a = torch.exp(-.5 * ((torch.log1p(256*u)/math.log(257))[..., None]
                             - self.centers).square() / width.square())
        b = torch.exp(-.5 * ((torch.log1p(256*v)/math.log(257))[..., None]
                             - self.centers).square() / width.square())
        return torch.cat((torch.stack((torch.ones_like(u), u, v, u*u, u*v, v*v), -1),
                          a, b, v[..., None]*a, u[..., None]*b), -1)

    def forward(self, context: Tensor, query: Tensor, *, steps: int | None = None,
                return_steps: bool = False) -> Prediction:
        single = query.ndim == 2
        if single:
            query = query[:, None]
        if (context.ndim != 3 or context.shape[1:] != (CONTEXT_SIZE, 3)
                or context.shape[0] < 1 or query.ndim != 3 or query.shape[-1] != 2
                or query.shape[0] != context.shape[0] or query.shape[1] < 1):
            raise ValueError('expected context[B,999,3] and query[B,2] or [B,Q,2]')
        if context.dtype != torch.float32 or query.dtype != torch.float32:
            raise ValueError('context and query must use float32 tensors')
        for inputs in (context[..., :2], query):
            if not torch.isfinite(inputs).all() or (inputs < 0).any():
                raise ValueError('a,b must be finite nonnegative float32 values')
        depth = self.config['steps'] if steps is None else steps
        if depth < 1:
            raise ValueError('steps must be positive')
        input_scale = context[..., :2].double().amax(1, keepdim=True).clamp_min(1.)
        x = context[..., :2].double() / input_scale
        q = (query.double() / input_scale).clamp(max=16.)
        y = context[..., 2].double()
        finite = torch.isfinite(y)
        safe = torch.where(finite, y, 0.)
        count = finite.sum(1, keepdim=True).clamp_min(1)
        mean = safe.sum(1, keepdim=True) / count
        scale = (torch.where(finite, (safe-mean).square(), 0.).sum(1, keepdim=True)
                 / count).sqrt().clamp_min(1e-4)
        values = torch.where(finite, (safe-mean)/scale, 0.)
        phi, psi = self.features(x), self.features(q)
        norm = phi.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        phi, psi = phi/norm, psi/norm
        phi = phi * finite[..., None]
        gram = phi.transpose(1, 2) @ phi / count[..., None]
        ridge = self.log_ridge.clamp(-28., -9.).exp()
        matrix = gram + ridge * torch.eye(phi.shape[-1], device=phi.device, dtype=phi.dtype)
        rhs = (phi.transpose(1, 2) @ values[..., None] / count[..., None]).squeeze(-1)
        z = torch.zeros_like(rhs)
        residual = rhs
        direction = residual
        squared = residual.square().sum(-1, keepdim=True)
        outputs: list[Tensor] = []
        errors: list[Tensor] = []
        for iteration in range(depth):
            product = (matrix @ direction[..., None]).squeeze(-1)
            active = squared > 1e-24
            alpha = torch.where(active, squared / (direction*product).sum(-1, keepdim=True).clamp_min(1e-30), 0.)
            z = z + alpha*direction
            residual = residual - alpha*product
            updated = residual.square().sum(-1, keepdim=True)
            direction = residual + torch.where(active, updated/squared.clamp_min(1e-30), 0.)*direction
            squared = updated
            if return_steps or iteration == depth-1:
                outputs.append(((psi @ z[..., None]).squeeze(-1)*scale + mean).clamp(-F32_MAX, F32_MAX))
                errors.append((((phi @ z[..., None]).squeeze(-1)-values).square().sum(1)/count[:, 0]))
        proportions = F.one_hot(kinds(context[..., 2]), 4).double().mean(1)
        status = self.kind_head(torch.cat((q, proportions[:, None].expand(-1, q.shape[1], -1)), -1))
        result = torch.stack(outputs, -1) if return_steps else outputs[-1]
        if return_steps:
            status = status[:, :, None].expand(-1, -1, depth, -1)
        if single:
            result, status = result[:, 0], status[:, 0]
        scales = scale.reshape(scale.shape[0], *([1]*(result.ndim-1))).expand_as(result)
        return Prediction(encode(result), status, result, scales, torch.stack(errors, -1))

    @torch.no_grad()
    def predict(self, examples: Sequence[Sequence[float]], query: Sequence[float], *, steps: int | None = None) -> float:
        device = next(self.parameters()).device
        context = torch.as_tensor(examples, dtype=torch.float32, device=device)
        pair = torch.as_tensor(query, dtype=torch.float32, device=device)
        if context.ndim != 2 or context.shape[-1] != 3 or pair.shape != (2,):
            raise ValueError('examples must be [K,3] and query [2]')
        if (context[:, :2] == pair).all(-1).any():
            raise ValueError('query must not occur in context')
        return float(self.forward(context[None], pair[None], steps=steps).values().item())
