"""Loss never performs arithmetic on nonfinite targets; metrics separate validity."""
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F

from .data import Episode
from .model import ContextModel, Prediction
from .numeric import encode, kinds

Metrics = dict[str, float | int | None]


def training_loss(prediction: Prediction, target: Tensor) -> Tensor:
    target_kind = kinds(target)
    values = encode(target)
    if prediction.encoded.ndim == target.ndim + 1:
        values = values[..., None].expand_as(prediction.encoded)
        target_kind = target_kind[..., None].expand_as(prediction.encoded)
    finite = target_kind == 0
    if prediction.finite_values is not None and prediction.output_scale is not None:
        safe_target = torch.where(torch.isfinite(target), target, 0).double()
        if prediction.encoded.ndim == target.ndim + 1:
            safe_target = safe_target[..., None]
        denominator = torch.maximum(safe_target.abs(), .01*prediction.output_scale).clamp_min(1e-30)
        squared = torch.log1p(((prediction.finite_values - safe_target) / denominator).square())
    else:
        squared = (prediction.encoded - values).square()
    regression = (squared * finite).sum() / finite.sum().clamp_min(1)
    classification = F.cross_entropy(prediction.kind_logits.reshape(-1, 4), target_kind.reshape(-1))
    return regression + .1 * classification


@torch.no_grad()
def measure(model: ContextModel, episodes: Sequence[Episode], *, steps: int | None = None,
            rtol: float = .01, atol: float = 0.,
            baseline: Literal['nearest', 'mean'] | None = None) -> Metrics:
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    encoded: list[Tensor] = []
    for episode in episodes:
        if baseline is None:
            prediction = model.forward(episode.context, episode.query, steps=steps)
        else:
            if baseline == 'nearest':
                x = episode.context[..., :2].double()
                scale = x.amax(1, keepdim=True).clamp_min(1.)
                q = episode.query[:, None] if episode.query.ndim == 2 else episode.query
                distances = ((x[:, :, None] - q[:, None].double()) / scale[:, :, None]).square().sum(-1)
                values = episode.context[..., 2].gather(1, distances.argmin(1))
                if episode.query.ndim == 2:
                    values = values[:, 0]
            else:
                values = episode.context[..., 2].double().mean(1).float()
                if episode.query.ndim == 3:
                    values = values[:, None].expand(-1, episode.query.shape[1])
            prediction = Prediction(encode(values), F.one_hot(kinds(values), 4).float(), values)
        predictions.append(prediction.values().reshape(-1).cpu())
        encoded.append(prediction.encoded.reshape(-1).cpu())
        targets.append(episode.target.reshape(-1).cpu())
    output, target = torch.cat(predictions), torch.cat(targets)
    predicted_kind, target_kind = kinds(output), kinds(target)
    finite = target_kind == 0
    status_ok = predicted_kind == target_kind
    # Float64 avoids overflow in differences/MAE across the float32 range.
    error = (output.double() - target.double()).abs()
    close = torch.where(finite, status_ok & (error <= atol + rtol * target.double().abs()), status_ok)
    finite_both = finite & (predicted_kind == 0)
    log_error = (torch.cat(encoded) - encode(target)).square()
    report: Metrics = {'count': len(target), 'accuracy': close.float().mean().item(),
                       'kind_accuracy': status_ok.float().mean().item(),
                       'finite_accuracy': close[finite].float().mean().item() if finite.any() else None,
                       'nonfinite_accuracy': close[~finite].float().mean().item() if (~finite).any() else None,
                       'finite_encoded_mse': log_error[finite].mean().item() if finite.any() else None,
                       'finite_pair_mae': error[finite_both].mean().item() if finite_both.any() else None,
                       'finite_pair_count': int(finite_both.sum()), 'rtol': rtol, 'atol': atol}
    for value, name in enumerate(('finite', 'pos_inf', 'neg_inf', 'undefined')):
        mask = target_kind == value
        report[f'{name}_count'] = int(mask.sum())
        report[f'{name}_kind_accuracy'] = status_ok[mask].float().mean().item() if mask.any() else None
    # Explicit low/mid/high magnitude strata prevent tiny or infinite answers hiding failures.
    for name, mask in [('zero', finite & (target == 0)),
                       ('tiny', finite & (target.abs() > 0) & (target.abs() < 1e-6)),
                       ('ordinary', finite & (target.abs() >= 1e-6) & (target.abs() <= 1e6)),
                       ('large', finite & (target.abs() > 1e6))]:
        report[f'{name}_count'] = int(mask.sum())
        report[f'{name}_accuracy'] = close[mask].float().mean().item() if mask.any() else None
    return report
