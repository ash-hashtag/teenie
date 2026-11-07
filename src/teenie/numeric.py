"""IEEE float32 values with stable finite encoding and explicit special-value kinds."""
from enum import IntEnum
import math

import numpy as np
import torch
from torch import Tensor

F32_MAX: float = float(np.finfo(np.float32).max)
F32_MIN: float = float(np.nextafter(np.float32(0), np.float32(1)))
LOG_SPAN: float = math.log1p(F32_MAX / F32_MIN)


class ValueKind(IntEnum):
    FINITE = 0
    POS_INF = 1
    NEG_INF = 2
    UNDEFINED = 3


def kinds(values: Tensor) -> Tensor:
    result = torch.zeros_like(values, dtype=torch.long)
    result = torch.where(torch.isposinf(values), int(ValueKind.POS_INF), result)
    result = torch.where(torch.isneginf(values), int(ValueKind.NEG_INF), result)
    return torch.where(torch.isnan(values), int(ValueKind.UNDEFINED), result)


def encode(values: Tensor) -> Tensor:
    safe = torch.where(torch.isfinite(values), values, 0).double()
    return (safe.sign() * torch.log1p(safe.abs() / F32_MIN) / LOG_SPAN).float()


def decode(encoded: Tensor, kind: Tensor) -> Tensor:
    # A FINITE prediction stays finite. Only the status head emits infinity/NaN.
    magnitude = (torch.expm1(encoded.double().abs().clamp(max=1) * LOG_SPAN) * F32_MIN).clamp(max=F32_MAX)
    result = (encoded.sign() * magnitude).float()
    result = torch.where(kind == int(ValueKind.POS_INF), float('inf'), result)
    result = torch.where(kind == int(ValueKind.NEG_INF), -float('inf'), result)
    return torch.where(kind == int(ValueKind.UNDEFINED), float('nan'), result)
