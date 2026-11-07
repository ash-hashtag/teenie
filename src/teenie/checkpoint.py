"""Portable versioned weights. V5 uses explicit residual fitting; older architectures are rejected."""
from pathlib import Path
from typing import cast

import torch

from .config import CONTEXT_SIZE
from .model import ContextModel, ModelConfig

Metadata = dict[str, object]


def save_model(path: str | Path, model: ContextModel, metadata: Metadata | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save({'format': 'teenie-residual-v5-fixed999', 'config': model.config, 'context_size': CONTEXT_SIZE,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'metadata': metadata or {}}, temporary)
    temporary.replace(destination)


def load_model(path: str | Path, device: str | torch.device = 'cpu') -> tuple[ContextModel, Metadata]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('format') != 'teenie-residual-v5-fixed999'
            or payload.get('context_size') != CONTEXT_SIZE):
        raise ValueError('not a fixed-999 float-loop checkpoint; retrain using the current CLI')
    config = cast(ModelConfig, payload['config'])
    model = ContextModel(**config)
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval(), cast(Metadata, payload['metadata'])
