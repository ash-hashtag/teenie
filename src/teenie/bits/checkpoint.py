"""Versioned bit-model weights, independent of the float residual checkpoints."""
from pathlib import Path
from typing import cast

import torch

from ..config import CONTEXT_SIZE
from .bit_token_model import BitTokenConfig, BitTokenModel
from .loop_model import LoopConfig, LoopModel
from .model import BitConfig, BitModel
from .program_model import ProgramConfig, ProgramModel
from .think_model import ThinkConfig, ThinkModel
from .wide_model import WideBitModel, WideConfig

Metadata = dict[str, object]
FORMAT = 'teenie-bits-v1-fixed999'  # bit-serial model
LOOP_FORMAT = 'teenie-bits-loop-v1-fixed999'  # all-at-once context + answer refinement loop
THINK_FORMAT = 'teenie-bits-think-v1-fixed999'  # tokens rewritten in place by one shared block per round
PROGRAM_FORMAT = 'teenie-bits-program-v1-fixed999'  # infers a small calculator from the examples
BIT_TOKEN_FORMAT = 'teenie-bits-bittoken-v1-fixed999'  # one token per answer bit, refined over rounds
WIDE_FORMAT = 'teenie-bits-wide-v1-fixed999'  # bit tokens with no width-sized layers: any width up to `span`
AnyBitModel = BitModel | LoopModel | ThinkModel | ProgramModel | BitTokenModel | WideBitModel
FORMATS: dict[type, str] = {BitModel: FORMAT, LoopModel: LOOP_FORMAT, ThinkModel: THINK_FORMAT,
                            ProgramModel: PROGRAM_FORMAT, BitTokenModel: BIT_TOKEN_FORMAT,
                            WideBitModel: WIDE_FORMAT}


def save_bit_model(path: str | Path, model: AnyBitModel, metadata: Metadata | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save({'format': FORMATS[type(model)], 'config': model.config,
                'context_size': CONTEXT_SIZE, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'metadata': metadata or {}}, temporary)
    temporary.replace(destination)


def load_bit_model(path: str | Path, device: str | torch.device = 'cpu') -> tuple[AnyBitModel, Metadata]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('format') not in FORMATS.values()
            or payload.get('context_size') != CONTEXT_SIZE):
        raise ValueError(f'not one of the {sorted(FORMATS.values())} checkpoints')
    model: AnyBitModel
    if payload['format'] == LOOP_FORMAT:
        model = LoopModel(**cast(LoopConfig, payload['config']))
    elif payload['format'] == BIT_TOKEN_FORMAT:
        model = BitTokenModel(**cast(BitTokenConfig, payload['config']))
    elif payload['format'] == WIDE_FORMAT:
        model = WideBitModel(**cast(WideConfig, payload['config']))
    elif payload['format'] == PROGRAM_FORMAT:
        model = ProgramModel(**cast(ProgramConfig, payload['config']))
    elif payload['format'] == THINK_FORMAT:
        model = ThinkModel(**cast(ThinkConfig, payload['config']))
    else:
        model = BitModel(**cast(BitConfig, payload['config']))
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval(), cast(Metadata, payload['metadata'])
