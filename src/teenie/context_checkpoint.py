"""Separate checkpoint format: single-pair models cannot read context sets."""

from pathlib import Path

import torch

from .context_model import ContextArithModel


def save_context_checkpoint(path, model, training_config, step, validation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {"format": "teenie-context-v1", "model_config": model.config,
                "training_config": training_config, "training_ops": [0, 1, 2],
                "step": step, "validation": validation,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def load_context_checkpoint(path, device="cpu"):
    contents = torch.load(path, map_location="cpu", weights_only=True)
    if contents.get("format") != "teenie-context-v1" or contents.get("training_ops") != [0, 1, 2]:
        raise ValueError("expected an add/sub/mul context-model checkpoint")
    model = ContextArithModel(**contents["model_config"])
    model.load_state_dict(contents["state_dict"], strict=True)
    return model.to(device).eval(), contents
