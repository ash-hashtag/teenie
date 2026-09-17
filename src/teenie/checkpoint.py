"""Portable model weights and experiment metadata; no pickled model objects."""

from pathlib import Path

import torch

from .model import RecursiveArithModel


def save_checkpoint(path, model, *, ops, adaptive, epoch, seed, seen_ops=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": 1,
        "model_config": {"num_range": model.num_range,
                         "hidden": model.hidden,
                         "max_steps": model.max_steps,
                         "representation": model.representation},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "ops": [int(op) for op in ops],
        "seen_ops": sorted({int(op) for op in (ops if seen_ops is None else seen_ops)}),
        "adaptive": bool(adaptive), "epoch": int(epoch), "seed": int(seed),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    model = RecursiveArithModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
