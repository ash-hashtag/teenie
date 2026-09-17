"""Learned arithmetic primitives; control flow is supplied by a separate VM."""

from pathlib import Path

import torch
from torch import nn


class ArithmeticPrimitives(nn.Module):
    """Two learned affine heads: instruction 0=ADD, 1=SUB. No DIV or MUL head.

    Linearity is an explicit architectural prior. Coefficients are optimized
    from data, not assigned the arithmetic formulas.
    """

    def __init__(self, operand_max=20):
        super().__init__()
        if operand_max < 1:
            raise ValueError("operand_max must be positive")
        self.operand_max = operand_max
        self.scale = max(2 * operand_max, operand_max**2)
        self.linear = nn.Linear(2, 2)

    def forward(self, left, right, instruction):
        values = torch.stack((left, right), -1).float() / self.scale
        output = self.linear(values)
        instruction = torch.broadcast_to(instruction, left.shape)
        return output.gather(-1, instruction.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def execute(self, left, right, instruction):
        # No clipping: the VM explicitly detects overflow rather than hiding it.
        return (self(left, right, instruction) * self.scale).round().long()


def primitive_transitions(operand_max=20):
    """Unique transitions from ADD/SUB examples and MUL repeated-add traces.

    No division answers or division execution traces are computed here.
    """
    if operand_max < 1:
        raise ValueError("operand_max must be positive")
    rows = set()
    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            rows.add((a, b, 0, a + b))
            rows.add((a, b, 1, a - b))
            accumulator = 0
            for counter in range(b):
                rows.add((accumulator, a, 0, accumulator + a))
                rows.add((counter, 1, 0, counter + 1))
                accumulator += a
            assert accumulator == a * b
    return torch.tensor(sorted(rows), dtype=torch.long)


def save_primitives(path, model, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "teenie-program-primitives-v1", "operand_min": 0,
               "operand_max": model.operand_max, "training_tasks": ["add", "sub", "mul"],
               "instructions": ["add", "sub"], "training": training,
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_primitives(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != "teenie-program-primitives-v1"
            or payload.get("training_tasks") != ["add", "sub", "mul"]
            or payload.get("instructions") != ["add", "sub"]):
        raise ValueError("expected add/sub/mul-only program-primitives checkpoint")
    model = ArithmeticPrimitives(payload["operand_max"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
