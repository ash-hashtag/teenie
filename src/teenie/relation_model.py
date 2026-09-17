"""Anonymous (a,b)->c context regression. No operator vocabulary or program menu."""

from pathlib import Path

import torch
from torch import nn


class RelationModel(nn.Module):
    """Meta-learned RBF similarity and ridge strength, with context-fit coefficients.

    The regression coefficients are temporary activations computed from the
    supplied examples, not model parameters updated by an optimizer at test time.
    This is a continuous kernel-regression baseline, not a symbolic executor.
    """

    def __init__(self, operand_max=20):
        super().__init__()
        if operand_max < 1:
            raise ValueError("operand_max must be positive")
        self.operand_max = operand_max
        self.log_length = nn.Parameter(torch.full((2,), -1.6094379124341003, dtype=torch.float64))
        self.log_ridge = nn.Parameter(torch.tensor(-11.512925464970229, dtype=torch.float64))

    def kernel(self, left, right):
        length = self.log_length.clamp(-3.5, 1.1).exp()
        delta = (left[:, :, None, :] - right[:, None, :, :]) / length
        return (-0.5 * delta.square().sum(-1)).exp()

    def forward(self, context_x, context_y, query_x):
        batch, count = context_x.shape[:2]
        if (context_x.shape != (batch, count, 2) or context_y.shape != (batch, count)
                or query_x.shape != (batch, 2) or batch < 1):
            raise ValueError("expected context_x[B,K,2], context_y[B,K], query_x[B,2]")
        for values in (context_x, context_y, query_x):
            if not torch.isfinite(values).all():
                raise ValueError("context and query values must be finite")
        for values in (context_x, query_x):
            if ((values < 0) | (values > self.operand_max)).any():
                raise ValueError("a,b must be in 0..operand_max")
        if count == 0:
            return torch.zeros(batch, dtype=torch.float64, device=query_x.device)
        x = context_x.double() / self.operand_max
        q = query_x.double()[:, None, :] / self.operand_max
        y = context_y.double()
        mean = y.mean(1, keepdim=True)
        covariance = self.kernel(x, x)
        ridge = self.log_ridge.clamp(-16, -4).exp()
        covariance = covariance + ridge * torch.eye(count, device=x.device, dtype=x.dtype)
        factor = torch.linalg.cholesky(covariance)
        coefficients = torch.cholesky_solve((y - mean).unsqueeze(-1), factor)
        return mean.squeeze(-1) + (self.kernel(q, x) @ coefficients).flatten()


def save_relation(path, model, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "teenie-anonymous-relation-v1", "operand_max": model.operand_max,
               "training_relations": ["add", "sub", "mul"], "training": training,
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_relation(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "teenie-anonymous-relation-v1":
        raise ValueError("expected anonymous relation checkpoint")
    model = RelationModel(payload["operand_max"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


@torch.no_grad()
def predict_document(model, document):
    examples, query = document["examples"], document["query"]
    if not examples or "c" in query or "result" in query:
        raise ValueError("provide labeled examples and a query WITHOUT its answer")
    for row in [*examples, query]:
        if "op" in row:
            raise ValueError("anonymous relation inputs do not take operation IDs")
    pairs = [(row["a"], row["b"]) for row in examples]
    if len(set(pairs)) != len(pairs):
        raise ValueError("context pairs must be distinct")
    if (query["a"], query["b"]) in pairs:
        raise ValueError("query must not appear in context")
    device = next(model.parameters()).device
    output = model(torch.tensor([pairs], dtype=torch.float64, device=device),
                   torch.tensor([[row["c"] for row in examples]], dtype=torch.float64, device=device),
                   torch.tensor([[query["a"], query["b"]]], dtype=torch.float64, device=device)).item()
    return {"prediction": output, "rounded_prediction": round(output), "context_size": len(pairs)}
