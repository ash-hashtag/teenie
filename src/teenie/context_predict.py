"""Predict from a user-supplied labeled context, without generating any answers."""

import argparse
import json
from pathlib import Path

import torch

from .context_checkpoint import load_context_checkpoint
from .train import select_device


def predict_from_context(model, document):
    """Schema: examples=[{a,b,op:[0,0,0,1],result}], query={a,b,op}.

    No arithmetic function or query result is needed by this interface.
    """
    examples, query = document["examples"], document["query"]
    if not examples:
        raise ValueError("provide at least one labeled example")
    if "result" in query:
        raise ValueError("query must not contain its answer")
    for row in [*examples, query]:
        if any(type(row[name]) is not int for name in ("a", "b")):
            raise ValueError("operands must be integers")
        if len(row["op"]) != 4 or any(value not in (0, 1) for value in row["op"]) or sum(row["op"]) != 1:
            raise ValueError("op must be a four-wide one-hot vector")
    keys = [(row["a"], row["b"], tuple(row["op"])) for row in examples]
    if len(set(keys)) != len(keys):
        raise ValueError("demonstrations must be distinct")
    if (query["a"], query["b"], tuple(query["op"])) in keys:
        raise ValueError("query must not appear in the demonstrations")
    device = next(model.parameters()).device
    inputs = (
        torch.tensor([[[row["a"], row["b"]] for row in examples]], dtype=torch.long, device=device),
        torch.tensor([[row["op"] for row in examples]], dtype=torch.float, device=device),
        torch.tensor([[row["result"] for row in examples]], dtype=torch.float, device=device),
        torch.tensor([[query["a"], query["b"]]], dtype=torch.long, device=device),
        torch.tensor([query["op"]], dtype=torch.float, device=device),
    )
    model.eval()
    with torch.no_grad():
        raw = (model(*inputs) * model.result_bound).item()
    return {"prediction": max(-model.result_bound, min(model.result_bound, round(raw))),
            "raw_prediction": raw, "context_size": len(examples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", required=True)
    parser.add_argument("--input", required=True, help="JSON with examples and an unanswered query")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    try:
        model, _ = load_context_checkpoint(args.load, select_device(args.device))
        document = json.loads(Path(args.input).read_text())
        prediction = predict_from_context(model, document)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(prediction), flush=True)


if __name__ == "__main__":
    main()
