"""Meta-learn continuous relation similarity, never an operation classifier."""

import argparse
import json
from pathlib import Path

import torch

from .relation_data import RelationSampler
from .relation_model import RelationModel, save_relation
from .train import select_device


@torch.no_grad()
def evaluate(model, batch, batch_size=8):
    device = next(model.parameters()).device
    correct = 0
    squared = absolute = 0.0
    for start in range(0, len(batch.targets), batch_size):
        subset = batch.slice(start, start + batch_size).to(device)
        prediction = model(*subset.inputs())
        correct += (prediction.round().long() == subset.targets).sum().item()
        squared += (prediction - subset.targets).square().sum().item()
        absolute += (prediction - subset.targets).abs().sum().item()
    n = len(batch.targets)
    return {"accuracy": correct / n, "mse": squared / n, "mae": absolute / n, "count": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--operand-max", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=99)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save", default="checkpoints/relation.pt")
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.log_every) < 1 or not 0 < args.lr < float("inf"):
        parser.error("updates, batch-size, log-every and finite lr must be positive")
    try:
        device = select_device(args.device)
        sampler = RelationSampler(args.operand_max, args.context_size, args.seed, args.seed + 1)
        val = RelationSampler(args.operand_max, args.context_size, args.seed, args.seed + 2).validation()
        torch.manual_seed(args.seed)
        model = RelationModel(args.operand_max).to(device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    path = Path(args.save)
    best_path = path.with_name(path.stem + ".best" + path.suffix)
    best = (-1.0, -float("inf"))
    for step in range(1, args.updates + 1):
        batch = sampler.batch(args.batch_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(*batch.inputs())
        scale = batch.context_y.double().std(1, correction=0).clamp_min(1)
        loss = ((predicted - batch.targets) / scale).square().mean()
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            metrics = evaluate(model, val)
            record = {"step": step, "val": metrics,
                      "lengthscales": model.log_length.detach().clamp(-3.5, 1.1).exp().cpu().tolist(),
                      "ridge": model.log_ridge.detach().clamp(-16, -4).exp().item()}
            print(json.dumps(record), flush=True)
            metadata = {"config": vars(args), **record}
            save_relation(path, model, metadata)
            score = (metrics["accuracy"], -metrics["mse"])
            if score > best:
                best = score
                save_relation(best_path, model, metadata)


if __name__ == "__main__":
    main()
