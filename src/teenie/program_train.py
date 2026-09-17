"""Train arithmetic transitions only, with no division feedback or program search."""

import argparse
import json
from pathlib import Path

import torch

from .program_model import ArithmeticPrimitives, primitive_transitions, save_primitives
from .train import select_device


@torch.no_grad()
def metrics(model, rows):
    raw = model(rows[:, 0], rows[:, 1], rows[:, 2]) * model.scale
    return {"accuracy": (raw.round().long() == rows[:, 3]).float().mean().item(),
            "mse": (raw - rows[:, 3]).square().mean().item(), "count": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--operand-max", type=int, default=20)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save", default="checkpoints/program-primitives.pt")
    args = parser.parse_args()
    if min(args.updates, args.log_every) < 1 or not 0 < args.lr < float("inf"):
        parser.error("updates, log-every and finite lr must be positive")
    try:
        device = select_device(args.device)
        rows = primitive_transitions(args.operand_max)
        torch.manual_seed(args.seed)
        model = ArithmeticPrimitives(args.operand_max).to(device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed))
    n = int(0.8 * len(rows))
    train, val = rows[permutation[:n]].to(device), rows[permutation[n:]].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.updates, eta_min=args.lr * 0.01)
    path = Path(args.save)
    best_path = path.with_name(path.stem + ".best" + path.suffix)
    best = (-1.0, -float("inf"))
    for step in range(1, args.updates + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        prediction = model(train[:, 0], train[:, 1], train[:, 2])
        loss = (prediction - train[:, 3].float() / model.scale).square().mean()
        loss.backward()
        opt.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            model.eval()
            record = {"step": step, "train": metrics(model, train), "val": metrics(model, val)}
            print(json.dumps(record), flush=True)
            metadata = {"config": vars(args), **record}
            save_primitives(path, model, metadata)
            score = (record["val"]["accuracy"], -record["val"]["mse"])
            if score > best:
                best = score
                save_primitives(best_path, model, metadata)


if __name__ == "__main__":
    main()
