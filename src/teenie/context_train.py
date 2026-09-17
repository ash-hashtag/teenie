"""Train ONLY add/sub/mul episodes; no division validation or model selection."""

import argparse
import json
from pathlib import Path
import sys

import torch

from .context_checkpoint import save_context_checkpoint
from .context_data import KnownEpisodeSampler
from .context_model import ContextArithModel
from .train import select_device


@torch.no_grad()
def episode_metrics(model, episodes, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    absolute_error = squared_error = 0.0
    for start in range(0, len(episodes.targets), batch_size):
        batch = episodes.slice(start, start + batch_size).to(device)
        raw = model(*batch.inputs()) * model.result_bound
        pred = raw.round().clamp(-model.result_bound, model.result_bound).long()
        correct += (pred == batch.targets).sum().item()
        absolute_error += (raw - batch.targets).abs().sum().item()
        squared_error += (raw - batch.targets).square().sum().item()
    n = len(episodes.targets)
    return {"accuracy": correct / n, "mae": absolute_error / n,
            "mse": squared_error / n, "count": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-range", type=int, default=10)
    parser.add_argument("--context-size", type=int, default=99)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--reasoning-steps", type=int, default=4)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--save", default="checkpoints/context.pt")
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.log_every) < 1 or not 0 < args.lr < float("inf"):
        parser.error("updates, batch-size, log-every and finite lr must be positive")
    try:
        device = select_device(args.device)
        sampler = KnownEpisodeSampler(args.num_range, args.context_size, args.seed, args.seed + 1)
        validation = KnownEpisodeSampler(args.num_range, args.context_size, args.seed, args.seed + 2).validation()
        torch.manual_seed(args.seed)
        model = ContextArithModel(args.num_range, args.hidden, args.heads, args.reasoning_steps).to(device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    print(f"device={device}; training_ops=add,sub,mul; context_size={args.context_size}; division excluded", file=sys.stderr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.updates, eta_min=args.lr * 0.01)
    path = Path(args.save)
    best_path = path.with_name(path.stem + ".best" + path.suffix)
    best_score = (-1.0, -float("inf"))
    for step in range(1, args.updates + 1):
        model.train()
        batch = sampler.batch(args.batch_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*batch.inputs())
        loss = (prediction - batch.targets.float() / model.result_bound).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            metrics = episode_metrics(model, validation, args.batch_size)
            per_op = {}
            n = len(validation.targets) // 3
            for index, name in enumerate(("add", "sub", "mul")):
                per_op[name] = episode_metrics(model, validation.slice(index * n, (index + 1) * n), args.batch_size)
            print(json.dumps({"step": step, "loss": loss.item(), "val": metrics, "val_per_op": per_op}), flush=True)
            config = vars(args).copy()
            save_context_checkpoint(path, model, config, step, metrics)
            score = (metrics["accuracy"], -metrics["mse"])
            if score > best_score:
                best_score = score
                save_context_checkpoint(best_path, model, config, step, metrics)
    print(f"best known-operation validation={best_score[0]:.4%}; checkpoint={best_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
