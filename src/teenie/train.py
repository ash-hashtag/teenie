"""Small reproducible training runner; emits JSONL metrics to stdout."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Op, make_dataset, split_dataset
from .loop import compute_loss
from .model import RecursiveArithModel
from .checkpoint import load_checkpoint, save_checkpoint


def parse_ops(value):
    aliases = {op.name.lower(): op for op in Op} | {str(int(op)): op for op in Op}
    try:
        ops = tuple(aliases[item.strip().lower()] for item in value.split(","))
    except KeyError as exc:
        raise argparse.ArgumentTypeError("use add,sub,mul or 0,1,2") from exc
    if len(set(ops)) != len(ops):
        raise argparse.ArgumentTypeError("operations must be unique")
    return ops


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu or check your GPU/driver")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader, adaptive=True):
    correct = total = steps_total = 0
    by_op = {}
    device = next(model.parameters()).device
    for a, b, op, target in loader:
        a, b, op, target = (x.to(device) for x in (a, b, op, target))
        result, steps = model.predict(a, b, op, adaptive=adaptive)
        correct += (result == target).sum().item()
        steps_total += steps.sum().item()
        total += len(target)
        for value in op.unique().tolist():
            mask = op == value
            counts = by_op.setdefault(Op(value).name.lower(), [0, 0])
            counts[0] += (result[mask] == target[mask]).sum().item()
            counts[1] += mask.sum().item()
    return {"accuracy": correct / total, "steps": steps_total / total,
            "per_op": {name: {"accuracy": c / n, "count": n} for name, (c, n) in by_op.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-range", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--representation", choices=("embedding", "numeric"),
                        help="Operand/output representation; default: embedding")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--ponder-weight", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--no-ponder", action="store_true", default=None,
                        help="Disable adaptive halting; train and evaluate the final step")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--ops", type=parse_ops, help="Training operations, e.g. add,sub,mul (default: add)")
    parser.add_argument("--save", default="checkpoints/model.pt", help="Output checkpoint path")
    parser.add_argument("--load", help="Load weights/config for evaluation or fresh-optimizer fine-tuning")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate --load without training or saving")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.log_every) < 1:
        parser.error("epochs, batch-size and log-every must be positive")
    if not args.lr > 0 or not args.weight_decay >= 0 or not args.ponder_weight >= 0:
        parser.error("lr must be positive; weight-decay and ponder-weight must be nonnegative")
    if args.eval_only and not args.load:
        parser.error("--eval-only requires --load")
    if args.load and (args.no_loop or any(x is not None for x in
                                        (args.num_range, args.hidden, args.max_steps, args.representation))):
        parser.error("--load restores architecture; omit range, hidden and loop overrides")
    if args.eval_only and any(x is not None for x in (args.ops, args.seed, args.no_ponder)):
        parser.error("--eval-only uses saved ops, seed and halting settings")
    try:
        device = select_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    checkpoint = None
    if args.load:
        model, checkpoint = load_checkpoint(args.load, device)
    seed = args.seed if args.seed is not None else (checkpoint["seed"] if checkpoint else 0)
    ops = args.ops if args.ops is not None else tuple(Op(x) for x in checkpoint["ops"]) if checkpoint else (Op.ADD,)
    adaptive = not args.no_ponder if args.no_ponder is not None else (checkpoint["adaptive"] if checkpoint else True)
    seen_ops = set(ops)
    if checkpoint:
        seen_ops.update(Op(x) for x in checkpoint.get("seen_ops", checkpoint["ops"]))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if checkpoint is None:
        model = RecursiveArithModel(
            10 if args.num_range is None else args.num_range,
            128 if args.hidden is None else args.hidden,
            1 if args.no_loop else (4 if args.max_steps is None else args.max_steps),
            representation=args.representation or "embedding",
        ).to(device)
    print(f"device={device}; ops={','.join(op.name.lower() for op in ops)}", file=sys.stderr)
    train, val = split_dataset(make_dataset(model.num_range, ops), seed=seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    loaders = {"train": train_loader, "val": val_loader}
    for op in Op:
        prefix = "all" if op in ops else ("previously_trained" if op in seen_ops else "zero_shot")
        loaders[f"{prefix}_{op.name.lower()}"] = DataLoader(
            make_dataset(model.num_range, (op,)), batch_size=args.batch_size)

    def report(epoch):
        model.eval()
        metrics = {name: evaluate(model, loader, adaptive) for name, loader in loaders.items()}
        print(json.dumps({"epoch": epoch, **metrics}), flush=True)
        return metrics["val"]["accuracy"]

    if args.eval_only:
        report(checkpoint["epoch"])
        return
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.01)
                 if args.lr_schedule == "cosine" else None)
    best_accuracy = -1.0
    save_path = Path(args.save)
    best_path = save_path.with_name(save_path.stem + ".best" + save_path.suffix)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for a, b, op, target in train_loader:
            a, b, op, target = (x.to(device) for x in (a, b, op, target))
            opt.zero_grad(set_to_none=True)
            logits, halts = model(a, b, op)
            if not adaptive:
                loss = compute_loss(logits[:, -1:], halts[:, -1:], target, model.result_bound,
                                    ponder_weight=0, representation=model.representation)
            else:
                loss = compute_loss(logits, halts, target, model.result_bound,
                                    ponder_weight=args.ponder_weight, representation=model.representation)
            loss.backward()
            opt.step()
        if scheduler:
            scheduler.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            accuracy = report(epoch)
            save_checkpoint(args.save, model, ops=ops, adaptive=adaptive, epoch=epoch,
                            seed=seed, seen_ops=seen_ops)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(best_path, model, ops=ops, adaptive=adaptive, epoch=epoch,
                                seed=seed, seen_ops=seen_ops)
    print(f"best validation accuracy={best_accuracy:.4%}; checkpoint={best_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
