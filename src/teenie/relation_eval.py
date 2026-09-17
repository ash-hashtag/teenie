"""Frozen anonymous-relation completion, including division only at evaluation."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .context_data import sample_support
from .relation_data import RelationBatch, RelationSampler
from .relation_model import RelationModel, load_relation, predict_document
from .relation_train import evaluate
from .train import select_device


def division_contexts(operand_max=20, context_size=99, seed=10000):
    pairs = torch.cartesian_prod(torch.arange(operand_max + 1), torch.arange(1, operand_max + 1))
    indices = torch.arange(len(pairs))
    support = sample_support(pairs, indices, indices, context_size, torch.Generator().manual_seed(seed))
    answers = torch.div(pairs[:, 0], pairs[:, 1], rounding_mode="floor")
    return RelationBatch(pairs[support], answers[support], pairs, answers)


def context_variant(batch, mode, seed):
    if mode == "correct":
        return batch
    if mode == "shuffled":
        order = torch.rand(batch.context_y.shape, generator=torch.Generator().manual_seed(seed)).argsort(-1)
        return replace(batch, context_y=batch.context_y.gather(1, order))
    if mode == "zero":
        return replace(batch, context_y=torch.zeros_like(batch.context_y))
    if mode == "none":
        return replace(batch, context_x=batch.context_x[:, :0], context_y=batch.context_y[:, :0])
    raise ValueError("unknown context variant")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--contexts", type=int, nargs="+", default=[99, 199, 419])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--input", help="JSON: examples [{a,b,c},...], query {a,b}")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    torch.set_num_threads(1)
    try:
        model, metadata = load_relation(args.load, select_device(args.device))
        model.requires_grad_(False)
        if args.input:
            report = predict_document(model, json.loads(Path(args.input).read_text()))
        else:
            config = metadata["training"]["config"]
            known = RelationSampler(model.operand_max, config["context_size"], config["seed"], config["seed"] + 2).validation()
            report = {"checkpoint": args.load, "model": "continuous_context_regression", "operation_menu": False,
                      "training_relations": metadata["training_relations"], "division_trained": False,
                      "known_validation": evaluate(model, known, args.batch_size), "context_seed": args.seed,
                      "division": {}, "untrained_kernel_baseline": {}}
            baseline = RelationModel(model.operand_max).to(next(model.parameters()).device).eval()
            baseline.requires_grad_(False)
            for count in args.contexts:
                batch = division_contexts(model.operand_max, count, args.seed)
                report["division"][str(count)] = {mode: evaluate(model, context_variant(batch, mode, args.seed), args.batch_size)
                                                  for mode in ("correct", "shuffled", "zero", "none")}
                report["untrained_kernel_baseline"][str(count)] = evaluate(baseline, batch, args.batch_size)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
