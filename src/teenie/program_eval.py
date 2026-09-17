"""Frozen program induction on contexts. Division labels are evaluation-only."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .context_data import sample_support
from .program_executor import ProgramInducer, trace_program
from .program_model import load_primitives
from .train import select_device


def task_episodes(operand_max, operation, context_size=99, seed=10000):
    pairs = torch.cartesian_prod(torch.arange(operand_max + 1), torch.arange(operand_max + 1))
    if operation == "floor_div":
        pairs = pairs[pairs[:, 1] != 0]
    a, b = pairs.unbind(-1)
    if operation == "add":
        results = a + b
    elif operation == "sub":
        results = a - b
    elif operation == "mul":
        results = a * b
    elif operation == "floor_div":
        results = torch.div(a, b, rounding_mode="floor")
    else:
        raise ValueError("unknown evaluation operation")
    queries = torch.arange(len(pairs))
    supports = sample_support(pairs, queries, queries, context_size, torch.Generator().manual_seed(seed))
    return pairs, results, supports


def evaluate_task(inducer, operation, context_size=99, seed=10000, ablation="correct"):
    pairs, targets, supports = task_episodes(inducer.primitives.operand_max, operation, context_size, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    records = []
    histogram = Counter()
    reasons = Counter()
    correct = answered = 0
    for index, query in enumerate(pairs):
        indices = supports[index]
        answers = targets[indices].clone()
        context_pairs = pairs[indices]
        if ablation == "shuffled_results":
            answers = answers[torch.randperm(context_size, generator=generator)]
        elif ablation == "zero_results":
            answers.zero_()
        elif ablation == "no_context":
            answers, context_pairs = answers[:0], context_pairs[:0]
        elif ablation != "correct":
            raise ValueError("unknown ablation")
        # The hidden target is not passed to prediction or program selection.
        prediction = inducer.predict(context_pairs, answers, query.tolist())
        target = targets[index].item()
        answered += prediction["prediction"] is not None
        correct += prediction["prediction"] == target
        reasons[prediction["reason"]] += 1
        if "program_index" in prediction:
            histogram[prediction["program_index"]] += 1
        records.append({"a": query[0].item(), "b": query[1].item(), "target": target, **prediction})
    return {"accuracy": correct / len(pairs), "correct": correct, "queries": len(pairs),
            "coverage": answered / len(pairs), "answered": answered,
            "accuracy_when_answered": correct / answered if answered else None,
            "reasons": dict(reasons), "program_histogram": dict(histogram), "predictions": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--context-size", type=int, default=99)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--input", help="Predict from arbitrary examples/query JSON instead of running the benchmark")
    parser.add_argument("--output", help="Optional full JSON report path")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        model, metadata = load_primitives(args.load, select_device(args.device))
        model.requires_grad_(False)
        inducer = ProgramInducer(model, args.budget)
        if args.input:
            document = json.loads(Path(args.input).read_text())
            report = inducer.predict_document(document)
            if "program_index" in report:
                report["trace"] = trace_program(model, inducer.programs[report["program_index"]],
                                                document["query"]["a"], document["query"]["b"], args.budget)
        else:
            report = {"checkpoint": args.load, "controller": "enumerative_symbolic_search",
                      "training_tasks": metadata["training_tasks"], "division_trained": False,
                      "operand_bounds": [0, model.operand_max], "register_bound": model.scale,
                      "candidate_count": len(inducer.programs), "budget": args.budget,
                      "context_size": args.context_size, "context_seed": args.seed,
                      "primitive_validation": metadata["training"]["val"],
                      "learned_weights": model.linear.weight.detach().cpu().tolist(),
                      "learned_bias": model.linear.bias.detach().cpu().tolist(),
                      "tasks": {}}
            for operation in ("add", "sub", "mul", "floor_div"):
                report["tasks"][operation] = {}
                for mode in ("correct", "shuffled_results", "zero_results", "no_context"):
                    report["tasks"][operation][mode] = evaluate_task(inducer, operation, args.context_size, args.seed, mode)
            # Example trace uses the selected program, not a DIV-specific branch.
            example = next((row for row in report["tasks"]["floor_div"]["correct"]["predictions"]
                            if row["a"] == 17 and row["b"] == 5), None)
            if example and "program_index" in example:
                report["selected_example"] = {**example, "trace": trace_program(
                    model, inducer.programs[example["program_index"]], 17, 5, args.budget)}
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    # Keep benchmark stdout concise; --output retains all individual predictions.
    if "tasks" in report:
        report = {**report, "tasks": {name: {mode: {k: v for k, v in result.items() if k != "predictions"}
                                                for mode, result in modes.items()}
                                        for name, modes in report["tasks"].items()}}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
