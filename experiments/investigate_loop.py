"""Read-only loop diagnostics, not a new training or division-solving model.

Positive controls deliberately supply the repeated-subtraction program. They
test its mechanics and MUST NOT be reported as model-inferred division.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from teenie.checkpoint import load_checkpoint
from teenie.context_checkpoint import load_context_checkpoint
from teenie.context_data import EpisodeBatch, KnownEpisodeSampler, operation_vectors, sample_support
from teenie.context_eval import division_episodes
from teenie.context_train import episode_metrics
from teenie.train import select_device


def repeated_subtraction(a, b, subtract, budget):
    """Host-supplied algorithm, with explicit remainder/count and a safety cap."""
    if a < 0 or b <= 0:
        raise ValueError("positive-control domain requires a>=0 and b>0")
    remainder, count = a, 0
    trace = [{"remainder": remainder, "count": count}]
    while remainder >= b:
        if count == budget:
            return {"answer": None, "failure": "budget", "trace": trace}
        updated = subtract(remainder, b)
        if not 0 <= updated < remainder:
            return {"answer": None, "failure": "invalid learned transition", "trace": trace}
        remainder, count = updated, count + 1
        trace.append({"remainder": remainder, "count": count})
    return {"answer": count, "failure": None, "trace": trace}


def nonnegative_episodes(upper=10, context_size=99, seed=10000):
    pairs = torch.cartesian_prod(torch.arange(upper + 1), torch.arange(1, upper + 1))
    queries = torch.arange(len(pairs))
    support = sample_support(pairs, queries, queries, context_size, torch.Generator().manual_seed(seed))
    # Reference labels are used solely for the evaluation context and scoring.
    results = torch.div(pairs[:, 0], pairs[:, 1], rounding_mode="floor")
    support_ops, query_ops = operation_vectors(torch.full((len(pairs),), 3), context_size)
    return EpisodeBatch(pairs[support], support_ops, results[support], pairs, query_ops, results)


@torch.no_grad()
def investigate(device):
    torch.set_num_threads(1)
    context_path = Path("checkpoints/context-long-seed0.best.pt")
    primitive_path = Path("checkpoints/numeric-long-seed0.best.pt")
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (context_path, primitive_path)}
    model, metadata = load_context_checkpoint(context_path, device)
    config = metadata["training_config"]
    known = KnownEpisodeSampler(model.num_range, config["context_size"], config["seed"], config["seed"] + 2).validation()
    signed = division_episodes(model.num_range, seed=10000)
    nonnegative = nonnegative_episodes(model.num_range)
    results = []
    original_steps = model.steps
    try:
        for steps in (1, 2, 4, 8, 16, 24):
            model.steps = steps
            row = {"steps": steps, "known": episode_metrics(model, known),
                   "signed_division": episode_metrics(model, signed),
                   "nonnegative_division_0_10": episode_metrics(model, nonnegative)}
            results.append(row)
    finally:
        model.steps = original_steps

    oracle_checks = []
    for budget in (4, 20):
        correct = failures = 0
        for a in range(21):
            for b in range(1, 21):
                out = repeated_subtraction(a, b, lambda x, y: x - y, budget)
                correct += out["answer"] == a // b
                failures += out["failure"] is not None
        oracle_checks.append({"budget": budget, "correct": correct, "queries": 420,
                              "failures": failures, "accuracy": correct / 420})

    primitive, primitive_metadata = load_checkpoint(primitive_path, device)
    primitive_correct = failures = 0

    def learned_subtract(a, b):
        values = [torch.tensor([v], dtype=torch.long, device=device) for v in (a, b, 1)]
        prediction, _ = primitive.predict(*values, adaptive=primitive_metadata["adaptive"])
        return prediction.item()

    for a in range(11):
        for b in range(1, 11):
            out = repeated_subtraction(a, b, learned_subtract, budget=10)
            primitive_correct += out["answer"] == a // b
            failures += out["failure"] is not None

    try:
        outside = nonnegative_episodes(20).slice(0, 1).to(device)
        model(*outside.inputs())
    except ValueError as exc:
        domain_status = str(exc)
    else:
        domain_status = "unexpectedly accepted out-of-range context"
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value
                    for p, value in original_hashes.items())
    assert unchanged
    return {"context_checkpoint": str(context_path), "trained_steps": original_steps,
            "step_sweep": results,
            "oracle_subtraction_0_20_HOST_PROGRAMMED": oracle_checks,
            "oracle_trace_17_by_5": repeated_subtraction(17, 5, lambda x, y: x - y, 20),
            "learned_subtraction_0_10_HOST_PROGRAMMED": {
                "primitive_checkpoint": str(primitive_path), "correct": primitive_correct,
                "queries": 110, "failures": failures, "accuracy": primitive_correct / 110},
            "existing_context_model_on_0_20": domain_status,
            "checkpoints_unchanged": unchanged,
            "caveats": [
                "Increasing inference steps exceeds the model's trained four-step distribution.",
                "Nonnegative subset changes both context composition and support density; it is not a retrained 0..20 comparison.",
                "Host-programmed positive controls supply comparison, loop, counter, primitive choice and return value; no division program was inferred.",
                "No training or checkpoint selection occurs in this diagnostic.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="experiments/loop-investigation.json")
    args = parser.parse_args()
    report = investigate(select_device(args.device))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
