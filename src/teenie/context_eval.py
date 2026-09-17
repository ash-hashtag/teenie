"""Frozen in-context evaluation. Floor-division labels are generated ONLY here."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .context_checkpoint import load_context_checkpoint
from .context_data import EpisodeBatch, KnownEpisodeSampler, operand_grid, operation_vectors, sample_support
from .context_train import episode_metrics
from .train import select_device


def division_episodes(num_range=10, context_size=99, seed=0, query=None):
    pairs = operand_grid(num_range)
    pairs = pairs[pairs[:, 1] != 0]
    if query is None:
        indices = torch.arange(len(pairs))
    else:
        match = (pairs == torch.tensor(query)).all(-1).nonzero().flatten()
        if len(match) != 1:
            raise ValueError("division query must be in range and have a nonzero divisor")
        indices = match
    generator = torch.Generator().manual_seed(seed)
    support = sample_support(pairs, indices, torch.arange(len(pairs)), context_size, generator)
    results = torch.div(pairs[:, 0], pairs[:, 1], rounding_mode="floor")
    support_ops, query_ops = operation_vectors(torch.full((len(indices),), 3), context_size)
    return EpisodeBatch(pairs[support], support_ops, results[support], pairs[indices], query_ops, results[indices])


def ablate(episodes, mode, seed=0):
    if mode == "correct":
        return episodes
    if mode == "shuffled_results":
        generator = torch.Generator().manual_seed(seed)
        order = torch.rand(episodes.support_results.shape, generator=generator).argsort(-1)
        return replace(episodes, support_results=episodes.support_results.gather(1, order))
    if mode == "zero_results":
        return replace(episodes, support_results=torch.zeros_like(episodes.support_results))
    if mode == "no_context":
        return replace(episodes, support_pairs=episodes.support_pairs[:, :0],
                       support_ops=episodes.support_ops[:, :0], support_results=episodes.support_results[:, :0])
    raise ValueError("unknown ablation")


@torch.no_grad()
def evaluate_context(model, episodes, batch_size=64, seed=0):
    return {mode: episode_metrics(model, ablate(episodes, mode, seed), batch_size)
            for mode in ("correct", "shuffled_results", "zero_results", "no_context")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--context-size", type=int, default=99)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query", nargs=2, type=int, metavar=("A", "B"))
    parser.add_argument("--output", help="Optional JSON report including each division prediction")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    try:
        device = select_device(args.device)
        model, metadata = load_context_checkpoint(args.load, device)
        division = division_episodes(model.num_range, args.context_size, args.seed, args.query)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    model.requires_grad_(False)
    config = metadata["training_config"]
    known = KnownEpisodeSampler(model.num_range, config["context_size"], config["seed"], config["seed"] + 2).validation()
    report = {"checkpoint": args.load, "checkpoint_step": metadata["step"],
              "context_size": args.context_size, "context_seed": args.seed,
              "training_ops": ["add", "sub", "mul"], "division_trained": False,
              "known_ops": evaluate_context(model, known, args.batch_size, args.seed),
              "floor_division": evaluate_context(model, division, args.batch_size, args.seed)}
    predictions = []
    for start in range(0, len(division.targets), args.batch_size):
        batch = division.slice(start, start + args.batch_size).to(device)
        predicted = model.predict(*batch.inputs()).cpu().tolist()
        for pair, result, target in zip(batch.query_pairs.cpu().tolist(), predicted, batch.targets.cpu().tolist()):
            predictions.append({"a": pair[0], "b": pair[1], "prediction": result, "target": target})
    report["predictions"] = predictions
    if args.query:
        report["demonstrations"] = [{"a": int(pair[0]), "b": int(pair[1]), "op": [0, 0, 0, 1], "result": int(result)}
                                    for pair, result in zip(division.support_pairs[0], division.support_results[0])]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
