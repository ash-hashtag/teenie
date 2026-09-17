"""Generate held-out relation JSON from a trusted local Python f(a, b)."""

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import random
import runpy


def generate_document(f, *, valid=None, operand_max=20, context_size=99, seed=0, query=None):
    """Return (model input, separate answer); f must be deterministic and finite."""
    if operand_max < 1 or context_size < 1:
        raise ValueError("operand-max and context-size must be positive")
    pairs = [(a, b) for a in range(operand_max + 1) for b in range(operand_max + 1)
             if valid is None or valid(a, b)]
    if len(pairs) < context_size + 1:
        raise ValueError(f"need {context_size + 1} valid pairs, found {len(pairs)}")
    rng = random.Random(seed)
    query = tuple(query) if query is not None else rng.choice(pairs)
    if query not in pairs:
        raise ValueError("query must be a valid pair in 0..operand-max")
    support = rng.sample([pair for pair in pairs if pair != query], context_size)

    def value(pair):
        result = f(*pair)
        if isinstance(result, bool) or not isinstance(result, Real) or not math.isfinite(result):
            raise ValueError(f"f{pair} must return a finite real number, got {result!r}")
        return int(result) if isinstance(result, int) else float(result)

    document = {"examples": [{"a": a, "b": b, "c": value((a, b))} for a, b in support],
                "query": {"a": query[0], "b": query[1]}}
    return document, {"query": dict(document["query"]), "expected": value(query)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", required=True, help="Trusted Python file defining f(a,b), optionally valid(a,b); executes code")
    parser.add_argument("--operand-max", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query", type=int, nargs=2, metavar=("A", "B"))
    parser.add_argument("--output", required=True, help="Model input JSON; answer saved beside it as NAME.answer.json")
    args = parser.parse_args()
    namespace = runpy.run_path(args.function)
    f, valid = namespace.get("f"), namespace.get("valid")
    if not callable(f) or (valid is not None and not callable(valid)):
        parser.error("function file must define f(a,b); optional valid(a,b) must be callable")
    try:
        document, answer = generate_document(f, valid=valid, operand_max=args.operand_max,
                                             context_size=args.context_size, seed=args.seed, query=args.query)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    answer_path = output.with_suffix(".answer.json")
    if output == answer_path:
        parser.error("output must not use the .answer.json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    answer_path.write_text(json.dumps(answer, indent=2, allow_nan=False) + "\n")
    print(f"Input: {output}\nAnswer (do not pass to model): {answer_path}")


if __name__ == "__main__":
    main()
