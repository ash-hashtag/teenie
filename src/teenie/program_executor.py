"""Abandoned operation-menu experiment; NOT used by anonymous relation models.

Small generic loop language and a context-only constraint-search controller.

This is neuro-symbolic program induction, NOT a learned neural controller.
Comparisons, register routing and enumeration are explicit architectural priors.
"""

from dataclasses import asdict, dataclass
from itertools import product

import torch


SOURCES = ("zero", "one", "a", "b")
COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")


@dataclass(frozen=True)
class Program:
    kind: str
    instruction: int
    left: str = "a"
    right: str = "b"
    guard_register: str = "counter"
    comparison: str = "lt"
    guard_right: str = "b"
    returns: str = "state"

    def describe(self):
        return asdict(self)


def enumerate_programs():
    """32 direct expressions and 3,072 loops, independent of any task labels."""
    direct = [Program("direct", op, left, right)
              for op, left, right in product(range(2), SOURCES, SOURCES)]
    loops = [Program("loop", op, initial, operand, register, comparison, bound, returns)
             for initial, register, comparison, bound, op, operand, returns in product(
                 SOURCES, ("counter", "state"), COMPARISONS, SOURCES,
                 range(2), SOURCES, ("state", "counter"))]
    return direct + loops


def _source(name, pairs):
    return {"zero": torch.zeros_like(pairs[:, 0]), "one": torch.ones_like(pairs[:, 0]),
            "a": pairs[:, 0], "b": pairs[:, 1]}[name]


def _guard(left, right, comparison_codes):
    return ((comparison_codes == 0) & (left < right)
            | (comparison_codes == 1) & (left <= right)
            | (comparison_codes == 2) & (left > right)
            | (comparison_codes == 3) & (left >= right)
            | (comparison_codes == 4) & (left == right)
            | (comparison_codes == 5) & (left != right))


@torch.no_grad()
def execute_programs(primitives, programs, pairs, budget=24):
    """Vectorized execution [program, pair]; invalid/nonhalting results stay invalid.

    For loops, `left` initializes state; `right` supplies the update operand.
    Each iteration writes primitive(state,right) and ADD(counter,1). The generic
    predicate controls termination; the return-register field chooses the output.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be an integer [N,2] tensor")
    if ((pairs < 0) | (pairs > primitives.operand_max)).any():
        raise ValueError("input operands outside 0..operand_max")
    if not programs or not len(pairs):
        raise ValueError("programs and pairs must be nonempty")
    device = next(primitives.parameters()).device
    pairs = pairs.to(device)
    state = torch.stack([_source(p.left, pairs) for p in programs])
    right = torch.stack([_source(p.right, pairs) for p in programs])
    bound = torch.stack([_source(p.guard_right, pairs) for p in programs])
    op = torch.tensor([p.instruction for p in programs], device=device)[:, None].expand_as(state)
    loop = torch.tensor([p.kind == "loop" for p in programs], device=device)[:, None].expand_as(state)
    state_guard = torch.tensor([p.guard_register == "state" for p in programs], device=device)[:, None]
    return_counter = torch.tensor([p.returns == "counter" for p in programs], device=device)[:, None]
    comparison = torch.tensor([COMPARISONS.index(p.comparison) for p in programs], device=device)[:, None]
    counter = torch.zeros_like(state)
    steps = torch.zeros_like(state)
    valid = torch.ones_like(state, dtype=torch.bool)
    # Direct expressions execute once; loops use their guard before each update.
    direct_output = primitives.execute(state, right, op)
    valid &= loop | (direct_output.abs() <= primitives.scale)
    state = torch.where(loop, state, direct_output)
    steps = torch.where(loop, steps, torch.ones_like(steps))
    active = loop.clone()
    for _ in range(budget + 1):
        predicate = _guard(torch.where(state_guard, state, counter), bound, comparison)
        active &= predicate & valid
        if not active.any():
            break
        exhausted = active & (steps >= budget)
        valid &= ~exhausted
        active &= ~exhausted
        if not active.any():
            break
        updated = primitives.execute(state, right, op)
        incremented = primitives.execute(counter, torch.ones_like(counter), torch.zeros_like(counter))
        bounded = (updated.abs() <= primitives.scale) & (incremented.abs() <= primitives.scale)
        valid &= ~active | bounded
        state = torch.where(active, updated, state)
        counter = torch.where(active, incremented, counter)
        steps += active.long()
    output = torch.where(loop & return_counter, counter, state)
    return output, valid, steps


@torch.no_grad()
def trace_program(primitives, program, a, b, budget=24):
    """Human-readable trace using the same learned arithmetic and generic guard."""
    device = next(primitives.parameters()).device
    pairs = torch.tensor([[a, b]], dtype=torch.long, device=device)
    _, validity, _ = execute_programs(primitives, [program], pairs, budget)
    sources = {"zero": 0, "one": 1, "a": a, "b": b}

    def apply(op, x, y):
        return primitives.execute(torch.tensor([x], device=device), torch.tensor([y], device=device),
                                  torch.tensor([op], device=device)).item()

    state, counter = sources[program.left], 0
    trace = [{"state": state, "counter": counter}]
    if program.kind == "direct":
        state = apply(program.instruction, state, sources[program.right])
        trace.append({"state": state, "counter": counter})
    else:
        for _ in range(budget):
            left = state if program.guard_register == "state" else counter
            predicate = _guard(torch.tensor(left), torch.tensor(sources[program.guard_right]),
                               torch.tensor(COMPARISONS.index(program.comparison))).item()
            if not predicate:
                break
            state = apply(program.instruction, state, sources[program.right])
            counter = apply(0, counter, 1)
            trace.append({"state": state, "counter": counter})
            if abs(state) > primitives.scale or abs(counter) > primitives.scale:
                break
    answer = counter if program.kind == "loop" and program.returns == "counter" else state
    return {"valid": validity.item(), "answer": answer if validity.item() else None, "states": trace}


class ProgramInducer:
    """Precompute candidate executions (no labels), then fit SUPPORT answers only."""

    def __init__(self, primitives, budget=24):
        self.primitives = primitives.eval()
        self.budget = budget
        self.programs = enumerate_programs()
        upper = primitives.operand_max
        self.pairs = torch.cartesian_prod(torch.arange(upper + 1), torch.arange(upper + 1))
        output, valid, steps = execute_programs(primitives, self.programs, self.pairs, budget)
        self.output, self.valid, self.steps = output.cpu(), valid.cpu(), steps.cpu()

    def matching_programs(self, pairs, results):
        """No query or query target is accepted by program selection."""
        if len(pairs) == 0:
            return torch.empty(0, dtype=torch.long)
        upper = self.primitives.operand_max
        pairs, results = torch.as_tensor(pairs), torch.as_tensor(results)
        if (pairs.dtype != torch.long or pairs.shape != (len(results), 2)
                or results.ndim != 1 or results.dtype != torch.long):
            raise ValueError("support pairs and answers must be integer tensors")
        if ((pairs < 0) | (pairs > upper)).any():
            raise ValueError("support operands outside configured range")
        if len(pairs.unique(dim=0)) != len(pairs):
            raise ValueError("support pairs must be distinct")
        columns = pairs[:, 0] * (upper + 1) + pairs[:, 1]
        return (self.valid[:, columns] & (self.output[:, columns] == results[None, :])).all(1).nonzero().flatten()

    def predict(self, pairs, results, query):
        upper = self.primitives.operand_max
        if len(query) != 2 or any(type(v) is not int or not 0 <= v <= upper for v in query):
            raise ValueError("query operands must be integers in the configured range")
        pairs = torch.as_tensor(pairs)
        if pairs.numel() and (pairs == torch.tensor(query)).all(-1).any():
            raise ValueError("query must not appear in support")
        matches = self.matching_programs(pairs, results)
        if not len(matches):
            return {"prediction": None, "reason": "no_consistent_program", "matches": 0}
        # Canonical ordering was fixed before looking at any context or query.
        selected = matches[0].item()
        column = query[0] * (upper + 1) + query[1]
        if not self.valid[matches, column].all():
            return {"prediction": None, "reason": "candidate_does_not_halt", "matches": len(matches)}
        predictions = self.output[matches, column].unique()
        if len(predictions) != 1:
            return {"prediction": None, "reason": "ambiguous_context", "matches": len(matches),
                    "candidate_predictions": predictions.tolist()}
        return {"prediction": predictions.item(), "reason": "consistent",
                "matches": len(matches), "program_index": selected,
                "program": self.programs[selected].describe(), "steps": self.steps[selected, column].item()}

    def predict_document(self, document):
        examples, query = document["examples"], document["query"]
        if "result" in query:
            raise ValueError("query must not include its answer")
        for row in [*examples, query]:
            if (len(row["op"]) != 4 or any(x not in (0, 1) for x in row["op"])
                    or sum(row["op"]) != 1):
                raise ValueError("op must be a four-wide one-hot label")
            if any(type(row[key]) is not int for key in ("a", "b")):
                raise ValueError("operands must be integers")
        matching = [row for row in examples if row["op"] == query["op"]]
        if any(type(row["result"]) is not int for row in matching):
            raise ValueError("context answers must be integers")
        return self.predict([[row["a"], row["b"]] for row in matching],
                            torch.tensor([row["result"] for row in matching], dtype=torch.long),
                            [query["a"], query["b"]])
