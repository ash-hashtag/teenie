# Investigation: recurrence is not yet a learned arithmetic executor

## Question

Can the model infer floor division as the number of successful subtractions?
Would operands in `0..20` make that easier than `-10..10`?

For `a >= 0` and `b > 0`, the proposed algorithm is valid:

```text
remainder = a
count = 0
while remainder >= b:
    remainder = subtract(remainder, b)
    count = add(count, 1)
return count
```

For `17 // 5`, the states `(remainder, count)` are
`(17,0) -> (12,1) -> (7,2) -> (2,3)`, then stop and return 3.
Excluding negative operands removes signed-floor corrections. Zero divisors
must still be excluded. Up to 20 successful subtractions are needed for `20/1`;
a controller that counts its final halt decision as a step needs 21 steps.

## What the current model actually does

The code in `src/teenie/context_model.py` encodes the query once, then repeatedly
reads the same demonstration set and applies a shared GRU to a hidden vector.
It projects that vector to an answer only after the configured four updates.

There is no explicit callable ADD/SUB primitive, register write-back,
remainder/count interface, learned stop action, or choice of which register to
return. The earlier accurate single-pair arithmetic checkpoint is a different
network; the context trainer never loads or calls it. Hidden state could in
principle represent an algorithm, but this implementation does not enforce or
supervise such a representation.

`context_data.py` trains one-step arithmetic input/output relationships in
contexts. `context_train.py` supervises only the final answer at fixed depth;
it never labels individual iterations as arithmetic transitions, counts, or
termination decisions. This makes three-function recognition a possible shortcut,
though that precise internal strategy has not been established by these tests.

## Read-only measurements

Checkpoint: `checkpoints/context-long-seed0.best.pt`, trained for four updates.
Same fixed known-task and signed-division evaluation episodes as before:

| Inference updates | Known operations | Signed division |
|---|---:|---:|
| 1 | 3.75% | 3.57% |
| 2 | 3.37% | 5.71% |
| **4 (trained depth)** | **99.25%** | **9.52%** |
| 8 | 3.00% | 1.43% |
| 16 | 2.25% | 0.95% |
| 24 | 1.50% | 0.95% |

More iterations do not produce reliable longer computation for these weights.
This is a depth-extrapolation diagnostic, not a comparison against models trained
at those depths, and not proof that recurrent networks cannot learn algorithms.

Restricting division contexts and queries to `a in 0..10`, `b in 1..10` gives
13/110 = **11.82%** at the trained depth. The 99 demonstrations now cover most
of the smaller pool. Both signs and support density/composition change, so this
does not isolate a causal effect of removing negative numbers. It is **not**
a freshly trained `0..20` experiment. The current checkpoint rejects values above
10; setting the existing `--num-range 20` would mean `-20..20`, not `0..20`.

### Positive controls — the program was supplied by us

| Controller/primitive | Domain | Result |
|---|---|---:|
| Host-written loop + exact subtraction, cap 4 | a=0..20, b=1..20 | 386/420, 34 budget failures |
| Host-written loop + exact subtraction, cap 20 | same | 420/420 |
| Host-written loop + earlier learned numeric subtraction | a=0..10, b=1..10 | 110/110 |

The learned primitive is `checkpoints/numeric-long-seed0.best.pt`, invoked with
SUB at every iteration. The host provides comparison, primitive selection,
state update, counter, stop rule and return value. Therefore the perfect score
shows that the arithmetic composition works, **not that the network discovered
division from context**. An explicit four-subtraction budget would already cover
91.90% of `0..20` division cases; lack of a large budget alone is not an adequate
explanation for the much lower contextual score on the different original task.

No model was trained, no checkpoint was selected, and checkpoint SHA-256 hashes
were unchanged. Reproduce with:

```bash
uv run python experiments/investigate_loop.py --device cpu
```

Raw metrics and traces: `experiments/loop-investigation.json`.

## Implications for a next experiment (not implemented)

1. Use explicit operand bounds `0..20`; continue allowing negative subtraction
   outputs and multiplication results up to 400.
2. Introduce a generic stateful arithmetic executor: registers, reusable learned
   ADD/SUB primitives, comparisons, a controller, a counter and a halt action.
   A budget of 24 or 32 allows 20 arithmetic updates plus termination overhead.
3. Train primitive reuse and control flow rather than only final answers—for
   example multiplication via repeated addition and variable-length compositions
   using the permitted operations. Division examples and division traces must
   remain excluded from training.
4. Condition the controller/program selection on the 99 demonstrations. Test
   whether it discovers subtraction, the guard, counter and return register.
   A fixed division-specific loop supplied by us is an oracle baseline, not
   evidence of such discovery.
5. Measure primitive accuracy, transition traces, halting correctness, and
   held-out composition transfer separately. Retain shuffled/no-context controls.

This is a stronger architectural hypothesis, not a promise that three-operation
training will make arbitrary functions identifiable. Broadening the permitted
composition curriculum changes the training task and should be explicit.

**Conclusion:** the user's repeated-subtraction idea is mathematically sound.
The current implementation demonstrates fixed-depth latent refinement, not
learned arithmetic control flow. Changing the range is sensible but does not by
itself repair that mismatch.
