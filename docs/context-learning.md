# Learning an operation from 99 demonstrations

This experiment tests—not assumes—whether a model trained on addition,
subtraction and multiplication can infer a new function from labeled examples.
It does not load the old single-pair model: the input contract and architecture
are different. Existing `teenie` commands and checkpoints remain unchanged.

Follow-up: [why the current internal loop does not implement repeated subtraction](loop-investigation.md).

## First measured result: division transfer did not succeed

One CUDA run, seed 0, 15,000 updates; selected checkpoint at update 13,500 using
only known-operation validation. Each evaluation episode has 99 demonstrations.

| Context | Known add/sub/mul (267 queries) | Unseen floor division (420 queries) |
|---|---:|---:|
| Correct answers | **99.25% (265/267)** | **9.52% (40/420)** |
| Shuffled answers | 22.10% | 8.57% |
| Zeroed answers | 4.49% | 6.19% |
| No demonstrations | 7.12% | 5.24% |

The context controls show reliance on demonstrations for the known tasks, but
**this run does not achieve general inference of division from examples**.
Learning three functions was not sufficient in this experiment. The result is
not a guarantee about every architecture or training run. No weights were
updated on division, and the division score was not used for checkpoint choice.
Do not describe this checkpoint as a working general-purpose operation learner.

Artifacts:
- `checkpoints/context-long-seed0.best.pt`: locally trained selected model.
- `experiments/context-train-long-seed0.jsonl`: known-task-only training log.
- `experiments/context-division-seed0.json`: full frozen evaluation, ablations
  and all 420 query predictions, using support-sampling seed 10000.
- `experiments/context-train-seed0.jsonl`: earlier 5k-update known-only run
  (87.27% validation); no division evaluation was used to decide the longer run.

To inspect the trained model immediately:

```bash
uv run python -m teenie.context_eval --load checkpoints/context-long-seed0.best.pt
```

## Run

```bash
# ONLY add/sub/mul. There is deliberately no --ops div training option.
uv run python -m teenie.context_train --device cuda --updates 15000 \
  --context-size 99 --num-range 10 --log-every 500 \
  --save checkpoints/context.pt

# Freeze the best known-operation checkpoint; test all 420 valid division queries
uv run python -m teenie.context_eval --device cuda \
  --load checkpoints/context.best.pt --output experiments/context-division.json

# One unseen query, with its 99 demonstrations included in the JSON report
uv run python -m teenie.context_eval --load checkpoints/context.best.pt \
  --query -7 2 --output experiments/division-query.json
```

`--device auto` is the default (CUDA when available). `--updates` counts optimizer
steps, not passes over a fixed dataset. Every step samples fresh episodes.
Training defaults: 64 episodes/batch, 99 examples/episode, hidden size 64, four
attention heads, four shared recurrent refinement steps, AdamW at 0.001 with
cosine decay to 0.00001. Numeric targets use MSE normalized by the result bound.
The model predicts an integer by rounding the rescaled scalar output.

## Episode contract

Each example supplies `(a, b, op[4], result)`. The query supplies only
`(a, b, op[4])`. Operands are signed integer tensors in `[-10,10]`, normalized
numerically inside the network—not tokenized into individual decimal digits.

Public model arguments:

| Tensor | Shape |
|---|---|
| support_pairs | `[batch, 99, 2]` |
| support_ops | `[batch, 99, 4]` one-hot |
| support_results | `[batch, 99]` |
| query_pairs | `[batch, 2]` |
| query_ops | `[batch, 4]` one-hot |

The query target is a separate loss/evaluation tensor; it is **not an argument
to the model**. The support set contains unique operand pairs and never contains
the query pair. Query identity can occur in other independent episodes; it is
hidden within its own episode.

Each training episode uses one actual operation, chosen from add/sub/mul. Its
categorical label is randomized among the first three slots, consistently for
the support set and query. The fourth slot is always zero during training.
Division evaluation consistently uses `[0,0,0,1]`.

Operation labels are used **only to match support examples to the query**, not
as learned semantic embeddings. This intentionally removes an untrained fourth
embedding and prevents a fixed opcode from revealing the function. Applying
the same permutation to all labels leaves the output unchanged. The generic
inference interface also accepts mixed-label contexts; attention masks out
examples that do not match the query's label.

## Architecture and limitations

A shared MLP encodes each `(a,b,result)` demonstration. There are no positional
embeddings: reordering the support set preserves the answer up to floating-point
roundoff. A query encoder and shared cross-attention/GRU cell refine the latent
query state four times, then a scalar head predicts its answer. There is no
hardcoded add/sub/mul/div calculation in the forward pass, no symbolic solver,
and no test-time parameter update. This variant uses fixed depth, not learned
adaptive halting.

Even with these safeguards, a network may merely recognize the three functions
seen during training rather than infer an arbitrary fourth function. High
known-operation validation accuracy is not evidence of division transfer.
No finite context uniquely determines every possible function on unseen inputs.

## Splits, checkpoint selection, and evaluation

- The 441 operand pairs are split into 352 training and 89 validation pairs.
  Training support and queries use only training pairs, excluding the query
  from its support. The same split applies to all three training operations.
- Known-operation validation queries all 89 held-out pairs for each operation
  (267 episodes). Support answers come **only from the training pool**. Fixed
  support sampling makes logging checkpoints comparable.
- The `.best.pt` checkpoint is selected only by known-operation validation
  exact accuracy, then lower MSE on ties. The normal path stores latest weights.
  Both files use a separate `teenie-context-v1` checkpoint format and record
  training configuration, operation whitelist, step and validation metrics.
- The division evaluator excludes `b=0`, yielding 420 possible queries. Each
  query gets 99 distinct pairs sampled without replacement from the other 419.
  Signed floor semantics apply: `-7 // 2 == -4` and `7 // -2 == -4`.
- Floor-division labels are generated only by the evaluation data builder.
  Model selection never reads them. Frozen weights consume support answers;
  the query answer is used only afterward for scoring.
- Metrics include exact rounded integer accuracy and raw numeric MAE/MSE.
  Reports compare correct context, independently shuffled support answers,
  zeroed support answers, and no context, for known operations and division.
  A context-sensitive model should react to ablations, but that alone does not
  establish that it inferred the correct new operation.

Only the 99 examples are visible to the model; it is not given the remaining
320 valid division pairs for that query. Repeated evaluations with different
`--seed` values change the support sample, not weights. Do not choose training
settings/checkpoints using division scores and then call division untouched.

## Supply an arbitrary context without a reference function

```bash
uv run python -m teenie.context_predict --load checkpoints/context.best.pt \
  --input my-context.json
```

`experiments/context-input-example.json` is a ready-to-run 99-demonstration
input with an unanswered query; use it with the saved
`checkpoints/context-long-seed0.best.pt`. The model's prediction can be wrong,
as shown by the division results above.

JSON schema (short example; supply 99 examples to match training):

```json
{
  "examples": [
    {"a": 7, "b": 2, "op": [0,0,0,1], "result": 3},
    {"a": -7, "b": 2, "op": [0,0,0,1], "result": -4}
  ],
  "query": {"a": 8, "b": 3, "op": [0,0,0,1]}
}
```

The predictor does not know the reference function or correct query answer.
It rejects supplied query answers, duplicate demonstrations and a query already
present among the matching-label demonstrations. It returns an integer prediction,
raw numeric prediction and context count. Other integer-valued functions can be
provided through the same interface, without a guarantee of success. Results
are bounded by the model's configured output range (default `[-100,100]`).

## Implementation

- `context_data.py`: known-operation-only sampling and held-out-pair splits.
- `context_model.py`: label-invariant set conditioning and recurrent attention.
- `context_checkpoint.py`: local portable context-model checkpoints.
- `context_train.py`: training and known-operation selection only.
- `context_eval.py`: frozen floor-division evaluation and context ablations.
- `context_predict.py`: inference from user-provided JSON, without an oracle.
- `tests/test_context.py`: leakage, reserved slot, floor semantics, invariances,
  context sensitivity, frozen evaluation, CLI and CPU/CUDA checkpoint checks.
