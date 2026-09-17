# Anonymous relation completion — current direction

The input is a set of examples from **one consistent relation**:

```text
(a1, b1, c1)
(a2, b2, c2)
...
query: (a, b, ?)
```

The model predicts the missing `c`. It receives no operation ID and does not
select an arithmetic primitive, named function, or program. Unlike the earlier
single-pair classifier, its output is a real value, not a fixed integer class.

## Run the trained model on your context

```bash
uv run python -m teenie.relation_eval \
  --load checkpoints/relation-seed0.best.pt \
  --input experiments/relation-input-example.json
```

JSON schema (use as many distinct examples as appropriate):

```json
{
  "examples": [
    {"a": 1, "b": 2, "c": 9},
    {"a": 3, "b": 4, "c": 17}
  ],
  "query": {"a": 5, "b": 6}
}
```

The two-example snippet illustrates the schema, not enough evidence to identify
a unique arbitrary relation. The supplied example file contains 99 examples
from `(a-7)^2 + 2*b`, a relation **not included in training**. Its formula and
query answer are not supplied to the predictor. The example's separate result
file is evaluation metadata only.

Inputs `a,b` are bounded to `0..20` by the saved model; continuous values within
those bounds are accepted. Context values `c` may be finite real numbers and
are not clipped to an arithmetic result vocabulary. Outputs include both the
raw prediction and its rounded integer. A query answer, operation ID, duplicate
context pair, or query already present in context is rejected.

## Actual implementation

This is a **learned-kernel context-regression baseline**:

1. Learn two continuous input-similarity lengthscales and a regularization
   strength using add/sub/mul training episodes.
2. At inference, form a smooth similarity matrix over the supplied `(a,b)` pairs.
3. Solve for temporary context-specific regression coefficients using the `c`
   observations, then predict the query through its similarity to the context.

The coefficients are intermediate values, not checkpoint parameters updated
with test-time gradient descent. The model uses one continuous regression path
for every relation; there is no `if operation == ...` branch in the model.
Numerical linear algebra and an RBF smoothness prior are explicit inductive
biases. This is **not** the proposed symbolic register executor, nor a claim of
learned recurrent control flow. The current baseline does not have a reasoning
loop; adding nominal iterations to an exact solve would not demonstrate reasoning.

No division formula, inverse-multiplication target, or missing-operand exercise
appears in training. The internal relation index used by the training generator
never enters the model. Pending permission for broader curricula, training remains
strictly add/sub/mul; only `c` is withheld from each query.

## Measured results

One CUDA run, seed 0, 1,000 optimizer updates. Training has 99 examples per episode
and 16 episodes per update. The checkpoint is selected only on held-out
add/sub/mul validation (265/267 exact = **99.25%**).

For division, all `a=0..20`, `b=1..20` pairs are queried: 420 episodes. Each gets
distinct demonstrations from other valid pairs; **its own answer never enters
its context**. At 419 examples, every other valid pair is supplied. Evaluation
support seed is 10000, with no gradient updates or division-based model selection.

| Context size | Trained, correct context | Untrained kernel, same context | Shuffled c values | Zero/no context |
|---:|---:|---:|---:|---:|
| 99 | **346/420 = 82.38%** | 75.48% | 19.52% | 50.00% |
| 199 | **372/420 = 88.57%** | 85.71% | 23.10% | 50.00% |
| 419 | **378/420 = 90.00%** | **398/420 = 94.76%** | 23.10% | 50.00% |

Accuracy uses exact equality after rounding the raw output, not an error
tolerance. MAE/MSE are also reported on the raw predictions. The 50% zero baseline
matters: half of these nonnegative floor-division answers are zero. Context
helps well beyond that baseline, but 90% is not perfect or evidence of universal
relation inference. In particular, it does not establish discovery of a division
algorithm or repeated subtraction.

The untrained baseline uses the identical regression architecture and initial
lengthscales/regularization, with the same contexts. **It beats the trained model
at 419 examples.** Meta-training helps at 99 and 199 examples but is not a
consistent transfer improvement. Much of the performance comes from context
interpolation and the smoothness prior, not demonstrated learned reasoning.
Both baselines are reported; the checkpoint was not changed based on these scores.

These results use a different operand domain and model than the earlier signed
attention experiment. They are not an isolated causal comparison against its
9.52% score. No architecture/hyperparameters were changed in response to this
division sweep. Additional relation families should be evaluated prospectively.

## Reproduce training and evaluation

```bash
uv run python -m teenie.relation_train --device cuda --operand-max 20 \
  --context-size 99 --updates 1000 --save checkpoints/relation-seed0.pt \
  > experiments/relation-train-seed0.jsonl

uv run python -m teenie.relation_eval --device cuda \
  --load checkpoints/relation-seed0.best.pt --contexts 99 199 419 \
  --output experiments/relation-eval-seed0.json
```

The 441 possible input pairs are split into 352 training and 89 validation pairs.
Known-operation validation covers each held-out pair for all three relations;
its support is training-only. Division evaluation is a separate unseen relation
whose labeled support is provided at inference, as requested. A given query can
occur as a demonstration in another independent episode, never its own.

Increasing context requires more memory and solve time (the implementation uses
double precision for stability). Context size 419 is the maximum for the 420-pair
division grid. Arbitrary JSON contexts are constrained by available memory and
the need for distinct pairs, not a fixed model input length.

## Files and status

- `src/teenie/relation_model.py`: operator-free continuous regression, checkpoint
  loading, arbitrary JSON relation prediction.
- `relation_data.py`, `relation_train.py`: strict three-relation meta-training.
- `relation_eval.py`: frozen division sweep and context controls.
- `tests/test_relations.py`: leakage, continuous values, context permutation,
  checkpoint preservation, CLI and CUDA checks.
- `experiments/relation-train-seed0.jsonl`, `relation-eval-seed0.json`: evidence.
- `experiments/relation-input-example.json`: 99 unlabeled-operation triples and
  one unanswered query for a different relation.
- `checkpoints/relation-seed0.best.pt`: local trained weights (gitignored).

The `program_*` files/checkpoint are **abandoned operation-menu experiment
artifacts**, not this solution. The user rejected that architecture before its
end-to-end program-induction benchmark completed; no program-search success is
claimed. Earlier models and checkpoints are retained unchanged for comparison.

## Generate inputs from your own function

Edit `examples/custom_relation.py` to define a deterministic `f(a, b)` returning
one finite real number. The supplied example is floor division. An optional
`valid(a, b)` predicate excludes undefined points (division excludes `b == 0`).
Other exceptions are not silently skipped. Only run function files you trust:
the generator executes them as ordinary Python code.

```bash
uv run python -m teenie.relation_generate \
  --function examples/custom_relation.py --context-size 99 --seed 0 \
  --query 17 5 --output experiments/my-input.json

uv run python -m teenie.relation_eval --device cuda \
  --load checkpoints/relation-seed0.best.pt --input experiments/my-input.json

cat experiments/my-input.answer.json
```

The answer file is separate and must not be supplied to the model. Context pairs
are unique and never include the query. Omit `--query` for a seeded random query.
Inputs default to the integer grid 0..20; match `--operand-max` to your checkpoint.
Use up to 419 examples for division, or 440 for a function valid on all 441 pairs.
Real-valued targets are supported; compare `prediction`, not `rounded_prediction`,
for noninteger answers. This generates evaluation data only; it does not retrain
or modify weights. Arbitrary functions can be supplied, but accurate inference
is not guaranteed (especially discontinuous or random-looking relations).
