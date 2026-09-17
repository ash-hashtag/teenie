# Teenie: recursive arithmetic PoC

Learn relations from examples `(a, b, c)` and predict a missing result for a new
pair `(a, b)`. Use `uv` (Python 3.12+). The original recurrent arithmetic
experiments are retained below as legacy workflows.

**Current experiment:** [anonymous relation completion](docs/relation-completion.md).
Supply `(a,b,c)` examples and a query `(a,b,?)`, with no operation IDs, primitive
menu or program selection. The frozen model reaches 82.38–90.00% floor-division
accuracy with 99–419 examples on `0..20`, without division training.
The untrained same-architecture baseline reaches 94.76% at 419 examples, so this
is context-interpolation evidence, not a claim of learned general reasoning.

## Quick start: train the relation model

```bash
uv sync
uv run pytest
uv run python -m teenie.relation_train --device cuda \
  --operand-max 20 --context-size 99 --updates 1000 \
  --batch-size 16 --lr 0.01 --seed 0 \
  --save checkpoints/relation.pt
```

Training uses **addition, subtraction and multiplication only**, with operands
in `0..20`. Division and your custom relations are never included in this training.
This current model learns a continuous similarity kernel for context regression;
it is not the older recurrent MLP or a symbolic operation selector.

- `checkpoints/relation.pt`: latest saved weights.
- `checkpoints/relation.best.pt`: best validation checkpoint, selected only on
  add/sub/mul (accuracy, then MSE).
- `--updates` controls training length; `--log-every` controls validation/logging.
- Use `--device cpu` without CUDA, or `--device auto` for automatic selection.
- Use a different `--save` path for each experiment to avoid overwriting weights.

### Evaluate unseen floor division

```bash
uv run python -m teenie.relation_eval --device cuda \
  --load checkpoints/relation.best.pt --contexts 99 199 419 \
  --output experiments/my-division-results.json
```

This loads frozen weights and tests `a // b` (for example, `7 // 2 = 3`),
excluding zero divisors. Each query is absent from its own context. The report
includes shuffled/absent-context controls and an untrained-kernel baseline.

## Test your own relation

### 1. Define a function

Edit [`examples/custom_relation.py`](examples/custom_relation.py), or create your
own Python file:

```python
def f(a, b):
    return a // b  # Replace with your formula, e.g. (a - 7)**2 + 2*b


def valid(a, b):
    return b != 0  # Optional: exclude points where f is undefined
```

Remove `valid` (or return `True`) if your function is defined everywhere.
`f` must be deterministic and return one finite real number. Only load function
files you trust: generation executes ordinary Python code, not a sandbox.

### 2. Generate examples and a held-out query

```bash
uv run python -m teenie.relation_generate \
  --function examples/custom_relation.py \
  --context-size 99 --seed 0 --query 17 5 \
  --output experiments/my-input.json
```

This writes two files:

- `my-input.json`: unique `(a,b,c)` examples plus query `(17,5)`, without its answer.
- `my-input.answer.json`: the expected result for comparison; **do not feed it to
  the model**.

Omit `--query` for a seeded random pair. Change `--seed` to sample another context.
The query pair is always excluded from the examples. The default integer grid is
`0..20`; keep `--operand-max` consistent with your checkpoint. On this grid you
can use up to 419 examples for division, or 440 for a function defined everywhere.
These are inference-context limits, not training-context limits.

### 3. Load the model and predict

```bash
uv run python -m teenie.relation_eval --device cuda \
  --load checkpoints/relation.best.pt \
  --input experiments/my-input.json \
  --output experiments/my-prediction.json

cat experiments/my-input.answer.json
```

Compare `prediction` with the expected value; use `rounded_prediction` for integer
relations. The function and expected answer are not passed to the model, and its
weights are not updated. Any finite numeric relation can be supplied, but accurate
inference is not guaranteed, especially for discontinuous or random-looking rules.

The earlier `experiments/relation-input-example.json` is a **quadratic** example
(`c = (a - 7)² + 2*b`), not division. See
[relation-completion details](docs/relation-completion.md) for measured results
and limitations.

## Legacy single-pair arithmetic experiments

The commands below use the older `teenie` CLI, not the anonymous relation model.
Their checkpoints and input formats are not interchangeable. The earlier
[context-aware recurrent experiment](docs/context-learning.md) is also retained
separately.

```bash
uv sync
uv run pytest
uv run teenie --epochs 100
uv run teenie --epochs 50000 --log-every 100 > metrics.jsonl
uv run teenie --epochs 100 --no-loop
uv run teenie --epochs 100 --no-ponder
```

The runner automatically uses CUDA when available, otherwise CPU, and a fixed
seed by default. Force either with `--device cuda` or `--device cpu`; explicitly
requesting unavailable CUDA raises an error rather than silently falling back.
Each JSONL record contains `train`/`val` metrics for the selected operations and
per-operation accuracy and mean inference steps. `all_<op>` evaluates the full
grid for a trained operation (not a held-out metric); `zero_shot_<op>` is used
only for operations never trained in the checkpoint lineage. Operations trained
in an earlier run but disabled now are labelled `previously_trained_<op>`.
Long training is opt-in; no measured grokking or transfer results are claimed.

## Legacy: higher-accuracy bounded arithmetic

Measured on operands `[-10,10]` with all three operations, using fresh runs:

| Model / seed | Validation exact accuracy | Addition | Subtraction | Multiplication |
|---|---:|---:|---:|---:|
| Original embedding model / 0 | 42.64% | 44.94% | 51.76% | 31.87% |
| Numeric, best of 15k epochs / 0 | **99.25% (263/265)** | 100% | 100% | 97.80% |
| Numeric, best of 15k epochs / 1 | **99.62% (264/265)** | 100% | 100% | 98.81% |

Seed 0 preserves the original split for comparison. These are validation-selected
results, not independent test scores. Different representations, objectives and
training schedules were used: this is an accuracy comparison, not an isolated
causal ablation. Details, commands and remaining errors: [experiments](experiments/README.md).

Use the **numeric** model when accuracy on unseen pairs is the priority:

```bash
uv run teenie --device cuda --representation numeric --ops add,sub,mul \
  --hidden 64 --max-steps 4 --no-ponder --batch-size 2048 \
  --epochs 15000 --lr-schedule cosine --log-every 500 \
  --save checkpoints/numeric.pt
uv run teenie --load checkpoints/numeric.best.pt --eval-only
```

This model projects normalized signed operand values with learned linear layers,
uses a shared two-layer tanh recurrent cell, and predicts a scalar rather than a
class label. Training minimizes squared error normalized by the result bound;
inference rescales, rounds to the nearest integer, and clips to the valid output
range. No `a+b`, `a-b`, or `a*b` formulas are used in the model's forward pass.
Exact integer equality is still the accuracy metric, not a tolerance score.

The example deliberately uses **fixed four-step reasoning**: it does not claim
learned adaptive halting. `--no-ponder` avoids the early-stop incentive while
learning precise numeric predictions. Adaptive mode is still available, but its
`--ponder-weight` must be tuned to the much smaller normalized regression loss;
the classifier's default `0.01` is not a calibrated regression setting.

The original embedding classifier remains the default and old checkpoints load
unchanged. Representation is stored in each new checkpoint. Other training
controls: `--lr` (default `0.001`), `--weight-decay` (default `0.0001`), and
`--lr-schedule constant|cosine` (default `constant`). Cosine decays to 1% of the
initial learning rate over the requested epochs.

`train.per_op` and `val.per_op` report exact accuracy and example counts for each
operation. The saved `*.best.pt` is selected by highest combined validation
accuracy **at logging intervals**, keeping the earlier checkpoint on ties. The
regular `--save` path still contains the latest weights. Validation-selected
results are not independent test accuracy or proof of arbitrary-range reasoning.

## Legacy: operations and local checkpoints

```bash
# Addition only (default), using CUDA; save at logging intervals and final epoch
uv run teenie --device cuda --epochs 1000 --save checkpoints/add.pt
# Toggle operations by name or ID (0,1,2)
uv run teenie --device cuda --ops add,sub,mul --save checkpoints/all.pt
# Reload architecture, weights, operation selection, seed and halting settings
uv run teenie --load checkpoints/all.pt --eval-only
# Fine-tune loaded weights with a fresh optimizer; epochs count from 1 again
uv run teenie --load checkpoints/add.pt --ops add,sub --epochs 100 --save checkpoints/transfer.pt
```

Default output is `checkpoints/model.pt` (overwritten at each save). Checkpoints
contain CPU-portable state dictionaries and model/experiment metadata, loaded
with `weights_only=True`. They can move between CPU and CUDA. Architecture flags
cannot override a loaded model. Evaluation-only uses saved experiment settings
and never saves or updates weights. Optimizer/RNG state is not saved: loading
for training is **fine-tuning, not exact training resumption**. This is supervised
training, not in-context transfer. Keep separate paths to preserve experiments.
Changing the seed or operation set during fine-tuning changes the split and can
put previously trained examples in validation. Such validation is not an unseen
test of the checkpoint's full training history; use fresh runs for comparisons.
Each training run also writes a sibling `model.best.pt` (or your chosen filename
with `.best` before its extension). Both files belong to that run and may be
overwritten; use a new output path when comparing configurations.

Python inference:

```python
import torch
from teenie.checkpoint import load_checkpoint

model, metadata = load_checkpoint("checkpoints/all.pt", device="cuda")
a = torch.tensor([3], device="cuda")
b = torch.tensor([2], device="cuda")
op = torch.tensor([1], device="cuda")  # subtraction
result, steps = model.predict(a, b, op, adaptive=metadata["adaptive"])
print(result.item(), steps.item())  # predictions, not guaranteed correct
```

## Legacy model contract

- Inputs are matching `torch.long` vectors of **signed integers** in `[-N,N]`
  and operation IDs (`0=ADD`, `1=SUB`, `2=MUL`). Embedding offsets are internal.
- `make_dataset()` defaults to addition only and exhaustively enumerates pairs.
  A seeded 80/20 split gives 352/89 examples for `N=10`.
- The embedding model's output vocabulary covers `[-R,R]`, where `R=max(2N,N²)`, supporting all
  three operations without clipping. Targets remain signed integers until loss.
  This reserves classes unused by addition-only training.
- `forward()` always returns all-step logits `[batch, steps, classes]` (or
  normalized scalar predictions `[batch, steps, 1]` in numeric mode) and
  conditional halt probabilities `[batch, steps]` for differentiable training.
- `predict()` returns `(integer_results, steps)` and stops each example at its
  first halt probability greater than 0.5, forcing a stop at the budget. The
  answer is the emission at that step, not a mixture of earlier answers.
- Adaptive training minimizes expected per-step task loss plus `ponder_weight * E[steps]`.
  Task loss is cross entropy for embeddings or normalized squared error for numeric mode.
  Survival-weighted halt probabilities sum to one; the final step absorbs the
  remainder. Penalizing the sum of halt probabilities instead would discourage
  stopping. This is an expected-step objective, not a PonderNet reproduction.
- `--no-loop` sets one step. `--no-ponder` disables adaptive halting and trains
  only the final emission, then always uses the full step budget at inference.

## Legacy interpretation and limits

An operation ID unseen during training has no learned subtraction semantics.
Addition examples alone do not identify what `op=1` should mean. Recurrence and
halting provide capacity and control flow, not proof of algorithmic reasoning.
Subtraction accuracy or extra steps alone would not establish algorithmic reuse;
compare seeds, baselines, and suitable held-out regimes before drawing conclusions.
Both modes enforce the configured integer input range; numeric mode does not
claim extrapolation beyond it.

The original single-pair MLPs cannot accept few-shot examples. The separate
`teenie.context_train` / `teenie.context_eval` experiment implements demonstration
conditioning and frozen division evaluation; it does not fine-tune on division.
Successful generalization to arbitrary new functions is not guaranteed.
Plotting notebooks and long-run experiments are follow-up work.
No synthetic success curves are presented.

## Files

- `src/teenie/relation_*.py`: current anonymous relation training, generation,
  inference and local checkpoints.
- `examples/custom_relation.py`: editable custom function for JSON generation.

- `src/teenie/data.py`: operations, exhaustive datasets, reproducible splits.
- `src/teenie/model.py`: shared cell, output heads, independent adaptive halting.
- `src/teenie/loop.py`: normalized halting distribution and training objective.
- `src/teenie/train.py`: training CLI and JSONL evaluation metrics.
- `src/teenie/checkpoint.py`: portable local weight/config save and load.
- `src/teenie/context_*.py`: independent context-aware training, checkpoints,
  division evaluation, and arbitrary labeled-context inference.
- `experiments/`: measured baseline, numeric run logs and reproducibility notes.
- `tests/test_model.py`: arithmetic bounds, splits, gradients, halt behavior,
  and a ten-example overfit check.
