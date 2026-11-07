# Teenie: context-driven arithmetic fitting

Given **exactly 999 `(a,b,c)` examples** and a new `(a,b)`, predict `c`.
No operation IDs, formula-selection menu, query labels, or inference-time weight
updates. Custom Python functions generate datasets; JSON input is unnecessary.

**The current model is a trainable numerical fitter, not a neural network that
has discovered an arithmetic algorithm.** Its explicit iterative solver is a
strong architectural prior. See the untrained comparison below.

## Train with uv and CUDA

```bash
uv sync
uv run teenie --device cuda --updates 100 --batch-size 8 \
  --queries 16 --steps 256 --log-every 25 --save checkpoints/residual.pt
```

- Default training relations: **add, subtract, multiply, divide**.
- Every episode has 999 unique supports and 16 distinct held-out queries. One
  context fit serves all queries; query answers only enter the training loss.
- Uniform float32 inputs progress `[0,16] → [0,64] → [0,256]`. Division's
  denominator starts at **1**, not zero.
- A stage advances after two consecutive validation checks reach **90% for
  every basic operation**, at 1% relative tolerance and zero absolute tolerance.
  This is a small validation gate, not statistical proof or a fixed-time schedule.
  Training may end before reaching the final range; logs record the actual range.
- Defaults: 256 solver iterations, 16 radial-basis knots, 16 queries/context.
  `--steps`, `--knots`, `--queries`, `--lr`, `--seed`, and `--log-every` are tunable.
  Context size **cannot** change. Use `--device cpu` for small tests.
- `residual.pt` is latest; `residual.best.pt` minimizes fixed **basic-relation
  validation loss at the final range**, independently of the curriculum gate.
  Unseen relations never select checkpoints. Optimizer-state resume is not supported.

The loss combines stable relative-aware regression and output-status prediction.
Its denominator has a scale-relative floor near zero; evaluation does **not**
relax the requested tolerance.

## What loops?

1. Normalize inputs and finite outputs using **support-only** statistics.
2. Encode inputs in a generic quadratic and log-coordinate radial-basis feature
   map (70 features with the default 16 knots).
3. Start a temporary coefficient vector at zero.
4. Repeatedly correct the coefficient state using **conjugate gradient** on the
   regularized support least-squares objective.
5. Evaluate the fitted function at the query.

No direct matrix-inverse/solve shortcut, operation dispatch, or query target is
used. The shared numerical update is **designed**, not learned. Outer training
adjusts basis centers, widths, regularization and a status head: **61 parameters**
by default. Solver arithmetic is float64; the public input/output contract is
float32. `return_steps=True` exposes intermediate predictions and normalized
support residuals. More iterations are not guaranteed to improve every query.

This replaces the previous GRU experiment, which remained far below the target.
It is a substantial change in inductive bias—not evidence that a tiny recurrent
neural network learned general recursive reasoning. In particular, the quadratic
basis already represents addition, subtraction and multiplication.

## Custom relations

Create a trusted Python file, for example `my_functions.py`:

```python
import numpy as np
from teenie import FunctionTask


def f(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a + 2*b)


TASKS = [FunctionTask("custom", f, (0., 256.), (0., 256.), "uniform")]
```

```bash
uv run teenie --device cuda --functions my_functions.py \
  --updates 100 --save checkpoints/custom.pt
```

The callable creates labels, never model inputs. A `--functions` file replaces
all default tasks **and** the range curriculum, using its declared ranges.
Files execute Python: load trusted code only. You can also call
`train(tasks, device="cuda", queries=16, save="checkpoints/custom.pt")` from
`teenie.train`. Additional relations are accepted, not guaranteed learnable.

## Load and predict

```python
from teenie import EpisodeSampler, load_model
from teenie.data import load_tasks

model, metadata = load_model("checkpoints/residual.best.pt", device="cuda")
episode = EpisodeSampler(load_tasks("examples/composite.py"), seed=42).sample(1)
answer = model.predict(episode.context[0].tolist(), episode.query[0].tolist())
print(answer, "expected:", episode.target.item())
```

Or pass your own 999 triples and one query to `predict`. It rejects a query
already present in the context. Tensor API:

```python
prediction = model.forward(context, query)  # [B,999,3], [B,2] or [B,Q,2]
values = prediction.values()                # [B] or [B,Q]
```

`EpisodeSampler(tasks).sample_queries(batch_size=8, queries=16)` generates
shared-context training episodes. All supports and held-out query pairs are
distinct after float32 rounding. Tensor callers must maintain this separation.
Inference changes only temporary coefficients, never saved parameters.

## Evaluate

```bash
# Basics only, fresh contexts, all controls:
uv run python -m teenie.eval --device cuda \
  --load checkpoints/residual.best.pt --functions examples/functions.py \
  --count 512 --seed 40000 --output experiments/basic-eval.json

# Broader relations and recurrence-depth probes:
uv run python -m teenie.eval --device cuda \
  --load checkpoints/residual.best.pt --count 256 \
  --output experiments/transfer-eval.json

# Separate full-float32 stress suite:
uv run python -m teenie.eval --device cuda \
  --load checkpoints/residual.best.pt --stress --count 128 \
  --output experiments/stress-eval.json
```

Reports include correct/shuffled/zeroed support outputs, untrained same-model,
nearest-example and mean baselines; 1/8/64/128/256 iterations; and explicit
boundary probes. Default accuracy means **relative error ≤1%, absolute
tolerance=0**. A zero answer must be exact. Tolerances are configurable explicitly.
The transfer suite includes `a*a+b/a`, other families and recursive relations.
Good approximation of their outputs does not prove execution of their recursion.

## Measured status

CUDA pilot: 100 updates, batch 8, 16 queries/context, 256 iterations, training
seed 0. Fresh evaluation: **1,536 queries per operation**, pooled across query
seeds 40000/40001/40002 (512 each; one training seed).

| Relation | Trained | Untrained same fitter |
|---|---:|---:|
| Add | 99.74% | 99.67% |
| Subtract | 99.22% | 99.35% |
| Multiply | 94.53% | 94.66% |
| Divide | 97.40% | 97.66% |

**These results primarily come from the numerical fitting prior, not training.**
On seed 40000, shuffling support labels reduces accuracy to ≤1.56%; zeroing them
gives 0%.
This establishes support dependence, not arbitrary-function generalization.

**Important limitation:** explicit boundary probes score only 6.25–25% in this
run. Zeros, subnormals and extrapolation remain weak. Random bounded accuracy
must not be interpreted as 90% accuracy everywhere, exact arithmetic, or
full-float32 mastery. See [experiment details and replications](experiments/README.md).

## Numeric contract and limitations

Inputs: finite nonnegative float32 values, `0..3.4028234663852886e38`.
Outputs: signed finite values, `+inf`, `-inf`, `NaN` (undefined).
NumPy float32 operations determine labels, including intermediate overflow and
rounding. Unexpected Python exceptions propagate instead of becoming NaN.

A separate status head represents nonfinite answers; the bounded training data
**does not teach** these cases. Wide-range and special-value performance are
separate stress results, not guaranteed capabilities. Query coordinates beyond
16 times their support-derived input scale are clipped internally; unrestricted
extrapolation is not supported. Finite contexts cannot uniquely identify an
arbitrary function.

## Validation and compatibility

```bash
uv run mypy
uv run pytest
```

Strict typing covers package/examples. Tests cover IEEE labels, fixed context,
query separation, recurrent residuals, gradients, shared fits, frozen inference,
checkpoint round trips and CUDA.

Current checkpoint format: `teenie-residual-v5-fixed999`. Older weights are
preserved but rejected, not silently converted. Previous model/train source is
in `experiments/archive/*_v4.py.txt`; historical results remain in `experiments/`.
Only the residual fitter is active in the package.

## Bit-vector track (`teenie.bits`)

A separate prototype. `a`, `b` and `c` are unsigned 8-bit integers given as
bits, results wrap modulo 256, and the score is **exact match on all 8 bits**.
The setup is still 999 examples plus questions that never appear among them.

```bash
uv run teenie-bits --device cuda --updates 3000 --save checkpoints/bits.pt \
  --log experiments/bits-seed0-train.jsonl
uv run teenie-bits-eval --load checkpoints/bits.pt --device cuda \
  --episodes 64 --seed 60000 --output experiments/bits-seed0-eval.json
```

How the model works: one shared GRU reads every example one bit position at a
time. At position `k` the question attends to the examples' states at the same
position and reads their `c_k`. Its own predicted bits feed back into its
state, so the GRU has to learn what to carry forward (a carry, a borrow, "a is
already larger"). One pass reads low bit first and another high bit first.
Each pass re-predicts 64 of the examples with that example hidden from itself,
and the pass that explains them better is trusted more. Weights never change
at inference, and there are no operation IDs.

Training draws from add, sub, b−a, xor/and/or, a+2b, 2a+b, max, a<<1, a+k,
b^k and random small state machines that read one bit position at a time.
`a+b+1`, `a−2b`, `min`, `nand` and `3a` are never trained. `a*b` is a stress test.

Seed 0, 3000 updates (~8 min on an RTX 3050), 1024 fresh questions per relation:

| relation | model exact | Hamming nearest | best hand-written window lookup |
|---|---|---|---|
| all 12 named training relations | **100%** | 2–33% | 71–100% |
| random state machines (trained family, new machines) | 74% | 20% | 66% |
| held out: a+b+1, nand | **100%** | 2–28% | 100% |
| held out: 3a / a−2b | 76% / 71% | 28% / 4% | 96.5% / 95% |
| held out: min | 32% | 22% | 72% |
| stress: a*b | 6% | 7% | 8% |

A visual walkthrough (model diagram, replays of real predictions, training replay, benchmarks)
lives in `site/` — `cd site && pnpm install && pnpm dev`. See `site/README.md`.

When the least certain bit is at ≥0.99, every answer in this evaluation was
correct. How often the model is that sure varies: 0.6% of the time for `max`,
even though it is 100% exact there, so the trust between passes is not sharp
enough yet. The model generalises within its training patterns, but not yet to
`min` or other new carry patterns.
