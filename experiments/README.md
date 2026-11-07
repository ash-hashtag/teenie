# Residual fitting experiment (v5)

## Scope and attribution

Fixed999 supports, float32 continuous bounded inputs, **1% relative tolerance,
zero absolute tolerance**. The active model now uses a designed CG residual
loop with trainable generic quadratic/log-RBF features. This is **not** the
previous learned-GRU mechanism. The untrained baseline is essential: most of
the improvement comes from the designed fitting prior, not outer training.

## Reproduce the CUDA pilot

```bash
uv run teenie --device cuda --updates 100 --batch-size 8 --queries 16 \
  --steps 256 --log-every 25 --validation-batches 4 \
  --save checkpoints/residual-pilot.pt
uv run python -m teenie.eval --device cuda \
  --load checkpoints/residual-pilot.best.pt --functions examples/functions.py \
  --count 512 --seed 40000 --output experiments/residual-pilot-basic-seed40000.json
```

Repeat evaluation with seeds40001 and40002. There is **one training seed**, not
three independently trained models. 100updates ×8contexts ×16queries =12,800
supervised queries; 800 contexts. 61 learned parameters;70features;256iterations.
Runtime44.6s on available CUDA device. Best selected at update100 by final-domain
basic validation loss. Gate and selection data are independent of test seeds.
Training range advances16→64 after update25,64→256 after update75.
The pilot used32validationqueries/operation; default is now128 to strengthen
the gate (`--validation-batches16` × batch8). No parameter updates at evaluation.

## Fresh bounded results

Each seed uses512fresh queries per relation; supports exclude each query.
Division uses a∈[0,256], b∈[1,256]; others both∈[0,256], uniform float32.

| Relation | Seed40000 | Seed40001 | Seed40002 | Pooled trained | Pooled untrained |
|---|---:|---:|---:|---:|---:|
| add |99.61%|99.80%|99.80%|99.74%|99.67%|
| sub |99.22%|99.41%|99.02%|99.22%|99.35%|
| mul |95.90%|93.36%|94.34%|94.53%|94.66%|
| div |98.05%|96.88%|97.27%|97.40%|97.66%|

**90% is exceeded on these random bounded tests; a learned accuracy advantage
is not established.** The fitted feature space already contains the first three
relations. Division is approximated from support data, without an explicit
reciprocal feature or operation identifier.

Seed40000 controls:

| Relation |1iteration|64iterations|256iterations|Nearest support|Shuffled labels|
|---|---:|---:|---:|---:|---:|
|add|8.01%|99.22%|99.61%|44.92%|1.56%|
|sub|2.54%|94.73%|99.22%|16.80%|0.59%|
|mul|1.37%|90.43%|95.90%|19.92%|0.59%|
|div|1.56%|91.21%|98.05%|21.48%|0.98%|

Zeroed labels:0% for all four. The 1/8/64/128-step probes differ from the trained
256-step budget. These controls show useful iterative context fitting, not
learned execution of arbitrary recursive programs.

## Failure boundaries and transfer provenance

Seed40000 explicit boundary probes: add18.75%,sub12.5%,mul6.25%,div25%.
Zeros/subnormals demand exact or extremely tight answers at atol0; support
sampling and query extrapolation make them much harder. Do not extrapolate the
random-test headline to boundary accuracy, exact arithmetic, or float32-wide
coverage. Nonfinite output representation is not mastery of those outputs.

The standalone design prototypes also inspected `a*a+b/a`; its later query
split is fresh but the expression is **not a blind architecture holdout**.
Other families and recursive relations are diagnostic transfer tasks, never
checkpoint-selection data. Their success does not prove that the model executes
the generating recursion. Results below are separate from the basic accuracy gate.

### Separate transfer results

Fresh seed41000,256queries/task; trained / untrained accuracy:

| Relation | Trained | Untrained |
|---|---:|---:|
| affine |99.61%|100.00%|
| quadratic |93.75%|95.31%|
| shifted product |95.31%|95.31%|
| shifted ratio |93.75%|94.14%|
| generated recurrence |91.80%|92.19%|
| `a*a+b/a` (development-exposed expression) |90.23%|89.84%|
| `(a*b+1)/(a+b+1)` |48.05%|48.44%|
| recurrence depth4 |91.41%|91.41%|
| recurrence depth8 |87.89%|87.89%|
| recurrence depth16 |94.92%|94.92%|

The cross-ratio failure and nonmonotonic recurrence-depth results rule out a
blanket generalization claim. Full-float32 stress (seed42000,128queries/task):
add7.81%,sub13.28%,mul/div/composite0%. This model remains bounded-domain only
in practical accuracy despite accepting the wider numeric interface.

## Artifacts

- `residual-pilot-train.jsonl`: losses, stage gates, fixed final-domain validation.
- `residual-pilot-basic-seed4000{0,1,2}.json`: complete paired controls/boundaries.
- `residual-pilot-transfer.json`: broader bounded suite,seed41000,count256.
- `residual-pilot-stress.json`: separate full-float32 suite,seed42000,count128.
- `checkpoints/residual-pilot{,.best}.pt`: local version-v5 weights (gitignored).
- `archive/model_v4.py.txt`, `archive/train_v4.py.txt`: previous active architecture.

---

# Historical v4 and earlier experiments

The material below describes older models, not the current residual fitter.

# Recurrent-memory curriculum experiments

## Protocol

All runs use **999 context examples**, scalar regression, hidden96, and a shared
attention/GRU loop. Training samples loop budgets4/8/16; default evaluation uses8.
Numeric support normalization and the architecture are identical across the three
pilots except for the number of persistent memory states. No operation labels,
expression trees or recursive execution traces enter the neural input.

Inputs progress through0..16,0..64,0..256; division denominators start at1.
Basic-only uses add/sub/mul/div. Diverse training mixes those with parameterized
affine, quadratic, shifted-product, safe-ratio and nonlinear recurrent families.
Recurrent data generation uses depths1..4. Held-out structures are `a*a+b/a` and
`(a*b+1)/(a+b+1)`; recursive depths8/16 are also withheld from training/selection.
The latter can approach a fixed point: these are **not proofs of obligatory
recursive execution** by the neural network.

Accuracy below means **1% relative error, zero absolute tolerance**. All tasks
have256 queries. Context/query pairs match across models, controls and loop
budgets. Shuffled labels, zero labels, an untrained network, nearest-example
retrieval and context mean provide controls. Nearest/mean are evaluation-only,
not alternate deployed models. Full float32 stress is separate.

## Matched short pilots

Seed0,1,000updates,batch8:8,000 supervised query episodes each. Compare **final**
`.pt` weights, not `.best`: basic-only and diverse runs have different validation
families, so comparing their validation-selected checkpoints would confound
selection. Query seed10000. Default8-loop accuracy:

| Training / memory | Add | Sub | Mul | Div | Held-out a²+b/a | Recurrence depth8 | Recurrence depth16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic / 1 state | 27.34% | 11.72% | 17.19% | 0.78% | 4.69% | 31.64% | 20.31% |
| Diverse / 1 state | 19.53% | 5.47% | 11.72% | 0.39% | 3.52% | 27.34% | 19.53% |
| Diverse / 4 states | 21.48% | 7.42% | 15.23% | 0.39% | 3.91% | 32.81% | 22.27% |
| Nearest context example | 36.33% | 15.23% | 17.97% | 17.19% | 12.11% | 43.36% | 54.30% |

Data diversity and four states **are not universal wins** at this budget.
Nearest-example retrieval is stronger here. More functions also reduce exposures
to each basic operation when the total query budget stays fixed.

Parameters:132,486 for1state;132,774 for4states. Observed training wall times were
61.3s/67.0s/64.7s respectively; these are single-run timings, **not** evidence that
four states are cheaper. Equal updates are not equal FLOPs; four states perform
more attention/recurrent work. Neither seed replication nor statistical
significance is established by this pilot.

### Do extra iterations help?

Diverse4-state model, identical queries:

| Task | 1 iteration | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| Addition | 14.84% | 19.14% | 21.48% | 21.88% |
| Held-out a²+b/a | 1.56% | 2.73% | 3.91% | 3.12% |
| Recurrence depth8 | 14.45% | 29.69% | 32.81% | 30.86% |
| Recurrence depth16 | 23.44% | 17.58% | 22.27% | 22.27% |

One iteration is outside the trained budget set. Results show some useful
iteration, but not monotonic improvement, not proof of learned recursion, and
not a compute-matched independently trained feed-forward ablation.

## Longer unchanged-architecture trial

After inspecting the pilots, extended training to3,000updates,batch16 (48,000
query episodes), keeping the same architecture. This is **6×** the query budget,
not an isolated update-count change. Best checkpoint chosen solely from known
training-family validation: update3,000. Runtime313.1s. No composite/deep-recursion
score selected the checkpoint. Fresh query seed30000; expression structures had
already been examined in the pilot, so this is not a new blind structural test.

| Task | Short M4, fresh queries | Longer M4 | Nearest example |
|---|---:|---:|---:|
| Add | 22.27% | 30.86% | 42.58% |
| Sub | 9.77% | 11.72% | 14.06% |
| Mul | 15.62% | 16.41% | 19.53% |
| Div | 1.17% | 1.56% | 16.41% |
| Held-out a²+b/a | 3.91% | 4.69% | 13.28% |
| Held-out (ab+1)/(a+b+1) | 11.72% | 12.11% | 27.73% |
| Recurrence depth8 | 31.25% | 32.03% | 43.75% |
| Recurrence depth16 | 25.39% | 30.08% | 62.89% |

Longer training improves several errors, but remains below nearest-example
retrieval on these tasks. **The99% objective is not met.** Increasing memory,
diversity and iteration count did not solve the precision/generalization problem.
Do not claim algorithm discovery from these results.

A small diagnostic on the short4-state model found mean within-example slot
standard deviation declining from0.041 at iteration1 to0.006 at iteration16
(4 basic-function episodes, seed333). States become more similar in this probe;
this is a possible bottleneck, not a demonstrated cause of the accuracy gap.

## Reproduce

```bash
# Three pilots: run each with the same remaining settings.
uv run teenie --device cuda --updates 1000 --batch-size 8 --hidden 96 \
  --memory-slots 1 --basic-only --log-every 250 --save checkpoints/curriculum-basic-m1.pt
uv run teenie --device cuda --updates 1000 --batch-size 8 --hidden 96 \
  --memory-slots 1 --log-every 250 --save checkpoints/curriculum-diverse-m1.pt
uv run teenie --device cuda --updates 1000 --batch-size 8 --hidden 96 \
  --memory-slots 4 --log-every 250 --save checkpoints/curriculum-diverse-m4.pt

# Repeat evaluation for each final pilot checkpoint.
uv run python -m teenie.eval --device cuda --load checkpoints/curriculum-diverse-m4.pt \
  --count 256 --output experiments/curriculum-diverse-m4-eval.json

uv run teenie --device cuda --updates 3000 --batch-size 16 --hidden 96 \
  --memory-slots 4 --log-every 500 --save checkpoints/curriculum-long-m4.pt
uv run python -m teenie.eval --device cuda --load checkpoints/curriculum-long-m4.best.pt \
  --seed 30000 --count 256 --output experiments/curriculum-long-m4-eval.json
uv run python -m teenie.eval --device cuda --load checkpoints/curriculum-long-m4.best.pt \
  --stress --seed 30000 --count 256 --output experiments/curriculum-long-m4-stress.json
```

Logs are `curriculum-*-train.jsonl`. All pilot reports and the matching fresh-query
short-run report are retained. Nearest/mean fields were added to the original
pilot reports by regenerating identical seeded episodes on CPU; trained metrics
were not modified. The evaluation CLI now emits these baselines directly.

## Historical experiments

`loop-*`, `float32-*`, and `fixed999-*` reports describe earlier architectures,
numeric objectives or context regimes. They remain for provenance. Do not join
them into a causal accuracy-improvement curve. Old weights are preserved but
are intentionally incompatible with the v4 working-memory model.
