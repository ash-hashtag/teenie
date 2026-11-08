# Findings: bit-vector relation inference

Task throughout: 999 distinct examples `(a, b, c)` of a hidden relation on 8-bit unsigned integers
(results wrap modulo 256) plus new questions `(a, b)` never in the context. A question counts only if
**all 8 bits** of `c` are right. Weights are frozen when answering; no operation names are given.

Unless noted: one training seed (0); evaluation on fresh episodes, `--seed 60000`, 64 episodes × 16
questions = 1,024 questions per relation (≈ ±1.5 points at 70–90%). "Sure" = least certain bit ≥ 0.99.
Raw numbers are in `experiments/`; reproduce with the commands in each section.

## 1. Before bits: float32 regression (superseded)

Recurrent/curriculum models on float32 values with 1% tolerance reached add 1.2% → 30.9% (still below
copying the closest example). A designed conjugate-gradient fitter reached ~99% on add/sub/mul/div, but an
untrained copy scored the same: the solver did the work. See `experiments/README.md`.

## 2. Is 999 examples enough? (no learning)

A hand-written lookup walking bits from the lowest, keeping examples whose bits at `k` and at the previous
position match the question, is **exact** for add/sub/xor; Hamming-nearest example gets 2–5%. Looking further
back fails (999 examples do not cover the combinations). `experiments/bits-baselines.json`.

## 3. Bit-serial model (`--architecture serial`)

Shared GRU reads one bit position at a time; at each bit the question attends to examples in the same
situation; low-first and high-first passes weighted by self-checks on 64 examples. 447K parameters, 8 min.

- All 12 named trained relations 100%; random 1–4 state rules 74%.
- Never seen: a+b+1, nand 100%; 3a 76%, a−2b 71%, **min 32%** (splits trust 63/37 between directions).
- `experiments/bits-seed0-{train.jsonl,eval.json}`, docs site in `site/`.

## 4. All bits at once + loop on the answer (`--architecture loop`)

Each example → one vector from all 24 bits, fixed for the run; the question holds a guess of all 8 bits
and corrects it each round (8 rounds, same weights). 388K parameters, 7 min.

| | bit-serial | all at once |
|---|---|---|
| trained named relations | 100% | 99.2–100% |
| 3a / a−2b / min (never seen) | 76 / 71 / 32% | **92 / 89 / 46%** |
| sure-coverage on max | 0.6% | 89.6% |

Rounds matter: one round solves bitwise relations; carries need two; flat after four.
`experiments/bits-loop-seed0-*.json(l)`.

## 5. Multiplication (`experiments/mul/`)

E1 = `a*b` in training (`--train-set mul`); E2 = only pieces (`a*const`, `a*(b&3)`, `a*(b&15)`;
`--train-set blocks`), test `a*(b&63)` and `a*b`; E0 = `a*b` only (`--train-set mul-only`, 3,000 updates).
6,000 updates unless noted.

| model | E0 a*b | E1 a*b | E2 a*(b&15) trained | E2 a*(b&63) / a*b unseen |
|---|---|---|---|---|
| loop (plain) | 52.3% | 18.8% | 72.6% | 21.7 / 16.1% |
| loop `--multiplicative` | **98.1%** | **45.3%** | 81.8% | 22.2 / 16.5% |

- **Multiplicative units are the key.** Bilinear inputs (`linear(x) + left(x) * right(x)`) and gated
  (SwiGLU) feed-forward layers let the network multiply its own signals. Trained on `a*b` alone it reaches
  98% exact; without them it stalls near 52%.
- **In mixed training `a*b` is starved**: 1 of 14 relations. The other relations stay at 99.9–100% and
  never-seen 3a / a−2b rise to 98.1 / 97.2%. Fix under test: `--adaptive` sampling (draw relations in
  proportion to how often they are still wrong).
- **No length extrapolation yet.** Learning `a*(b&15)` does not transfer to `a*(b&63)` or `a*b`.
- **Confidently wrong on unseen multiplication**: E2 "sure" answers on unseen `a*b` were right only
  31–48% of the time (elsewhere ≈ 100%). Confidence is not trustworthy outside the training distribution.
- **More rounds at test time do not help** (`a*b` 43–45% for 4…24 rounds): the loop is not used as a
  step-by-step computation.
- **`--clock`** (fixed round-number signal, loss mostly on the last round), E1, 8 rounds: `a*b` 50.0%
  (vs 45.3%), others unchanged. Still not step-by-step: 4 rounds 53.3%, 8 → 51.2%, 12 → 42.6%,
  16 → 37.7% (round numbers past training are unfamiliar; even `add` drops to 98.2%).

Dead ends, kept for the record:

- **Context as memory** (`--architecture think`: one shared block rewrites all 999 + question tokens
  each round): 0.9 s/update, 28% on trained relations after 1,000 updates; stopped. Too slow for this GPU.
- **Program inference** (`--architecture program`: examples set the weights of a tiny calculator with
  products and a cosine bit readout): can represent `a*b`, `a+b`, `xor`, `max` exactly (checked on all
  65,536 pairs) but does not train — the cosine over values up to 65,025 makes gradients effectively
  random (loss rose 2.9 → 3.7). `experiments/mul/stopped/`.

## 6. Procedurally generated relations (`--train-set mix`)

Training adds random expression trees with 1–3 operations (`+ − × xor and or min max`, `<<1 >>1 not`,
constants) to the named relations and `a*b`. Eight **hidden** compositions are never generated —
checked by behaviour on fixed probe inputs, so rewrites are caught too: `a*b+a`, `a*a`, `(a+1)*(b+1)`,
`(a^b)*3`, `|a−b|`, `(a+b)>>1`, `(a*b)^b`, `(a<<1)+(b>>1)`. The old held-out relations are excluded the
same way. `deeper` tests fresh 5-operation expressions. Multiplicative loop, adaptive sampling,
10,000 updates.

Validation at update 6,500 (128 questions per relation; not yet the full 1,024-question evaluation):

- **Adaptive sampling fixes the starvation**: `a*b` 52% at update 2,000, 98% at 3,000, 100% at 6,500,
  while every named relation stays at 98–100%. Random expressions: 1 op 99%, 2 ops 86%, 3 ops 95%;
  random state machines 86%.
- **Old weak spots recover**: `min` 95% (≈45% in all earlier models), `a−2b` 98%, `3a` 99%.
- **Hidden compositions** (mean 48%): `(a+b)>>1` 92%, `(a^b)*3` 77%, `a*a` 67%, `|a−b|` 53%,
  `(a<<1)+(b>>1)` 45%, but multiplication combined with something else stays low: `a*b+a` 20%,
  `(a*b)^b` 16%, `(a+1)*(b+1)` 14%. Fresh 5-operation expressions: 70%.
- Sampling share at the end went mostly to state machines and 2-operation expressions.

The run was paused by request at 6,500 and continued with `--init` for 3,500 more updates at a lower
peak learning rate (3e-4) to reach 10,000 (`checkpoints/mix/mix-mloop-cont.pt`,
`experiments/mix/mix-mloop-cont-train.jsonl`).

**Full evaluation at 10,000 updates** (1,024 questions per relation, `experiments/mix/mix-mloop-eval.json`):

| group | relation | exact | bits | sure | right when sure | closest example | best lookup |
|---|---|---|---|---|---|---|---|
| trained | 12 named relations | 99.9–100% | 100% | 94–100% | 100% | 2–33% | 71–100% |
| trained | `a*b` | **99.6%** | 100% | 72% | 100% | 6.6% | 7.8% |
| trained | random state machines | 86.5% | 97.8% | 46% | 99.4% | 20% | 66% |
| trained | expressions, 1 / 2 / 3 ops | 94.5 / 83.7 / 77.6% | 95–99% | 49–87% | ≥99.6% | 25–30% | 55–84% |
| held out | a+b+1, nand | 100% | | | | | |
| held out | a−2b / min / 3a | 99.8 / **97.7** / 99.3% | | | 100% | | |
| hidden | `(a+b)>>1` | 92.0% | 98.6% | 9.9% | 100% | 3.8% | 35% |
| hidden | `(a^b)*3` | 82.4% | 97.2% | 7.0% | 100% | 4.7% | 97% |
| hidden | `a*a` | 70.9% | 95.8% | 0.4% | 100% | 41% | 18% |
| hidden | `\|a−b\|` | 60.9% | 88.5% | 7.0% | 95.8% | 3.2% | 35% |
| hidden | `(a<<1)+(b>>1)` | 49.7% | 89.9% | 0.1% | 100% | 4.6% | 5.4% |
| hidden | `(a*b)^b` / `a*b+a` / `(a+1)*(b+1)` | 15.4 / 13.6 / 14.7% | ~80% | 0% | — | 5–7% | 7% |
| deeper | fresh 5-op expressions | 68.7% | 92.7% | 30% | 99.0% | 26% | 47% |
| stress | `a*(b&63)` | 50.2% | 92.3% | 74% | **54.5%** | 9.3% | 8.0% |

Conclusions:

- **The multiplicative loop with adaptive sampling is the best model so far**: `a*b` 99.6% inside mixed
  training, every earlier relation at 98–100% (including `min`, 32–47% in all previous models).
- **Hidden compositions average 50%**, far above both baselines (closest example ≈ 9%, lookup ≈ 28%).
  The model reasons its way to new combinations of shifts, xor, add and squaring, but **multiplication
  combined with another operation is not solved** (14–15%).
- **Confidence is now honest on unseen relations**: on hidden compositions it is rarely "sure" (0–10%),
  and when it is, it is right 96–100% of the time — unlike the E2 multiplication runs. The exception is
  `a*(b&63)`, which it confidently mistakes for `a*b` (sure 74%, right 55%).

## 7. Why "a product, then another step" fails (`experiments/mix2/`)

Diagnosis of the `mix` model: `a*b+a` fails exactly like relatives training may produce (`a*b+b`, `a*b+5`,
`a*b−a`, `(a*b)^a`: 12–17% exact). The low 4 bits are right (0.3% wrong), the high 4 bits wrong 37% of
the time — the carries through the product's result. Only ~1% of training episodes contained such a
composition.

**More steps in the data (`--train-set mix2`)**: expressions with 1–5 operations, `*` twice as likely,
plus compositions of two multi-step pieces; 10,000 updates, multiplicative loop, adaptive sampling.

| | `mix` | `mix2`, 8 rounds | `mix2`, 12 rounds + `--clock` |
|---|---|---|---|
| hidden mean | 50.0% | 53.5% | 54.1% |
| `(a+b)>>1` / `a*a` / `(a<<1)+(b>>1)` | 92 / 71 / 50% | 98.6 / 80.3 / 67.6% | 99.5 / 79.8 / — |
| near group (`a*b+b`, `a*b+5`, `a*b−a`, `(a*b)^a`) | 12–17% | 21.5% | 20.1% |
| `a*b+a` | 13.6% | 19.0% | 21.6% |
| deeper (fresh 7-op) | — | 53.4% | 53.7% |

More multi-step data helps multi-step reasoning in general, but not product-then-step; extra rounds with a
round counter change nothing.

**Capacity check (`--train-set near-only`)**: trained on only the four near relations for 3,000 updates,
the model reaches just 38–40% on them (bits 90.6%), and 6.0% on the unseen `a*b+a` — whereas `a*b` alone
reached 98% in the same budget. **The limit is computation, not data or recognition**: the loop has no
good way to carry through an intermediate result. Next: give each round per-bit-position state with a
carry scan across positions, and scratch slots that are not scored as the answer.

## 8. Loops and the difficulty ladder (`--train-set mix3`, `experiments/mix3/`)

Goal: recognise patterns in increasingly complex functions, including loop-defined ones, at **≥50% exact on
unseen relations** with honest confidence. A tier is **won** at ≥50% mean exact with sure answers right
≥95% of the time (pooled).

New data (`src/teenie/bits/loops.py`): loops whose repeat count is n = b & 15 in training — accumulations
`acc = init; n×: acc = body(acc, a, i)`, repeated application `x = step(x)`, two-term recurrences — plus
bit loops (popcount, reverse, parity). Loops whose result barely depends on n are redrawn. Expression trees
now also hold out whole **combinations**: a xor of a product, a product of a difference, a min of a sum,
a shift of a xor (`HELD_PAIRS`).

Ladder (`src/teenie/bits/sets.py`, `TIERS`):

- T1 / T2 / T3: 2 / 3–4 / 5–7 operations, always containing a held-out combination.
- T4: new operation types never trained — `a//b`, `a%b`, `rotl(a,1)`, `rotl(a,b&7)`, three conditionals.
- T5: hidden loop bodies (sum of a..a+n−1, aⁿ, Fibonacci-like, x→3x+1, acc=(acc<<1)^a) at n ≤ 15, 63, 255;
  trained loop kinds at n ≤ 63 and 255 (longer than trained); hidden bit loops.

**Baseline — the `mix2` model, never trained on loops** (`baseline-mix2-mloop-eval.*`):

| tier | exact | sure right | |
|---|---|---|---|
| T1 | 58.8% | 100% | won |
| T2 | 44.6% | 99.1% | |
| T3 | 47.0% | 95.6% | |
| T4 | 69.9% | 100% | won |
| T5 | 40.3% | 98.8% | |

T1–T3 flatter this model (its training did not hold combinations out). T4 is already inferred well without
training: `a//b` 72%, `a%b` 59%, conditionals 66–93%, `rotl(a,1)` 100% (`rotl(a,b&7)` 23%). Hidden loops
split: `acc=(acc<<1)^a` (53–94%) and `aⁿ` (53–61%) collapse to simple patterns modulo 256; the sum of a
sequence (7–20%), Fibonacci-like (4–16%) and x→3x+1 (13–16%) do not. Random loops at longer lengths reach
52–77% partly because many settle into fixed points (closest example: 28–50%).

**B — the same architecture trained on `mix3`** (multiplicative loop, 8 rounds, adaptive sampling, 10,000 updates;
`mix3-mloop-*`): trained relations 87.7%, trained loop kinds 78–84%, bit loops 83%.

| tier | `mix2` model | B (`mix3`) |
|---|---|---|
| T1 | 58.8% won | 59.9% won |
| T2 | 44.6% | 43.0% |
| T3 | 47.0% | 45.0% |
| T4 | 69.9% won | 68.9% won |
| T5 | 40.3% | **46.4%** |
| hidden | 53.5% | 52.8% |

Loop training lifts hidden loop bodies at the trained length — sum of a..a+n−1 20 → 31%, Fibonacci-like 16 → 33%,
x→3x+1 16 → 22% — and the unseen bit loop `reverse(a^b)` reaches 93%. It does **not** carry over to longer loops
(the sum: 9% at counts ≤ 63, 5% at ≤ 255): the model has neither learned the loop as a procedure nor found the
closed form. T1–T3 hold steady although the combinations are now truly unseen.

Cost of breadth: with many more relation kinds sharing the same 10,000 updates, `a*b` falls to **53.9%** in B
(99.6% in the `mix` run). Named relations stay at 99.6–100%.

**S — B plus 4 scratch slots** (`--scratch 4`, 732K parameters; `mix3-mloop-scratch4-*`), same data and budget:

| | B | S |
|---|---|---|
| trained relations | 87.7% | 89.7% |
| `a*b` (trained) | 53.9% | **80.8%** |
| T1 / T2 / T3 | 59.9 / 43.0 / 45.0% | 58.9 / 43.7 / 44.0% |
| T4 | 68.9% | 71.9% |
| T5 | 46.4% | 48.1% |
| hidden | 52.8% | 53.4% |
| product, then a step | 21.5% | 19.2% |

Slots help where intermediate values matter — multiplication +27 points, Fibonacci-like loop 33 → 48%,
x→3x+1 22 → 30% — but not "product, then another step", and the tiers won stay at 2 of 5. One seed each:
likely, not yet confirmed. S becomes the site's current model.

## 9. Try it: your own functions (`uv run teenie-serve`)

A local relay (`src/teenie/bits/serve.py`) serves the built site and two endpoints: `/api/sample` generates
999 distinct examples from a user expression plus an unseen question and returns the model's answer after
every round; `/api/evaluate` answers 1,000 fresh questions over 10 different example sets (≈0.3 s on CPU).
Expressions are checked against a whitelist (arithmetic, bit operations, comparisons, conditionals,
min/max/abs, one `sum(... for i in range(n))` loop of ≤ 256 steps, popcount/reverse/rotl). With B: `a + b`
99.8%, `max(a,b) − min(a,b)` 67.5%, the sum loop 31.9%, `a*b + a` 24.9%.

## 10. What actually limits "product, then a step" (`experiments/arch/`)

Training data: only ~5% of `mix3` episodes contain a product followed by another operation (2–5 operation
expressions: 12 / 23 / 33 / 41% of draws). But the capacity check — training on **nothing but** `a*b+b`,
`a*b+5`, `a*b−a`, `(a*b)^a` for 3,000 updates — separates data from architecture:

| architecture | those four | `a*b+a` (never shown) |
|---|---|---|
| loop (467K) | 37.6–39.8% | 6.0% |
| loop + 4 scratch slots (732K) | 43.5–47.7% | 7.0% |
| **bit tokens** (666K; one token per answer bit) | **99.3–100%** | 12.3% (sure 90%, right 13%) |

Parameters were never the limit: carries have to move between bit positions through an intermediate result,
and a single question vector has no structure for that. Giving each answer bit its own token — attending to
the other bits and to the examples every round — solves it. The narrowly trained model forces the unseen
`a*b+a` into a known relation, confidently; generalisation needs broad training (next section).

The bit-token model is ~5× slower per update (8 tokens per question read the examples). Next: bit tokens
and the loop model both with step-by-step training (`--stepwise`: round t scored against intermediate
step t, plus a "done" signal), width 96, 64 questions per context, on `mix4` (adds a product-then-step
family), tested on the ladder and on loops 4–16× longer than trained.

## 11. Step-by-step training (`--stepwise`, `experiments/steps/`) — not helping as built

Every training relation carries its intermediate values (each operation of an expression, each iteration of a
loop); round t is scored against step t, then the final answer, and a "done" output learns when the steps end.
Both runs used width 96, 64 questions per context, 16 probes, 16 rounds, `mix4`, adaptive sampling; both were
stopped at 5,000 of 10,000 updates after flattening. Validation (128 questions per relation):

| update 5,000 | loop + steps | bit tokens + steps | B, no steps (width 128, `mix3`) |
|---|---|---|---|
| trained relations | 73% | 72% | 87% |
| hidden compositions | 31% | 26% | 45% |
| T1 / T2 / T3 | 47 / 42 / 26% | 33 / 48 / 29% | 55 / 46 / 44% |
| T4 / T5 | 52 / 35% | 47 / 33% | 64 / 46% |
| product, then a step | 14% | 13% | 20% |

Step supervision held both architectures back. Likely reason: the 999 examples only ever show the final `c`, so
in early rounds the model is scored on intermediate values it cannot see and would have to guess the
decomposition to match; most of the training effort goes there. (Runs differ in width and data too, so this is
not a clean ablation.) Next: bit tokens **without** step supervision on `mix4`, width 128, 8 rounds — the
configuration that reached 99–100% in the capacity check — with bf16 training (3.5× faster, same loss).

## 12. Bit tokens without step supervision — the new best model (`experiments/steps/bittoken-mix4-*`)

Bit tokens (width 128, 8 rounds, 666K parameters) trained on final answers only, `mix4`, 64 questions per
context, 16 probes, adaptive sampling, bf16, 10,000 updates (23 minutes). Full evaluation, 1,024 questions per
relation, against S (loop + 4 scratch slots, 732K, `mix3`):

| | S | bit tokens |
|---|---|---|
| trained relations | 89.7% | 88.1% |
| hidden compositions | 53.4% | **57.5%** |
| product, then a step | 19.2% | **30.7%** |
| `a*b+a` (hidden) / `(a*b)^b` (hidden) | 20.1 / 19.5% | **37.4 / 35.8%** |
| `a*b` | 80.8% | **91.8%** |
| T1 / T2 / T3 | 58.9 / 43.7 / 44.0% | **61.3 / 46.9 / 45.6%** |
| T4 | 71.9% | **72.7%** |
| T5 | **48.1%** | 46.0% |

Better on every non-loop measure with fewer parameters; tiers won stay 2 of 5 (T1, T4). Loops are slightly
worse (Fibonacci-like 48 → 36%, `reverse(a^b)` 84 → 25%). One seed, and the training data differs (`mix4`
adds the product-step family), so part of the gain may be data. Bit tokens become the site's current model.

## 13. Second seed and extra rounds (`experiments/bt/`)

**Seed 1** (same configuration as section 12): trained 89.2%, hidden 52.6%, product-then-step 23.4%,
`a*b+a` 20.9%, `a*b` 98.4%, T1–T5 59.2 / 43.8 / 45.1 / 72.3 / 46.1%. Across both seeds, bit tokens are
**clearly better than S at multiplication (92–98% vs 81%) and at product-then-step (23–31% vs 19%)** with fewer
parameters; on the ladder tiers they are **level** with S — the differences are within seed-to-seed spread.
Seed 0's lead on hidden compositions and `a*b+a` (37%) was partly luck. Validation during training (128
questions per relation) ran optimistic: seed 1 logged T5 50% at 8,000 updates but scores 46.1% on the full
evaluation.

**16 rounds at answer time** for the 8-round model: every tier within ±0.8 points — the model settles within its
trained rounds and does not use extra ones.

## 14. Answer-time improvements, no training (`teenie.bits.vote`, `experiments/bt/vote-seeds-eval.*`)

**Voting** — averaging the bit probabilities of the two bit-token seeds:

| | seed 0 | seed 1 | vote |
|---|---|---|---|
| hidden | 57.5% | 52.6% | **58.4%** |
| T1 / T2 / T3 | 61.3 / 46.9 / 45.6% | 59.2 / 43.8 / 45.1% | **61.5** / 46.7 / **46.7%** |
| T4 / T5 | 72.7 / 46.0% | 72.3 / 46.1% | **74.5 / 47.4%** |

At or above the better seed almost everywhere (+1–2 points), sure answers right 98.6–100%; costs only twice the
answer time.

**Fit check** — "sure" also requires reproducing ≥75% of the probe examples (each hidden from itself). It nearly
removes the confidently wrong answers on `a*(b&63)`, which the model mistakes for `a*b` (sure 24% at 75% right →
sure 0.5%), at the cost of being sure less often overall (mean coverage 16.4% → 11.3%).

## Guardrails against fitting our own benchmarks (from 2026-11-08)

Every test in sections 6–14 has been used to choose architectures and data, so repeated selection slowly tunes
choices to those tests even though no run trains on them. From here on:

- **Sealed set** (`src/teenie/bits/sealed.py`, `--tests sealed`): 24 everyday functions written independently —
  e.g. `(a+b)*(a-b)`, `(a*b)>>4`, `gcd(a,b)`, a sum of squares, `popcount(a)+popcount(b)`, division by repeated
  subtraction — several with operations the generator never produces. It is evaluated only at the end of a round of
  experiments and never used to make a decision.
- **Decide on broad averages**, not single relations; differences under ~5 points are within seed-to-seed spread.
- **No data families aimed at one failing test.** General coverage instead (`mix5`: every allowed operation pair
  equally often). The product-step family in `mix4` was motivated by one weak test group and is flagged as such.
- **Try it** (`uv run teenie-serve`): any function a person types remains the final check of generality.
