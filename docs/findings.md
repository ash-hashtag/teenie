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

**Longer training** — seed 0 continued for 10,000 more updates at a lower peak learning rate (3e-4), 20,000 in all
(`bt-continue`): trained 89.3%, hidden **62.3%** (57.5% at 10,000), product-then-step 34.1%, T1–T5 65.3 / 49.1 /
47.6 / 74.9 / 47.5%. Every measure rises together; T2 ends 0.9 points short of 50%.

**Fine-tuning with 16 rounds** (2,000 updates from the 10,000-update model, `bt-rounds16`): every tier within a
point of the base model (T5 46.5% vs 46.0%). Loops are not short of rounds; more rounds alone do not teach the
model to iterate.

## 15. Bit-aligned reading — the best result so far (`experiments/bt/bt-aligned-*`)

Same bit-token model, 10,000 updates on `mix4`, plus `--aligned`: every example is also split into one token per
bit position (its vector, the position, and its own bits there and one position below), and answer bit k reads only
the examples' bit-k tokens — the first model's per-position lookup, combined with bit tokens. +2,176 parameters
(668,409 in all). The run was interrupted by a restart at ~7,000 updates and resumed exactly from its saved state.

| | current (10k) | seed 1 (10k) | trained 20k | **aligned (10k)** |
|---|---|---|---|---|
| hidden | 57.5% | 52.6% | 62.3% | **64.2%** |
| product-then-step | 30.7% | 23.4% | 34.1% | **35.4%** |
| T1 / T2 / T3 | 61.3 / 46.9 / 45.6% | 59.2 / 43.8 / 45.1% | 65.3 / 49.1 / 47.6% | **77.0 / 55.1 / 54.3%** |
| T4 | 72.7% | 72.3% | 74.9% | **76.4%** |
| T5a / T5b | 39.7 / 49.2% | 41.3 / 48.5% | 41.8 / 50.4% | 38.5 / 51.4% |
| a*b | 91.8% | 98.4% | 96.2% | 99.5% |

At the same budget it beats both 10k seeds by **8–16 points on every held-out-combination tier** (T1–T3), well
outside the ~5-point seed spread, and beats the 20,000-update model with half the training. It is the first model
to clear the bar (≥50% with sure answers right ≥95%) on T1, T2, T3 and T4 together. Sure answers stay honest
(99–100% right on every test group). Unseen loop bodies (T5a) do not improve. One seed so far.

## 16. Every operation pair equally often (`mix5`, `experiments/bt/bt-pairs-*`)

`mix4` plus a family that picks every allowed (parent, child) operation pair uniformly — general coverage of
two-step shapes, including "a product, then another step", instead of a family aimed at one test. Bit tokens,
10,000 updates: hidden 58.7%, product-then-step 32.1%, T1–T5b 59.7 / 47.7 / 45.9 / 73.3 / 39.3 / 49.1%. **No
change** — every group within ~2 points of the same-budget `mix4` model, well inside seed spread. Two-step shapes
are already common enough in the data; more of them does not help.

## 17. Training loops up to 31 iterations (`mix6`, `experiments/bt/bt-loops31-*`)

`mix4` with the trained loop kinds run up to 31 times instead of 15 (63 and 255 stay longer than trained). Bit
tokens, 10,000 updates: hidden 56.1%, T1–T4 61.3 / 46.3 / 46.0 / 71.6%, **T5a 36.0%** (39.7% for `mix4`), **T5b
49.4%** (49.2%). **No help** for loops — longer training loops do not teach the model to run a step further.

**T5b mostly measures loops whose answer stops depending on the count.** Per relation (this run / `mix4`):
`acc=(acc<<1)^a` 81 / 85% at 63 and 91 / 94% at 255 — after 8 iterations every earlier bit has been shifted out,
so the answer no longer depends on the count; `a^n` 62–65% — powers of an even number reach 0 mod 256. The loops
whose answer keeps changing with the count stay near chance: sum of a..a+n−1 at 255 iterations 5.5%, `fib(a,a)`
5.6%. T5b's ~49% is therefore not evidence of iteration; the honest loop measures are T5a and these per-relation
scores.

## 18. 12-bit numbers (`experiments/bt/bt-width12-*`) — width is costly

Same bit-token model and budget as `mix4` (10,000 updates, 8 rounds, no bit alignment), on 12-bit numbers (12
answer-bit tokens; 999 examples now cover 0.006% of all (a, b) pairs instead of 1.5%):

| | 8-bit | 12-bit |
|---|---|---|
| trained | 89.6% | 76.2% |
| hidden | 57.5% | 15.0% |
| product-then-step | 30.7% | 2.1% |
| T1 / T2 / T3 | 61.3 / 46.9 / 45.6% | 27.7 / 28.2 / 24.3% |
| T4 | 72.7% | 62.3% |
| a*b | 91.8% | 14.1% |

Bitwise operations, addition and subtraction carry over unchanged (99–100%); **multiplication collapses** and so
does everything built on it. Confidence stays honest (sure answers 98–100% right). This fails the bar set for moving
to 16 bits (within ~10 points of 8-bit). A product's carries run across every bit, so wider numbers need more
sequential work per round or more rounds; simply widening the current model does not scale. The next attempt at
width should combine bit-aligned reading (section 15), mixed widths in training, and more or adaptive rounds.

## 19. Step-by-step hints that fade out (`--stepwise --hint-fade 5000`, `experiments/bt/bt-hintfade-*`)

Bit tokens on `mix4`, trained to produce one step of the computation per round (as in section 11), but the
step-by-step targets fade out linearly over the first 5,000 updates so the rounds are free to organise themselves
afterwards; answers are read at the first round that says it is done. 10,000 updates: trained 82.1%, hidden
38.0%, product-then-step 18.1%, T1–T5b 50.2 / 36.2 / 39.0 / 64.3 / 32.4 / 42.1%, a*b 71.4%. **Worse on every
group** — 7 to 20 points below the same-budget model trained on final answers only — and sure answers on
product-then-step are right only 87.5% of the time. Step supervision, whether kept (section 11) or faded, does
not help: intermediate values never appear in the 999 examples, so the model cannot check them there.

## 20. More math concepts (`mix7`, `experiments/bt/bt-mix7-*`)

`mix4` plus five families of 1–5 operation expressions that may also use division, remainder, comparisons, shifts
by a variable amount (`x<<(y&7)`, `x>>(y&7)`), squares and bit counts. Rotations and conditionals stay unseen;
every T4 relation and every sealed function is excluded by behaviour, so `a//b` itself is never drawn. Bit tokens,
10,000 updates.

- **The new concepts are learned:** generated expressions using them 95 / 76 / 71 / 66 / 59% exact at 1–5
  operations (the original families: 94% at 1, 82% at 3).
- **Nothing else is lost:** hidden 56.0%, T1–T5b 60.9 / 47.3 / 47.8 / 74.2 / 40.7 / 48.7% — all within ~2 points of
  `mix4` (57.5%, 61.3 / 46.9 / 45.6 / 72.7 / 39.7 / 49.2%). Product-then-step 27.9% (30.7%).
- `a//b` 79.9% (73.2% without the concepts), `a%b` 61.3% (60.1%): knowing the operation helps little with the exact
  unseen function. `a*b` 78.9% — single-relation scores swing widely between runs (92–98% for `mix4` seeds).

The broader vocabulary costs nothing at the same budget, so the final configuration uses it.

## 21. The sealed set, evaluated once (`experiments/sealed/`, seed 61000)

All candidates of this round, on the 24 everyday functions of `sealed.py` that were never used for a decision:

| model | sealed exact | sure answers right |
|---|---|---|
| bit tokens, `mix4` (current) | 56.0% | 98.2% |
| second seed | 54.6% | 97.5% |
| trained 20,000 updates | **58.1%** | 98.6% |
| bit-aligned | 55.7% | 99.6% |
| every pair equally often | 56.2% | 98.8% |
| loops up to 31 | 55.0% | 98.9% |
| more concepts (`mix7`) | 57.6% | **99.8%** |
| baselines (best of five per function, chosen after the fact) | 34.7% | — |

- **Every model clears the bar on functions it was never selected on** (≥50% exact, sure answers ≥95% right), and
  the two seeds agree within 1.5 points. This is the first evidence that the model is general rather than fitted
  to our benchmarks.
- **The ranking is flat.** Only longer training (+2.1) stands out; the other differences are within seed spread.
- **Bit-aligned reading's +8–16 points on T1–T3 do not carry over** (55.7% vs 56.0%). It gains on some functions
  (sum of squares 95% vs 70%, `((a+b)>>1)^b` 83% vs 69%) and loses on squares (`(a+b)^2` 29% vs 49%). Its large
  gain is specific to relations shaped like our generator's; its honest confidence (99.6%) does carry over.
- **`mix7`'s +1.6 comes mostly from one function that uses a trained concept**: `a>>(b&7)` 70.5% (46–57% for the
  others). The exact function was excluded from training, but variable shifts were not. Without it, `mix7` is level
  with the rest.
- Hard for every model: `(a*b)>>4` (0–2%, the high half of a product), `(a+b)*(a-b)`, `a*b-a-b`, `7a+3b`,
  `(a|1)*(b|1)` (19–31%), popcounts (25–34%).

## 22. A width-free model, trained on 8, 12 and 16 bits (`experiments/wide/wide-mix-*`)

From here on the headline is wide numbers: at 8 bits the 999 examples cover 1.5% of all (a, b) pairs and lookup
baselines answer 35% of the sealed set; at 32 bits they cover ~0.00000002% and a wrong guess is 1 in 4 billion.
Results are quoted next to the best of five lookup baselines at the same width.

**Model** (`wide_model.py`, 939,001 parameters at every width): every answer bit and every example bit sees the
numbers relative to its own position (±31 bits) and from the lowest bit (bits 0–31), with an "exists" channel,
through a bilinear layer computed as a 1-D convolution — no layer depends on the width, so the same weights answer
any width up to 32. Bit-aligned reading, 2 layers, 8 rounds. Trained 12,000 updates on `mix7`, cycling 8-, 12- and
16-bit episodes (0.25 s per update at 8 bits, 0.52 s at 16 on the RTX 3050).

| | 8 bits | 12 bits | 16 bits | 24 (untrained) | 32 (untrained) | lookup, 16 / 32 bits |
|---|---|---|---|---|---|---|
| target: `((a*b)^(a>>3))\|(b+9)` | 17.8% | 4.5% | 1.0% | 0% | 0% | 0.8 / 0.1% |
| target: T6, random 4–6 ops | 53.0% | 31.2% | 22.9% | 8.6% | 6.7% | ~10 / ~11% |
| T2 / T3 | 51.5 / 52.1% | 37.3 / 30.8% | 24.1 / 16.7% | 18.2 / 7.0% | 7.7 / 7.1% | 25 / 10%, 17 / 14% |
| T4 (unseen operations) | 79.0% | 73.3% | 70.8% | 20.0% | 4.3% | |
| product, then a step | 37.2% | 9.1% | 2.2% | 0% | 0% | |
| sealed | | | 43.5% | | | 24% (16 bits) |

Sure answers stay 98–100% right at every width.

- **One model for every width it was trained on.** At 8 bits it matches the best 8-bit-only model (hidden 65.8%
  vs 64.2%). Bit distances beyond 15 were never trained, so 24 and 32 bits are not free.
- **Multiplication on its own reached the top bit at 16 bits** suddenly, between 7,000 and 10,000 updates (a*b
  exact at 16 bits 7% → 73%); before that the low bits were right at every width and accuracy fell off at bits 5–7.
- **Composition is the block.** Per bit at 16 bits, `a*b` is 96–100% right on every bit, but `a*b+b` and
  `(a*b)^a` are right only up to bit 4 and coin flips from bit 6 up — the same frontier as before multiplication was
  learned. For a relation that is not one it has learned to recognise whole, the model reads the answer from the
  examples bit by bit, which only reaches low bits; it does not use its product as an intermediate value. At 8
  bits that fallback covers most bits, which hid the problem. The target needs exactly this: a product, then more
  steps.

## 23. Why composition fails at 16 bits: identifying, not computing (`experiments/wide/wide-ps16*`, `wide-near16`)

All three start from the stage-2 model (24/32-bit episodes added, `wide-32`), which computes `a*b` at 16 bits
(99.4%) but `a*b+b` at 0.1%. Fine-tuned at 16 bits only, peak learning rate 3e-4; scored on 64 episodes.

| | random product-then-step (fit) | `a*b+b`, `a*b-a`, `(a*b)^a` | unseen `a*b+a`, `(a*b)^b`, `(a+1)*(b+1)` |
|---|---|---|---|
| start | 19.9% | 0.1–1.2% | 0–0.6% |
| 3,000 updates on random product-then-step | 23.8% | 0.4–0.6% | 0–0.5% |
| same, questions also read deep probes (`--peers`) | 23.8% | 0.1–0.5% | 0–0.2% |
| 2,000 updates on four fixed relations only | — | **98.2–99.5%** | — |

- **Computing is not the problem.** Given only four candidates, the model computes product-then-step at 16 bits
  with its existing 8 rounds within 500 updates (98–100%).
- **Identifying is.** Drawn from a broad random family, it cannot tell from the 999 examples which
  product-then-step relation it faces, even when that family is all it trains on — so it is not a training budget
  problem either. `a*b` alone is recognisable because it is one relation seen often; conditionals and rotations
  leave local fingerprints in the examples.
- **Deep probes do not help** (`--peers`: questions read 16 probes worked through every round, with their true
  answers): identical results. Depth for checking examples is not enough without a way to search among candidates.
- The four-relation model is not general (held-out basic relations 2.9%, sure answers right 10.6%); it is only a
  diagnostic.

Next: search at answer time (Latent Program Network, Bonnet & Macfarlane): encode the examples into a program
vector, compute answers from it, and at answer time refine the vector by gradient descent until it reproduces the
examples, with the weights frozen.

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
