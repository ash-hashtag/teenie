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

## 24. Deep example processing does not help either; relations are learned one by one (`wide-ps16-deep`)

**Example layers** (`--example-layers 4`, 1.6M parameters): before the questions read them, each example's bit
tokens attend to each other and pass through a gated layer, four times, so an example can be worked through
(e.g. its own a*b) instead of read through one shallow layer. Added to the stage-2 model with zero outputs, then
trained on random product-then-step relations at 16 bits like section 23. The layers trained (their weights moved
well off zero), but the loss tracked the plain run to within 0.001 at every check, and every product-then-step
relation stayed at 0–1% through 3,000 updates (checked every 500 updates, 16 episodes each). Stopped at the
halfway point.

**Why nothing has helped.** `a*b+b` is itself in the random product-then-step family: 0.35% of its episodes
(`a*b-a` 0.25%), about 170 episodes over a run — and the model still scores 0% on it. Given four relations at 25%
of episodes each, it learns all four in 500 updates. The model learns these relations one at a time, each needing
on the order of a thousand episodes, rather than one way to compose "product, then a step" that transfers to
relations seen rarely or never. More training, deep probes and deep examples change none of that.

## 25. Can it combine parts it knows? A composition grid (`experiments/wide/wide-grid-*`, `wide-pool16-*`)

**Pool of 16.** Training on a fixed pool of 16 random product-then-step relations (2,000 updates at 16 bits): the
pool itself only reached 35.9%, fresh relations from the same family stayed at the starting 18.5%, unseen ones at 0.
Stopped there; it repeats section 24.

**The grid.** 8 product-like first steps (`a*b`, `a*a`, `b*b`, `(a+1)*b`, `a*(b>>1)`, `(a^b)*a`, `(a|b)*b`,
`(a-b)*(a+b)`) × 8 second steps (`+a`, `+b`, `^a`, `^b`, `-a`, `-b`, `|b`, `>>1`). 52 cells trained, each about 900
episodes over 6,000 updates at 16 bits from the stage-2 model; 12 held out so that every first and every second step
is still trained in other cells, all checked to match no trained cell. (A diagnostic: real relations are not limited
to a grid like this.)

- **Trained:** 9 of 52 cells reached 50% or more — nearly all on the two products the model already handled, `a*b`
  and `(a+1)*b`, with several second steps each (87–97%). The other first steps barely started.
- **Held out, where both parts are learned:** `(a*b)+a` **7.7%** — while `a*b` with `+b`, `^a`, `^b`, `|b` is
  87–96% and `+a` after `(a+1)*b` is 88%. `((a+1)*b)^b` **10.1%** with its row and column learned. The rest 0–15%.
- **Reading:** a little transfer, far below the cells themselves. The model memorises combinations rather than
  composing known parts. Next: separate guessing the relation from applying it (section 26).

## 26. Guess the relation first, then apply it: a relation-code bottleneck (`experiments/wide/wide-grid-code*`)

`--code-slots 8 --code-layers 2`: 8 code tokens per bit position read that bit of every example, share across bit
positions and among themselves (2 layers), and the answer bits read only these code tokens, never the examples —
no direct lookup of similar examples, so an answer can only come from what the code holds about the relation. One
forward pass, nothing fitted at answer time, and the code is free vectors, not a choice among operations. Added to
the stage-2 model (its weights reused, the code layers new), trained on the same grid as section 25:

| 6,000 updates at 16 bits | plain | bottleneck |
|---|---|---|
| trained cells at ≥50% | 9 of 52 | **11 of 52** |
| mean on trained cells | 18.0% | **23.6%** |
| mean on held-out cells | 3.9% | 1.4% |
| time per update | 0.58 s | **0.30 s** |

The bottleneck learns trained combinations faster, in half the time (answer bits read 8 code tokens instead of
999 examples). Neither combines known parts yet — but both learned only about a fifth of the grid, so most held-out
cells have parts that are learned in few combinations. A longer run (4,000 more updates) showed why: learning stalled at about 11 cells,
nearly all of two complete rows:

| first step | trained second steps |
|---|---|
| `a*b` | 95 96 96 93 96 35 |
| `(a+1)*b` | 96 97 94 91 97 92 |
| `a*(b>>1)` | 27 29 30 26 24 41 1 |
| `a*a`, `b*b`, `(a\|b)*b` | 5–29 |
| `(a^b)*a`, `(a-b)*(a+b)` | 0–11 |

**In the two learned rows the held-out cells stay near zero**: `(a*b)+a` 6% and `((a+1)*b)^b` 1%, although `a*b` works
with five other second steps and `+a` and `^b` work after the other product. Even where both parts are known
thoroughly, the model does not put them together: it memorises each combination as its own relation, with or
without the bottleneck. With 52 combinations, memorising each is the cheapest solution. Next: data with far too
many combinations to memorise while every part stays frequent (section 27).

## 27. Multi-step chains work at 32 bits — except through a product (`chain_split.py`, `experiments/wide/chains-split-wide-32.json`)

**Chains**: a start (`a*b`, `a+b`, `a-b`, `a^b`, `a&b`, `a|b`, `a*a`, `b*b`, `min`, `max`) followed by one or two steps, each
one of 21 kinds (`+ - ^ | & *` with `a`, `b` or a fresh random constant; `>>1`, `<<1`, not). About one in seven
(start, first step) pairs never appear together in chain training — including `a*b` then `+a` — and are the test.

**A new measure, beyond lookup** (`teenie-bits-eval`, `beyond_lookup`): the model's exact rate on the questions that
every lookup baseline gets wrong, so what it gets there cannot come from copying examples. On the never-trained
conditional `a>b ? a-b : a+b` at 16 bits, lookup fails 62.5% of questions and the model gets 90% of those.

**The stage-2 model, never trained on chains**, on held-out pairs at 16 bits (48 episodes per family):

| held-out chains | exact | right where lookup fails |
|---|---|---|
| no product, 1 step | 99.0% | 96.7% of 152 |
| no product, 2 steps | 93.1% | 84.3% of 121 |
| with a product, 1 step | 10.7% | 9.3% of 257 |
| with a product, 2 steps | 11.0% | 7.9% of 302 |

**Chains without a product work; chains through a product fail.** Chains of additions, subtractions, bitwise
operations, shifts and constants are answered almost perfectly at 16 bits, mostly on questions lookup cannot answer.
A caution on "held out": this model never trained on chains, but it trained on random expressions of up to 5
operations over the same operations, which can include the same shapes with other constants; the relations here are
new (fresh random constants), the shapes not necessarily. What is clear: **a product as an intermediate step**
fails, although products followed by a step were in its training data — everything sections 22–26 called a
composition failure was this one case.

**At 32 bits, with confidence** (`chains-split-wide-32-w32.json`):

| chains | exact | right where lookup fails | sure | right when sure |
|---|---|---|---|---|
| no product, 1 step | 91.8% | 86.0% of 136 | 56.2% | 100% |
| no product, 2 steps | 83.0% | 66.5% of 188 | 51.0% | 100% |
| with a product, 1 step | 0.0% | 0.0% of 415 | 0.0% | — |
| with a product, 2 steps | 0.0% | 0.0% of 366 | 0.0% | — |

Where it answers, it knows it: about half its answers on chains without a product are sure, and every sure answer
was right; through a product it is never sure, so it never claims a wrong answer.

## 28. Training on chains: products start to compose (`experiments/wide/wide-chains-*`)

The stage-2 model trained 8,000 updates at 16 bits on chains (section 27), with the held-out (start, first step)
pairs never shown; a third of the starts are products and `*` is one of the steps. Held-out chain pairs, 16 bits:

| held-out chains | before | after | right where lookup fails | sure (right when sure) |
|---|---|---|---|---|
| no product, 1 step | 99.0% | 100.0% | 100.0% | 98.8% (100%) |
| no product, 2 steps | 93.1% | 97.0% | 92.6% | 86.1% (100%) |
| through a product, 1 step | 10.7% | 21.7% | 19.1% | 0.4% (100%) |
| through a product, 2 steps | 11.0% | 15.5% | 11.9% | 0.6% (100%) |

Chains without a product become nearly perfect, and the model is sure of almost all of them. **Chains through a
product roughly double** — the first movement on composing through a product, on pairs never trained together —
but stay far below; the model is almost never sure of them, and right when it is. The fixed checks `(a*b)+a` and
`(a*b)^b` stayed at 0%, so the gain is on other product chains.

## 29. Checking worked examples, on chains (`experiments/wide/wide-chains-peers-*`)

Same chain training as section 28, plus deep checking: 64 examples per episode are worked through every round like
questions, and the questions read them with their true answers (`--peers --probes 64`), so a guess such as "a
product, then a step" can be checked against examples the model has worked through. Held-out pairs, 16 bits:

| held-out chains | plain | with checking | sure, with checking (right when sure) |
|---|---|---|---|
| no product, 1 step | 100.0% | 100.0% | 98.6% (100%) |
| no product, 2 steps | 97.0% | 97.7% | 84.5% (100%) |
| through a product, 1 step | 21.7% | 22.8% | 5.1% (100%) |
| through a product, 2 steps | 15.5% | 17.0% | 1.8% (100%) |

One to one and a half points on product chains — within noise. Checking worked examples is not what products are
missing. It does make the model sure more often on product chains it gets right (5.1% of questions instead of 0.4%),
and those sure answers stayed right.

## 30. Product-heavy chains (`experiments/wide/wide-chains-products-*`)

Continuing the chain-trained model (section 28) for 8,000 updates at 16 bits on `chains-products`: half general
chains, half chains from ten product starts (`a*b`, `a*a`, `b*b`, `(a+1)*b`, `a*(b+1)`, `(a^b)*b`, `(a|b)*a`,
`(a-b)*b`, `(a+b)*(a-b)`, `(a>>1)*b`) — adaptive sampling settled at 62% product chains. Held-out pairs follow one
rule across all chain families (`chain_held`), so no pair held out of one family is trained in another.

| held-out | 16 bits | 32 bits |
|---|---|---|
| no product 1 step | 100.0% (sure 98%, right 100%) | 95.2% (sure 77%, right 100%) |
| no product 2 steps | 96.5% (sure 85%, right 100%) | 84.5% (sure 61%, right 100%) |
| product 1 step | 25.4% (sure 4%, right 100%) | 0.0% (sure 0%) |
| product 2 steps | 17.6% (sure 0%, right 100%) | 0.3% (sure 0%) |
| product start 1 step | 2.9% (sure 0%) | 0.0% (sure 0%) |
| product start 2 steps | 7.8% (sure 0%, right 100%) | 2.1% (sure 2%, right 100%) |

- Product chains moved only a little at 16 bits (21.7 → 25.4% for one step, 15.5 → 17.6% for two) and stay at zero
  at 32 bits; held-out chains from the new product starts are 3–8%. **The product chains it trains on most stayed
  at 11–14% for all 8,000 updates** — data alone teaches products far too slowly.
- Why it stops where it does: on `(a*b)+a` the model gets bits 0–4 right and is at chance from bit 6. Without a
  recognised rule it matches examples on the input bits that matter; a product's bit k depends on 2(k+1) input bits,
  i.e. 4^(k+1) patterns, and 999 examples (about 2^10) run out of matches around bit 4–5 at any width. `a*b` alone
  is right at every bit because a learned product computation handles it; that computation does not switch on once
  another step follows.
- Chains without a product kept improving: 95.2% / 84.5% at 32 bits, sure on 77% / 61% of them and right 99.6%.

Next: room for intermediate values that examples and questions can match on — if the model keeps `a*b` in a
scratch register, `(a*b)+a` becomes as local as addition at every bit.

## 31. Scratch registers: computing through a product (`experiments/wide/wide-chains-registers-*`)

The product-heavy model (section 30) with two scratch registers per question and per worked example (numbers it
writes and reads back every round) and 64 worked examples the questions read (`--registers 2 --peers --probes 64`),
8,000 more updates on product-heavy chains at 16 bits. Nothing about products is built in.

| held-out | before (16 bits) | with registers, 16 bits | with registers, 32 bits |
|---|---|---|---|
| no product 1 step | 100.0% | 100.0% (sure 98%, right 100%) | 93.2% (sure 64%, right 100%) |
| no product 2 steps | 96.5% | 96.8% (sure 84%, right 100%) | 82.2% (sure 49%, right 100%) |
| product 1 step | 25.4% | 31.6% (sure 15%, right 100%) | 0.0% (sure 0%) |
| product 2 steps | 17.6% | 22.0% (sure 2%, right 100%) | 0.0% (sure 0%) |
| product start 1 step | 2.9% | 5.2% (sure 0%) | 0.0% (sure 0%) |
| product start 2 steps | 7.8% | 9.0% (sure 0%, right 100%) | 2.2% (sure 2%, right 100%) |

- **Product chains improved across the board at 16 bits** — 25.4 → 31.6% for one step, 17.6 → 22.0% for two — and the
  model is now sure of 15% of the one-step answers, all of them right.
- **It computes through the product, not just matches:** on the never-trained `(a*b)+a`, bits 7–15 rose from 61–73% to
  74–84% (50% is guessing; matching examples cannot get past bit 5), and its exact rate climbed through the run
  (4 → 27% in the checks). `(a*b)^b` did not move: xor after a product is not learned yet.
- At 32 bits products stay at zero; these runs train at 16 bits only, and even `a*b` alone was never learned at 32 bits.

Since chain training began, held-out one-step product chains at 16 bits went 10.7 → 21.7 → 25.4 → 31.6%.

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
