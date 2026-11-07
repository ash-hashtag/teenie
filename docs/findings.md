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
  step-by-step computation. `--clock` (round number signal + loss mostly on the last round) is under test.

Dead ends, kept for the record:

- **Context as memory** (`--architecture think`: one shared block rewrites all 999 + question tokens
  each round): 0.9 s/update, 28% on trained relations after 1,000 updates; stopped. Too slow for this GPU.
- **Program inference** (`--architecture program`: examples set the weights of a tiny calculator with
  products and a cosine bit readout): can represent `a*b`, `a+b`, `xor`, `max` exactly (checked on all
  65,536 pairs) but does not train — the cosine over values up to 65,025 makes gradients effectively
  random (loss rose 2.9 → 3.7). `experiments/mul/stopped/`.

## 6. Procedurally generated relations (`--train-set mix`) — in progress

Training adds random expression trees with 1–3 operations (`+ − × xor and or min max`, `<<1 >>1 not`,
constants) to the named relations and `a*b`. Eight **hidden** compositions are never generated —
checked by behaviour on fixed probe inputs, so rewrites are caught too: `a*b+a`, `a*a`, `(a+1)*(b+1)`,
`(a^b)*3`, `|a−b|`, `(a+b)>>1`, `(a*b)^b`, `(a<<1)+(b>>1)`. The old held-out relations are excluded the
same way. `deeper` tests fresh 5-operation expressions. Multiplicative loop, adaptive sampling,
10,000 updates.
