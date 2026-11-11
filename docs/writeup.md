# Teenie: inferring hidden integer rules from examples, in one pass

*Draft, 2026-11-11. Numbers are for the current release model (the mixed run, update 85,000) unless stated.*

## The task

Teenie is shown 999 examples `(a, b, c)` of a hidden rule `c = f(a, b)` on 16- or 32-bit unsigned integers, then new
questions `(a, b)`. It answers every bit of every question in a single forward pass, with frozen weights and no search
at answer time, and says how sure it is of each bit. It is never told the rule, and the rules it is scored on are held
out of training.

At 32 bits a blind guess is right once in four billion tries, and 999 examples cover a vanishing fraction of the
2^64 possible inputs, so a right answer has to come from inferring something about the rule, not from copying.

## The model

A width-free transformer of 1,145,339 parameters: one token per answer bit, the same weights at every width up to
32 bits. Each answer bit reads the examples bit by bit with attention, the whole answer is refined over 12 rounds,
each question keeps two scratch registers it writes and reads back every round, and it reads 64 worked examples
alongside. Nothing about any operation is built in. It was trained on a 4 GB laptop GPU.

## How we score it

Exact match (all 32 bits right) hides most of the progress: a guess that gets 28 of 32 bits right with honest
confidence is far better than chance, but scores zero. We use **bits recovered**:

> width + log2 of the probability the model gives the true answer

0 is a blind guess, 32 is certain and right, and a confident mistake costs more than an honest "don't know". 10 bits
means the true answer is about 1,000 times more likely under the model than by chance.

We give the same score to two kinds of baselines, on exactly the same episodes:

- **Lookups** that understand nothing: the closest example, or matching one or two neighbouring bits.
- **Models fitted to each task's own 999 examples:** gradient-boosted trees (one per answer bit) and a small neural
  network, calibrated on held-back examples so overconfidence does not cost them.

## The middle band

Rules people can see at a glance (addition, xor) are not the point; bitwise and additive chains collapse to a tiny
per-bit truth table or carry automaton, and lookups solve them too. The interesting rules have a definite rule but no
textbook way to work it out from examples. We built a held-out set of 37: random 6-10 step expressions, deep
expressions with constants, a random S-box inside arithmetic, data-dependent steps (Collatz-style, branches inside
loops), cellular automata (rules 30 and 110), rotation mixes, add-rotate-xor rounds, loops, and products as a floor.
Hashes and 6-8 mixing rounds are a control: nothing should beat chance there, and nothing does.

Bits recovered at 32 bits (of 32):

| held-out group | Teenie | best lookup | trees fitted per task | small net fitted per task |
|---|---|---|---|---|
| Random 6-10 step expressions | **19.9** | 7.7 | 18.5 | 12.9 |
| Products (floor) | **11.7** | 6.0 | 3.8 | 0.4 |
| Cellular automata | **9.1** | 0.7 | 2.9 | 0.0 |
| Deep expressions with constants | **8.7** | 1.6 | 2.0 | 0.0 |
| A random S-box inside arithmetic | **7.8** | 0.5 | 1.2 | 0.0 |
| Data-dependent steps | **6.4** | 1.5 | 1.5 | 0.1 |
| Loops | **3.8** | 1.0 | 2.4 | 0.2 |
| Rotation mixes | 3.6 | 2.6 | **8.6** | 0.0 |
| Add-rotate-xor, 1-4 rounds | 0.0 | 0.0 | 0.5 | 0.0 |
| Control: hashes | 0.0 | 0.0 | 0.0 | 0.0 |

Teenie leads wherever the rule needs real computation, and does it in one pass where the fitted baselines train 32
classifiers per task. It loses on rotations, an operation it had never trained on, which a tree reads directly.
Individual examples: rule 30 for two steps 12.2 bits (trees 0.7), `3*(a^b)` 21.0 (1.2), `a+b` 32 (2.4), `a*b` about 30
(3.3).

Beyond the middle band, at 32 bits: `a*b` is 89% exactly right, never-trained `(a*b)+a` 85%, chains of one or two
operations without products 99% and 96%, and when the model is sure, it is right.

## What it cannot do yet

**Multiply values it computed.** It multiplies its inputs, but `(a+b)*a`, `a*a-b*b` or `(a-b)^2` get only their lowest
six or so bits right, at 16 and 32 bits. Bit k of a product depends on every lower bit of both operands; the
multiplication the model learned reads the input tokens and does not take its own intermediate values as operands.
This is the frontier we are working on, without handing the model intermediate values.

**Rotations and real mixing.** Rotations are new to it; even one add-rotate-xor round is beyond it.

**Answers wrap at every step**, like 32-bit registers. For addition, multiplication and bitwise operations this changes
nothing; for rules that compare or shift an overflowed intermediate it makes the rule stranger than the mathematics.

## How it got here

Most of the gains came from changing the structure and the data, not from more training of the same thing: one token
per answer bit (width-free), refining the answer over rounds, scratch registers (the first change that let it compute
through a product), 32-bit episodes, and 12 rounds instead of 8 (which moved the correct bits of a 32-bit product from
22 to 29). Training only on product chains quietly eroded loops and multi-concept rules; mixing the general set back
in fixed it. Everything, including what did not work, is in [the findings](https://teenie.lupyd.com/findings/).

## Where this fits

Teenie belongs with models that learn to infer a function from examples in context: Garg et al. (2022) and looped
transformers (Yang et al., 2024) on real-valued function classes, Boolean in-context learning (Bhattamishra et al.,
2024), and above all TabPFN (Hollmann et al., Nature 2025), which pretrains on synthetic tasks and predicts in one
pass. What we have not seen elsewhere: arbitrary composed integer and bitwise rules at 32 bits from about a thousand
examples, a width-free bit-level model, bits recovered against lookups and fitted models scored the same way, and a
held-out set of rules with no textbook solution. We have not yet compared against TabPFN.

## Try it

[teenie.lupyd.com/try](https://teenie.lupyd.com/try/) runs the model in your browser: type any rule of `a` and `b`,
and it reads 999 generated examples and answers 16 new questions, including your own.
