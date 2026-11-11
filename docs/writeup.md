# Teenie: inferring hidden integer rules from examples, in one pass

*November 2026. Numbers are for the release model (the mixed run, update 85,000) unless we say otherwise.*

## The task

We show Teenie 999 examples `(a, b, c)` of a hidden rule `c = f(a, b)` on 16- or 32-bit unsigned integers and then ask
it new questions `(a, b)`. It answers every bit of every question in one forward pass, with frozen weights and no search
at answer time, and it reports how sure it is of each bit. It never sees the rule, and none of the rules we score it on
appear in training.

At 32 bits a blind guess lands once in four billion tries, while 999 examples cover a vanishing slice of the 2^64
possible inputs. A right answer therefore has to come from inferring something about the rule; there's nothing to copy.

## The model

It's a width-free transformer with 1,145,339 parameters and one token per answer bit, so the same weights work at any
width up to 32 bits. Each answer bit reads the examples bit by bit with attention, and the whole answer gets refined
over 12 rounds. Every question also keeps two scratch registers that it writes and reads back each round, and 64 worked
examples run alongside the questions. We didn't build in any operation. All of the training ran on a 4 GB laptop GPU.

## How we score it

Exact match (all 32 bits right) hides most of the progress. A guess that gets 28 of 32 bits right, and knows which four
it's unsure of, is far better than chance, yet it scores zero. So we use **bits recovered**:

> width + log2 of the probability the model gives the true answer

A blind guess scores 0 and certain-and-right scores 32. Confident mistakes cost more than an honest "don't know", and 10
bits means the model thinks the true answer is about 1,000 times likelier than chance would.

Two kinds of baseline get the same score on exactly the same episodes. The first are lookups that understand nothing:
copy the closest example, or match one or two neighbouring bits across the examples. The second are models we fit to
each task's own 999 examples, namely gradient-boosted trees (one per answer bit) and a small neural network, both
calibrated on held-back examples so that overconfidence doesn't cost them.

## The middle band

Rules you can see at a glance, like addition or xor, aren't the interesting case. Any chain of bitwise and additive
steps collapses into a tiny per-bit truth table or a small carry automaton, and the lookups solve those too. What we
care about are rules that are perfectly definite but have no textbook way of working them out from examples, so we built
a held-out set of 37: random expressions of 6 to 10 steps, deep expressions with constants, a random S-box inside
arithmetic, Collatz-style loops and branches inside loops, cellular automata (rules 30 and 110), rotation mixes,
add-rotate-xor rounds, loops, and products as a floor. Hashes and long mixing act as the control. Nothing should beat
chance on them, and nothing does.

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

Teenie leads wherever the rule needs real computation, and it gets there in one pass while each fitted baseline trains
32 classifiers per task. Rotations are where it loses. It never trained on them, and a tree can read a rotated bit
straight off the input. A few single rules show the gap: rule 30 run for two steps gets 12.2 bits against the trees'
0.7, `3*(a^b)` gets 21.0 against 1.2, and plain `a+b` gets the full 32 against 2.4.

Outside the middle band, at 32 bits, `a*b` comes out exactly right 89% of the time and the never-trained `(a*b)+a` 85%.
Chains of one or two steps without a product reach 99% and 96%. And when the model says it's sure, it's right.

## What it can't do yet

It can't **multiply values it computed itself**. `a*b` works, but `(a+b)*a`, `a*a-b*b` or `(a-b)^2` get only their
lowest six or so bits right, at 16 bits as well as 32. Bit k of a product depends on every lower bit of both operands,
and the multiplication the model learned reads the input tokens; it doesn't take its own intermediate values as
operands. We tried training on exactly these products, with final answers only. Mixed into the general set at 6% and
then 30% of episodes, the held-out ones stayed near 6 bits while `a*b` started to erode. On their own, with twice the
scratch registers, they crept up by about 0.3 bits per 1,000 updates. That points at how the model learns to use its own
intermediates, so the next attempt has to change the model rather than the data. We won't hand it the intermediates.

Rotations are new to it, and real mixing is out of reach: even one add-rotate-xor round scores zero.

Numbers also wrap at every step, like 32-bit registers. For addition, multiplication and bitwise operations that makes no
difference to the answer, but a rule that compares or shifts an overflowed intermediate ends up stranger than the
mathematics it was written as.

## How it got here

Most of the gains came from changing the structure or the data rather than training the same thing for longer: one
token per answer bit, refining the answer over rounds, scratch registers (the first change that let it compute through
a product), 32-bit episodes, and 12 rounds instead of 8, which moved the correct bits of a 32-bit product from 22 to 29.
Training on product chains alone quietly wore down loops and multi-concept rules until we mixed the general set back
in. The [findings](https://teenie.lupyd.com/findings/) record every experiment, including the ones that went nowhere.

## Where this fits

Teenie sits with models that learn to infer a function from examples in context: Garg et al. (2022) and looped
transformers (Yang et al., 2024) on real-valued function classes, Boolean in-context learning (Bhattamishra et al., 2024)
and, closest of all, TabPFN (Hollmann et al., Nature 2025), which pretrains on synthetic tasks and predicts in one pass.
We haven't seen arbitrary composed integer and bitwise rules at 32 bits inferred from about a thousand examples before,
nor a width-free bit-level model for it, nor this way of scoring against lookups and fitted models on a held-out set of
rules with no textbook solution. We haven't compared against TabPFN yet, and until we do we won't claim to beat it.

## Try it

[teenie.lupyd.com/try](https://teenie.lupyd.com/try/) runs the model in your browser. Type a rule of `a` and `b` and it
reads 999 generated examples, then answers 16 new questions, a few of them yours if you like.
