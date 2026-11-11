"""Run Teenie on a rule of your choice with only numpy and onnxruntime.

    pip install numpy onnxruntime huggingface_hub
    python example.py

The model sees 999 examples (a, b, c) of a hidden rule and answers new questions (a, b) in one pass, giving each answer
bit's probability of being 1. Numbers are unsigned and wrap at the width (16 or 32 bits).
"""
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

WIDTH = 32
CONTEXT = 999
MASK = (1 << WIDTH) - 1


def rule(a: int, b: int) -> int:
    return (a * b + a) & MASK  # try your own; wrap at the width after every step to match training


def signed_bits(x: int) -> list[float]:
    """Least significant bit first, as +1 / -1."""
    return [1.0 if (x >> k) & 1 else -1.0 for k in range(WIDTH)]


rng = np.random.default_rng(0)
pairs, seen = [], set()
while len(pairs) < CONTEXT + 16:  # 999 distinct examples, then 16 questions not among them
    a, b = (int(v) for v in rng.integers(0, 1 << WIDTH, 2, dtype=np.uint64))
    if (a, b) not in seen:
        seen.add((a, b))
        pairs.append((a, b))
examples, questions = pairs[:CONTEXT], pairs[CONTEXT:]

bits = np.array([[signed_bits(a), signed_bits(b), signed_bits(rule(a, b))] for a, b in examples], np.float32)[None]
asked = np.array([[signed_bits(a), signed_bits(b)] for a, b in questions], np.float32)[None]

session = ort.InferenceSession(hf_hub_download('TheLupyd/teenie', f'teenie-w{WIDTH}.onnx'))
p = session.run(None, {'bits': bits, 'asked': asked})[0][0]  # [16, WIDTH]: probability that each bit is 1

right = 0
for (a, b), probs in zip(questions, p):
    answer = sum(1 << k for k in range(WIDTH) if probs[k] > 0.5)
    truth = rule(a, b)
    recovered = WIDTH + sum(np.log2(max(min(q if (truth >> k) & 1 else 1 - q, 1 - 2**-20), 2**-20)) for k, q in enumerate(probs))
    right += answer == truth
    print(f'{a:>10} {b:>10}  true {truth:>10}  answer {answer:>10}  {"right" if answer == truth else "wrong"}  '
          f'{recovered:5.1f} bits recovered')
print(f'{right} of {len(questions)} exactly right')
