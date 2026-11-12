"""Puzzles and a 'rules it cracks' wall for the landing page, from the release model on small numbers.

uv run python -m teenie.bits.export_showcase --checkpoint checkpoints/wide/wide-mix7-chains.pt --device cuda

Each rule is a chain of steps the model handles with small inputs (a, b below 256, answers 16 bits wide). For each one
this records a real episode (a few of its 999 examples and the model's answers to its questions, right or wrong) and the
model's exact rate over 32 fresh episodes of 16 questions, so the page shows measured numbers, not picked ones.
"""
import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_bit_model
from .data import BitSampler
from .relations import fixed

WIDTH = 16


def _sum_one_to(n: torch.Tensor) -> torch.Tensor:
    return n * (n + 1) // 2


# (shown to people, the Try it formula, the rule, a puzzle?) Puzzles run from easy to hard.
RULES = [
    ('a × 10 + b', 'a * 10 + b', lambda a, b, w: a * 10 + b, True),
    ('(a + 1) × (b + 1)', '(a + 1) * (b + 1)', lambda a, b, w: (a + 1) * (b + 1), True),
    ('a + (1 + 2 + … + n), with n = b mod 16', 'a + sum(i + 1 for i in range(b & 15))',
     lambda a, b, w: a + _sum_one_to(b & 15), True),
    ('((a + b) × 3 − a) xor 5', '((a + b) * 3 - a) ^ 5', lambda a, b, w: ((a + b) * 3 - a) ^ 5, True),
    ('(a xor 85) + (b xor 15) + 3', '(a ^ 85) + (b ^ 15) + 3', lambda a, b, w: (a ^ 0x55) + (b ^ 0x0F) + 3, True),
    ('(2a + 3b + 7) xor (a and 15)', '(2 * a + 3 * b + 7) ^ (a & 15)',
     lambda a, b, w: (2 * a + 3 * b + 7) ^ (a & 15), True),
    ('((a + 7) xor b) × 2', '((a + 7) ^ b) << 1', lambda a, b, w: ((a + 7) ^ b) << 1, False),
    ('((a xor b) + a) × 2 + b', '((a ^ b) + a) * 2 + b', lambda a, b, w: ((a ^ b) + a) * 2 + b, False),
    ('(a + b) × 2 + (a xor 3) − b', '(a + b) * 2 + (a ^ 3) - b', lambda a, b, w: (a + b) * 2 + (a ^ 3) - b, False),
    ('(a × b + a) xor 9', '(a * b + a) ^ 9', lambda a, b, w: (a * b + a) ^ 9, False),
    ('a × 2^(b mod 4)', 'a << (b & 3)', lambda a, b, w: a << (b & 3), False),
    ('(a + 1 + 2 + … + n) × 2, with n = b mod 16', '(a + sum(i + 1 for i in range(b & 15))) * 2',
     lambda a, b, w: (a + _sum_one_to(b & 15)) * 2, False),
    ('a × b', 'a * b', lambda a, b, w: a * b, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', default='checkpoints/wide/wide-mix7-chains.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='site/src/data/showcase.json')
    args = parser.parse_args()
    model, metadata = load_bit_model(args.checkpoint, args.device)
    model.eval()
    out = []
    for index, (shown, formula, rule, puzzle) in enumerate(RULES):
        family = fixed(shown, rule)
        with torch.no_grad():
            episode = BitSampler([family], WIDTH, 500 + index, small=1.).sample(1, 16).to(args.device)
            prediction = model(episode.context, episode.query, WIDTH)
            answers, sure = prediction.values()[0], prediction.confidence()[0]
            right = total = 0
            scorer = BitSampler([family], WIDTH, 9000 + index, small=1.)
            for _ in range(32):
                e = scorer.sample(1, 16).to(args.device)
                right += int((model(e.context, e.query, WIDTH).values() == e.target).sum())
                total += e.target.numel()
        context = episode.context[0].tolist()
        examples = sorted(context[:7], key=lambda row: (row[0], row[1]))
        questions = [{'a': int(q[0]), 'b': int(q[1]), 'truth': int(t), 'answer': int(p), 'sure': float(c)}
                     for q, t, p, c in zip(episode.query[0].tolist(), episode.target[0].tolist(), answers.tolist(),
                                           sure.tolist())]
        out.append({'shown': shown, 'formula': formula, 'puzzle': puzzle, 'examples': examples,
                    'questions': questions, 'right': right, 'total': total})
        print(f'{shown:45s} {100 * right / total:5.1f}%  episode questions right: '
              f'{sum(q["answer"] == q["truth"] for q in questions)}/16')
    Path(args.output).write_text(json.dumps({'checkpoint': args.checkpoint, 'update': metadata.get('update'),
                                             'width': WIDTH, 'rules': out}, indent=1) + '\n')


if __name__ == '__main__':
    main()
