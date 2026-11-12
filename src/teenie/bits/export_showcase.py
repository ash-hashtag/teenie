"""Puzzles and a 'rules it cracks' wall for the landing page, from the release model on small numbers.

uv run python -m teenie.bits.export_showcase --checkpoint checkpoints/wide/wide-mix7-chains.pt --device cuda

Each rule is a chain of steps the model handles with small inputs (a, b below 256, answers 16 bits wide). For each one
this records four real episodes (a few of their 999 examples, one question, and the model's answer, right or wrong) and
the model's exact rate over 32 fresh episodes of 16 questions, with how many bits its wrong answers were off by, so the
page shows measured numbers, not picked ones.
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


# (shown to people, the Try it formula, the rule, a puzzle?, on the wall?) Puzzles run from easy to hard; the wall shows
# the chains a person would struggle with.
RULES = [
    ('2a + 3b + 7', '2 * a + 3 * b + 7', lambda a, b, w: 2 * a + 3 * b + 7, True, False),
    ('(a + 1) × (b + 1)', '(a + 1) * (b + 1)', lambda a, b, w: (a + 1) * (b + 1), True, False),
    ('a + (1 + 2 + … + n), with n = b mod 16', 'a + sum(i + 1 for i in range(b & 15))',
     lambda a, b, w: a + _sum_one_to(b & 15), True, True),
    ('((a + b) × 3 − a) xor 5', '((a + b) * 3 - a) ^ 5', lambda a, b, w: ((a + b) * 3 - a) ^ 5, True, True),
    ('(a xor 85) + (b xor 15) + 3', '(a ^ 85) + (b ^ 15) + 3', lambda a, b, w: (a ^ 0x55) + (b ^ 0x0F) + 3, True, True),
    ('(2a + 3b + 7) xor (a and 15)', '(2 * a + 3 * b + 7) ^ (a & 15)',
     lambda a, b, w: (2 * a + 3 * b + 7) ^ (a & 15), True, True),
    ('((a + 7) xor b) × 2', '((a + 7) ^ b) << 1', lambda a, b, w: ((a + 7) ^ b) << 1, True, True),
    ('((a xor b) + a) × 2 + b', '((a ^ b) + a) * 2 + b', lambda a, b, w: ((a ^ b) + a) * 2 + b, True, True),
    ('(a + b) × 2 + (a xor 3) − b', '(a + b) * 2 + (a ^ 3) - b', lambda a, b, w: (a + b) * 2 + (a ^ 3) - b, False,
     True),
    ('(a × b + a) xor 9', '(a * b + a) ^ 9', lambda a, b, w: (a * b + a) ^ 9, True, True),
    ('(a + 1 + 2 + … + n) × 2, with n = b mod 16', '(a + sum(i + 1 for i in range(b & 15))) * 2',
     lambda a, b, w: (a + _sum_one_to(b & 15)) * 2, False, True),
    ('a × b', 'a * b', lambda a, b, w: a * b, True, False),
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
    for index, (shown, formula, rule, puzzle, wall) in enumerate(RULES):
        family = fixed(shown, rule)
        episodes = []
        with torch.no_grad():
            for k in range(4):
                e = BitSampler([family], WIDTH, 500 + 10 * index + k, small=1.).sample(1, 16).to(args.device)
                answers = model(e.context, e.query, WIDTH).values()[0].tolist()
                rows = sorted(e.context[0, :7].tolist(), key=lambda row: (row[0], row[1]))
                a, b = e.query[0, 0].tolist()
                episodes.append({'examples': rows, 'a': a, 'b': b, 'truth': int(e.target[0, 0]), 'answer': int(answers[0])})
            right = total = 0
            off: list[int] = []
            scorer = BitSampler([family], WIDTH, 9000 + index, small=1.)
            for _ in range(32):
                e = scorer.sample(1, 16).to(args.device)
                values = model(e.context, e.query, WIDTH).values()
                right += int((values == e.target).sum())
                total += e.target.numel()
                for v, t in zip(values.flatten().tolist(), e.target.flatten().tolist()):
                    if v != t:
                        off.append(bin(v ^ t).count('1'))
        out.append({'shown': shown, 'formula': formula, 'puzzle': puzzle, 'wall': wall, 'episodes': episodes,
                    'right': right, 'total': total,
                    'bits_off': sorted(off)[len(off) // 2] if off else 0, 'bits_off_max': max(off, default=0)})
        print(f'{shown:45s} {100 * right / total:5.1f}%  misses off by a median {out[-1]["bits_off"]} bits '
              f'(at most {out[-1]["bits_off_max"]}); shown answers right: '
              f'{sum(x["answer"] == x["truth"] for x in episodes)}/4')
    Path(args.output).write_text(json.dumps({'checkpoint': args.checkpoint, 'update': metadata.get('update'),
                                             'width': WIDTH, 'rules': out}, indent=1) + '\n')


if __name__ == '__main__':
    main()
