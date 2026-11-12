"""Record the release model answering, round by round, for the Model page (site/public/data/rounds.json).

uv run python -m teenie.bits.export_rounds --device cuda

For a few rules from the landing page (export_showcase.RULES) plus a product it can do and one it cannot, at 16 bits
with a, b below 256: one episode of 999 examples and four questions. For every question and every round it records
the answer bits' probabilities, the two scratch registers as the model writes them, and where the question's bits look
among the 999 examples in the last bit layer (averaged over heads and bit positions).
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .checkpoint import load_bit_model
from .data import BitSampler
from .export_showcase import RULES, WIDTH
from .relations import fixed

EXTRA = (
    ('a × b', lambda a, b, w: a * b),
    ('(a + b) × a', lambda a, b, w: (a + b) * a),  # a failure: products of computed values
)
QUESTIONS = 4


def quantize(weights: torch.Tensor) -> list[int]:
    """Attention as 0..255 on a log scale of 'times an even share' (0 = even or less, 255 = 256x)."""
    ratio = (weights * len(weights)).clamp(min=1e-12)
    return (ratio.log() / math.log(256) * 255).clamp(0, 255).round().int().tolist()


@torch.no_grad()
def record(model, shown: str, rule, seed: int, device: str) -> dict[str, object]:
    episode = BitSampler([fixed(shown, rule)], WIDTH, seed, small=1.).sample(1, QUESTIONS).to(device)
    layers = list(model.layers)
    for layer in layers:
        layer.attention = []
    prediction = model(episode.context, episode.query, WIDTH)
    rounds = len(layers[-1].attention)
    last = layers[-1].attention  # one [1, Q + P, N] per round
    for layer in layers:
        layer.attention = None
    # The registers: run the rounds again, keeping everything the readout writes (answer and registers) per round.
    written: list[torch.Tensor] = []
    original = model._round

    def keep(*args):
        state, out = original(*args)
        written.append(out)
        return state, out

    model._round = keep
    try:
        model(episode.context, episode.query, WIDTH)
    finally:
        del model._round
    probabilities = torch.sigmoid(prediction.logits[:, 0, :QUESTIONS]).cpu()  # [T, Q, W]
    values, confidence = prediction.values()[0], prediction.confidence()[0]
    examples = []
    for q in range(QUESTIONS):
        steps = []
        for t in range(rounds):
            weights = last[t][0, q].float().cpu()
            top = weights.topk(6)
            registers = torch.sigmoid(written[t][0, q, 1:]).cpu()  # [R, W]
            steps.append({'p': [round(float(x), 4) for x in probabilities[t, q]],
                          'registers': [[round(float(x), 3) for x in r] for r in registers],
                          'attention': quantize(weights),
                          'top': [{'i': int(j), 'w': round(float(v), 4)} for v, j in zip(top.values, top.indices)]})
        examples.append({'a': int(episode.query[0, q, 0]), 'b': int(episode.query[0, q, 1]),
                         'target': int(episode.target[0, q]), 'prediction': int(values[q]),
                         'confidence': round(float(confidence[q]), 4), 'rounds': steps})
    return {'relation': shown, 'width': WIDTH, 'context': episode.context[0].cpu().tolist(), 'examples': examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', default='checkpoints/wide/wide-mix7-chains.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='site/public/data/rounds.json')
    args = parser.parse_args()
    model, metadata = load_bit_model(args.checkpoint, args.device)
    model.eval()
    chosen = [(shown, rule) for shown, _, rule, puzzle, _ in RULES if puzzle] + list(EXTRA)
    traces = [record(model, shown, rule, 72000 + i, args.device) for i, (shown, rule) in enumerate(chosen)]
    for t in traces:
        right = sum(e['prediction'] == e['target'] for e in t['examples'])
        print(f'{t["relation"]:42s} {right}/{QUESTIONS} right, {len(t["examples"][0]["rounds"])} rounds')
    Path(args.output).write_text(json.dumps({'checkpoint': args.checkpoint, 'update': metadata.get('update'),
                                             'width': WIDTH, 'traces': traces}, separators=(',', ':')))
    print(f'{args.output}: {Path(args.output).stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
