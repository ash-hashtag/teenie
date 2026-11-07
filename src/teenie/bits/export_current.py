"""Export the current model for the docs site: its architecture, parameter counts and recorded predictions.

uv run python -m teenie.bits.export_current --checkpoint checkpoints/mix3/mix3-mloop.pt \
    --evaluation experiments/mix3/mix3-mloop-eval.json

Writes site/src/data/current.json (architecture, parameters per part, evaluation summary) and
site/public/data/rounds.json (for a few relations: the answer guess after every round and where the
question looks among the 999 examples).
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from .checkpoint import load_bit_model
from .data import BitSampler, to_bits
from .eval import summarise
from .bit_token_model import BitTokenModel
from .export_progress import bit_token_parts
from .loop_model import LoopModel
from .sets import ALL_NAMED

TRACED = ('add', 'a*b', 'max', '(a+b)>>1', 'a*a', 'a//b', 'sum a..a+n-1 n15', 'a*b+a')
WIDTH = 8


def count(module: nn.Module | None) -> int:
    return 0 if module is None else sum(p.numel() for p in module.parameters())


def architecture(model: LoopModel | BitTokenModel, metadata: dict[str, object]) -> dict[str, object]:
    if isinstance(model, BitTokenModel):
        return {'kind': 'bittoken', **bit_token_parts(model), 'probes': model.config['probes'], 'scratch': 0,
                'multiplicative': True, 'train_set': metadata.get('train_set'), 'updates': metadata.get('update'),
                'adaptive': metadata.get('adaptive')}
    config = model.config
    looks = list(model.looks)
    parts = [
        {'part': 'Example encoder', 'detail': '24 bits → bilinear layer → GELU → linear → norm (run once)',
         'parameters': count(model.example)},
        {'part': 'Question input', 'detail': '(a, b, current guess) → bilinear layer', 'parameters': count(model.ask)},
        *[{'part': f'Look layer {i + 1}', 'detail': f'cross-attention to the examples ({config["heads"]} heads) '
           '+ gated (SwiGLU) feed-forward', 'parameters': count(look)} for i, look in enumerate(looks)],
        {'part': 'Memory', 'detail': 'GRU cell carrying the question’s state between rounds', 'parameters': count(model.cell)},
        {'part': 'Readout', 'detail': 'norm → linear → 8 bit logits', 'parameters': count(model.readout)},
    ]
    if config.get('scratch', 0):
        parts.append({'part': f'{config["scratch"]} scratch slots',
                      'detail': 'slot identities, self-attention among the question’s tokens, slot GRU cell',
                      'parameters': count(model.within) + count(model.slot_cell) + model.slot_identity.numel()})
    return {'kind': 'loop', 'hidden': config['hidden'], 'heads': config['heads'], 'layers': config['layers'],
            'rounds': config['steps'], 'probes': config['probes'], 'width': config['width'],
            'multiplicative': config.get('multiplicative', False), 'scratch': config.get('scratch', 0),
            'parameters': count(model), 'parts': parts, 'train_set': metadata.get('train_set'),
            'updates': metadata.get('update'), 'adaptive': metadata.get('adaptive')}


def quantize(weights: torch.Tensor) -> list[int]:
    """Attention as 0..255 on a log scale of 'times an even share' (0 = even or less, 255 = 256x)."""
    ratio = (weights * len(weights)).clamp(min=1e-12)
    return (ratio.log() / math.log(256) * 255).clamp(0, 255).round().int().tolist()


@torch.no_grad()
def traces(model: LoopModel | BitTokenModel, device: str, queries: int = 2) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for i, name in enumerate(TRACED):
        episode = BitSampler([ALL_NAMED[name]], WIDTH, 71000 + i).sample(1, queries).to(device)
        attention = model.record_attention()
        prediction = model.forward(episode.context, episode.query, WIDTH)
        model.record_attention(False)
        probabilities = torch.sigmoid(prediction.logits[:, 0, :queries]).cpu()  # [T, Q, W]
        last = attention[-1]  # last look layer, one entry per round: [1, Q + P, N]
        examples = []
        for q in range(queries):
            rounds = []
            for t in range(len(probabilities)):
                weights = last[t][0, q].cpu()
                top = weights.topk(6)
                rounds.append({'p': [round(float(x), 4) for x in probabilities[t, q]], 'attention': quantize(weights),
                               'top': [{'i': int(j), 'w': round(float(v), 4)} for v, j in zip(top.values, top.indices)]})
            examples.append({'a': int(episode.query[0, q, 0]), 'b': int(episode.query[0, q, 1]),
                             'target': int(episode.target[0, q]), 'prediction': int(prediction.values()[0, q]),
                             'confidence': round(float(prediction.confidence()[0, q]), 4), 'rounds': rounds})
        result.append({'relation': name, 'context': episode.context[0].cpu().tolist(), 'examples': examples})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Export the current model for the docs site')
    parser.add_argument('--checkpoint', default='checkpoints/mix3/mix3-mloop-scratch4.pt')
    parser.add_argument('--evaluation', default='experiments/mix3/mix3-mloop-scratch4-eval.json')
    parser.add_argument('--name', default='Multiplicative loop with scratch slots')
    parser.add_argument('--out', default='site')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    model, metadata = load_bit_model(args.checkpoint, args.device)
    if not isinstance(model, (LoopModel, BitTokenModel)):
        raise SystemExit('the current-model export expects a loop or bit-token model checkpoint')
    evaluation = json.loads(Path(args.evaluation).read_text())
    summary = {group: summarise(reports) for group, reports in evaluation['results'].items()}
    families = {name: report.get('exact') for reports in evaluation['results'].values() for name, report in reports.items()}
    out = Path(args.out)
    (out / 'src/data').mkdir(parents=True, exist_ok=True)
    (out / 'public/data').mkdir(parents=True, exist_ok=True)
    (out / 'src/data/current.json').write_text(json.dumps({
        'name': args.name, 'checkpoint': args.checkpoint, 'architecture': architecture(model, metadata),
        'summary': summary, 'families': families}, indent=1))
    (out / 'public/data/rounds.json').write_text(json.dumps(traces(model, args.device), separators=(',', ':')))
    print('exported current model', args.checkpoint)


if __name__ == '__main__':
    main()
