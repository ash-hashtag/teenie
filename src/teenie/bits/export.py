"""Export training logs, evaluations and recorded model behaviour as JSON for the docs site.

uv run python -m teenie.bits.export --checkpoint checkpoints/bits.pt \
    --snapshots checkpoints/bits-snapshots --out site
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .checkpoint import load_bit_model
from .data import BitEpisodes, BitSampler, to_bits
from .model import BitModel, BitPrediction
from .relations import NAMED, HELDOUT_FAMILIES, STRESS_FAMILIES, TRAIN_FAMILIES

TRACE_FAMILIES = ('add', 'max', 'a+2b', 'xor', 'a-2b', 'min', '3a', 'a*b')
TIMELINE_FAMILIES = ('add', 'max', 'a+2b', 'min')
WIDTH = 8
Json = dict[str, object]


def episode_for(name: str, seed: int, queries: int) -> BitEpisodes:
    return BitSampler([NAMED[name]], WIDTH, seed).sample(1, queries)


def quantize(weights: torch.Tensor) -> list[int]:
    """Attention as 0..255 on a log scale of "times an even share": 0 = even spread or less, 255 = 256x or more."""
    ratio = (weights * len(weights)).clamp(min=1e-12)
    return (ratio.log() / math.log(256) * 255).clamp(0, 255).round().int().tolist()


@torch.no_grad()
def record(model: BitModel, episode: BitEpisodes, device: str) -> tuple[BitPrediction, list[dict[int, torch.Tensor]]]:
    attention = model.record_attention()
    episode = episode.to(device)
    prediction = model.forward(episode.context, episode.query, WIDTH)
    model.record_attention(False)
    return prediction, attention


def rounded(x: torch.Tensor) -> list[float]:
    return [round(float(v), 4) for v in x]


def traces(model: BitModel, device: str, queries: int = 3) -> list[Json]:
    result: list[Json] = []
    for i, name in enumerate(TRACE_FAMILIES):
        episode = episode_for(name, 70000 + i, queries)
        prediction, attention = record(model, episode, device)
        probabilities = torch.sigmoid(prediction.logits[:, 0]).cpu()
        values, confidence = prediction.values()[0].cpu(), prediction.confidence()[0].cpu()
        combined = prediction.probabilities()[0].cpu()
        context = episode.context[0]
        c_bits = to_bits(context[:, 2], WIDTH)
        examples: list[Json] = []
        for q in range(queries):
            passes = []
            for d, label in enumerate(('low bit first', 'high bit first')):
                bits = []
                for k in range(WIDTH):
                    w = attention[d][k][0, q].cpu()
                    top = w.topk(6)
                    bits.append({'k': k, 'p': round(float(probabilities[d, q, k]), 4),
                                 'vote': round(float((w * c_bits[:, k]).sum()), 4),
                                 'attention': quantize(w),
                                 'top': [{'i': int(j), 'w': round(float(v), 4)} for v, j in zip(top.values, top.indices)]})
                passes.append({'label': label, 'bits': bits})
            examples.append({'a': int(episode.query[0, q, 0]), 'b': int(episode.query[0, q, 1]),
                             'target': int(episode.target[0, q]),
                             'prediction': int(values[q]), 'confidence': round(float(confidence[q]), 4),
                             'combined': rounded(combined[q]),
                             'passes': passes})
        result.append({'relation': name, 'heldout': NAMED[name] in HELDOUT_FAMILIES,
                       'stress': NAMED[name] in STRESS_FAMILIES,
                       'trust': rounded(prediction.weights[0]), 'context': context.tolist(), 'examples': examples})
    return result


@torch.no_grad()
def timeline(snapshot_dir: Path, device: str, queries: int = 64) -> Json:
    episodes = {name: episode_for(name, 80000 + i, queries) for i, name in enumerate(TIMELINE_FAMILIES)}
    paths = sorted(snapshot_dir.glob('update-*.pt'))
    rows: dict[str, list[Json]] = {name: [] for name in TIMELINE_FAMILIES}
    for path in paths:
        model, metadata = load_bit_model(path, device)
        assert isinstance(model, BitModel)
        for name, episode in episodes.items():
            prediction, attention = record(model, episode, device)
            values = prediction.values()[0].cpu()
            rows[name].append({
                'update': metadata['update'], 'exact': float((values == episode.target[0]).float().mean()),
                'combined': rounded(prediction.probabilities()[0, 0]), 'prediction': int(values[0]),
                'trust': rounded(prediction.weights[0]),
                # Low-bit-first pass, bit 3: where the first query looks among the 999 examples.
                'attention': quantize(attention[0][3][0, 0].cpu())})
    return {'bit': 3, 'relations': {name: {
        'a': int(episode.query[0, 0, 0]), 'b': int(episode.query[0, 0, 1]), 'target': int(episode.target[0, 0]),
        'context': episode.context[0, :, :2].tolist(), 'snapshots': rows[name]} for name, episode in episodes.items()}}


@torch.no_grad()
def pass_trust(model: BitModel, device: str, episodes: int = 16) -> dict[str, float]:
    """Mean weight on the high-bit-first pass per relation."""
    result = {}
    for family in TRAIN_FAMILIES + HELDOUT_FAMILIES + STRESS_FAMILIES:
        sampler = BitSampler([family], WIDTH, 90000)
        weights = [model.forward(e.context, e.query, WIDTH).weights[0, 1]
                   for e in (sampler.sample(1, 1).to(device) for _ in range(episodes))]
        result[family.name] = round(float(torch.stack(weights).mean()), 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Export JSON data for the docs site')
    parser.add_argument('--checkpoint', default='checkpoints/bits.pt')
    parser.add_argument('--snapshots', default='checkpoints/bits-snapshots')
    parser.add_argument('--train-log', default='experiments/bits-seed0-train.jsonl')
    parser.add_argument('--evaluation', default='experiments/bits-seed0-eval.json')
    parser.add_argument('--out', default='site')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    out = Path(args.out)
    (out / 'src/data').mkdir(parents=True, exist_ok=True)
    (out / 'public/data').mkdir(parents=True, exist_ok=True)
    model, metadata = load_bit_model(args.checkpoint, args.device)
    if not isinstance(model, BitModel):
        raise SystemExit('export currently records the bit-serial model only')
    summary = {'parameters': sum(p.numel() for p in model.parameters()), 'config': dict(model.config),
               'metadata': metadata, 'pass_trust': pass_trust(model, args.device),
               'families': {'train': [f.name for f in TRAIN_FAMILIES], 'heldout': [f.name for f in HELDOUT_FAMILIES],
                            'stress': [f.name for f in STRESS_FAMILIES]}}
    training = [json.loads(line) for line in Path(args.train_log).read_text().splitlines() if line]
    (out / 'src/data/model.json').write_text(json.dumps(summary, indent=1))
    (out / 'src/data/training.json').write_text(json.dumps(training))
    (out / 'src/data/evaluation.json').write_text(Path(args.evaluation).read_text())
    (out / 'public/data/traces.json').write_text(json.dumps(traces(model, args.device), separators=(',', ':')))
    (out / 'public/data/timeline.json').write_text(
        json.dumps(timeline(Path(args.snapshots), args.device), separators=(',', ':')))
    print('exported to', out)


if __name__ == '__main__':
    main()
