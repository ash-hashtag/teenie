"""Export a width-free checkpoint to ONNX for the in-browser Try it page (onnxruntime-web).

uv run --extra web python -m teenie.bits.export_web --checkpoint checkpoints/wide/wide-mix7-chains.pt

The browser turns integers into bits itself (the shift that does it has no ONNX form), so the graph takes
bits [1, 999, 3, W] and asked [1, Q, 2, W] in {-1, 1} and returns each answer bit's probability of being 1,
[1, Q, W], after the model's last round. The question count Q is free; the export fixes the width, so there is one
self-contained file per width (teenie-w16.onnx, teenie-w32.onnx; weights inside, about 8 MB each, well under the
25 MB Cloudflare asset limit) plus a small JSON description. Each file is checked against PyTorch.
"""
import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_bit_model
from .data import BitSampler, to_bits
from .sets import ALL_NAMED


class WebModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, bits: torch.Tensor, asked: torch.Tensor) -> torch.Tensor:
        logits = self.model.forward_bits(bits, asked, self.model.config['steps'])
        return torch.sigmoid(logits[-1, :, :asked.shape[1]])


def signed_bits(numbers: torch.Tensor, width: int) -> torch.Tensor:
    return to_bits(numbers, width) * 2 - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', default='checkpoints/wide/wide-mix7-chains.pt')
    parser.add_argument('--output', default='site/public/weights')
    parser.add_argument('--widths', type=int, nargs='+', default=[16, 32])
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, metadata = load_bit_model(args.checkpoint, 'cpu')
    model.eval()
    web = WebModel(model)

    def episode(width: int, queries: int):
        e = BitSampler([ALL_NAMED['a*b']], width, 1).sample(1, queries)
        bits = torch.stack([signed_bits(e.context[..., i], width) for i in range(3)], -2)
        asked = torch.stack([signed_bits(e.query[..., i], width) for i in range(2)], -2)
        return e, bits, asked

    import onnxruntime
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for w in args.widths:
        _, bits, asked = episode(w, 16)
        output = folder / f'teenie-w{w}.onnx'
        queries = torch.export.Dim('queries', min=1, max=64)
        program = torch.onnx.export(web, (bits, asked), dynamo=True, opset_version=18,
                                    input_names=['bits', 'asked'], output_names=['probabilities'],
                                    dynamic_shapes={'bits': {}, 'asked': {1: queries}})
        program.save(str(output), external_data=False)
        session = onnxruntime.InferenceSession(str(output))
        for q in (16, 4):
            e, bits, asked = episode(w, q)
            with torch.no_grad():
                reference = model(e.context, e.query, w).probabilities()
            got = torch.from_numpy(session.run(None, {'bits': bits.numpy(), 'asked': asked.numpy()})[0])
            difference = float((got - reference).abs().max())
            print(f'width {w}, {q} questions: max difference from PyTorch {difference:.2e}')
            assert difference < 1e-4, difference
        files[str(w)] = {'file': output.name, 'bytes': output.stat().st_size}
        print(f'{output} ({output.stat().st_size / 1e6:.1f} MB)')

    parameters = sum(p.numel() for p in model.parameters())
    info = {'checkpoint': args.checkpoint, 'update': metadata.get('update'), 'train_set': metadata.get('train_set'),
            'parameters': parameters, 'rounds': model.config['steps'], 'context': 999, 'widths': files}
    (folder / 'teenie.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'{parameters:,} parameters')


if __name__ == '__main__':
    main()
