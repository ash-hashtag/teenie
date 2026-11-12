"""Export a width-free checkpoint to ONNX for the in-browser Try it page (onnxruntime-web).

uv run --extra web python -m teenie.bits.export_web --checkpoint checkpoints/wide/wide-mix7-chains.pt

The browser turns integers into bits itself (the shift that does it has no ONNX form), so the graph takes
bits [1, 999, 3, W] and asked [1, Q, 2, W] in {-1, 1} and returns each answer bit's probability of being 1,
[1, Q, W], after the model's last round. The question count Q is free; the export fixes the width, so there is one
self-contained file per width, named with a hash of its contents (teenie-w32-<hash>.onnx; weights inside, about 8 MB
each, well under the 25 MB Cloudflare asset limit), plus a small JSON description the site reads at build time. A new
export gets new names, so the files can be cached forever (site/public/_headers) and older versions are removed.
Each file is checked against PyTorch.

WebGPU: onnxruntime-web's WebGPU convolution returns wrong numbers (no error) when the input channel count is not a
multiple of 4, which the ask layer's 6-channel Conv1d is. The export pads every such Conv with zero channels, which
leaves the maths unchanged. Each width also gets a self-check (a fixed small episode the browser can rebuild and the
probabilities this file gives for it): the page runs it on WebGPU and falls back to WebAssembly on any mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

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


def pad_conv_channels(model: onnx.ModelProto, multiple: int = 4) -> onnx.ModelProto:
    """Give every Conv a multiple of `multiple` input channels by adding zero channels to its input and weight."""
    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    padded: dict[str, str] = {}
    nodes = []
    for node in graph.node:
        if node.op_type == 'Conv' and node.input[1] in initializers:
            weight = numpy_helper.to_array(initializers[node.input[1]])
            extra = -weight.shape[1] % multiple
            if extra:
                if node.input[1] not in padded:
                    zeros = np.zeros((weight.shape[0], extra, *weight.shape[2:]), weight.dtype)
                    padded[node.input[1]] = node.input[1] + '_padded'
                    graph.initializer.append(numpy_helper.from_array(np.concatenate([weight, zeros], 1),
                                                                     padded[node.input[1]]))
                rank = weight.ndim
                pads = helper.make_tensor(f'{node.name}_channel_pads', onnx.TensorProto.INT64, [2 * rank],
                                          [0] * rank + [0, extra] + [0] * (rank - 2))
                graph.initializer.append(pads)
                widened = f'{node.input[0]}_channels_{node.name}'
                nodes.append(helper.make_node('Pad', [node.input[0], pads.name], [widened], name=f'{node.name}_channels'))
                node.input[0], node.input[1] = widened, padded[node.input[1]]
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    for name in padded:
        graph.initializer.remove(initializers[name])
    return model


CHECK_QUESTIONS = 4


def check_episode(width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The self-check episode, rebuilt the same way by the browser (infer.worker.ts): a, b below 256 from a
    multiplicative hash, c = a*b + a."""
    hashed = lambda i: ((i + 1) * 0x9E3779B1) & 0xFFFFFFFF
    pairs = torch.tensor([[hashed(2 * n) & 255, hashed(2 * n + 1) & 255] for n in range(999 + CHECK_QUESTIONS)])
    a, b = pairs[:, 0], pairs[:, 1]
    c = (a * b + a) % (1 << width)
    bits = torch.stack([signed_bits(x[:999][None], width) for x in (a, b, c)], -2)
    asked = torch.stack([signed_bits(x[999:][None], width) for x in (a, b)], -2)
    return bits.float(), asked.float()


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
        onnx.save(pad_conv_channels(onnx.load(str(output))), str(output))
        session = onnxruntime.InferenceSession(str(output))
        for q in (16, 4):
            e, bits, asked = episode(w, q)
            with torch.no_grad():
                reference = model(e.context, e.query, w).probabilities()
            got = torch.from_numpy(session.run(None, {'bits': bits.numpy(), 'asked': asked.numpy()})[0])
            difference = float((got - reference).abs().max())
            print(f'width {w}, {q} questions: max difference from PyTorch {difference:.2e}')
            assert difference < 1e-4, difference
        bits, asked = check_episode(w)
        check = session.run(None, {'bits': bits.numpy(), 'asked': asked.numpy()})[0]
        digest = hashlib.sha256(output.read_bytes()).hexdigest()[:10]
        versioned = folder / f'teenie-w{w}-{digest}.onnx'
        for old in folder.glob(f'teenie-w{w}-*.onnx'):
            if old != versioned:
                old.unlink()
        output.replace(versioned)
        files[str(w)] = {'file': versioned.name, 'bytes': versioned.stat().st_size,
                         'check': [round(float(x), 5) for x in check.flatten()]}
        print(f'{versioned} ({versioned.stat().st_size / 1e6:.1f} MB)')

    parameters = sum(p.numel() for p in model.parameters())
    info = {'checkpoint': args.checkpoint, 'update': metadata.get('update'), 'train_set': metadata.get('train_set'),
            'parameters': parameters, 'rounds': model.config['steps'], 'context': 999, 'widths': files}
    (folder / 'teenie.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'{parameters:,} parameters')


if __name__ == '__main__':
    main()
