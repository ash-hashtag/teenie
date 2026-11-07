"""A local relay between the docs site and the model: try your own functions.

uv run teenie-serve                          # http://127.0.0.1:8000 — the built site plus the API
uv run teenie-serve --checkpoint checkpoints/mix3/mix3-mloop.pt --device cuda

POST /api/sample    {"function": "a*b + a", "seed": 1}
    -> 999 generated examples (a, b, c), one new question and the model's answer after every round
POST /api/evaluate  {"function": "a*b + a", "questions": 1000, "contexts": 10, "seed": 1}
    -> exact-match accuracy over fresh questions, spread over several sets of 999 examples

Functions are Python expressions in `a` and `b`, checked before use: only arithmetic, bit operations,
comparisons, `x if cond else y`, min/max/abs, `sum(... for i in range(n))` loops and the helpers popcount,
reverse and rotl are allowed (one loop of at most 256 steps). Results wrap modulo 256. The server listens on localhost only by default.
"""
import argparse
import ast
from collections.abc import Callable
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import threading
import time

import torch

from ..config import CONTEXT_SIZE
from .checkpoint import load_bit_model
from .data import to_bits
from .loop_model import LoopModel

WIDTH = 8
MASK = (1 << WIDTH) - 1
MAX_RANGE = 256


def popcount(x: int) -> int:
    return bin(x & MASK).count('1')


def reverse(x: int) -> int:
    return int(format(x & MASK, f'0{WIDTH}b')[::-1], 2)


def rotl(x: int, k: int) -> int:
    k %= WIDTH
    x &= MASK
    return ((x << k) | (x >> (WIDTH - k))) & MASK


def bounded_range(*args: int) -> range:
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise ValueError(f'range longer than {MAX_RANGE}')
    return r


HELPERS: dict[str, Callable[..., int | range]] = {
    'min': min, 'max': max, 'abs': abs, 'sum': sum, 'range': bounded_range,
    'popcount': popcount, 'reverse': reverse, 'rotl': rotl,
}
ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name, ast.Load,
           ast.Constant, ast.GeneratorExp, ast.comprehension, ast.Store,
           ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr,
           ast.BitXor, ast.Invert, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
           ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class Function:
    """A user expression in a and b, checked against a small whitelist, evaluated modulo 256."""

    def __init__(self, source: str) -> None:
        source = source.strip()
        if not source or len(source) > 300:
            raise ValueError('write an expression in a and b (at most 300 characters)')
        tree = ast.parse(source, mode='eval')
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.comprehension)]
        if len(loops) > 1:
            raise ValueError('at most one loop (one `for` inside sum)')
        loop_names = {node.target.id for node in loops if isinstance(node.target, ast.Name)}
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED):
                raise ValueError(f'not allowed: {type(node).__name__}')
            if isinstance(node, ast.Name) and node.id not in {'a', 'b'} | loop_names | set(HELPERS):
                raise ValueError(f'unknown name: {node.id}')
            if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in HELPERS):
                raise ValueError('only min, max, abs, sum, range, popcount, reverse and rotl can be called')
            if isinstance(node, ast.Constant) and not isinstance(node.value, int):
                raise ValueError('only integer constants are allowed')
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and not (
                    isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and 0 <= node.right.value <= 16):
                raise ValueError('powers need a constant exponent between 0 and 16')
        self.source = source
        self.code = compile(tree, '<function>', 'eval')

    def __call__(self, a: int, b: int) -> int:
        try:
            # Globals, not locals: a loop (generator) has its own scope and only sees globals. No builtins.
            value = eval(self.code, {'__builtins__': {}, **HELPERS, 'a': a, 'b': b})  # noqa: S307 — whitelisted AST
        except ZeroDivisionError:
            value = MASK  # x // 0 and x % 0 read as all ones, as in many hardware dividers
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int):
            raise ValueError('the expression must produce an integer')
        return value & MASK


def episode(function: Function, questions: int, rng: random.Random) -> tuple[list[list[int]], list[list[int]]]:
    """999 distinct examples plus distinct questions that are not among them."""
    pairs = rng.sample(range(1 << (2 * WIDTH)), CONTEXT_SIZE + questions)
    rows = [[p >> WIDTH, p & MASK, function(p >> WIDTH, p & MASK)] for p in pairs]
    return rows[:CONTEXT_SIZE], rows[CONTEXT_SIZE:]


class Relay:
    def __init__(self, checkpoint: str, device: str) -> None:
        model, metadata = load_bit_model(checkpoint, device)
        if not isinstance(model, LoopModel):
            raise SystemExit('teenie-serve expects a loop model checkpoint')
        self.model, self.metadata, self.device, self.checkpoint = model, metadata, device, checkpoint
        self.lock = threading.Lock()

    @torch.no_grad()
    def run(self, context: list[list[int]], questions: list[list[int]]) -> torch.Tensor:
        """Probabilities after every round, [T, Q, W] (bit index order)."""
        c = torch.tensor([context], dtype=torch.long, device=self.device)
        q = torch.tensor([[row[:2] for row in questions]], dtype=torch.long, device=self.device)
        with self.lock:
            prediction = self.model.forward(c, q, WIDTH)
        return torch.sigmoid(prediction.logits[:, 0, :len(questions)]).cpu()

    def sample(self, request: dict[str, object]) -> dict[str, object]:
        function = Function(str(request.get('function', '')))
        rng = random.Random(int(str(request.get('seed', time.time_ns()))))
        context, questions = episode(function, 1, rng)
        probabilities = self.run(context, questions)[:, 0]  # [T, W]
        final = probabilities[-1]
        answer = int(((final > .5).long() * (1 << torch.arange(WIDTH))).sum())
        confidence = float(torch.maximum(final, 1 - final).min())
        a, b, truth = questions[0]
        return {'function': function.source, 'context': context, 'question': {'a': a, 'b': b, 'truth': truth},
                'rounds': [[round(float(x), 4) for x in p] for p in probabilities], 'answer': answer,
                'correct': answer == truth, 'confidence': round(confidence, 4)}

    def evaluate(self, request: dict[str, object]) -> dict[str, object]:
        function = Function(str(request.get('function', '')))
        total = max(1, min(int(str(request.get('questions', 1000))), 5000))
        contexts = max(1, min(int(str(request.get('contexts', 10))), total))
        rng = random.Random(int(str(request.get('seed', time.time_ns()))))
        started = time.perf_counter()
        exact = bits = sure = sure_right = 0
        mistakes: list[dict[str, int | float]] = []
        for k in range(contexts):
            count = total // contexts + (1 if k < total % contexts else 0)
            context, questions = episode(function, count, rng)
            final = self.run(context, questions)[-1]  # [Q, W]
            guessed = (final > .5).long()
            truth_bits = to_bits(torch.tensor([row[2] for row in questions]), WIDTH).long()
            right = (guessed == truth_bits).all(-1)
            confident = torch.maximum(final, 1 - final).amin(-1) >= .99
            exact += int(right.sum())
            bits += int((guessed == truth_bits).sum())
            sure += int(confident.sum())
            sure_right += int((confident & right).sum())
            for i in torch.nonzero(~right).flatten().tolist():
                if len(mistakes) < 12:
                    a, b, truth = questions[i]
                    answer = int((guessed[i] * (1 << torch.arange(WIDTH))).sum())
                    mistakes.append({'a': a, 'b': b, 'truth': truth, 'answer': answer,
                                     'confidence': round(float(torch.maximum(final[i], 1 - final[i]).min()), 4)})
        return {'function': function.source, 'questions': total, 'contexts': contexts,
                'exact': exact / total, 'bits': bits / (total * WIDTH), 'sure_coverage': sure / total,
                'sure_accuracy': sure_right / sure if sure else None, 'mistakes': mistakes,
                'seconds': round(time.perf_counter() - started, 2)}

    def info(self) -> dict[str, object]:
        return {'checkpoint': self.checkpoint, 'parameters': sum(p.numel() for p in self.model.parameters()),
                'config': dict(self.model.config), 'train_set': self.metadata.get('train_set'),
                'device': self.device}


class Handler(SimpleHTTPRequestHandler):
    relay: Relay

    def end_headers(self) -> None:
        self.send_header('Access-Control-Allow-Origin', '*')  # the Astro dev server runs on another port
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def _json(self, status: int, body: dict[str, object]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith('/api/info'):
            return self._json(200, self.relay.info())
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        routes = {'/api/sample': self.relay.sample, '/api/evaluate': self.relay.evaluate}
        route = routes.get(self.path.split('?')[0])
        if route is None:
            return self._json(404, {'error': 'unknown endpoint'})
        try:
            length = min(int(self.headers.get('Content-Length', 0)), 10_000)
            request = json.loads(self.rfile.read(length) or b'{}')
            return self._json(200, route(request))
        except (ValueError, SyntaxError, TypeError) as error:
            return self._json(400, {'error': str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve the docs site and relay "try it" requests to the model')
    parser.add_argument('--checkpoint', help='default: the current model named in site/src/data/current.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--site', default='site/dist', help='built site to serve alongside the API')
    args = parser.parse_args()
    checkpoint = args.checkpoint or json.loads(Path('site/src/data/current.json').read_text())['checkpoint']
    Handler.relay = Relay(checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=args.site))
    print(f'teenie relay on http://{args.host}:{args.port}  (model {checkpoint}, {args.device})', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
