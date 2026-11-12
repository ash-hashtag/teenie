"""Teenie on SyGuS-Comp bit-vector benchmarks, against cvc5 (experiments/sygus/).

    uv run python -m teenie.bits.sygus hd            Teenie on the Hacker's Delight General-track functions
    uv run python -m teenie.bits.sygus check-cvc5    cvc5's programs for those, checked on held-out inputs
    uv run python -m teenie.bits.sygus deep-make     write the deep-carry candidates as PBE problems for cvc5
    uv run python -m teenie.bits.sygus deep-teenie   Teenie on the candidates cvc5 timed out on

Benchmarks: github.com/SyGuS-Org/benchmarks at 13c8deb, lib/General_Track/from_2018/hd-NN-dK-prog.sl (the SyGuS-Comp
2017-2019 General track Hacker's Delight set: 15 functions, each with grammars d0/d1/d5). The PBE_BV track is not
used: all its problems are 64-bit and the model's widths stop at 32.

What "success" means differs, and the report says so. cvc5 succeeds when it returns a program, which it has checked
against the specification. Teenie never writes a program: it reads 999 examples and answers questions. Its success
on a function is answering all of 1,000 fresh random questions exactly, a sample, not a proof.

Deep-carry protocol, fixed before any result was seen:
  1. 40 random expressions in x, y from seed 0 (`deep_candidates`): trees of the ops bvadd, bvsub, bvxor, bvand,
     bvor, bvnot, bvshl/bvlshr by a constant 1..15, and random 32-bit constants; at least 4 additions or subtractions
     and depth 5 or more; no multiplication or division (the model's known weakness, and not what this set is about);
     dropped if it gives fewer than 500 different values on the 999 examples (a nearly constant rule).
  2. Each becomes a PBE problem with the same 999 examples Teenie reads, the grammar above with any 32-bit constant,
     for cvc5 with default options and 300 s.
  3. Teenie runs once on every candidate cvc5 timed out on, with 1,000 fresh questions; all are reported.
"""
import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import torch

ROOT = Path('experiments/sygus')
WIDTH = 32
MASK = (1 << WIDTH) - 1
QUESTIONS = 1000
CHUNK = 250  # questions per forward pass (they don't see each other, so this only bounds memory)


# ---- SMT-LIB terms --------------------------------------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    text = re.sub(r';[^\n]*', '', text)
    return re.findall(r'\(|\)|[^\s()]+', text)


def sexps(text: str) -> list:
    stack: list[list] = [[]]
    for t in tokens(text):
        if t == '(':
            stack.append([])
        elif t == ')':
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    return stack[0]


def signed(v: int) -> int:
    return v - (1 << WIDTH) if v >> (WIDTH - 1) else v


def evaluate(term, env: dict[str, int], defs: dict) -> int:
    """A 32-bit bit-vector (or Boolean, as 0/1) term under `env`, with SMT-LIB semantics."""
    if isinstance(term, str):
        if term in env:
            return env[term]
        if term.startswith('#x'):
            return int(term[2:], 16)
        if term.startswith('#b'):
            return int(term[2:], 2)
        if term in ('true', 'false'):
            return int(term == 'true')
        if term in defs and not defs[term][0]:
            return evaluate(defs[term][1], {}, defs)
        raise ValueError(f'unknown symbol {term}')
    head = term[0]
    if isinstance(head, list) and head[0] == '_' and head[1].startswith('bv'):
        raise ValueError(term)
    if head == '_':  # (_ bvN W)
        return int(term[1][2:]) & MASK
    if head == 'let':
        inner = dict(env)
        for name, value in term[1]:
            inner[name] = evaluate(value, env, defs)
        return evaluate(term[2], inner, defs)
    if head == 'ite':
        return evaluate(term[2] if evaluate(term[1], env, defs) else term[3], env, defs)
    args = [evaluate(t, env, defs) for t in term[1:]]
    if head in defs:
        params, body = defs[head]
        return evaluate(body, dict(zip(params, args)), defs)
    x = args[0]
    y = args[1] if len(args) > 1 else None
    match head:
        case 'bvnot': return ~x & MASK
        case 'bvneg': return -x & MASK
        case 'bvadd': return sum(args) & MASK
        case 'bvsub': return (x - y) & MASK
        case 'bvmul': return (x * y) & MASK
        case 'bvudiv': return MASK if y == 0 else x // y
        case 'bvurem': return x if y == 0 else x % y
        case 'bvsdiv':
            if y == 0:
                return 1 if signed(x) < 0 else MASK
            q = abs(signed(x)) // abs(signed(y))
            return (-q if (signed(x) < 0) != (signed(y) < 0) else q) & MASK
        case 'bvsrem':
            if y == 0:
                return x
            r = abs(signed(x)) % abs(signed(y))
            return (-r if signed(x) < 0 else r) & MASK
        case 'bvand': return x & y
        case 'bvor': return x | y
        case 'bvxor': return x ^ y
        case 'bvshl': return (x << y) & MASK if y < WIDTH else 0
        case 'bvlshr': return x >> y if y < WIDTH else 0
        case 'bvashr': return (signed(x) >> min(y, WIDTH - 1)) & MASK
        case '=': return int(x == y)
        case 'bvult': return int(x < y)
        case 'bvule': return int(x <= y)
        case 'bvugt': return int(x > y)
        case 'bvuge': return int(x >= y)
        case 'bvslt': return int(signed(x) < signed(y))
        case 'bvsle': return int(signed(x) <= signed(y))
        case 'bvsgt': return int(signed(x) > signed(y))
        case 'bvsge': return int(signed(x) >= signed(y))
        case 'not': return int(not x)
        case 'and': return int(all(args))
        case 'or': return int(any(args))
    raise ValueError(f'unsupported operator {head}')


def definitions(forms: list) -> dict:
    """define-fun name -> (parameter names, body)."""
    return {f[1]: ([p[0] for p in f[2]], f[4]) for f in forms if isinstance(f, list) and f and f[0] == 'define-fun'}


def reference(path: Path) -> tuple[str, list[str], dict]:
    """A General-track Hacker's Delight file: its reference function's name, parameters, and all definitions."""
    defs = definitions(sexps(path.read_text()))
    name = next(n for n in defs if n.startswith('hd'))
    return name, defs[name][0], defs


# ---- Rules as site formulas (generate-csv.ts reads them with the Try page's expression.ts) --------------------------

SITE = {'bvadd': '+', 'bvsub': '-', 'bvmul': '*', 'bvudiv': '//', 'bvand': '&', 'bvor': '|', 'bvxor': '^',
        'bvshl': '<<', 'bvlshr': '>>'}


def to_formula(term, names: dict[str, str]) -> str:
    """An SMT-LIB term as a Try-page formula, exactly: bvneg is 0 - x, bvashr by 31 is all ones when the top bit is set
    (the only arithmetic shift these benchmarks use), and the rest map one to one."""
    if isinstance(term, str):
        if term in names:
            return names[term]
        if term.startswith('#x'):
            return str(int(term[2:], 16))
        raise ValueError(f'cannot translate {term}')
    head, args = term[0], [to_formula(t, names) for t in term[1:]]
    if head == 'bvnot':
        return f'(~{args[0]})'
    if head == 'bvneg':
        return f'(0 - {args[0]})'
    if head == 'bvashr':
        if term[2] != '#x0000001F':
            raise ValueError('only bvashr by 31 is translated')
        return f'({MASK} if {args[0]} >> 31 else 0)'
    if head in SITE:
        return f'({args[0]} {SITE[head]} {args[1]})'
    raise ValueError(f'cannot translate {head}')


def rules() -> list[dict]:
    """Every rule in the comparison: the Hacker's Delight reference functions with at most two inputs, then the
    deep-carry candidates; each with its SMT-LIB term (for checking) and its site formula (for generate-csv)."""
    out = []
    seen = set()
    for path in sorted((ROOT / 'benchmarks/hd').glob('hd-*-prog.sl')):
        name, params, defs = reference(path)
        if name in seen:
            continue
        seen.add(name)
        entry = {'set': 'hd', 'name': name, 'inputs': len(params), 'params': params, 'term': show(defs[name][1])}
        if len(params) <= 2:
            entry['formula'] = to_formula(defs[name][1], dict(zip(params, 'ab')))
        out.append(entry)
    for n, t in enumerate(deep_candidates()):
        out.append({'set': 'deep', 'name': f'deep-{n:02d}', 'inputs': 2, 'params': ['x', 'y'], 'term': show(t),
                    'formula': to_formula(t, {'x': 'a', 'y': 'b'})})
    return out


def truth(rule: dict):
    term = sexps(rule['term'])[0]
    return lambda a, b: evaluate(term, dict(zip(rule['params'], (a, b))), {})


def read_csv(path: Path) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """A generate-csv file: the 999 examples (a, b, c) and the questions (line, a, b)."""
    lines = path.read_text().splitlines()
    rows = [(n + 1, line.split(',')) for n, line in enumerate(lines[1:], 1) if line.strip()]
    examples = [(int(c[0]), int(c[1]), int(c[2])) for _, c in rows[:999]]
    return examples, [(n, int(c[0]), int(c[1])) for n, c in rows[999:]]


def verify(paths: list[Path]) -> None:
    """The examples generate-csv wrote (through expression.ts) against the SMT-LIB reference: the translation check."""
    by_name = {r['name']: r for r in rules()}
    for path in paths:
        rule = by_name[path.stem.rsplit('-s', 1)[0]]
        f = truth(rule)
        examples, _ = read_csv(path)
        wrong = sum(f(a, b) != c for a, b, c in examples)
        print(f'{path.name}: {wrong} of {len(examples)} examples differ from the SMT-LIB reference')
        if wrong:
            raise SystemExit(1)


# ---- Teenie ---------------------------------------------------------------------------------------------------------

def load_model(device: str):
    from .checkpoint import load_bit_model
    model, _ = load_bit_model('checkpoints/wide/wide-mix7-chains.pt', device)
    return model.eval()


@torch.no_grad()
def answer(paths: list[Path], device: str) -> None:
    """Teenie answers each generate-csv file, writing <name>-answers.csv in the Try page's download format (so
    generate-csv --check scores it) and the seconds each file took to <folder>/timings.json."""
    model = load_model(device)
    model(torch.ones(1, 999, 3, dtype=torch.int64, device=device), torch.ones(1, 1, 2, dtype=torch.int64,
                                                                                device=device), WIDTH)  # warm-up
    for path in paths:
        examples, questions = read_csv(path)
        context = torch.tensor([examples], dtype=torch.int64, device=device)
        probabilities = []
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(questions), CHUNK):
            query = torch.tensor([[(a, b) for _, a, b in questions[start:start + CHUNK]]], dtype=torch.int64,
                                 device=device)
            probabilities.append(model(context, query, WIDTH).probabilities()[0].float().cpu())
        if device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        write_answers(path, questions, torch.cat(probabilities).tolist(), seconds, 'answers',
                      torch.cuda.get_device_name() if device == 'cuda' else 'cpu')


def write_answers(path: Path, questions: list, p: list[list[float]], seconds: float, kind: str, device: str) -> None:
    """<name>-<kind>.csv in the Try page's download format, so generate-csv --check scores it; p is [Q][W], bit k at
    index k. The seconds go to <folder>/<kind>-timings.json."""
    out = ['line,a,b,answer,confidence,sure,' + ','.join(f'p_bit{k}' for k in range(WIDTH - 1, -1, -1))]
    for (line, a, b), bits in zip(questions, p):
        value = sum(1 << k for k, q in enumerate(bits) if q > .5)
        confidence = min(max(q, 1 - q) for q in bits)
        out.append(f'{line},{a},{b},{value},{confidence:.4f},{int(confidence >= .99)},'
                   + ','.join(f'{bits[k]:.4f}' for k in range(WIDTH - 1, -1, -1)))
    target = path.with_name(f'{path.stem}-{kind}.csv')
    target.write_text('\n'.join(out) + '\n')
    timings_path = path.parent / f'{kind}-timings.json'
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    timings[path.stem] = {'seconds': round(seconds, 3), 'questions': len(questions), 'device': device}
    timings_path.write_text(json.dumps(timings, indent=1))
    print(f'{path.name}: {len(questions)} answers in {seconds:.2f} s -> {target.name}')


def tabpfn(paths: list[Path]) -> None:
    """TabPFN on each generate-csv file, the way learners.py runs it (GPU, one classifier per answer bit on the 64
    input bits, 4 estimators): the same 999 examples and the same questions as Teenie. Not calibrated: calibration only
    pulls probabilities toward 1/2, which changes no answer except to make a coin-toss bit 0."""
    import os
    token = Path.home() / '.tabpfn_token'
    if 'TABPFN_TOKEN' not in os.environ and token.exists():
        os.environ['TABPFN_TOKEN'] = token.read_text().strip()
    import numpy as np
    from .learners import fit_tabpfn

    def features(pairs: list[tuple[int, int]]) -> np.ndarray:
        return np.array([[(v >> k) & 1 for v in pair for k in range(WIDTH)] for pair in pairs], dtype=np.float32)

    for path in paths:
        examples, questions = read_csv(path)
        x = features([(a, b) for a, b, _ in examples])
        y = np.array([[(c >> k) & 1 for k in range(WIDTH)] for _, _, c in examples])
        torch.cuda.synchronize()
        started = time.perf_counter()
        p = fit_tabpfn(x, y, features([(a, b) for _, a, b in questions]))
        torch.cuda.synchronize()
        write_answers(path, questions, p.tolist(), time.perf_counter() - started, 'tabpfn', torch.cuda.get_device_name())


# ---- cvc5 -----------------------------------------------------------------------------------------------------------

def check_cvc5() -> None:
    """Each cvc5 program for a Hacker's Delight file against its reference on 10,000 random inputs."""
    rows = []
    for path in sorted((ROOT / 'benchmarks/hd').glob('hd-*-prog.sl')):
        out = ROOT / 'cvc5-hd' / f'{path.stem}.out'
        name, params, defs = reference(path)
        if not out.exists() or 'define-fun' not in out.read_text():
            continue
        program = definitions(sexps(out.read_text())[0])['f']
        rng = random.Random(path.stem)
        wrong = 0
        for _ in range(10_000):
            env = {p: rng.getrandbits(WIDTH) for p in params}
            wrong += evaluate(program[1], dict(zip(program[0], env.values())), defs) != evaluate(defs[name][1], env, defs)
        rows.append((path.stem, wrong))
        print(f'{path.stem}: {wrong} of 10,000 disagree')
    (ROOT / 'cvc5-hd-check.json').write_text(json.dumps(dict(rows), indent=1))


# ---- Deep-carry set -------------------------------------------------------------------------------------------------

def deep_candidates(count: int = 40, seed: int = 0) -> list:
    rng = random.Random(seed)

    def tree(depth: int):
        if depth == 0 or (depth < 3 and rng.random() < 0.25):
            return rng.choice(['x', 'y', 'x', 'y', f'#x{rng.getrandbits(WIDTH):08x}'])
        op = rng.choices(['bvadd', 'bvsub', 'bvxor', 'bvand', 'bvor', 'bvnot', 'shift'], [5, 3, 3, 1, 1, 1, 2])[0]
        if op == 'bvnot':
            return ['bvnot', tree(depth - 1)]
        if op == 'shift':
            return [rng.choice(['bvshl', 'bvlshr']), tree(depth - 1), f'#x{rng.randint(1, 15):08x}']
        return [op, tree(depth - 1), tree(depth - 1)]

    def measure(t) -> tuple[int, int]:  # (additions and subtractions, depth)
        if isinstance(t, str):
            return 0, 0
        parts = [measure(c) for c in t[1:]]
        return (sum(p[0] for p in parts) + (t[0] in ('bvadd', 'bvsub')), 1 + max(p[1] for p in parts))

    found = []
    while len(found) < count:
        t = tree(rng.randint(5, 6))
        carries, depth = measure(t)
        if carries >= 4 and depth >= 5 and 'x' in str(t) and 'y' in str(t):
            found.append(t)
    return found


def show(t) -> str:
    return t if isinstance(t, str) else f'({" ".join(show(c) for c in t)})'


GRAMMAR = """((Start (_ BitVec 32)))
    ((Start (_ BitVec 32) (x y (Constant (_ BitVec 32)) (bvnot Start) (bvadd Start Start) (bvsub Start Start)
                           (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvshl Start Start)
                           (bvlshr Start Start))))"""


def pbe(paths: list[Path], folder: Path) -> None:
    """A PBE problem for cvc5 from each generate-csv file: its 999 examples, the grammar above; and the share of
    distinct answers, for the protocol's filter (fewer than 500 different values: dropped)."""
    folder.mkdir(parents=True, exist_ok=True)
    for path in paths:
        examples, _ = read_csv(path)
        distinct = len({c for _, _, c in examples})
        if distinct < 500:
            print(f'{path.stem}: dropped, {distinct} different values')
            continue
        lines = ['(set-logic BV)', f'(synth-fun f ((x (_ BitVec 32)) (y (_ BitVec 32))) (_ BitVec 32)\n    {GRAMMAR})']
        lines += [f'(constraint (= (f #x{a:08x} #x{b:08x}) #x{c:08x}))' for a, b, c in examples]
        lines.append('(check-synth)')
        (folder / f'{path.stem}.sl').write_text('\n'.join(lines) + '\n')
    print(f'{len(list(folder.glob("*.sl")))} PBE problems in {folder}')


def cvc5_held_out(paths: list[Path], results: Path) -> None:
    """For each deep problem cvc5 solved: its program on the file's questions, against the rule (should all agree)."""
    by_name = {r['name']: r for r in rules()}
    for path in paths:
        out = results / f'{path.stem}.out'
        if not out.exists() or 'define-fun' not in out.read_text():
            continue
        program = definitions(sexps(out.read_text())[0])['f']
        f = truth(by_name[path.stem.rsplit('-s', 1)[0]])
        _, questions = read_csv(path)
        wrong = sum(evaluate(program[1], dict(zip(program[0], (a, b))), {}) != f(a, b) for _, a, b in questions)
        print(f'{path.stem}: cvc5 program wrong on {wrong} of {len(questions)} questions')


def export(output: Path) -> None:
    """Everything the site's SyGuS page shows, from the files the runs wrote (no number is computed anywhere else)."""
    import statistics
    rules_by = {r['name']: r for r in json.loads((ROOT / 'rules.json').read_text())}

    def tsv(path: Path) -> dict[str, tuple[str, float]]:
        if not path.exists():
            return {}
        return {name: (status, int(ms) / 1000) for name, status, ms in
                (line.split('\t') for line in path.read_text().splitlines())}

    def checks(stem: str, suffix: str = '') -> dict | None:
        path = ROOT / 'checks' / f'{stem}{suffix}.json'
        return json.loads(path.read_text()) if path.exists() else None

    def method(stems: list[str], suffix: str, timings: dict) -> dict | None:
        found = [(s, checks(s, suffix)) for s in stems]
        found = [(s, c) for s, c in found if c]
        if not found:
            return None
        return {'episodes': len(found), 'all_exact': sum(c['exact'] == c['answers'] for _, c in found),
                'exact': [round(c['exact_share'], 4) for _, c in found],
                'bits_right': round(statistics.mean(c['bits_right'] for _, c in found), 4),
                'seconds': [timings[s]['seconds'] for s, _ in found if s in timings]}

    def timings(folder: str, kind: str) -> dict:
        path = ROOT / 'episodes' / folder / f'{kind}-timings.json'
        return json.loads(path.read_text()) if path.exists() else {}

    capped = tsv(ROOT / 'cvc5-hd/results.tsv')
    verified = json.loads((ROOT / 'cvc5-hd-check.json').read_text())
    files = []
    for name, (status, seconds) in sorted(capped.items()):
        entry = {'file': f'{name}.sl', 'function': 'hd' + name.split('-')[1], 'grammar': name.split('-')[2],
                 'status': 'failed' if status.startswith('error') else status, 'seconds': seconds}
        if name in verified:
            entry['wrong_of_10000'] = verified[name]
        files.append(entry)
    teenie_hd, tabpfn_hd = timings('hd', 'answers'), timings('hd', 'tabpfn')
    functions = []
    for r in rules_by.values():
        if r['set'] != 'hd':
            continue
        stems = [f'{r["name"]}-s{s}' for s in range(5)]
        functions.append({'name': r['name'], 'inputs': r['inputs'], 'term': r['term'], 'formula': r.get('formula'),
                          'teenie': method(stems, '', teenie_hd) if 'formula' in r else None,
                          'tabpfn': method(stems, '-tabpfn', tabpfn_hd) if 'formula' in r else None})
    deep_cvc5 = tsv(ROOT / 'cvc5-deep/results.tsv')
    teenie_deep, tabpfn_deep = timings('deep', 'answers'), timings('deep', 'tabpfn')
    problems = []
    for r in rules_by.values():
        if r['set'] != 'deep':
            continue
        stem = f'{r["name"]}-s0'
        status, seconds = deep_cvc5.get(stem, ('not run', None))
        problems.append({'name': r['name'], 'formula': r['formula'], 'cvc5': status, 'cvc5_seconds': seconds,
                         'teenie': method([stem], '', teenie_deep), 'tabpfn': method([stem], '-tabpfn', tabpfn_deep)})
    profile = [0] * WIDTH
    count = 0
    for r in rules_by.values():
        answers = ROOT / 'episodes/deep' / f'{r["name"]}-s0-answers.csv'
        if r['set'] != 'deep' or not answers.exists():
            continue
        f = truth(r)
        for line in answers.read_text().splitlines()[1:]:
            cells = line.split(',')
            off = f(int(cells[1]), int(cells[2])) ^ int(cells[3])
            count += 1
            for k in range(WIDTH):
                profile[k] += not (off >> k) & 1
    device = next(iter(teenie_hd.values()), {}).get('device')
    output.write_text(json.dumps({
        'source': {'repository': 'https://github.com/SyGuS-Org/benchmarks', 'commit': '13c8deb',
                   'path': 'lib/General_Track/from_2018/hd-NN-dK-prog.sl'},
        'solver': {'name': 'cvc5 1.4.2', 'options': 'default (--lang=sygus2)', 'timeout': 300},
        'teenie': {'checkpoint': 'checkpoints/wide/wide-mix7-chains.pt', 'device': device},
        'tabpfn': {'version': 'tabpfn 9.2.0', 'setup': 'one classifier per answer bit on the 64 input bits, 4 estimators',
                   'device': next(iter(tabpfn_hd.values()), {}).get('device')} if tabpfn_hd else None,
        'questions': QUESTIONS,
        'hd': {'files': files, 'functions': functions},
        'deep': {'problems': problems, 'bit_profile': [round(v / count, 4) for v in profile] if count else None},
        'pbe_bv': {'problems': 753, 'reason': 'every problem is 64-bit; the model takes at most 32 bits'},
    }, indent=1))
    print(f'wrote {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['rules', 'verify', 'answer', 'tabpfn', 'pbe', 'check-cvc5', 'cvc5-held-out', 'export'])
    parser.add_argument('files', nargs='*', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--folder', type=Path, default=ROOT / 'deep')
    args = parser.parse_args()
    if args.command == 'rules':
        (ROOT / 'rules.json').write_text(json.dumps(rules(), indent=1))
        for r in rules():
            print(f'{r["name"]}\t{r.get("formula", "(skipped: three inputs)")}')
    elif args.command == 'verify':
        verify(args.files)
    elif args.command == 'answer':
        answer(args.files, args.device)
    elif args.command == 'tabpfn':
        tabpfn(args.files)
    elif args.command == 'pbe':
        pbe(args.files, args.folder)
    elif args.command == 'check-cvc5':
        check_cvc5()
    elif args.command == 'export':
        export(Path('site/src/data/sygus.json'))
    else:
        cvc5_held_out(args.files, args.folder)


if __name__ == '__main__':
    main()
