// A small, safe evaluator for c = f(a, b) on unsigned `width`-bit integers (BigInt), wrapping like the training
// relations. Python-style syntax; parsed by hand, never passed to eval.
//
//   numbers 12, 0xff, 0b101; variables a, b
//   + - * // (or /) % **   & | ^ ~ << >>   == != < <= > >= (1 or 0)   not and or
//   x if cond else y   and   cond ? x : y
//   min(x, y)  max(x, y)  abs(x)  popcount(x)  reverse(x)  rotl(x, k)  rotr(x, k)
//   sum(f for i in range(n))      f summed over i = 0 .. n-1
//   loop(n, start, f)             x = start, then n times x = f (f may use x and the step i)
//   rec(n, x0, y0, f)             x, y = x0, y0, then n times x, y = f, x (f may use x, y, i); the answer is x
//
// Division by zero gives all ones and x % 0 gives x, as in training. A loop runs at most 255 times and loops can't be
// nested, so evaluating 1,015 examples stays fast.

type Node =
  | { kind: 'num'; value: bigint }
  | { kind: 'var'; name: Variable }
  | { kind: 'sum'; count: Node; body: Node }
  | { kind: 'loop'; count: Node; start: Node; body: Node }
  | { kind: 'rec'; count: Node; start: Node; second: Node; body: Node }
  | { kind: 'unary'; op: string; arg: Node }
  | { kind: 'binary'; op: string; left: Node; right: Node }
  | { kind: 'if'; cond: Node; then: Node; otherwise: Node }
  | { kind: 'call'; name: string; args: Node[] };

type Variable = 'a' | 'b' | 'i' | 'x' | 'y';
const MAX_STEPS = 255n;

const FUNCTIONS: Record<string, number> = { min: 2, max: 2, abs: 1, popcount: 1, reverse: 1, rotl: 2, rotr: 2 };

function tokenize(source: string): string[] {
  const tokens: string[] = [];
  const pattern = /\s*(0x[0-9a-fA-F]+|0b[01]+|\d+|[A-Za-z_]\w*|\*\*|\/\/|<<|>>|<=|>=|==|!=|[-+*\/%&|^~()<>,?:])/y;
  let index = 0;
  while (index < source.length) {
    if (/^\s*$/.test(source.slice(index))) break;
    pattern.lastIndex = index;
    const match = pattern.exec(source);
    if (!match) throw new Error(`Unexpected "${source.slice(index).trim()[0]}" at position ${index + 1}`);
    tokens.push(match[1]!);
    index = pattern.lastIndex;
  }
  return tokens;
}

export function parse(source: string): Node {
  const tokens = tokenize(source);
  let i = 0;
  let scope: Set<string> = new Set(['a', 'b']); // the loop variables become usable inside a loop's body
  const peek = () => tokens[i];
  const take = (expected?: string) => {
    const token = tokens[i++];
    if (token === undefined || (expected !== undefined && token !== expected)) {
      throw new Error(expected ? `Expected "${expected}"${token ? ` but found "${token}"` : ' at the end'}` : 'Unexpected end of the formula');
    }
    return token;
  };

  // Lowest to highest: x if c else y / c ? x : y, or, and, not, comparisons, |, ^, &, shifts, + -, * / // %,
  // unary - ~, **.
  function conditional(): Node {
    const value = orExpr();
    if (peek() === 'if') {
      take();
      const cond = orExpr();
      take('else');
      return { kind: 'if', cond, then: value, otherwise: conditional() };
    }
    if (peek() === '?') {
      take();
      const then = conditional();
      take(':');
      return { kind: 'if', cond: value, then, otherwise: conditional() };
    }
    return value;
  }
  function chain(next: () => Node, ops: string[]): () => Node {
    return () => {
      let left = next();
      while (ops.includes(peek() ?? '')) {
        const op = take();
        left = { kind: 'binary', op, left, right: next() };
      }
      return left;
    };
  }
  function notExpr(): Node {
    if (peek() === 'not') {
      take();
      return { kind: 'unary', op: 'not', arg: notExpr() };
    }
    return comparison();
  }
  function unary(): Node {
    const token = peek();
    if (token === '-' || token === '~' || token === '+') {
      take();
      return { kind: 'unary', op: token, arg: unary() };
    }
    return power();
  }
  function power(): Node {
    const base = atom();
    if (peek() === '**') {
      take();
      return { kind: 'binary', op: '**', left: base, right: unary() };
    }
    return base;
  }
  function atom(): Node {
    const token = take();
    if (token === '(') {
      const inner = conditional();
      take(')');
      return inner;
    }
    if (/^(0x|0b)?\d|^0[xb]/.test(token)) return { kind: 'num', value: BigInt(token) };
    if (scope.has(token)) return { kind: 'var', name: token as Variable };
    if (token === 'sum' || token === 'loop' || token === 'rec') return loopForm(token);
    if (token === 'i' || token === 'x' || token === 'y') {
      throw new Error(`"${token}" only exists inside sum(...), loop(...) or rec(...)`);
    }
    if (token in FUNCTIONS) {
      take('(');
      const args = [conditional()];
      while (peek() === ',') {
        take();
        args.push(conditional());
      }
      take(')');
      if (args.length !== FUNCTIONS[token]) throw new Error(`${token}() takes ${FUNCTIONS[token]} argument${FUNCTIONS[token] === 1 ? '' : 's'}`);
      return { kind: 'call', name: token, args };
    }
    throw new Error(`Unknown name "${token}": use a, b, numbers and ${Object.keys(FUNCTIONS).join(', ')}`);
  }
  function inside<T>(names: string[], parse: () => T): T {
    if (scope.size > 2) throw new Error('Loops can\'t be nested');
    const outer = scope;
    scope = new Set([...outer, ...names]);
    try {
      return parse();
    } finally {
      scope = outer;
    }
  }
  function loopForm(name: 'sum' | 'loop' | 'rec'): Node {
    if (scope.size > 2) throw new Error('Loops can\'t be nested');
    take('(');
    if (name === 'sum') {
      const body = inside(['i'], conditional);
      take('for');
      take('i');
      take('in');
      take('range');
      take('(');
      const count = conditional();
      take(')');
      take(')');
      return { kind: 'sum', count, body };
    }
    const count = conditional();
    take(',');
    const start = conditional();
    take(',');
    if (name === 'loop') {
      const body = inside(['x', 'i'], conditional);
      take(')');
      return { kind: 'loop', count, start, body };
    }
    const second = conditional();
    take(',');
    const body = inside(['x', 'y', 'i'], conditional);
    take(')');
    return { kind: 'rec', count, start, second, body };
  }
  const bitOps = chain(chain(chain(chain(chain(chain(unary, ['*', '/', '//', '%']), ['+', '-']), ['<<', '>>']), ['&']), ['^']), ['|']);
  const comparison = chain(bitOps, ['==', '!=', '<', '<=', '>', '>=']);
  const andExpr = chain(notExpr, ['and']);
  const orExpr = chain(andExpr, ['or']);

  if (tokens.length === 0) throw new Error('Write a formula in a and b');
  const tree = conditional();
  if (i < tokens.length) throw new Error(`Unexpected "${tokens[i]}"`);
  if (!usesVariable(tree)) throw new Error('The formula must use a or b');
  return tree;
}

function usesVariable(node: Node): boolean {
  switch (node.kind) {
    case 'var': return node.name === 'a' || node.name === 'b';
    case 'sum': return usesVariable(node.count) || usesVariable(node.body);
    case 'loop': return usesVariable(node.count) || usesVariable(node.start) || usesVariable(node.body);
    case 'rec': return [node.count, node.start, node.second, node.body].some(usesVariable);
    case 'num': return false;
    case 'unary': return usesVariable(node.arg);
    case 'binary': return usesVariable(node.left) || usesVariable(node.right);
    case 'if': return usesVariable(node.cond) || usesVariable(node.then) || usesVariable(node.otherwise);
    case 'call': return node.args.some(usesVariable);
  }
}

function popcount(x: bigint): bigint {
  let count = 0n;
  for (; x > 0n; x >>= 1n) count += x & 1n;
  return count;
}

export function evaluate(node: Node, a: bigint, b: bigint, width: number): bigint {
  const mask = (1n << BigInt(width)) - 1n;
  const w = BigInt(width);
  const wrap = (x: bigint) => ((x % (mask + 1n)) + (mask + 1n)) & mask;
  const rotl = (x: bigint, k: bigint) => {
    const r = ((k % w) + w) % w;
    return wrap((x << r) | (x >> ((w - r) % w)));
  };
  const env: Record<Variable, bigint> = { a, b, i: 0n, x: 0n, y: 0n };
  const steps = (count: Node) => {
    const n = run(count);
    return n > MAX_STEPS ? MAX_STEPS : n;
  };
  const run = (n: Node): bigint => {
    switch (n.kind) {
      case 'num': return wrap(n.value);
      case 'var': return env[n.name];
      case 'sum': {
        let total = 0n;
        for (let k = 0n, count = steps(n.count); k < count; k++) {
          env.i = k;
          total = wrap(total + run(n.body));
        }
        return total;
      }
      case 'loop': {
        const count = steps(n.count);
        env.x = run(n.start);
        for (let k = 0n; k < count; k++) {
          env.i = k;
          env.x = run(n.body);
        }
        return env.x;
      }
      case 'rec': {
        const count = steps(n.count);
        env.x = run(n.start);
        env.y = run(n.second);
        for (let k = 0n; k < count; k++) {
          env.i = k;
          const next = run(n.body);
          env.y = env.x;
          env.x = next;
        }
        return env.x;
      }
      case 'if': return run(n.cond) !== 0n ? run(n.then) : run(n.otherwise);
      case 'unary': {
        const x = run(n.arg);
        return n.op === '-' ? wrap(-x) : n.op === '~' ? wrap(~x) : n.op === 'not' ? (x === 0n ? 1n : 0n) : x;
      }
      case 'call': {
        const [x, y] = n.args.map(run) as [bigint, bigint];
        switch (n.name) {
          case 'min': return x < y ? x : y;
          case 'max': return x > y ? x : y;
          case 'abs': return x; // values are unsigned
          case 'popcount': return popcount(x);
          case 'reverse': {
            let out = 0n;
            for (let k = 0n; k < w; k++) out |= ((x >> k) & 1n) << (w - 1n - k);
            return out;
          }
          case 'rotl': return rotl(x, y);
          case 'rotr': return rotl(x, w - (y % w));
        }
        throw new Error(n.name);
      }
      case 'binary': {
        if (n.op === 'and') return run(n.left) !== 0n && run(n.right) !== 0n ? 1n : 0n;
        if (n.op === 'or') return run(n.left) !== 0n || run(n.right) !== 0n ? 1n : 0n;
        const x = run(n.left);
        const y = run(n.right);
        switch (n.op) {
          case '+': return wrap(x + y);
          case '-': return wrap(x - y);
          case '*': return wrap(x * y);
          case '/': case '//': return y === 0n ? mask : x / y;
          case '%': return y === 0n ? x : x % y;
          case '**': {
            let result = 1n;
            let base = x;
            for (let e = y; e > 0n; e >>= 1n) {
              if (e & 1n) result = wrap(result * base);
              base = wrap(base * base);
            }
            return result;
          }
          case '&': return x & y;
          case '|': return x | y;
          case '^': return x ^ y;
          case '<<': return y >= w ? 0n : wrap(x << y);
          case '>>': return y >= w ? 0n : x >> y;
          case '==': return x === y ? 1n : 0n;
          case '!=': return x !== y ? 1n : 0n;
          case '<': return x < y ? 1n : 0n;
          case '<=': return x <= y ? 1n : 0n;
          case '>': return x > y ? 1n : 0n;
          case '>=': return x >= y ? 1n : 0n;
        }
        throw new Error(n.op);
      }
    }
  };
  return run(node);
}
