// A small, safe evaluator for c = f(a, b) on unsigned `width`-bit integers (BigInt), wrapping like the training
// relations. Python-style syntax; parsed by hand, never passed to eval.
//
//   numbers 12, 0xff, 0b101; variables a, b
//   + - * // (or /) % **   & | ^ ~ << >>   == != < <= > >= (1 or 0)   not and or
//   x if cond else y   and   cond ? x : y
//   min(x, y)  max(x, y)  abs(x)  popcount(x)  reverse(x)  rotl(x, k)  rotr(x, k)
//
// Division by zero gives all ones and x % 0 gives x, as in training.

type Node =
  | { kind: 'num'; value: bigint }
  | { kind: 'var'; name: 'a' | 'b' }
  | { kind: 'unary'; op: string; arg: Node }
  | { kind: 'binary'; op: string; left: Node; right: Node }
  | { kind: 'if'; cond: Node; then: Node; otherwise: Node }
  | { kind: 'call'; name: string; args: Node[] };

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
    if (token === 'a' || token === 'b') return { kind: 'var', name: token };
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
    case 'var': return true;
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
  const run = (n: Node): bigint => {
    switch (n.kind) {
      case 'num': return wrap(n.value);
      case 'var': return n.name === 'a' ? a : b;
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
