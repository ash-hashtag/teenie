// Random rules for the Try it page, written in the page's formula language (expression.ts) and generated on the fly:
// long chains, loops (x updated n times), sums, two-term recurrences and deep random expressions. Each one is checked
// to take many different values, so nothing trivially constant reaches the list.
import { evaluate, parse } from './expression';

export type Kind = 'chain' | 'loop' | 'sum' | 'recurrence' | 'deep';
export const KINDS: { kind: Kind; label: string }[] = [
  { kind: 'chain', label: 'Long chains' },
  { kind: 'loop', label: 'Loops' },
  { kind: 'sum', label: 'Sums' },
  { kind: 'recurrence', label: 'Recurrences' },
  { kind: 'deep', label: 'Deep expressions' },
];

const pick = <T>(items: readonly T[]): T => items[Math.floor(Math.random() * items.length)]!;
/** Like pick, with weights. */
function weighted<T>(items: readonly (readonly [T, number])[]): T {
  let roll = Math.random() * items.reduce((total, [, weight]) => total + weight, 0);
  for (const [item, weight] of items) {
    roll -= weight;
    if (roll <= 0) return item;
  }
  return items[items.length - 1]![0];
}
// Multiplication is the model's weak spot when its operands are computed, so random rules use it sparingly.
const BINARY = [['+', 4], ['-', 3], ['^', 4], ['&', 2], ['|', 2], ['*', 1]] as const;
const between = (low: number, high: number) => low + Math.floor(Math.random() * (high - low + 1));
const constant = () => (Math.random() < 0.8 ? String(between(2, 255)) : `0x${between(0x1000, 0xffff).toString(16).toUpperCase()}`);

/** A random expression with `ops` operations over the given leaves. */
function tree(ops: number, leaves: readonly string[]): string {
  if (ops === 0) return Math.random() < 0.15 ? constant() : pick(leaves);
  if (Math.random() < 0.15) {
    const inner = tree(ops - 1, leaves);
    return pick([`~${inner}`, `(${inner} >> ${between(1, 5)})`, `(${inner} << ${between(1, 3)})`]);
  }
  const left = between(0, ops - 1);
  const op = weighted(BINARY);
  return `(${tree(left, leaves)} ${op} ${tree(ops - 1 - left, leaves)})`;
}

const count = () => pick(['b & 15', 'b & 15', 'b & 7', 'b & 31', 'a & 15']);
const uses = (text: string, name: string) => new RegExp(`\\b${name}\\b`).test(text);
const strip = (text: string) => (text.startsWith('(') && text.endsWith(')') && balanced(text.slice(1, -1)) ? text.slice(1, -1) : text);
function balanced(text: string): boolean {
  let depth = 0;
  for (const c of text) {
    depth += c === '(' ? 1 : c === ')' ? -1 : 0;
    if (depth < 0) return false;
  }
  return depth === 0;
}

const STARTS = [['a + b', 3], ['a ^ b', 3], ['a - b', 3], ['a | b', 2], ['a & b', 2], ['max(a, b)', 1], ['min(a, b)', 1],
  ['a * b', 1]] as const;
const STEP_WEIGHTS = [3, 3, 3, 1, 2, 2, 2, 2, 2, 3, 1];
const STEPS: ((e: string) => string)[] = [
  (e) => `(${e} + ${constant()})`,
  (e) => `(${e} ^ b)`,
  (e) => `(${e} ^ a)`,
  (e) => `(${e} * ${pick(['3', '5', '7', '9'])})`,
  (e) => `(${e} >> ${between(1, 6)})`,
  (e) => `(${e} << ${between(1, 3)})`,
  (e) => `(${e} | a)`,
  (e) => `(${e} - b)`,
  (e) => `(${e} & ${constant()})`,
  (e) => `(${e} + a)`,
  (e) => `~${e}`,
];

function draft(kind: Kind): string {
  switch (kind) {
    case 'chain': {
      let e = `(${weighted(STARTS)})`;
      for (let k = between(4, 8); k > 0; k--) e = weighted(STEPS.map((step, i) => [step, STEP_WEIGHTS[i]!] as const))(e);
      return strip(e);
    }
    case 'loop': {
      let body = '';
      while (!uses(body, 'x')) body = tree(between(1, 3), ['x', 'x', 'x', 'i', 'a', 'b']);
      return `loop(${count()}, ${pick(['a', 'b', 'a ^ b', 'a + b', '1'])}, ${strip(body)})`;
    }
    case 'sum': {
      let body = '';
      while (!uses(body, 'i')) body = tree(between(1, 3), ['i', 'i', 'a', 'b']);
      return `sum(${strip(body)} for i in range(${count()}))`;
    }
    case 'recurrence': {
      let body = '';
      while (!uses(body, 'x') || !uses(body, 'y')) body = tree(between(1, 3), ['x', 'y', 'x', 'y', 'a', 'b', 'i']);
      return `rec(${count()}, ${pick(['a', 'b', '1'])}, ${pick(['a', 'b', '0', '1'])}, ${strip(body)})`;
    }
    case 'deep':
      return strip(tree(between(5, 9), ['a', 'b']));
  }
}

const PROBE = Array.from({ length: 48 }, () => [BigInt(Math.floor(Math.random() * 2 ** 32)), BigInt(Math.floor(Math.random() * 2 ** 32))] as const);

/** A random rule of this kind that takes at least 24 different values on 48 random 32-bit inputs. */
export function generate(kind: Kind): string {
  for (let attempt = 0; attempt < 60; attempt++) {
    const text = draft(kind);
    if (/\((\w) [-^&|] \1\)/.test(text)) continue; // x - x, a ^ a and the like cancel out
    try {
      const rule = parse(text);
      const values = new Set(PROBE.map(([a, b]) => evaluate(rule, a, b, 32)));
      if (values.size >= 24) return text;
    } catch {
      // a draw the parser rejects is simply skipped
    }
  }
  return draft(kind);
}
