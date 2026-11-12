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
// Weights measured on the release model (1,500 generated rules at 32 bits): rules with * were 11% exact against 51%
// without (products of computed values are its weak spot), shifts cost about 10 points, chains past 7 steps about 20,
// and loops counting to b & 7 did best. So * is rare, shifts are rare and mostly by 1, chains and bodies are short.
const BINARY = [['+', 4], ['-', 3], ['^', 4], ['&', 2], ['|', 2], ['*', 0.25]] as const;
const between = (low: number, high: number) => low + Math.floor(Math.random() * (high - low + 1));
/** A constant of random size: anywhere from 2 to 32 bits long, so some are small and some large. */
function constant(): string {
  const bits = between(2, 32);
  const value = Math.floor(2 ** (bits - 1) + Math.random() * 2 ** (bits - 1));
  return value < 10000 ? String(value) : `0x${value.toString(16).toUpperCase()}`;
}
/** Shifts beyond 1 rarely give the model anything to find, so shifts are rare and mostly by 1, both ways. */
const shift = () => weighted([[1, 8], [2, 1], [3, 1]] as const);

/** A random expression with `ops` operations over the given leaves. */
function tree(ops: number, leaves: readonly string[]): string {
  if (ops === 0) return Math.random() < 0.15 ? constant() : pick(leaves);
  if (Math.random() < 0.15) {
    const inner = tree(ops - 1, leaves);
    return weighted([[`~${inner}`, 4], [`(${inner} >> ${shift()})`, 0.5], [`(${inner} << ${shift()})`, 0.5]] as const);
  }
  const left = between(0, ops - 1);
  const op = weighted(BINARY);
  return `(${tree(left, leaves)} ${op} ${tree(ops - 1 - left, leaves)})`;
}

const count = () => weighted([['b & 7', 4], ['b & 15', 2], ['a & 7', 1], ['b & 3', 1]] as const);
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
const STEP_WEIGHTS = [3, 3, 3, 0.25, 0.5, 0.5, 2, 2, 2, 3, 1];
const STEPS: ((e: string) => string)[] = [
  (e) => `(${e} + ${constant()})`,
  (e) => `(${e} ^ b)`,
  (e) => `(${e} ^ a)`,
  (e) => `(${e} * ${pick(['3', '5', '7', '9'])})`,
  (e) => `(${e} >> ${shift()})`,
  (e) => `(${e} << ${shift()})`,
  (e) => `(${e} | a)`,
  (e) => `(${e} - b)`,
  (e) => `(${e} & ${constant()})`,
  (e) => `(${e} + a)`,
  (e) => `~${e}`,
];

function draft(kind: Kind): string {
  switch (kind) {
    case 'chain': {
      let e = `(${weighted(STARTS)})`; // a * b is a fine start: multiplying the inputs is what it does well
      let last = -1;
      for (let k = between(3, 6); k > 0; k--) {
        // the same xor twice in a row undoes itself, so it can't follow itself
        const choices = STEPS.map((_, i) => [i, i === last && (i === 1 || i === 2) ? 0 : STEP_WEIGHTS[i]!] as const);
        last = weighted(choices);
        e = STEPS[last]!(e);
      }
      return strip(e);
    }
    case 'loop': {
      let body = '';
      while (!uses(body, 'x')) body = tree(between(1, 2), ['x', 'x', 'x', 'i', 'a', 'b']);
      return `loop(${count()}, ${pick(['a', 'b', 'a ^ b', 'a + b', '1'])}, ${strip(body)})`;
    }
    case 'sum': {
      let body = '';
      while (!uses(body, 'i')) body = tree(between(1, 2), ['i', 'i', 'a', 'b']);
      return `sum(${strip(body)} for i in range(${count()}))`;
    }
    case 'recurrence': {
      let body = '';
      while (!uses(body, 'x') || !uses(body, 'y')) body = tree(between(1, 2), ['x', 'y', 'x', 'y', 'a', 'b', 'i']);
      return `rec(${count()}, ${pick(['a', 'b', '1'])}, ${pick(['a', 'b', '0', '1'])}, ${strip(body)})`;
    }
    case 'deep':
      return strip(tree(between(4, 7), ['a', 'b']));
  }
}

const MASK = (1n << 32n) - 1n;
const SIMPLE: ((a: bigint, b: bigint) => bigint)[] = [
  (a) => a, (_a, b) => b, (a) => ~a & MASK, (_a, b) => ~b & MASK, (a, b) => (a + b) & MASK, (a, b) => (a - b) & MASK,
  (a, b) => (b - a) & MASK, (a, b) => a ^ b, (a, b) => a & b, (a, b) => a | b,
];
/** True when the rule is really something trivial: it ignores a or b, or it's a constant (added or xored) away from a,
 * b, ~a, ~b, a+b, a-b, b-a, a^b, a&b or a|b. Such rules come out of random trees that cancel themselves. */
function collapses(values: bigint[], probe: readonly (readonly [bigint, bigint])[]): boolean {
  for (const simple of SIMPLE) {
    const base = probe.map(([a, b]) => simple(a, b));
    const plus = new Set(values.map((v, k) => (v - base[k]! + (1n << 32n)) & MASK));
    const xor = new Set(values.map((v, k) => v ^ base[k]!));
    if (plus.size === 1 || xor.size === 1) return true;
  }
  return false;
}
function ignores(rule: ReturnType<typeof parse>, which: 'a' | 'b'): boolean {
  return PROBE.slice(0, 16).every(([a, b]) => {
    const other = BigInt(Math.floor(Math.random() * 2 ** 32));
    return evaluate(rule, a, b, 32) === (which === 'a' ? evaluate(rule, other, b, 32) : evaluate(rule, a, other, 32));
  });
}

const PROBE = Array.from({ length: 48 }, () => [BigInt(Math.floor(Math.random() * 2 ** 32)), BigInt(Math.floor(Math.random() * 2 ** 32))] as const);

/** A random rule of this kind that takes at least 24 different values on 48 random 32-bit inputs. */
export function generate(kind: Kind): string {
  for (let attempt = 0; attempt < 60; attempt++) {
    const text = draft(kind);
    if (/\((\w) [-^&|] \1\)/.test(text)) continue; // x - x, a ^ a and the like cancel out
    try {
      const rule = parse(text);
      const outputs = PROBE.map(([a, b]) => evaluate(rule, a, b, 32));
      if (new Set(outputs).size >= 24 && !collapses(outputs, PROBE) && !ignores(rule, 'a') && !ignores(rule, 'b')) return text;
    } catch {
      // a draw the parser rejects is simply skipped
    }
  }
  return draft(kind);
}
