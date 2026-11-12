// Make a CSV for the Try page's "Use your own data": 999 examples of a rule, then questions with c left empty; or
// check the page's downloaded answers against the rule.
//
//   bun scripts/generate-csv.ts 'a * b + a'                   999 examples + 200 questions at 32 bits, to stdout
//   bun scripts/generate-csv.ts 'a * b + a' -o rule.csv       to a file
//   bun scripts/generate-csv.ts '(a ^ b) * 3' --width 16 --small --questions 1000
//   bun scripts/generate-csv.ts 'a * b + a' --check rule-answers.csv    how accurate the downloaded answers are
//
// Options:
//   --width 16|32     number width; values wrap at it after every step, as on the page (default 32)
//   --small           a and b under 256
//   --questions N     question rows after the 999 examples (default 200)
//   --seed N          the same file every time for the same N (default: a new random file each run)
//   -o, --output F    write to F instead of stdout
//   --check F         read F, the answers downloaded from the page, and report how many are right. The width comes from
//                     its p_bit columns (or --width). With -o, also writes each question with the true c and its verdict.
//
// The formula language is the page's own (src/lib/expression.ts): + - * // % **, & | ^ ~ << >>, comparisons,
// x if cond else y, min max popcount reverse rotl rotr, sum/loop/rec. Every (a, b) pair in the file is distinct.
import { readFileSync, writeFileSync } from "node:fs";
import { evaluate, parse } from "../src/lib/expression";

const CONTEXT = 999;

function usage(message?: string): never {
  if (message) console.error(`error: ${message}`);
  console.error(
    "usage: bun scripts/generate-csv.ts '<formula>' [--width 16|32] [--small] [--questions N] [--seed N] [-o file]\n" +
      "       bun scripts/generate-csv.ts '<formula>' --check answers.csv [--width 16|32] [-o checked.csv]",
  );
  process.exit(message ? 1 : 0);
}

const args = process.argv.slice(2);
let formula: string | undefined;
let width = 32;
let small = false;
let questions = 200;
let seed: number | undefined;
let output: string | undefined;
let check: string | undefined;
let widthGiven = false;
for (let i = 0; i < args.length; i++) {
  const arg = args[i]!;
  const value = () => args[++i] ?? usage(`${arg} needs a value`);
  if (arg === "-h" || arg === "--help") usage();
  else if (arg === "--width") {
    width = Number(value());
    widthGiven = true;
  } else if (arg === "--check") check = value();
  else if (arg === "--small") small = true;
  else if (arg === "--questions") questions = Number(value());
  else if (arg === "--seed") seed = Number(value());
  else if (arg === "-o" || arg === "--output") output = value();
  else if (formula === undefined) formula = arg;
  else usage(`unexpected argument ${arg}`);
}
if (formula === undefined) usage("give a formula, e.g. 'a * b + a'");
if (width !== 16 && width !== 32) usage("--width must be 16 or 32");
if (!Number.isInteger(questions) || questions < 1)
  usage("--questions must be a whole number of 1 or more");
if (seed !== undefined && !Number.isSafeInteger(seed))
  usage("--seed must be a whole number");

let tree: ReturnType<typeof parse>;
try {
  tree = parse(formula);
} catch (error) {
  usage(`the formula has a mistake: ${(error as Error).message}`);
}
if (check !== undefined) {
  verify(check);
  process.exit(0);
}

const inputBits = small ? 8 : width;
// Uniform random 32-bit words: from crypto, or with --seed from SplitMix32, which steps a counter by a fixed odd
// number and scrambles it with a bijection, so over its 2^32 period it gives every 32-bit value exactly once.
let state = (seed ?? 0) >>> 0;
const word = (): number => {
  if (seed === undefined) return crypto.getRandomValues(new Uint32Array(1))[0]!;
  state = (state + 0x9e3779b9) >>> 0;
  let z = state;
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
  return (z ^ (z >>> 16)) >>> 0;
};
// a and b: the top inputBits bits of a word each (>>> keeps them unsigned).
const pair = (): [bigint, bigint] => [
  BigInt(word() >>> (32 - inputBits)),
  BigInt(word() >>> (32 - inputBits)),
];

const total = CONTEXT + questions;
if (total > 2 ** (2 * inputBits))
  usage(
    `only ${2 ** (2 * inputBits)} distinct (a, b) pairs exist; ask fewer questions`,
  );
const seen = new Set<string>();
const lines = ["a,b,c"];
const answers = new Set<string>();
while (lines.length <= total) {
  const [a, b] = pair();
  const key = `${a},${b}`;
  if (seen.has(key)) continue;
  seen.add(key);
  if (lines.length > CONTEXT) {
    lines.push(`${a},${b},`); // a question: c left empty for the model to answer
    continue;
  }
  const c = evaluate(tree, a, b, width);
  answers.add(String(c));
  lines.push(`${a},${b},${c}`);
}
const text = `${lines.join("\n")}\n`;
if (output) writeFileSync(output, text);
else process.stdout.write(text);
console.error(
  `c = ${formula} at ${width} bits${small ? ", a and b under 256" : ""}: ${CONTEXT} examples (${answers.size} different c), ` +
    `${questions} questions with c left empty${output ? ` → ${output}` : ""}`,
);
if (answers.size < 24)
  console.error(
    "note: few different answers among the examples; the rule may ignore most of its input",
  );

// ---- --check: the page's downloaded answers (line, a, b, [c], answer, confidence, sure, ..., p_bitN) against the rule.
function verify(path: string): void {
  const rows = readFileSync(path, "utf8").split(/\r?\n/).filter((line) => line.trim());
  const head = rows[0]?.split(",").map((cell) => cell.trim().toLowerCase()) ?? [];
  const col = (name: string) => head.indexOf(name);
  const [ia, ib, ianswer] = [col("a"), col("b"), col("answer")];
  if (ia < 0 || ib < 0 || ianswer < 0)
    usage(`${path} needs columns a, b and answer (the page's "Download the answers" file)`);
  const bitColumns = new Map<number, number>(); // bit -> column
  head.forEach((name, i) => {
    if (/^p_bit\d+$/.test(name)) bitColumns.set(Number(name.slice(5)), i);
  });
  if (bitColumns.size && widthGiven && bitColumns.size !== width)
    usage(`${path} has ${bitColumns.size} p_bit columns, so it is ${bitColumns.size}-bit, not ${width}-bit`);
  const w = bitColumns.size || width;
  const mask = (1n << BigInt(w)) - 1n;
  const [iline, iconfidence, ic] = [col("line"), col("confidence"), col("c")];

  const bands = [
    { name: "0.99 and above (sure)", low: 0.99, n: 0, right: 0 },
    { name: "0.9 to 0.99", low: 0.9, n: 0, right: 0 },
    { name: "0.7 to 0.9", low: 0.7, n: 0, right: 0 },
    { name: "below 0.7", low: -1, n: 0, right: 0 },
  ];
  let n = 0;
  let exact = 0;
  let bitsRight = 0;
  let recovered = 0;
  let disagreeC = 0;
  const misses: { line: string; a: bigint; b: bigint; truth: bigint; answer: bigint; off: number }[] = [];
  const out = ["line,a,b,c,answer,correct,bits_off"];
  for (let r = 1; r < rows.length; r++) {
    const cells = rows[r]!.split(",").map((cell) => cell.trim());
    const number = (i: number, name: string) => {
      if (!/^\d+$/.test(cells[i] ?? "")) usage(`${path} line ${r + 1}: ${name} = "${cells[i] ?? ""}" is not a whole number`);
      return BigInt(cells[i]!);
    };
    const [a, b, answer] = [number(ia, "a"), number(ib, "b"), number(ianswer, "answer")];
    const truth = evaluate(tree, a, b, w);
    if (ic >= 0 && cells[ic] && BigInt(cells[ic]!) !== truth) disagreeC++;
    const off = ((truth ^ answer) & mask).toString(2).split("").filter((d) => d === "1").length;
    const right = off === 0;
    n++;
    if (right) exact++;
    bitsRight += w - off;
    if (bitColumns.size) {
      let score = w;
      for (const [k, i] of bitColumns) {
        const one = Number(cells[i]);
        score += Math.log2(Math.min(Math.max((truth >> BigInt(k)) & 1n ? one : 1 - one, 2 ** -20), 1 - 2 ** -20));
      }
      recovered += score;
    }
    if (iconfidence >= 0 && cells[iconfidence]) {
      const band = bands.find((x) => Number(cells[iconfidence]) >= x.low)!;
      band.n++;
      if (right) band.right++;
    }
    const line = iline >= 0 ? cells[iline]! : String(r + 1);
    if (!right) misses.push({ line, a, b, truth, answer, off });
    out.push(`${line},${a},${b},${truth},${answer},${right ? 1 : 0},${off}`);
  }
  if (!n) usage(`${path} has no answers`);

  const pct = (x: number, of: number) => `${((100 * x) / of).toFixed(1)}%`;
  console.log(`c = ${formula} at ${w} bits: ${n.toLocaleString("en-US")} answers in ${path}`);
  console.log(`  exactly right   ${exact.toLocaleString("en-US")} of ${n.toLocaleString("en-US")} (${pct(exact, n)})`);
  console.log(`  bits right      ${pct(bitsRight, n * w)}`);
  if (bitColumns.size)
    console.log(`  bits recovered  ${(recovered / n).toFixed(1)} of ${w} on average (${w} = certain and right, 0 = a blind guess)`);
  if (bands.some((x) => x.n)) {
    console.log("  by confidence (its least certain bit):");
    for (const x of bands)
      if (x.n) console.log(`    ${x.name.padEnd(22)} ${String(x.n).padStart(7)} answers, ${pct(x.right, x.n).padStart(6)} right`);
  }
  if (disagreeC)
    console.log(`  note: ${disagreeC} rows have a c column that differs from this formula; is it the same rule and width?`);
  if (misses.length) {
    console.log(`  most bits off (of ${misses.length.toLocaleString("en-US")} wrong):`);
    for (const m of misses.sort((x, y) => y.off - x.off).slice(0, 5))
      console.log(`    line ${m.line}: a = ${m.a}, b = ${m.b}, true ${m.truth}, answered ${m.answer} (${m.off} bits off)`);
  }
  if (output) {
    writeFileSync(output, `${out.join("\n")}\n`);
    console.log(`  each question with its true c → ${output}`);
  }
}
