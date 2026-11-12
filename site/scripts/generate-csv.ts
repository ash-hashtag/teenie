// Make a CSV for the Try page's "Use your own data": 999 examples of a rule, then questions.
//
//   bun scripts/generate-csv.ts 'a * b + a'                   999 examples + 200 questions at 32 bits, to stdout
//   bun scripts/generate-csv.ts 'a * b + a' -o rule.csv       to a file
//   bun scripts/generate-csv.ts '(a ^ b) * 3' --width 16 --small --questions 1000 --blank
//
// Options:
//   --width 16|32     number width; values wrap at it after every step, as on the page (default 32)
//   --small           a and b under 256
//   --questions N     question rows after the 999 examples (default 200)
//   --blank           leave c empty on the question rows (otherwise filled in, so the page can check its answers)
//   -o, --output F    write to F instead of stdout
//
// The formula language is the page's own (src/lib/expression.ts): + - * // % **, & | ^ ~ << >>, comparisons,
// x if cond else y, min max popcount reverse rotl rotr, sum/loop/rec. Every (a, b) pair in the file is distinct.
import { writeFileSync } from "node:fs";
import { evaluate, parse } from "../src/lib/expression";

const CONTEXT = 999;

function usage(message?: string): never {
  if (message) console.error(`error: ${message}`);
  console.error(
    "usage: bun scripts/generate-csv.ts '<formula>' [--width 16|32] [--small] [--questions N] [--blank] [-o file]",
  );
  process.exit(message ? 1 : 0);
}

const args = process.argv.slice(2);
let formula: string | undefined;
let width = 32;
let small = false;
let questions = 200;
let blank = false;
let output: string | undefined;
for (let i = 0; i < args.length; i++) {
  const arg = args[i]!;
  const value = () => args[++i] ?? usage(`${arg} needs a value`);
  if (arg === "-h" || arg === "--help") usage();
  else if (arg === "--width") width = Number(value());
  else if (arg === "--small") small = true;
  else if (arg === "--questions") questions = Number(value());
  else if (arg === "--blank") blank = true;
  else if (arg === "-o" || arg === "--output") output = value();
  else if (formula === undefined) formula = arg;
  else usage(`unexpected argument ${arg}`);
}
if (formula === undefined) usage("give a formula, e.g. 'a * b + a'");
if (width !== 16 && width !== 32) usage("--width must be 16 or 32");
if (!Number.isInteger(questions) || questions < 1)
  usage("--questions must be a whole number of 1 or more");

let tree;
try {
  tree = parse(formula);
} catch (error) {
  usage(`the formula has a mistake: ${(error as Error).message}`);
}

const inputBits = small ? 8 : width;
// Two random 32-bit words, cut down to their top inputBits bits (>>> keeps them unsigned).
const pair = (): [bigint, bigint] => {
  const [a, b] = crypto.getRandomValues(new Uint32Array(2));
  return [BigInt(a! >>> (32 - inputBits)), BigInt(b! >>> (32 - inputBits))];
};

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
  const c = evaluate(tree, a, b, width);
  const example = lines.length <= CONTEXT;
  if (example) answers.add(String(c));
  lines.push(example || !blank ? `${a},${b},${c}` : `${a},${b},`);
}
const text = `${lines.join("\n")}\n`;
if (output) writeFileSync(output, text);
else process.stdout.write(text);
console.error(
  `c = ${formula} at ${width} bits${small ? ", a and b under 256" : ""}: ${CONTEXT} examples (${answers.size} different c), ` +
    `${questions} questions${blank ? " with c left empty" : " with c filled in"}${output ? ` → ${output}` : ""}`,
);
if (answers.size < 24)
  console.error(
    "note: few different answers among the examples; the rule may ignore most of its input",
  );
