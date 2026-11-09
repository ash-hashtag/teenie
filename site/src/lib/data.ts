// Shapes of the JSON written by `uv run python -m teenie.bits.export`.
import evaluationJson from '../data/evaluation.json';
import modelJson from '../data/model.json';
import currentJson from '../data/current.json';
import progressJson from '../data/progress.json';
import trainingJson from '../data/training.json';

export type Report = {
  exact?: number | null;
  bit?: number | null;
  sure_coverage?: number | null;
  sure_accuracy?: number | null;
  nearest?: number | null;
  window1L?: number | null;
  window1H?: number | null;
  window2L?: number | null;
  window2H?: number | null;
};

export type Group = 'train' | 'heldout' | 'stress';

export type Evaluation = {
  checkpoint: string;
  width: number;
  episodes: number;
  queries: number;
  seed: number;
  sure_threshold: number;
  results: Record<Group, Record<string, Report>>;
};

export type TrainingRecord = {
  update: number;
  seconds: number;
  loss: number;
  known_exact: number;
  heldout_exact: number;
  exact: Record<string, number>;
};

export type ModelSummary = {
  parameters: number;
  config: { hidden: number; heads: number; max_width: number; probes: number };
  metadata: Record<string, unknown>;
  pass_trust: Record<string, number>;
  families: Record<Group, string[]>;
};

export type TraceBit = {
  k: number;
  p: number;
  vote: number;
  attention: number[];
  top: { i: number; w: number }[];
};

export type TraceExample = {
  a: number;
  b: number;
  target: number;
  prediction: number;
  confidence: number;
  combined: number[];
  passes: { label: string; bits: TraceBit[] }[];
};

export type Trace = {
  relation: string;
  heldout: boolean;
  stress: boolean;
  trust: number[];
  context: [number, number, number][];
  examples: TraceExample[];
};

export type TimelineSnapshot = {
  update: number;
  exact: number;
  combined: number[];
  prediction: number;
  trust: number[];
  attention: number[];
};

export type Timeline = {
  bit: number;
  relations: Record<
    string,
    { a: number; b: number; target: number; context: [number, number][]; snapshots: TimelineSnapshot[] }
  >;
};

export const evaluation = evaluationJson as Evaluation;
export const model = modelJson as ModelSummary;
export const training = trainingJson as TrainingRecord[];

export const WIDTH = 8;

/** Best hand-written window lookup on the same episodes (any window/direction). */
export function bestWindow(r: Report): number {
  return Math.max(r.window1L ?? 0, r.window1H ?? 0, r.window2L ?? 0, r.window2H ?? 0);
}

export function dataUrl(file: string): string {
  return `${import.meta.env.BASE_URL.replace(/\/$/, '')}/data/${file}`;
}

// Written by `uv run python -m teenie.bits.export_progress`: every experiment so far.
export type GroupSummary = { exact: number | null; sure_accuracy: number | null; won: boolean };

export type CurvePoint = { update: number; known: number } & Record<string, number>;

export type Experiment = {
  id: string;
  chapter: string;
  title: string;
  train_set: string | null;
  tests: string | null;
  groups: Record<string, GroupSummary>;
  highlights: Record<string, number | null>;
  families: Record<string, { exact: number | null; bit: number | null; sure_coverage: number | null; sure_accuracy: number | null }>;
  curve: CurvePoint[];
};

export type Architecture = {
  title: string;
  checkpoint: string;
  hidden: number;
  heads: number;
  layers: number;
  rounds: number;
  width: number;
  parameters: number;
  parts: { part: string; detail: string; parameters: number }[];
};

export type WideWidth = {
  groups: Record<string, GroupSummary>;
  families: Record<string, number | null>;
  lookup: Record<string, number>;
  sealed?: GroupSummary;
  sealed_lookup?: number | null;
  chains?: Record<string, { exact: number; beyond_lookup: number; questions: number; lookup_fails: number; sure_coverage: number; sure_accuracy: number | null }>;
};

export type Progress = {
  ladder: string[];
  tiers: Record<string, string[]>;
  experiments: Experiment[];
  architectures: Record<string, Architecture>;
  wide: { checkpoint: string; parameters: number | null; widths: Record<string, WideWidth> };
  current: {
    checkpoint: string;
    update: number | null;
    hidden: number;
    heads: number;
    layers: number;
    rounds: number;
    span: number;
    registers: number;
    worked_examples: number;
    parameters: number;
    parts: { part: string; detail: string; parameters: number }[];
  } | null;
  products: { stage: string; summary: Record<string, { exact: number; sure_coverage?: number; sure_accuracy?: number | null }> }[];
};

export const progress = progressJson as unknown as Progress;

export function experimentById(id: string): Experiment | undefined {
  return progress.experiments.find((e) => e.id === id);
}

/** Runs evaluated on the difficulty ladder, oldest first. */
export function ladderRuns(): Experiment[] {
  return progress.experiments.filter((e) => 'T1' in e.groups);
}

export const TIER_LABELS: Record<string, string> = {
  T1: 'T1 · 2 operations',
  T2: 'T2 · 3–4 operations',
  T3: 'T3 · 5–7 operations',
  T4: 'T4 · new operation types',
  target: 'target · unseen multi-step expressions',
  T5a: 'T5a · unseen loop bodies',
  T5b: 'T5b · longer loops (stretch)',
  hidden: 'hidden compositions',
  near: 'product, then a step',
};

// Written by `uv run python -m teenie.bits.export_current`: the model the main pages describe.
export type Current = {
  name: string;
  checkpoint: string;
  architecture: {
    kind?: 'loop' | 'bittoken';
    hidden: number;
    heads: number;
    layers: number;
    rounds: number;
    probes: number;
    width: number;
    multiplicative: boolean;
    scratch: number;
    parameters: number;
    parts: { part: string; detail: string; parameters: number }[];
    train_set: string | null;
    updates: number | null;
    adaptive: boolean | null;
  };
  summary: Record<string, GroupSummary>;
  families: Record<string, number | null>;
};

export const current = currentJson as unknown as Current;
