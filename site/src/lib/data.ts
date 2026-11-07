// Shapes of the JSON written by `uv run python -m teenie.bits.export`.
import evaluationJson from '../data/evaluation.json';
import modelJson from '../data/model.json';
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
