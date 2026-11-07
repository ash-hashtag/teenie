// The 999 examples drawn on the (a, b) plane, sized and shaded by how much the question attends to each.
import { token } from './tooltip';

export type PlaneInput = {
  points: [number, number, ...number[]][];
  /** Attention per example, 0..255 on a log scale: 0 = an even share (1/999) or less, 255 = 256x an even share. */
  weights: number[];
  query: [number, number];
  /** Bit position whose (a_k, b_k) bands are shaded; omit for none. */
  bit?: number;
};

const SIZE = 256;

export function drawPlane(canvas: HTMLCanvasElement, input: PlaneInput): void {
  const css = canvas.clientWidth || 300;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(css * ratio);
  canvas.height = Math.round(css * ratio);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(ratio, ratio);
  const scale = css / SIZE;
  const x = (a: number) => (a + 0.5) * scale;
  const y = (b: number) => css - (b + 0.5) * scale;

  ctx.fillStyle = token('--surface');
  ctx.fillRect(0, 0, css, css);

  const { bit, query } = input;
  if (bit !== undefined) {
    // Shade the columns whose a_k matches the question's a_k, and the rows whose b_k matches its b_k.
    ctx.fillStyle = token('--wash-1');
    const span = 1 << bit;
    const qa = (query[0] >> bit) & 1;
    const qb = (query[1] >> bit) & 1;
    for (let start = 0; start < SIZE; start += span) {
      if (((start >> bit) & 1) === qa) ctx.fillRect(start * scale, 0, span * scale, css);
      if (((start >> bit) & 1) === qb) ctx.fillRect(0, css - (start + span) * scale, css, span * scale);
    }
  }

  const order = input.weights.map((w, i) => [w, i] as const).sort((p, q) => p[0] - q[0]);
  const accent = token('--series-1');
  const muted = token('--muted');
  for (const [w, i] of order) {
    const [a, b] = input.points[i]!;
    const strength = w / 255;
    ctx.beginPath();
    if (strength < 0.05) {
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = muted;
      ctx.arc(x(a), y(b), 1.4, 0, Math.PI * 2);
    } else {
      ctx.globalAlpha = 0.3 + 0.7 * strength;
      ctx.fillStyle = accent;
      ctx.arc(x(a), y(b), 1.8 + 3.6 * Math.sqrt(strength), 0, Math.PI * 2);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = token('--series-2');
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x(query[0]), y(query[1]), 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x(query[0]) - 11, y(query[1]));
  ctx.lineTo(x(query[0]) + 11, y(query[1]));
  ctx.moveTo(x(query[0]), y(query[1]) - 11);
  ctx.lineTo(x(query[0]), y(query[1]) + 11);
  ctx.stroke();
}

/** Inverse of the export's log scale: how many times an even share of attention this example gets. */
export function evenShares(value: number): number {
  return Math.pow(256, value / 255);
}

/** Nearest example under the pointer (in canvas CSS pixels), or -1. */
export function hit(canvas: HTMLCanvasElement, points: [number, number, ...number[]][], px: number, py: number): number {
  const css = canvas.clientWidth;
  const scale = css / SIZE;
  let best = -1;
  let bestDistance = 64;
  points.forEach(([a, b], i) => {
    const dx = (a + 0.5) * scale - px;
    const dy = css - (b + 0.5) * scale - py;
    const d = dx * dx + dy * dy;
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  });
  return best;
}
