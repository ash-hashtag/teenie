export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined) return '—';
  const v = x * 100;
  return `${v === 100 || v === 0 ? v.toFixed(0) : v.toFixed(digits)}%`;
}

/** Most significant bit first, as people read binary. */
export function binary(x: number, width = 8): string {
  return x.toString(2).padStart(width, '0');
}

export function bit(x: number, k: number): number {
  return (x >> k) & 1;
}

export function thousands(x: number): string {
  return x.toLocaleString('en-US');
}
