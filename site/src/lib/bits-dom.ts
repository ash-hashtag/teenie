// Client-side twin of BitStrip.astro (uses the same global classes).
export type StripOptions = {
  label?: string;
  highlight?: number;
  probs?: number[];
  unknown?: boolean;
  width?: number;
  wrong?: boolean;
};

export function strip(value: number, options: StripOptions = {}): string {
  const { label, highlight, probs, unknown = false, width = 8, wrong = false } = options;
  const cells: string[] = [];
  for (let k = width - 1; k >= 0; k--) {
    const on = (value >> k) & 1;
    // NaN in `probs` marks a bit not decided yet.
    const raw = probs?.[k];
    const pending = raw !== undefined && Number.isNaN(raw);
    const p = pending ? undefined : raw;
    const classes = ['cell'];
    if (!unknown && !pending && p === undefined && on) classes.push('on');
    if (p !== undefined) classes.push('soft');
    if (unknown || pending) classes.push('unknown');
    if (highlight === k) classes.push('hi');
    const style = p === undefined ? '' : ` style="--p:${p}"`;
    const tip = p === undefined ? '' : ` data-tip="bit ${k}: p(1) = ${p.toFixed(3)}"`;
    cells.push(`<span class="${classes.join(' ')}"${style}${tip}>${unknown || pending ? '?' : p === undefined ? on : ''}</span>`);
  }
  const decimal = unknown || probs ? '' : `<span class="dec num${wrong ? ' wrong' : ''}">${value}</span>`;
  return `<div class="strip">${label ? `<span class="label">${label}</span>` : ''}<span class="cells">${cells.join('')}</span>${decimal}</div>`;
}
