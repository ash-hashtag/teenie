// One shared tooltip: any element with `data-tip` shows it on hover or keyboard focus.
export function installTooltips(): void {
  const tip = document.getElementById('tooltip');
  if (!tip) return;
  let active: Element | null = null;

  const place = (x: number, y: number): void => {
    const pad = 14;
    const { width, height } = tip.getBoundingClientRect();
    const left = Math.min(x + pad, window.innerWidth - width - 8);
    const top = y + pad + height > window.innerHeight ? y - height - pad : y + pad;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  };

  const show = (target: Element, x: number, y: number): void => {
    const text = target.getAttribute('data-tip');
    if (!text) return;
    active = target;
    tip.textContent = text;
    tip.classList.add('on');
    place(x, y);
  };

  const hide = (): void => {
    active = null;
    tip.classList.remove('on');
  };

  document.addEventListener('pointermove', (event) => {
    const target = (event.target as Element | null)?.closest('[data-tip]');
    if (!target) {
      if (active) hide();
      return;
    }
    if (target !== active) show(target, event.clientX, event.clientY);
    else {
      // Canvases rewrite their data-tip as the pointer moves.
      const text = target.getAttribute('data-tip') ?? '';
      if (tip.textContent !== text) tip.textContent = text;
      place(event.clientX, event.clientY);
    }
  });
  document.addEventListener('focusin', (event) => {
    const target = (event.target as Element | null)?.closest('[data-tip]');
    if (!target) return;
    const box = target.getBoundingClientRect();
    show(target, box.left + box.width / 2, box.bottom);
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('scroll', hide, { passive: true });
}

/** Reads a CSS custom property from :root (re-read after a theme change). */
export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
