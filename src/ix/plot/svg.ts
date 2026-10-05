import { NDArray } from '../core/ndarray.js';

/**
 * Generates a lightweight standalone SVG for inline previews in browsers and editors.
 */
export function plotSVG(
  x: NDArray | number[],
  y: NDArray | number[],
  opts: { width?: number; height?: number; color?: string; strokeWidth?: number } = {}
): string {
  const width = opts.width ?? 500;
  const height = opts.height ?? 250;
  const color = opts.color ?? '#2563eb';
  const strokeWidth = opts.strokeWidth ?? 2;
  const pad = 30;

  const xs = x instanceof NDArray ? x.data : new Float64Array(x);
  const ys = y instanceof NDArray ? y.data : new Float64Array(y);
  const n = Math.min(xs.length, ys.length);

  if (n < 2) return `<svg width="${width}" height="${height}"></svg>`;

  let minX = xs[0], maxX = xs[0];
  let minY = ys[0], maxY = ys[0];
  for (let i = 1; i < n; i++) {
    if (xs[i] < minX) minX = xs[i];
    if (xs[i] > maxX) maxX = xs[i];
    if (ys[i] < minY) minY = ys[i];
    if (ys[i] > maxY) maxY = ys[i];
  }

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const toPxX = (v: number) => pad + ((v - minX) / rangeX) * (width - 2 * pad);
  const toPxY = (v: number) => height - pad - ((v - minY) / rangeY) * (height - 2 * pad);

  let pathD = `M ${toPxX(xs[0])} ${toPxY(ys[0])}`;
  for (let i = 1; i < n; i++) {
    pathD += ` L ${toPxX(xs[i])} ${toPxY(ys[i])}`;
  }

  return `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background:#0f172a; border-radius:8px;">
  <!-- Axes -->
  <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#475569" stroke-width="1" />
  <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#475569" stroke-width="1" />
  <!-- Curve -->
  <path d="${pathD}" fill="none" stroke="${color}" stroke-width="${strokeWidth}" stroke-linecap="round" />
</svg>`.trim();
}
