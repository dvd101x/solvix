/**
 * @file adapters.ts
 * Puentes de exportación y renderizado para librerías de visualización web:
 * - toChartJS: Formato datasets para Chart.js
 * - toPlotly: Formato traces para Plotly.js (scatter, bar, heatmap)
 * - toObservablePlot: Formato de objetos/arrays para Observable Plot
 * - plotSVG: Generador de SVG puro (Zero-Dependencies) para incrustación inline instantánea
 */
import { NDArray } from '../core/ndarray.js';

export interface PlotOptions {
  title?: string;
  xLabel?: string;
  yLabel?: string;
  color?: string;
  fill?: boolean;
}

/**
 * Convierte series x, y en la estructura de datasets que consume Chart.js
 */
export function toChartJS(
  x: NDArray | number[],
  y: NDArray | number[],
  opts: PlotOptions & { label?: string } = {}
) {
  const xArr = x instanceof NDArray ? Array.from(x.data) : x;
  const yArr = y instanceof NDArray ? Array.from(y.data) : y;

  return {
    type: 'line',
    data: {
      labels: xArr,
      datasets: [
        {
          label: opts.label ?? 'Series 1',
          data: yArr,
          borderColor: opts.color ?? 'rgb(75, 192, 192)',
          fill: opts.fill ?? false,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: Boolean(opts.title), text: opts.title },
      },
      scales: {
        x: { title: { display: Boolean(opts.xLabel), text: opts.xLabel } },
        y: { title: { display: Boolean(opts.yLabel), text: opts.yLabel } },
      },
    },
  };
}

/**
 * Convierte series a formato de trazas de Plotly.js (data + layout)
 */
export function toPlotly(
  x: NDArray | number[],
  y: NDArray | number[],
  opts: PlotOptions & { type?: 'scatter' | 'bar'; mode?: 'lines' | 'markers' | 'lines+markers' } = {}
) {
  const xArr = x instanceof NDArray ? Array.from(x.data) : x;
  const yArr = y instanceof NDArray ? Array.from(y.data) : y;

  return {
    data: [
      {
        x: xArr,
        y: yArr,
        type: opts.type ?? 'scatter',
        mode: opts.mode ?? 'lines',
        line: { color: opts.color ?? '#1f77b4' },
      },
    ],
    layout: {
      title: opts.title ?? '',
      xaxis: { title: opts.xLabel ?? '' },
      yaxis: { title: opts.yLabel ?? '' },
    },
  };
}

/**
 * Convierte pares ordenados a formato tabular para Observable Plot (Plot.line(data, { x: 'x', y: 'y' }))
 */
export function toObservablePlot(x: NDArray | number[], y: NDArray | number[]) {
  const xArr = x instanceof NDArray ? Array.from(x.data) : x;
  const yArr = y instanceof NDArray ? Array.from(y.data) : y;
  const len = Math.min(xArr.length, yArr.length);

  const data: { x: number; y: number }[] = new Array(len);
  for (let i = 0; i < len; i++) {
    data[i] = { x: xArr[i], y: yArr[i] };
  }
  return data;
}

/**
 * Generador SVG autónomo y ligero (<2KB) para previsualización inmediata en navegadores y editores
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
  <!-- Ejes -->
  <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#475569" stroke-width="1" />
  <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#475569" stroke-width="1" />
  <!-- Curva -->
  <path d="${pathD}" fill="none" stroke="${color}" stroke-width="${strokeWidth}" stroke-linecap="round" />
</svg>`.trim();
}
