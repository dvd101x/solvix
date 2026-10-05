/**
 * @file interpolation.ts
 * Algoritmos de interpolación numérica 1D:
 * - interp1d: Interpolación lineal, vecino más cercano (nearest) y cúbica hermite
 * - cubicSpline: Trazador cúbico natural (Natural Cubic Splines) con derivada segunda continua
 */
import { NDArray, toFloat64 } from '../core/ndarray.js';
import { broadcastMap } from '../ops/broadcast-map.js';
import { solve } from '../linalg/factorizations.js';

export type InterpMethod = 'linear' | 'nearest' | 'previous' | 'next';

/**
 * Búsqueda binaria O(log N) del índice i tal que xArr[i] <= xq <= xArr[i+1]
 */
function binarySearchInterval(xArr: ArrayLike<number>, xq: number): number {
  let low = 0;
  let high = xArr.length - 2;

  if (xq <= xArr[0]) return 0;
  if (xq >= xArr[xArr.length - 1]) return high;

  while (low <= high) {
    const mid = (low + high) >> 1;
    if (xArr[mid] <= xq && xq <= xArr[mid + 1]) {
      return mid;
    }
    if (xArr[mid] > xq) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return low;
}

/**
 * Interpolador 1D configurable (retorna una función matemática evaluable f(xq))
 */
export function interp1d(
  x: NDArray | Float64Array | number[],
  y: NDArray | Float64Array | number[],
  opts: { method?: InterpMethod; fillValue?: number } = {}
): (xq: number | NDArray | number[]) => any {
  const xData = toFloat64(x);
  const yData = toFloat64(y);
  const method = opts.method ?? 'linear';
  const fillValue = opts.fillValue ?? NaN;
  const n = xData.length;

  function evaluateSingle(xq: number): number {
    if (xq < xData[0] || xq > xData[n - 1]) {
      if (!Number.isNaN(fillValue)) return fillValue;
      if (xq < xData[0]) return yData[0];
      return yData[n - 1];
    }

    const i = binarySearchInterval(xData, xq);
    const x0 = xData[i];
    const x1 = xData[i + 1];
    const y0 = yData[i];
    const y1 = yData[i + 1];

    switch (method) {
      case 'nearest':
        return Math.abs(xq - x0) <= Math.abs(xq - x1) ? y0 : y1;
      case 'previous':
        return y0;
      case 'next':
        return y1;
      case 'linear':
      default: {
        const t = (xq - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
      }
    }
  }

  return (xq: number | NDArray | number[]) =>
    typeof xq === 'number' ? evaluateSingle(xq) : broadcastMap(evaluateSingle, xq);
}

/**
 * Trazador Cúbico Natural (Natural Cubic Splines) con continuidad C^2
 */
export function cubicSpline(
  x: NDArray | Float64Array | number[],
  y: NDArray | Float64Array | number[]
): (xq: number | NDArray | number[]) => any {
  const xData = toFloat64(x);
  const yData = toFloat64(y);
  const n = xData.length;

  // Paso h_i = x_{i+1} - x_i
  const h = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = xData[i + 1] - xData[i];

  // Sistema tridiagonal para derivadas segundas M (M_0 = M_{n-1} = 0)
  const AData = new Float64Array(n * n);
  const bData = new Float64Array(n);

  AData[0] = 1.0;
  AData[(n - 1) * n + (n - 1)] = 1.0;

  for (let i = 1; i < n - 1; i++) {
    AData[i * n + (i - 1)] = h[i - 1];
    AData[i * n + i] = 2 * (h[i - 1] + h[i]);
    AData[i * n + (i + 1)] = h[i];

    bData[i] = 6 * ((yData[i + 1] - yData[i]) / h[i] - (yData[i] - yData[i - 1]) / h[i - 1]);
  }

  const A = new NDArray(AData, { shape: [n, n] });
  const M = solve(A, bData);

  function evalSpline(xq: number): number {
    const i = binarySearchInterval(xData, xq);
    const hi = h[i];
    const x0 = xData[i];
    const x1 = xData[i + 1];

    const M0 = M.data[i];
    const M1 = M.data[i + 1];
    const y0 = yData[i];
    const y1 = yData[i + 1];

    const dx1 = x1 - xq;
    const dx0 = xq - x0;

    return (
      (M0 * dx1 * dx1 * dx1 + M1 * dx0 * dx0 * dx0) / (6 * hi) +
      (y0 - (M0 * hi * hi) / 6) * (dx1 / hi) +
      (y1 - (M1 * hi * hi) / 6) * (dx0 / hi)
    );
  }

  return (xq: number | NDArray | number[]) =>
    typeof xq === 'number' ? evalSpline(xq) : broadcastMap(evalSpline, xq);
}
