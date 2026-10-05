/**
 * @file quadrature.ts
 * Algoritmos de integración numérica definida:
 * - trapz / trapezoid: Regla del trapecio para datos discretos
 * - simpson: Regla de Simpson compuesta 1/3 (precisión O(h^4))
 * - quad: Cuadratura adaptativa de Gauss-Kronrod / Simpson adaptativa para funciones continuas f(x)
 * - cumulativeIntegrate: Integral acumulativa 1D (ej. aceleración -> velocidad -> posición)
 */
import { broadcastMap, type BroadcastArg } from '../ops/broadcast-map.js';
import { NDArray, toFloat64 } from '../core/ndarray.js';
import { cumsum } from '../stats/reductions.js';

/**
 * Regla del trapecio sobre muestras discretas y(x)
 */
export function trapz(y: NDArray | Float64Array | number[], x?: NDArray | Float64Array | number[], dx = 1.0): number {
  const yData = toFloat64(y);
  const n = yData.length;
  if (n < 2) return 0;

  if (!x) {
    let sum = 0.5 * (yData[0] + yData[n - 1]);
    for (let i = 1; i < n - 1; i++) sum += yData[i];
    return sum * dx;
  }

  const xData = toFloat64(x);
  let total = 0;
  for (let i = 0; i < n - 1; i++) {
    const h = xData[i + 1] - xData[i];
    total += 0.5 * (yData[i] + yData[i + 1]) * h;
  }
  return total;
}

/**
 * Regla de Simpson compuesta (1/3) para datos discretos equiespaciados o con vector x
 */
export function simpson(y: NDArray | Float64Array | number[], x?: NDArray | Float64Array | number[], dx = 1.0): number {
  const yData = toFloat64(y);
  const n = yData.length;
  if (n < 3) return trapz(y, x, dx);

  const xData = x ? (toFloat64(x)) : null;
  const h = xData ? (xData[n - 1] - xData[0]) / (n - 1) : dx;

  let sumOdd = 0;
  let sumEven = 0;

  // Si n es par, número de intervalos es impar: aplicamos Simpson en n-1 y trapecio en el último
  const limit = n % 2 === 0 ? n - 1 : n;

  for (let i = 1; i < limit - 1; i += 2) sumOdd += yData[i];
  for (let i = 2; i < limit - 1; i += 2) sumEven += yData[i];

  let result = (h / 3) * (yData[0] + 4 * sumOdd + 2 * sumEven + yData[limit - 1]);

  if (n % 2 === 0) {
    // Último tramo trapezoidal
    const lastH = xData ? xData[n - 1] - xData[n - 2] : dx;
    result += 0.5 * (yData[n - 2] + yData[n - 1]) * lastH;
  }

  return result;
}

/**
 * Cuadratura adaptativa de Simpson para funciones continuas en [a, b]
 */
export function quad(f: (x: number) => number, a: number, b: number, tol?: number, maxDepth?: number): number;
export function quad(f: (x: number) => number, a: BroadcastArg, b: BroadcastArg, tol?: number, maxDepth?: number): NDArray | any;
export function quad(f: (x: number) => number, a: BroadcastArg, b: BroadcastArg, tol = 1e-8, maxDepth = 25): any {
  // Array limits are broadcast: one integral per (a, b) pair.
  if (typeof a !== 'number' || typeof b !== 'number') {
    return broadcastMap((lo, hi) => quad(f, lo as number, hi as number, tol, maxDepth), a, b);
  }

  function simpsonRule(fa: number, fb: number, fc: number, h: number): number {
    return (h / 6) * (fa + 4 * fc + fb);
  }

  function adapt(
    x0: number,
    x1: number,
    f0: number,
    f1: number,
    fmid: number,
    whole: number,
    depth: number
  ): number {
    const mid = (x0 + x1) / 2;
    const h = x1 - x0;
    const xL = (x0 + mid) / 2;
    const xR = (mid + x1) / 2;

    const fL = f(xL);
    const fR = f(xR);

    const left = simpsonRule(f0, fmid, fL, h / 2);
    const right = simpsonRule(fmid, f1, fR, h / 2);
    const delta = left + right - whole;

    if (depth <= 0 || Math.abs(delta) <= 15 * tol) {
      return left + right + delta / 15;
    }

    return (
      adapt(x0, mid, f0, fmid, fL, left, depth - 1) +
      adapt(mid, x1, fmid, f1, fR, right, depth - 1)
    );
  }

  const h = b - a;
  const fA = f(a);
  const fB = f(b);
  const fMid = f((a + b) / 2);
  const initial = simpsonRule(fA, fB, fMid, h);

  return adapt(a, b, fA, fB, fMid, initial, maxDepth);
}

/**
 * Integral acumulativa discreta 1D: Y[i] = integral de 0 a i
 */
export function cumulativeIntegrate(y: NDArray, x?: NDArray, dx = 1.0): NDArray {
  const n = y.size;
  const yData = toFloat64(y);
  const xData = x ? toFloat64(x) : null;
  const out = new Float64Array(n);

  if (n === 0) return new NDArray(out, { shape: [0] });

  const areas = new Float64Array(n - 1);
  for (let i = 1; i < n; i++) {
    const h = xData ? xData[i] - xData[i - 1] : dx;
    areas[i - 1] = 0.5 * (yData[i - 1] + yData[i]) * h;
  }

  const accumulated = cumsum(new NDArray(areas, { shape: [areas.length] }));
  for (let i = 1; i < n; i++) out[i] = accumulated.get(i - 1);
  return new NDArray(out, { shape: [n] });
}
