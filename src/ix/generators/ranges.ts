/**
 * @file ranges.ts
 * Generadores numéricos y de mallas espaciales estilo NumPy (linspace, arange, logspace, meshgrid, eye).
 */
import { NDArray } from '../core/ndarray.js';

/**
 * Genera 'num' valores linealmente espaciados en el intervalo cerrado [start, stop].
 */
export function linspace(start: number, stop: number, num: number = 50): NDArray {
  if (num <= 0) throw new RangeError(`Number of samples must be positive, got ${num}`);
  const data = new Float64Array(num);

  if (num === 1) {
    data[0] = start;
    return new NDArray(data, { shape: [1] });
  }

  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    data[i] = start + i * step;
  }
  // Garantizar precisión numérica exacta en el extremo final
  data[num - 1] = stop;

  return new NDArray(data, { shape: [num] });
}

/**
 * Genera valores dentro del intervalo semiabierto [start, stop) con un paso 'step' constante.
 */
export function arange(start: number, stop?: number, step: number = 1): NDArray {
  let s = start;
  let e = stop;
  if (stop === undefined) {
    s = 0;
    e = start;
  }

  if (step === 0) throw new RangeError('Step cannot be zero');
  if ((step > 0 && s >= e!) || (step < 0 && s <= e!)) {
    return new NDArray(new Float64Array(0), { shape: [0] });
  }

  const num = Math.ceil((e! - s) / step);
  const data = new Float64Array(num);
  let cur = s;
  for (let i = 0; i < num; i++) {
    data[i] = cur;
    cur += step;
  }

  return new NDArray(data, { shape: [num] });
}

/**
 * Genera 'num' valores espaciados logarítmicamente entre base^start y base^stop.
 */
export function logspace(start: number, stop: number, num: number = 50, base: number = 10): NDArray {
  const exponents = linspace(start, stop, num);
  const data = new Float64Array(num);
  for (let i = 0; i < num; i++) {
    data[i] = Math.pow(base, exponents.data[i]);
  }
  return new NDArray(data, { shape: [num] });
}

/**
 * Crea una matriz identidad cuadrada o rectangular de tamaño n x m con unos en la diagonal principal.
 */
export function eye(n: number, m?: number): NDArray {
  const cols = m ?? n;
  const data = new Float64Array(n * cols);
  const minDim = Math.min(n, cols);
  for (let i = 0; i < minDim; i++) {
    data[i * cols + i] = 1.0;
  }
  return new NDArray(data, { shape: [n, cols] });
}

/**
 * Genera cuadrículas de coordenadas 2D (meshgrid) a partir de dos vectores 1D.
 * Devuelve [X, Y] donde X replica las filas e Y replica las columnas.
 */
export function meshgrid(x: NDArray, y: NDArray): [NDArray, NDArray] {
  const nx = x.size;
  const ny = y.size;

  const xData = new Float64Array(ny * nx);
  const yData = new Float64Array(ny * nx);

  for (let r = 0; r < ny; r++) {
    const yVal = y.data[r];
    for (let c = 0; c < nx; c++) {
      xData[r * nx + c] = x.data[c];
      yData[r * nx + c] = yVal;
    }
  }

  const X = new NDArray(xData, { shape: [ny, nx] });
  const Y = new NDArray(yData, { shape: [ny, nx] });
  return [X, Y];
}
