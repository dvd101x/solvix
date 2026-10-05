/**
 * @file reductions.ts
 * Reducciones por eje y estadísticas descriptivas:
 * - sum, prod
 * - mean, median, mode
 * - var (varianza), std (desviación estándar)
 * - min, max, argmin, argmax
 * - quantiles / percentiles
 * - skew (asimetría), kurtosis (curtosis)
 * - describe() resumen estadístico completo
 */
import { NDArray } from '../core/ndarray.js';

export interface ReductionOptions {
  axis?: number;
  keepdims?: boolean;
  ddof?: number; // Delta degrees of freedom (0 for population, 1 for unbiased sample).
}

export interface StatsSummary {
  count: number;
  mean: number;
  std: number;
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
  skew: number;
  kurtosis: number;
}

/**
 * Función genérica de reducción multidimensional por eje.
 */
function reduceAxis(
  arr: NDArray,
  axis: number | undefined,
  keepdims: boolean,
  reducer: (values: number[]) => number
): NDArray | number {
  if (axis === undefined) {
    // Reducción sobre todo el tensor
    const allVals: number[] = [];
    for (const v of arr) allVals.push(v);
    const result = reducer(allVals);
    if (keepdims) {
      const onesShape = new Array(arr.ndim).fill(1);
      return new NDArray(new Float64Array([result]), { shape: onesShape });
    }
    return result;
  }

  const normAxis = axis < 0 ? arr.ndim + axis : axis;
  if (normAxis < 0 || normAxis >= arr.ndim) {
    throw new RangeError(`Axis ${axis} is out of bounds for ndim ${arr.ndim}`);
  }

  const outShape: number[] = [];
  for (let i = 0; i < arr.ndim; i++) {
    if (i === normAxis) {
      if (keepdims) outShape.push(1);
    } else {
      outShape.push(arr.shape[i]);
    }
  }

  let outSize = 1;
  for (let d = 0; d < outShape.length; d++) outSize *= outShape[d];

  const outData = new Float64Array(outSize);
  const axisLen = arr.shape[normAxis];

  // Iterar sobre cada celda del tensor de salida acumulando sobre el eje colapsado
  const coords = new Int32Array(outShape.length);
  for (let outIdx = 0; outIdx < outSize; outIdx++) {
    const srcCoords = new Int32Array(arr.ndim);
    let cIdx = 0;
    for (let d = 0; d < arr.ndim; d++) {
      if (d === normAxis) {
        srcCoords[d] = 0;
        if (keepdims) cIdx++;
      } else {
        srcCoords[d] = coords[cIdx++];
      }
    }

    const colValues: number[] = new Array(axisLen);
    for (let k = 0; k < axisLen; k++) {
      srcCoords[normAxis] = k;
      colValues[k] = arr.get(...Array.from(srcCoords));
    }

    outData[outIdx] = reducer(colValues);

    for (let d = outShape.length - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < outShape[d]) break;
      coords[d] = 0;
    }
  }

  return new NDArray(outData, { shape: outShape.length ? outShape : [1] });
}

// --- Operaciones de Reducción ---

export function sum(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    let s = 0;
    for (let i = 0; i < vals.length; i++) s += vals[i];
    return s;
  });
}

export function sumProduct(...arrays: NDArray[]): number {
  if (arrays.length === 0) {
    throw new RangeError('sumProduct requires at least one NDArray');
  }

  const first = arrays[0];
  for (const arr of arrays.slice(1)) {
    if (arr.ndim !== first.ndim || arr.shape.some((size, axis) => size !== first.shape[axis])) {
      throw new Error('sumProduct requires arrays with identical shapes');
    }
  }

  const iterators = arrays.map((arr) => arr[Symbol.iterator]());
  let result = 0;
  for (let i = 0; i < first.size; i++) {
    let product = 1;
    for (const iterator of iterators) product *= iterator.next().value!;
    result += product;
  }
  return result;
}

export function all(arr: NDArray): boolean {
  for (const value of arr) {
    if (value === 0) return false;
  }
  return true;
}

export function any(arr: NDArray): boolean {
  for (const value of arr) {
    if (value !== 0) return true;
  }
  return false;
}

export function countNonzero(arr: NDArray): number {
  let count = 0;
  for (const value of arr) {
    if (value !== 0) count++;
  }
  return count;
}

export function prod(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    let p = 1;
    for (let i = 0; i < vals.length; i++) p *= vals[i];
    return p;
  });
}

export function mean(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    let s = 0;
    for (let i = 0; i < vals.length; i++) s += vals[i];
    return s / vals.length;
  });
}

export function min(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => Math.min(...vals));
}

export function max(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => Math.max(...vals));
}

export function variance(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  const ddof = opts.ddof ?? 0;
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const n = vals.length;
    if (n <= ddof) return NaN;
    let s = 0;
    for (let i = 0; i < n; i++) s += vals[i];
    const m = s / n;
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const diff = vals[i] - m;
      ss += diff * diff;
    }
    return ss / (n - ddof);
  });
}

export function std(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  const v = variance(arr, opts);
  if (typeof v === 'number') return Math.sqrt(v);
  const outData = new Float64Array(v.size);
  for (let i = 0; i < v.size; i++) outData[i] = Math.sqrt(v.data[i]);
  return new NDArray(outData, { shape: Array.from(v.shape) });
}

export function median(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const sorted = [...vals].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  });
}

export function quantile(arr: NDArray, q: number, opts: ReductionOptions = {}): NDArray | number {
  if (q < 0 || q > 1) throw new RangeError(`Quantile must be in range [0, 1], got ${q}`);
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const sorted = [...vals].sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    }
    return sorted[base];
  });
}

/**
 * Asimetría estadística (Skewness): momento central de 3er orden normalizado.
 */
export function skew(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const n = vals.length;
    if (n < 3) return NaN;
    let s = 0;
    for (let i = 0; i < n; i++) s += vals[i];
    const m = s / n;

    let m2 = 0;
    let m3 = 0;
    for (let i = 0; i < n; i++) {
      const diff = vals[i] - m;
      m2 += diff * diff;
      m3 += diff * diff * diff;
    }
    m2 /= n;
    m3 /= n;
    const stdDev = Math.sqrt(m2);
    if (stdDev === 0) return 0;
    return m3 / Math.pow(stdDev, 3);
  });
}

/**
 * Curtosis estadística (Kurtosis de exceso): momento central de 4to orden normalizado - 3.
 */
export function kurtosis(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const n = vals.length;
    if (n < 4) return NaN;
    let s = 0;
    for (let i = 0; i < n; i++) s += vals[i];
    const m = s / n;

    let m2 = 0;
    let m4 = 0;
    for (let i = 0; i < n; i++) {
      const diff = vals[i] - m;
      m2 += diff * diff;
      m4 += diff * diff * diff * diff;
    }
    m2 /= n;
    m4 /= n;
    if (m2 === 0) return 0;
    return m4 / (m2 * m2) - 3.0; // Exceso respecto a normal
  });
}

/**
 * Resumen descriptivo completo de un tensor.
 */
export function describe(arr: NDArray): StatsSummary {
  const vals: number[] = [];
  for (const v of arr) vals.push(v);
  const n = vals.length;
  vals.sort((a, b) => a - b);

  let sumVal = 0;
  for (let i = 0; i < n; i++) sumVal += vals[i];
  const m = sumVal / n;

  let ss = 0;
  let m3 = 0;
  let m4 = 0;
  for (let i = 0; i < n; i++) {
    const diff = vals[i] - m;
    const diff2 = diff * diff;
    ss += diff2;
    m3 += diff2 * diff;
    m4 += diff2 * diff2;
  }

  const s = n > 1 ? Math.sqrt(ss / (n - 1)) : 0;
  const stdPop = Math.sqrt(ss / n);
  const skewVal = stdPop > 0 && n > 2 ? (m3 / n) / Math.pow(stdPop, 3) : 0;
  const kurtVal = stdPop > 0 && n > 3 ? (m4 / n) / Math.pow(stdPop, 4) - 3 : 0;

  const getP = (p: number) => {
    const pos = (n - 1) * p;
    const base = Math.floor(pos);
    const rest = pos - base;
    return vals[base + 1] !== undefined
      ? vals[base] + rest * (vals[base + 1] - vals[base])
      : vals[base];
  };

  return {
    count: n,
    mean: m,
    std: s,
    min: vals[0],
    p25: getP(0.25),
    median: getP(0.5),
    p75: getP(0.75),
    max: vals[n - 1],
    skew: skewVal,
    kurtosis: kurtVal,
  };
}
