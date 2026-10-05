/**
 * @file reductions.ts
 * Reducciones por eje y estadísticas descriptivas:
 * - sum, prod
 * - mean, median, variance, std
 * - min, max
 * - quantile / percentile
 * - skew (asimetría), kurtosis (curtosis)
 * - describe() resumen estadístico completo
 */
import { NDArray } from '../core/ndarray.js';
import { Quantity } from '../units/units.js';

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

function sumValues(values: number[]): number {
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i];
  return total;
}

function centralMoments(values: number[]) {
  const count = values.length;
  const mean = sumValues(values) / count;
  let sumSquared = 0;
  let sumCubed = 0;
  let sumFourth = 0;

  for (let i = 0; i < count; i++) {
    const diff = values[i] - mean;
    const squared = diff * diff;
    sumSquared += squared;
    sumCubed += squared * diff;
    sumFourth += squared * squared;
  }

  return {
    count,
    mean,
    second: sumSquared / count,
    third: sumCubed / count,
    fourth: sumFourth / count,
  };
}

function interpolateQuantile(sorted: number[], q: number): number {
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const fraction = position - base;
  return sorted[base + 1] !== undefined
    ? sorted[base] + fraction * (sorted[base + 1] - sorted[base])
    : sorted[base];
}

/**
 * Función genérica de reducción multidimensional por eje.
 */
function reduceAxis(arr: NDArray, axis: number | undefined, keepdims: boolean, reducer: (values: number[]) => number): NDArray | number;
function reduceAxis(arr: NDArray, axis: number | undefined, keepdims: boolean, reducer: (values: number[]) => number, keepUnit: true): NDArray | number | Quantity;
function reduceAxis(arr: NDArray, axis: number | undefined, keepdims: boolean, reducer: (values: number[]) => number, keepUnit: false, allowUnit: true): NDArray | number;
function reduceAxis(arr: NDArray, axis: number | undefined, keepdims: boolean, reducer: (values: number[]) => number, keepUnit: boolean, allowUnit: boolean): NDArray | number | Quantity;
function reduceAxis(
  arr: NDArray,
  axis: number | undefined,
  keepdims: boolean,
  reducer: (values: number[]) => number,
  keepUnit = false,
  allowUnit = false
): NDArray | number | Quantity {
  if (arr.isComplex) {
    throw new TypeError('Reductions are not defined for complex arrays; reduce real(z), imag(z) or abs(z) instead');
  }
  if (arr.unit && !keepUnit && !allowUnit) {
    throw new TypeError('This reduction is not supported for arrays with units; convert with .to(unit) first');
  }
  const unit = keepUnit ? arr.unit : undefined;
  if (axis === undefined) {
    // Reducción sobre todo el tensor
    const allVals: number[] = [];
    for (const v of arr) allVals.push(v);
    const result = reducer(allVals);
    if (keepdims) {
      const onesShape = new Array(arr.ndim).fill(1);
      return new NDArray(new Float64Array([result]), { shape: onesShape, unit });
    }
    return unit ? new Quantity(result, unit) : result;
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

  return new NDArray(outData, { shape: outShape.length ? outShape : [1], unit });
}

// --- Operaciones de Reducción ---

export function sum(arr: NDArray, opts: ReductionOptions = {}): NDArray | number | Quantity {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, sumValues, true);
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

export function all(arr: NDArray): boolean;
export function all(arr: NDArray, opts: ReductionOptions): boolean | NDArray;
export function all(arr: NDArray, opts: ReductionOptions = {}): boolean | NDArray {
  const result = reduceAxis(
    arr,
    opts.axis,
    opts.keepdims ?? false,
    (values) => values.every((value) => value !== 0) ? 1 : 0,
    false,
    true
  );
  return typeof result === 'number' ? result !== 0 : result;
}

export function any(arr: NDArray): boolean;
export function any(arr: NDArray, opts: ReductionOptions): boolean | NDArray;
export function any(arr: NDArray, opts: ReductionOptions = {}): boolean | NDArray {
  const result = reduceAxis(
    arr,
    opts.axis,
    opts.keepdims ?? false,
    (values) => values.some((value) => value !== 0) ? 1 : 0,
    false,
    true
  );
  return typeof result === 'number' ? result !== 0 : result;
}

export function countNonzero(arr: NDArray): number;
export function countNonzero(arr: NDArray, opts: ReductionOptions): NDArray | number;
export function countNonzero(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(
    arr,
    opts.axis,
    opts.keepdims ?? false,
    (values) => values.reduce((count, value) => count + (value !== 0 ? 1 : 0), 0),
    false,
    true
  );
}

export function prod(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    let p = 1;
    for (let i = 0; i < vals.length; i++) p *= vals[i];
    return p;
  });
}

export function mean(arr: NDArray, opts: ReductionOptions = {}): NDArray | number | Quantity {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    return sumValues(vals) / vals.length;
  }, true);
}

export function min(arr: NDArray, opts: ReductionOptions = {}): NDArray | number | Quantity {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => Math.min(...vals), true);
}

export function max(arr: NDArray, opts: ReductionOptions = {}): NDArray | number | Quantity {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => Math.max(...vals), true);
}

export function variance(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  const ddof = opts.ddof ?? 0;
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const n = vals.length;
    if (n <= ddof) return NaN;
    return centralMoments(vals).second * n / (n - ddof);
  });
}

export function std(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  const v = variance(arr, opts);
  if (typeof v === 'number') return Math.sqrt(v);
  const outData = new Float64Array(v.size);
  for (let i = 0; i < v.size; i++) outData[i] = Math.sqrt(v.data[i]);
  return new NDArray(outData, { shape: Array.from(v.shape) });
}

export function median(arr: NDArray, opts: ReductionOptions = {}): NDArray | number | Quantity {
  return quantile(arr, 0.5, opts);
}

export function quantile(arr: NDArray, q: number, opts: ReductionOptions = {}): NDArray | number | Quantity {
  if (!Number.isFinite(q) || q < 0 || q > 1) throw new RangeError(`Quantile must be in range [0, 1], got ${q}`);
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const sorted = [...vals].sort((a, b) => a - b);
    return interpolateQuantile(sorted, q);
  }, true);
}

export function percentile(arr: NDArray, p: number, opts: ReductionOptions = {}): NDArray | number | Quantity {
  if (!Number.isFinite(p) || p < 0 || p > 100) {
    throw new RangeError(`Percentile must be in range [0, 100], got ${p}`);
  }
  return quantile(arr, p / 100, opts);
}

/**
 * Asimetría estadística (Skewness): momento central de 3er orden normalizado.
 */
export function skew(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const moments = centralMoments(vals);
    if (moments.count < 3) return NaN;
    const stdDev = Math.sqrt(moments.second);
    if (stdDev === 0) return 0;
    return moments.third / Math.pow(stdDev, 3);
  });
}

/**
 * Curtosis estadística (Kurtosis de exceso): momento central de 4to orden normalizado - 3.
 */
export function kurtosis(arr: NDArray, opts: ReductionOptions = {}): NDArray | number {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, (vals) => {
    const moments = centralMoments(vals);
    if (moments.count < 4) return NaN;
    if (moments.second === 0) return 0;
    return moments.fourth / (moments.second * moments.second) - 3.0;
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

  if (n === 0) {
    return {
      count: 0, mean: NaN, std: NaN, min: NaN, p25: NaN,
      median: NaN, p75: NaN, max: NaN, skew: NaN, kurtosis: NaN,
    };
  }

  const moments = centralMoments(vals);
  const sampleStd = n > 1 ? Math.sqrt(moments.second * n / (n - 1)) : 0;
  const stdPop = Math.sqrt(moments.second);

  return {
    count: n,
    mean: moments.mean,
    std: sampleStd,
    min: vals[0],
    p25: interpolateQuantile(vals, 0.25),
    median: interpolateQuantile(vals, 0.5),
    p75: interpolateQuantile(vals, 0.75),
    max: vals[n - 1],
    skew: stdPop > 0 && n > 2 ? moments.third / Math.pow(stdPop, 3) : 0,
    kurtosis: stdPop > 0 && n > 3 ? moments.fourth / (moments.second * moments.second) - 3 : 0,
  };
}
