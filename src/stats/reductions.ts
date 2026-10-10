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
import { broadcastMap } from '../ops/broadcast-map.js';
import { reduceTensorAxis, type Axis } from './reduction-utils.js';
import { scanTensorAxis } from './scan-utils.js';
export type { Axis } from './reduction-utils.js';

export interface ReductionOptions {
  axis?: Axis;
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

export type AxisStatsSummary = { [K in keyof StatsSummary]: NDArray };

export interface SumProductOptions {
  axis?: Axis;
  keepdims?: boolean;
}

export interface DescribeOptions {
  axis?: Axis;
  keepdims?: boolean;
}

export interface ScanOptions {
  axis?: Axis;
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

function minValue(values: number[]): number {
  if (values.length === 0) return Infinity;
  let result = values[0];
  for (let i = 1; i < values.length; i++) {
    if (Number.isNaN(values[i])) return NaN;
    if (values[i] < result) result = values[i];
  }
  return result;
}

function maxValue(values: number[]): number {
  if (values.length === 0) return -Infinity;
  let result = values[0];
  for (let i = 1; i < values.length; i++) {
    if (Number.isNaN(values[i])) return NaN;
    if (values[i] > result) result = values[i];
  }
  return result;
}

/**
 * Función genérica de reducción multidimensional por eje.
 */
function reduceAxis(arr: NDArray, axis: Axis | undefined, keepdims: boolean, reducer: (values: number[]) => number): NDArray | number;
function reduceAxis(arr: NDArray, axis: Axis | undefined, keepdims: boolean, reducer: (values: number[]) => number, keepUnit: true): NDArray | number | Quantity;
function reduceAxis(arr: NDArray, axis: Axis | undefined, keepdims: boolean, reducer: (values: number[]) => number, keepUnit: false): NDArray | number;
function reduceAxis(
  arr: NDArray,
  axis: Axis | undefined,
  keepdims: boolean,
  reducer: (values: number[]) => number,
  keepUnit = false,
): NDArray | number | Quantity {
  if (arr.isComplex) {
    throw new TypeError('Reductions are not defined for complex arrays; reduce real(z), imag(z) or abs(z) instead');
  }
  if (arr.unit && !keepUnit) {
    throw new TypeError('This reduction is not supported for arrays with units; convert with .to(unit) first');
  }
  const result = reduceTensorAxis(arr, axis, keepdims, reducer);
  if (typeof result === 'number') {
    return keepUnit && arr.unit ? new Quantity(result, arr.unit) : result;
  }
  return keepUnit && arr.unit
    ? new NDArray(result.data, { shape: result.shape, unit: arr.unit })
    : result;
}

// --- Operaciones de Reducción ---

export function sum(arr: NDArray, opts: ReductionOptions = {}): NDArray | number | Quantity {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, sumValues, true);
}

export function cumsum(arr: NDArray, opts: ScanOptions = {}): NDArray {
  return scanTensorAxis(arr, opts.axis, (accumulated, value) => accumulated + value, true);
}

export function cumprod(arr: NDArray, opts: ScanOptions = {}): NDArray {
  return scanTensorAxis(arr, opts.axis, (accumulated, value) => accumulated * value, false);
}

export function cummin(arr: NDArray, opts: ScanOptions = {}): NDArray {
  return scanTensorAxis(arr, opts.axis, Math.min, true);
}

export function cummax(arr: NDArray, opts: ScanOptions = {}): NDArray {
  return scanTensorAxis(arr, opts.axis, Math.max, true);
}

export function sumProduct(...arrays: [NDArray, ...NDArray[]]): number;
export function sumProduct(...args: [...NDArray[], SumProductOptions]): number | NDArray;
export function sumProduct(...args: (NDArray | SumProductOptions)[]): number | NDArray {
  let opts: SumProductOptions = {};
  const last = args[args.length - 1];
  if (last && !(last instanceof NDArray)) {
    opts = args.pop() as SumProductOptions;
    if (!opts || typeof opts !== 'object' || Array.isArray(opts)) {
      throw new TypeError('sumProduct options must be an object');
    }
    for (const key of Object.keys(opts)) {
      if (key !== 'axis' && key !== 'keepdims') {
        throw new TypeError(`Unknown sumProduct option: ${key}`);
      }
    }
  }
  const arrays = args as NDArray[];
  if (arrays.length === 0) {
    throw new RangeError('sumProduct requires at least one NDArray');
  }

  for (const arr of arrays) {
    if (!(arr instanceof NDArray)) throw new TypeError('sumProduct arguments must be NDArrays');
    if (arr.isComplex) throw new TypeError('sumProduct is only defined for real arrays');
  }

  const products = broadcastMap((...values: number[]) => {
    let product = 1;
    for (const value of values) product *= value;
    return product;
  }, ...arrays) as NDArray;
  return reduceAxis(products, opts.axis, opts.keepdims ?? false, sumValues);
}

export function all(arr: NDArray): boolean;
export function all(arr: NDArray, opts: ReductionOptions): boolean | NDArray;
export function all(arr: NDArray, opts: ReductionOptions = {}): boolean | NDArray {
  const result = reduceAxis(
    arr,
    opts.axis,
    opts.keepdims ?? false,
    (values) => values.every((value) => value !== 0) ? 1 : 0,
    false
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
    false
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
    false
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
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, minValue, true);
}

export function max(arr: NDArray, opts: ReductionOptions = {}): NDArray | number | Quantity {
  return reduceAxis(arr, opts.axis, opts.keepdims ?? false, maxValue, true);
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
export function describe(arr: NDArray): StatsSummary;
export function describe(arr: NDArray, opts: DescribeOptions & { axis: Axis }): AxisStatsSummary;
export function describe(arr: NDArray, opts: DescribeOptions): StatsSummary | AxisStatsSummary;
export function describe(arr: NDArray, opts: DescribeOptions = {}): StatsSummary | AxisStatsSummary {
  if (opts.axis !== undefined) {
    const { axis, keepdims = false } = opts;
    const reduce = (reducer: (values: number[]) => number) =>
      reduceTensorAxis(arr, axis, keepdims, reducer) as NDArray;

    return {
      count: reduce((values) => values.length),
      mean: reduce((values) => centralMoments(values).mean),
      std: reduce((values) => {
        const moments = centralMoments(values);
        return moments.count > 1
          ? Math.sqrt(moments.second * moments.count / (moments.count - 1))
          : 0;
      }),
      min: reduce((values) => {
        return values.length === 0 ? NaN : minValue(values);
      }),
      p25: reduce((values) => interpolateQuantile([...values].sort((a, b) => a - b), 0.25)),
      median: reduce((values) => interpolateQuantile([...values].sort((a, b) => a - b), 0.5)),
      p75: reduce((values) => interpolateQuantile([...values].sort((a, b) => a - b), 0.75)),
      max: reduce((values) => {
        return values.length === 0 ? NaN : maxValue(values);
      }),
      skew: reduce((values) => {
        const moments = centralMoments(values);
        if (moments.count < 3) return 0;
        const stdPop = Math.sqrt(moments.second);
        return stdPop > 0 ? moments.third / Math.pow(stdPop, 3) : 0;
      }),
      kurtosis: reduce((values) => {
        const moments = centralMoments(values);
        if (moments.count < 4 || moments.second === 0) return 0;
        return moments.fourth / (moments.second * moments.second) - 3;
      }),
    };
  }

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
