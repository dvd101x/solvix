/**
 * @file one-based.ts
 * Julia-style ranges, array literals and 1-based indexing, used by the expression parser.
 * Every function here is 1-based and inclusive, unlike the 0-based NDArray API.
 */

import { NDArray } from '../core/ndarray.js';
import { Complex } from '../types/complex.js';
import { Quantity } from '../units/units.js';
import { isArrayLike, toNDArray, wrapLike } from '../ops/elementwise.js';

/** An index along one axis: a scalar, a list of positions, or `':'` for the whole axis. */
export type AxisIndex = number | number[] | ':';

/** Inclusive range `start:step:stop` as a 1D array (Julia `start:step:stop`). */
export function colonRange(start: number, stop: number, step: number = 1): NDArray {
  if (step === 0) throw new RangeError('Range step cannot be zero');
  if (![start, stop, step].every(Number.isFinite)) throw new RangeError('Range bounds must be finite numbers');
  const n = Math.max(0, Math.floor((stop - start) / step + 1e-10) + 1);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = start + i * step;
  return new NDArray(out, { shape: [n] });
}

type Scalar = number | Complex;

function toGrid(x: unknown): Scalar[][] {
  if (typeof x === 'number' || x instanceof Complex) return [[x]];
  if (!isArrayLike(x)) throw new TypeError('Array literals can only contain numbers, complex numbers and arrays');
  const a = toNDArray(x);
  if (a.unit) throw new TypeError('Array literals with units are not supported; use withUnit() on the result');
  const nested = a.toNestedArray() as any;
  if (a.ndim === 1) return (nested as Scalar[]).map((v) => [v]); // vectors are columns, as in Julia
  if (a.ndim === 2) return nested as Scalar[][];
  throw new TypeError('Array literals only support scalars, vectors and matrices');
}

/**
 * Builds an array from rows of evaluated elements, with Julia semantics:
 * `[1,2,3]` and `[1;2;3]` are vectors, `[1 2;3 4]` is a matrix, and array elements are concatenated.
 */
export function buildArray(rows: unknown[][], commaSeparated: boolean): NDArray {
  if (rows.length === 0 || rows.every((r) => r.length === 0)) return new NDArray(new Float64Array(0), { shape: [0] });
  if (commaSeparated && rows.length > 1) throw new SyntaxError('Cannot mix "," and ";" in an array literal');
  const allScalar = rows.every((r) => r.every((e) => typeof e === 'number' || e instanceof Complex));
  const scalarVector = allScalar && (commaSeparated || rows.every((r) => r.length === 1));
  if (scalarVector) return NDArray.fromArray(rows.flat() as Scalar[]);
  const isVectorLike = (e: unknown) => typeof e === 'number' || e instanceof Complex || (isArrayLike(e) && toNDArray(e).ndim === 1);
  if (!commaSeparated && rows.every((r) => r.length === 1 && isVectorLike(r[0]))) {
    return NDArray.fromArray(rows.flatMap((r) => toGrid(r[0]).map((g) => g[0])));
  }
  if (commaSeparated) {
    // [A, B] stacks arrays along a new leading axis only for scalars; blocks are not allowed here
    throw new SyntaxError('Use spaces and ";" to concatenate arrays: [A B; C D]');
  }

  const stacked: Scalar[][] = [];
  for (const row of rows) {
    const blocks = row.map(toGrid);
    const height = blocks[0].length;
    if (blocks.some((b) => b.length !== height)) {
      throw new RangeError('Horizontal concatenation requires equal heights');
    }
    for (let r = 0; r < height; r++) stacked.push(blocks.flatMap((b) => b[r]));
  }
  const width = stacked[0].length;
  if (stacked.some((r) => r.length !== width)) {
    throw new RangeError('Vertical concatenation requires equal widths');
  }
  return NDArray.fromArray(stacked);
}

function positions(idx: AxisIndex, size: number, axis: number): number[] {
  const list = idx === ':' ? Array.from({ length: size }, (_, i) => i + 1) : typeof idx === 'number' ? [idx] : idx;
  for (const p of list) {
    if (!Number.isInteger(p) || p < 1 || p > size) {
      throw new RangeError(`Index ${p} out of bounds for axis ${axis + 1} of size ${size}`);
    }
  }
  return list.map((p) => p - 1);
}

/**
 * `A[i]`, `A[i, j]`, `A[:, 2]`, `A[2:3, :]`. Scalar indices drop their axis;
 * all-scalar indices return a number, Complex or Quantity.
 */
export function indexOneBased(source: unknown, indices: AxisIndex[]): any {
  if (!isArrayLike(source)) throw new TypeError('Only arrays can be indexed');
  const arr = toNDArray(source);
  if (indices.length !== arr.ndim) {
    throw new RangeError(`Expected ${arr.ndim} indices for a ${arr.ndim}D array, got ${indices.length}`);
  }
  const lists = indices.map((idx, d) => positions(idx, arr.shape[d], d));
  const keptAxes = indices.map((idx, d) => (typeof idx === 'number' ? -1 : d)).filter((d) => d >= 0);
  const outShape = keptAxes.map((d) => lists[d].length);
  const total = outShape.reduce((p, n) => p * n, 1);

  const re = new Float64Array(total);
  const im = arr.isComplex ? new Float64Array(total) : undefined;
  const counter = new Array(arr.ndim).fill(0);
  for (let k = 0; k < total; k++) {
    const src = arr.indexOf(...counter.map((c, d) => lists[d][c]));
    re[k] = arr.data[src];
    if (im) im[k] = arr.imag![src];
    for (let d = arr.ndim - 1; d >= 0; d--) {
      if (++counter[d] < lists[d].length) break;
      counter[d] = 0;
    }
  }

  if (keptAxes.length === 0) {
    if (arr.unit) return new Quantity(re[0], arr.unit);
    return im ? new Complex(re[0], im[0]) : re[0];
  }
  const result = new NDArray(re, { shape: outShape, imag: im, unit: arr.unit });
  return wrapLike(result, source);
}
