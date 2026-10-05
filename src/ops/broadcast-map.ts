/**
 * @file broadcast-map.ts
 * Applies an N-ary function over several arrays and scalars with broadcasting (Julia `f.(a, b, c)`).
 */

import { NDArray } from '../core/ndarray.js';
import { Complex } from '../types/complex.js';
import { broadcastShapes } from './broadcast.js';
import { isArrayLike, toNDArray, wrapLike, type ArrayLike_ } from './elementwise.js';

export type Element = number | Complex;
export type BroadcastArg = ArrayLike_ | Element;

/**
 * Calls `fn(a[i], b[i], ...)` for every position of the broadcast shape and collects the results.
 *
 * - Arguments may be arrays (strided `NDArray`, `NestedArray`, nested `any[]`) or scalars; scalars and
 *   size-1 axes are repeated, following NumPy/Julia broadcasting rules.
 * - `fn` receives ONLY the element values, one per argument. It never receives indices or arrays,
 *   so extra optional parameters on `fn` stay untouched.
 * - Elements are numbers, or `Complex` when the source array is complex. `fn` may return a number,
 *   a boolean (stored as 1/0) or a `Complex`; the result becomes complex if any element is.
 * - Arrays with units are rejected, because `fn` cannot know about dimensions.
 * - The result is a new `NDArray`, or the container type of the first array argument when every
 *   array argument is a nested array.
 * - With no array arguments, `fn` is called once and its result returned as is.
 */
export function broadcastMap(fn: (...values: any[]) => Element | boolean, ...args: BroadcastArg[]): any {
  const plan = planBroadcast(args);
  if (!plan) return fn(...args);
  const { shape, size } = plan;

  const re = new Float64Array(size);
  let im: Float64Array | undefined;
  forEachBroadcast(plan, args, (n, values) => {
    const out = fn(...values);
    if (out instanceof Complex) {
      im ??= new Float64Array(size);
      re[n] = out.re;
      im[n] = out.im;
    } else {
      re[n] = toReal(out);
    }
  });

  const result = new NDArray(re, { shape: Array.from(shape), imag: im });
  const firstArray = args.find(isArrayLike);
  const allNested = args.filter(isArrayLike).every((a) => !(a instanceof NDArray));
  return allNested ? wrapLike(result, firstArray) : result;
}

/**
 * Like `broadcastMap`, but writes into an existing array (any strides or offset) without allocating.
 * The broadcast shape of `args` must equal `out.shape`. `out` may alias an argument, because every
 * element is read before it is written. `out` must be complex if any result is complex.
 */
export function broadcastInto(
  out: NDArray,
  fn: (...values: any[]) => Element | boolean,
  ...args: BroadcastArg[]
): NDArray {
  if (out.unit) throw new TypeError('In-place operations on arrays with units are not supported');
  const plan = planBroadcast(args);
  // Only scalars: the output has no shape to infer, so fill it with the single result
  const shape = plan ? plan.shape : out.shape;
  if (!plan) {
    const filled = broadcastInto(out, () => fn(...args), out);
    return filled;
  }
  if (shape.length !== out.ndim || Array.from(shape).some((n, d) => n !== out.shape[d])) {
    throw new RangeError(
      `Output shape [${Array.from(out.shape)}] does not match broadcast shape [${Array.from(shape)}]`
    );
  }

  forEachBroadcast(plan, args, (_n, values, coords) => {
    const result = fn(...values);
    let at = out.offset;
    for (let d = 0; d < coords.length; d++) at += coords[d] * out.strides[d];
    if (result instanceof Complex) {
      if (!out.imag) throw new TypeError('The result is complex but the output array is real');
      out.data[at] = result.re;
      out.imag[at] = result.im;
    } else {
      out.data[at] = toReal(result);
      if (out.imag) out.imag[at] = 0;
    }
  });
  return out;
}

function toReal(v: Element | boolean): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  throw new TypeError('The mapped function must return a number, boolean or Complex');
}

interface Plan {
  arrays: (NDArray | undefined)[];
  shape: ArrayLike<number>;
  size: number;
  strides: Int32Array[];
}

/** Validates the arguments and computes the broadcast shape; undefined when no argument is an array. */
function planBroadcast(args: BroadcastArg[]): Plan | undefined {
  const arrays = args.map((a) => (isArrayLike(a) ? toNDArray(a) : undefined));
  if (arrays.every((a) => a === undefined)) return undefined;

  for (const a of arrays) {
    if (a?.unit) throw new TypeError('Mapping a function over arrays with units is not supported');
  }
  for (const a of args) {
    if (!isArrayLike(a) && typeof a !== 'number' && !(a instanceof Complex)) {
      throw new TypeError('Broadcast arguments must be arrays, numbers or Complex values');
    }
  }

  let shape: ArrayLike<number> = [];
  for (const a of arrays) if (a) shape = broadcastShapes(shape, a.shape);
  const ndim = shape.length;
  const size = Array.from(shape).reduce((p, n) => p * n, 1);

  // Per-argument strides aligned to the result rank; broadcast axes get stride 0.
  const strides = arrays.map((a) => {
    const s = new Int32Array(ndim);
    if (a) {
      const pad = ndim - a.ndim;
      for (let d = 0; d < a.ndim; d++) s[pad + d] = a.shape[d] === 1 ? 0 : a.strides[d];
    }
    return s;
  });
  return { arrays, shape, size, strides };
}

/** Visits every position of the broadcast shape in row-major order with the element of each argument. */
function forEachBroadcast(
  plan: Plan,
  args: BroadcastArg[],
  visit: (n: number, values: Element[], coords: Int32Array) => void
): void {
  const { arrays, shape, size, strides } = plan;
  const ndim = shape.length;
  const coords = new Int32Array(ndim);
  const values: Element[] = new Array(args.length);

  for (let n = 0; n < size; n++) {
    for (let k = 0; k < args.length; k++) {
      const a = arrays[k];
      if (!a) {
        values[k] = args[k] as Element;
        continue;
      }
      let at = a.offset;
      for (let d = 0; d < ndim; d++) at += coords[d] * strides[k][d];
      values[k] = a.imag ? new Complex(a.data[at], a.imag[at]) : a.data[at];
    }
    visit(n, values, coords);
    for (let d = ndim - 1; d >= 0; d--) {
      if (++coords[d] < shape[d]) break;
      coords[d] = 0;
    }
  }
}

/** Applies a UNARY function: `fn(value)`. Same rules as `broadcastMap` with a single array. */
export function mapElements(x: ArrayLike_, fn: (value: Element) => Element | boolean): any {
  return broadcastMap((v) => fn(v), x);
}

/**
 * Like `Array.prototype.map`: `fn(value, index, array)`, where `index` is the 0-based multi-index
 * (one entry per axis) and `array` is the original input.
 */
export function mapIndexed(
  x: ArrayLike_,
  fn: (value: Element, index: number[], array: ArrayLike_) => Element | boolean
): any {
  const plan = planBroadcast([x])!;
  const re = new Float64Array(plan.size);
  let im: Float64Array | undefined;
  forEachBroadcast(plan, [x], (n, [v], coords) => {
    const out = fn(v, Array.from(coords), x);
    if (out instanceof Complex) {
      im ??= new Float64Array(plan.size);
      re[n] = out.re;
      im[n] = out.im;
    } else {
      re[n] = toReal(out);
    }
  });
  return wrapLike(new NDArray(re, { shape: Array.from(plan.shape), imag: im }), x);
}
