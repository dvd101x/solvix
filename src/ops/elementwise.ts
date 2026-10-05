/**
 * @file elementwise.ts
 * Broadcasting element-wise arithmetic for strided arrays, nested arrays and scalars.
 * Supports complex numbers (split re/im storage) and physical dimensions (units).
 *
 * Operand kinds: number | Complex | Quantity | NDArray | NestedArray | plain nested JS arrays.
 * Nested inputs are converted to NDArray and the result is converted back to nested form.
 */
import { NDArray, TypedArray } from '../core/ndarray.js';
import { NestedArray } from '../core/nested-array.js';
import { Complex } from '../types/complex.js';
import { Quantity, Dimensions } from '../units/units.js';
import { broadcastShapes } from './broadcast.js';

export type Scalar = number | Complex | Quantity;
export type ArrayLike_ = NDArray | NestedArray | any[];
export type Operand = Scalar | ArrayLike_;

export function isArrayLike(x: unknown): x is ArrayLike_ {
  return x instanceof NDArray || x instanceof NestedArray || Array.isArray(x);
}

/** Converts any array-like operand to a strided NDArray. */
export function toNDArray(x: ArrayLike_): NDArray {
  if (x instanceof NDArray) return x;
  if (x instanceof NestedArray) return NDArray.fromArray(x.data);
  return NDArray.fromArray(x);
}

/** Converts an NDArray result back to the same container flavour as `like`. */
export function wrapLike(result: NDArray, like: unknown): any {
  if (like instanceof NestedArray) return new NestedArray(result.toNestedArray() as any[]);
  if (Array.isArray(like)) return result.toNestedArray();
  return result;
}

function dimsOf(q: Quantity): Dimensions | undefined {
  return q.isDimensionless() ? undefined : q.dims;
}

export function mulDims(a: Dimensions | undefined, b: Dimensions | undefined, sign: 1 | -1): Dimensions | undefined {
  if (!a && !b) return undefined;
  const qa = new Quantity(1, a ?? {});
  const qb = new Quantity(1, b ?? {});
  return (sign === 1 ? qa.mul(qb) : qa.div(qb)).dims;
}

export function sameDims(a: Dimensions | undefined, b: Dimensions | undefined): boolean {
  return new Quantity(1, a ?? {}).hasSameDimensions(new Quantity(1, b ?? {}));
}

export function fmt(d: Dimensions | undefined): string {
  return new Quantity(1, d ?? {}).formatDimensions();
}

interface Op {
  data: TypedArray;
  imag?: TypedArray;
  offset: number;
  shape: ArrayLike<number>;
  strides: ArrayLike<number>;
  unit?: Dimensions;
  isArray: boolean;
}

function toOp(x: Scalar | NDArray): Op {
  if (x instanceof NDArray) {
    return { data: x.data, imag: x.imag, offset: x.offset, shape: x.shape, strides: x.strides, unit: x.unit, isArray: true };
  }
  if (x instanceof Complex) {
    return { data: new Float64Array([x.re]), imag: new Float64Array([x.im]), offset: 0, shape: [], strides: [], isArray: false };
  }
  if (x instanceof Quantity) {
    return { data: new Float64Array([x.value]), offset: 0, shape: [], strides: [], unit: dimsOf(x), isArray: false };
  }
  return { data: new Float64Array([x]), offset: 0, shape: [], strides: [], isArray: false };
}

/** Strides of `op` aligned to the broadcast shape (0 on broadcast axes). */
function alignedStrides(op: Op, ndim: number): Int32Array {
  const out = new Int32Array(ndim);
  const pad = ndim - op.shape.length;
  for (let d = 0; d < op.shape.length; d++) {
    out[pad + d] = op.shape[d] === 1 ? 0 : op.strides[d];
  }
  return out;
}

export type BinaryName = 'add' | 'sub' | 'mul' | 'div' | 'pow';

function resultUnit(name: BinaryName, a: Op, b: Op): Dimensions | undefined {
  switch (name) {
    case 'add':
    case 'sub':
      if (!sameDims(a.unit, b.unit)) {
        throw new TypeError(`Dimensional mismatch in ${name}: [${fmt(a.unit)}] vs [${fmt(b.unit)}]`);
      }
      return a.unit;
    case 'mul':
      return mulDims(a.unit, b.unit, 1);
    case 'div':
      return mulDims(a.unit, b.unit, -1);
    case 'pow': {
      if (b.unit) throw new TypeError('The exponent must be dimensionless');
      if (!a.unit) return undefined;
      if (b.data.length !== 1 || b.imag) {
        throw new TypeError('An array with units can only be raised to a real scalar power');
      }
      const e = b.data[b.offset];
      return new Quantity(1, a.unit).pow(e).dims;
    }
  }
}

function complexPow(ar: number, ai: number, br: number, bi: number): [number, number] {
  if (ar === 0 && ai === 0) return br === 0 && bi === 0 ? [1, 0] : [0, 0];
  const logR = Math.log(Math.hypot(ar, ai));
  const theta = Math.atan2(ai, ar);
  const er = br * logR - bi * theta;
  const ei = bi * logR + br * theta;
  const m = Math.exp(er);
  return [m * Math.cos(ei), m * Math.sin(ei)];
}

function dot(coords: Int32Array, strides: Int32Array, n: number): number {
  let sum = 0;
  for (let d = 0; d < n; d++) sum += coords[d] * strides[d];
  return sum;
}

/**
 * Applies a binary operation with broadcasting. At least one operand must be an NDArray.
 */
export function binaryOp(name: BinaryName, left: Scalar | NDArray, right: Scalar | NDArray): NDArray {
  const a = toOp(left);
  const b = toOp(right);
  const unit = resultUnit(name, a, b);
  const shape = broadcastShapes(a.shape, b.shape);
  const ndim = shape.length;
  const sa = alignedStrides(a, ndim);
  const sb = alignedStrides(b, ndim);

  let size = 1;
  for (let d = 0; d < ndim; d++) size *= shape[d];

  const complex = !!(a.imag || b.imag);
  const outRe = new Float64Array(size);
  const outIm = complex ? new Float64Array(size) : undefined;
  if (size === 0) return new NDArray(outRe, { shape: Array.from(shape), imag: outIm, unit });

  // Walk the result one "line" (last axis) at a time: only the outer coordinates need index
  // arithmetic, and the operation is chosen once per line instead of once per element.
  const inner = ndim === 0 ? 1 : shape[ndim - 1];
  const stepA = ndim === 0 ? 0 : sa[ndim - 1];
  const stepB = ndim === 0 ? 0 : sb[ndim - 1];
  const outerDims = Math.max(ndim - 1, 0);
  const coords = new Int32Array(outerDims);
  const ad = a.data;
  const bd = b.data;

  for (let n = 0; n < size; n += inner) {
    const baseA = a.offset + dot(coords, sa, outerDims);
    const baseB = b.offset + dot(coords, sb, outerDims);

    if (!complex) {
      switch (name) {
        case 'add': for (let k = 0, ia = baseA, ib = baseB; k < inner; k++, ia += stepA, ib += stepB) outRe[n + k] = ad[ia] + bd[ib]; break;
        case 'sub': for (let k = 0, ia = baseA, ib = baseB; k < inner; k++, ia += stepA, ib += stepB) outRe[n + k] = ad[ia] - bd[ib]; break;
        case 'mul': for (let k = 0, ia = baseA, ib = baseB; k < inner; k++, ia += stepA, ib += stepB) outRe[n + k] = ad[ia] * bd[ib]; break;
        case 'div': for (let k = 0, ia = baseA, ib = baseB; k < inner; k++, ia += stepA, ib += stepB) outRe[n + k] = ad[ia] / bd[ib]; break;
        case 'pow': for (let k = 0, ia = baseA, ib = baseB; k < inner; k++, ia += stepA, ib += stepB) outRe[n + k] = Math.pow(ad[ia], bd[ib]); break;
      }
    } else {
      for (let k = 0, ia = baseA, ib = baseB; k < inner; k++, ia += stepA, ib += stepB) {
        const ar = ad[ia];
        const br = bd[ib];
        const ai = a.imag ? a.imag[ia] : 0;
        const bi = b.imag ? b.imag[ib] : 0;
        const o = n + k;
        switch (name) {
          case 'add': outRe[o] = ar + br; outIm![o] = ai + bi; break;
          case 'sub': outRe[o] = ar - br; outIm![o] = ai - bi; break;
          case 'mul': outRe[o] = ar * br - ai * bi; outIm![o] = ar * bi + ai * br; break;
          case 'div': {
            const den = br * br + bi * bi;
            outRe[o] = (ar * br + ai * bi) / den;
            outIm![o] = (ai * br - ar * bi) / den;
            break;
          }
          case 'pow': {
            const [r, i] = complexPow(ar, ai, br, bi);
            outRe[o] = r;
            outIm![o] = i;
            break;
          }
        }
      }
    }

    for (let d = outerDims - 1; d >= 0; d--) {
      if (++coords[d] < shape[d]) break;
      coords[d] = 0;
    }
  }

  return new NDArray(outRe, { shape: Array.from(shape), imag: outIm, unit });
}

function isScalar(x: unknown): x is Scalar {
  return typeof x === 'number' || x instanceof Complex || x instanceof Quantity;
}

function dispatch(name: BinaryName, l: any, r: any): any {
  if (isScalar(l) && isScalar(r)) return scalarOp(name, l, r);
  if (isScalar(l) || isScalar(r) || isArrayLike(l) || isArrayLike(r)) {
    if (!(isArrayLike(l) || isScalar(l)) || !(isArrayLike(r) || isScalar(r))) {
      throw new TypeError(`Unsupported operand types for ${name}`);
    }
    const a = isArrayLike(l) ? toNDArray(l) : l;
    const b = isArrayLike(r) ? toNDArray(r) : r;
    const out = binaryOp(name, a, b);
    return wrapLike(out, isArrayLike(l) ? l : r);
  }
  // Foreign scalar types (e.g. Fraction) that implement the method themselves.
  if (l && typeof l[name] === 'function') return l[name](r);
  throw new TypeError(`Unsupported operand types for ${name}`);
}

function scalarOp(name: BinaryName, l: Scalar, r: Scalar): Scalar {
  if (typeof l === 'number' && typeof r === 'number') {
    switch (name) {
      case 'add': return l + r;
      case 'sub': return l - r;
      case 'mul': return l * r;
      case 'div': return l / r;
      case 'pow': return Math.pow(l, r);
    }
  }
  if (l instanceof Quantity || r instanceof Quantity) {
    if (l instanceof Complex || r instanceof Complex) {
      throw new TypeError('Quantities with complex values are only supported inside arrays');
    }
    const ql = l instanceof Quantity ? l : new Quantity(l as number);
    if (name === 'pow') {
      if (typeof r !== 'number') throw new TypeError('The exponent must be a real number');
      return ql.pow(r);
    }
    const qr = r instanceof Quantity ? r : new Quantity(r as number);
    return ql[name](qr);
  }
  const zl = l instanceof Complex ? l : new Complex(l as number, 0);
  return name === 'pow' ? zl.pow(r as Complex | number) : zl[name](r as Complex | number);
}

export const sub = (a: Operand, b: Operand): any => dispatch('sub', a, b);
/** Element-wise product (Julia `.*`). */
export const mul = (a: Operand, b: Operand): any => dispatch('mul', a, b);
/** Element-wise division (Julia `./`). */
export const div = (a: Operand, b: Operand): any => dispatch('div', a, b);
/** Element-wise power (Julia `.^`). */
export const pow = (a: Operand, b: Operand): any => dispatch('pow', a, b);
export const addElementwise = (a: Operand, b: Operand): any => dispatch('add', a, b);

// --- Unary operations ---

function unaryMap(
  x: NDArray,
  fn: (re: number, im: number) => [number, number],
  opts: { keepUnit?: boolean; realResult?: boolean } = {}
): NDArray {
  const out = x.copy();
  const re = out.data as Float64Array;
  const im = out.imag as Float64Array | undefined;
  const outIm = opts.realResult ? undefined : new Float64Array(re.length);
  for (let i = 0; i < re.length; i++) {
    const [r, m] = fn(re[i], im ? im[i] : 0);
    re[i] = r;
    if (outIm) outIm[i] = m;
  }
  return new NDArray(re, {
    shape: Array.from(out.shape),
    imag: outIm && (x.isComplex || !outIm.every((v) => v === 0)) ? outIm : undefined,
    unit: opts.keepUnit ? x.unit : undefined,
  });
}

function mapAny(x: any, fn: (a: NDArray) => NDArray): any {
  return isArrayLike(x) ? wrapLike(fn(toNDArray(x)), x) : undefined;
}

export function neg(x: Operand): any {
  if (typeof x === 'number') return -x;
  if (x instanceof Complex) return new Complex(-x.re, -x.im);
  if (x instanceof Quantity) return new Quantity(-x.value, x.dims);
  return mapAny(x, (a) => unaryMap(a, (r, i) => [-r, -i], { keepUnit: true }));
}

export function conj(x: Operand): any {
  if (typeof x === 'number') return x;
  if (x instanceof Complex) return x.conj();
  if (x instanceof Quantity) return x;
  return mapAny(x, (a) => (a.isComplex ? unaryMap(a, (r, i) => [r, -i], { keepUnit: true }) : a));
}

export function real(x: Operand): any {
  if (typeof x === 'number') return x;
  if (x instanceof Complex) return x.re;
  if (x instanceof Quantity) return x;
  return mapAny(x, (a) => unaryMap(a, (r) => [r, 0], { keepUnit: true, realResult: true }));
}

export function imag(x: Operand): any {
  if (typeof x === 'number') return 0;
  if (x instanceof Complex) return x.im;
  if (x instanceof Quantity) return new Quantity(0, x.dims);
  return mapAny(x, (a) => unaryMap(a, (_r, i) => [i, 0], { keepUnit: true, realResult: true }));
}

/** Modulus |z| (or absolute value for real inputs), element-wise. */
export function abs(x: Operand): any {
  if (typeof x === 'number') return Math.abs(x);
  if (x instanceof Complex) return x.abs();
  if (x instanceof Quantity) return new Quantity(Math.abs(x.value), x.dims);
  return mapAny(x, (a) => unaryMap(a, (r, i) => [Math.hypot(r, i), 0], { keepUnit: true, realResult: true }));
}

/** Phase angle arg(z), element-wise. */
export function angle(x: Operand): any {
  if (typeof x === 'number') return x >= 0 ? 0 : Math.PI;
  if (x instanceof Complex) return x.arg();
  return mapAny(x, (a) => unaryMap(a, (r, i) => [Math.atan2(i, r), 0], { realResult: true }));
}

/** Applies a real function to every element of a real, unit-free array (Julia `f.(x)`). */
export function mapReal(x: ArrayLike_, fn: (v: number) => number): any {
  const a = toNDArray(x);
  if (a.isComplex) throw new TypeError('This function is not defined for complex arrays');
  if (a.unit) throw new TypeError('This function requires a dimensionless array');
  return wrapLike(unaryMap(a, (r) => [fn(r), 0], { realResult: true }), x);
}

/** Applies complex-valued and real functions to every element of an array (Julia `f.(x)`). */
export function mapComplex(
  x: ArrayLike_,
  fn: (re: number, im: number) => [number, number]
): any {
  const a = toNDArray(x);
  if (a.unit) throw new TypeError('This function requires a dimensionless array');
  return wrapLike(unaryMap(a, fn), x);
}
