/**
 * @file matrix.ts
 * Matrix fundamentals (Julia-style): products, norms, structural helpers, decompositions
 * and solvers. Every function accepts strided NDArrays and nested arrays
 * (NestedArray / number[][]); nested inputs return nested outputs.
 * Complex arrays and arrays with units are supported wherever it is mathematically meaningful;
 * the remaining functions reject them with an explicit error.
 */
import { NDArray } from '../core/ndarray.js';
import { NestedArray } from '../core/nested-array.js';
import { Complex } from '../types/complex.js';
import { Quantity, Dimensions } from '../units/units.js';
import {
  Operand,
  isArrayLike,
  toNDArray,
  wrapLike,
  mulDims,
  div,
} from '../ops/elementwise.js';
import { svd } from './factorizations.js';
import { gaussSolve, gaussInv } from './solver.js';

type MatrixLike = NDArray | NestedArray | any[];

function nd(x: MatrixLike): NDArray {
  if (!isArrayLike(x)) throw new TypeError('Expected an array');
  return toNDArray(x);
}

function plain(a: NDArray, fn: string): void {
  if (a.isComplex) throw new TypeError(`${fn} is not supported for complex arrays`);
  if (a.unit) throw new TypeError(`${fn} is not supported for arrays with units`);
}

function scalarOut(re: number, im: number, complex: boolean, unit?: Dimensions): number | Complex | Quantity {
  if (unit) {
    if (complex) throw new TypeError('Complex scalars with units are not supported');
    return new Quantity(re, unit);
  }
  return complex ? new Complex(re, im) : re;
}

function matrix2D(a: NDArray, fn: string): void {
  if (a.ndim !== 2) throw new Error(`${fn} requires a 2D matrix`);
}

// --- Products ---

interface MatView {
  rows: number;
  cols: number;
  s0: number;
  s1: number;
}

function view(a: NDArray, role: 'left' | 'right'): MatView {
  if (a.ndim === 2) return { rows: a.shape[0], cols: a.shape[1], s0: a.strides[0], s1: a.strides[1] };
  if (a.ndim === 1) {
    return role === 'left'
      ? { rows: 1, cols: a.shape[0], s0: 0, s1: a.strides[0] }
      : { rows: a.shape[0], cols: 1, s0: a.strides[0], s1: 0 };
  }
  throw new Error('matmul supports 1D and 2D operands only');
}

/**
 * Matrix product. 2D·2D → 2D, 2D·1D → 1D, 1D·2D → 1D and 1D·1D → scalar inner product (no conjugation).
 */
export function matmul(left: MatrixLike, right: MatrixLike): any {
  const A = nd(left);
  const B = nd(right);
  const va = view(A, 'left');
  const vb = view(B, 'right');
  if (va.cols !== vb.rows) {
    throw new Error(`Dimension mismatch in matrix product: [${Array.from(A.shape)}] and [${Array.from(B.shape)}]`);
  }

  const m = va.rows;
  const k = va.cols;
  const n = vb.cols;
  const complex = A.isComplex || B.isComplex;
  const outRe = new Float64Array(m * n);
  const outIm = complex ? new Float64Array(m * n) : undefined;

  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const ia = A.offset + i * va.s0 + p * va.s1;
      const ar = A.data[ia];
      const ai = A.imag ? A.imag[ia] : 0;
      for (let j = 0; j < n; j++) {
        const ib = B.offset + p * vb.s0 + j * vb.s1;
        const br = B.data[ib];
        if (!complex) {
          outRe[i * n + j] += ar * br;
        } else {
          const bi = B.imag ? B.imag[ib] : 0;
          outRe[i * n + j] += ar * br - ai * bi;
          outIm![i * n + j] += ar * bi + ai * br;
        }
      }
    }
  }

  const unit = mulDims(A.unit, B.unit, 1);
  if (A.ndim === 1 && B.ndim === 1) return scalarOut(outRe[0], outIm ? outIm[0] : 0, complex, unit);

  const shape = A.ndim === 1 ? [n] : B.ndim === 1 ? [m] : [m, n];
  const out = new NDArray(outRe, { shape, imag: outIm, unit });
  return wrapLike(out, isArrayLike(left) && !(left instanceof NDArray) ? left : right);
}

/** Inner product conj(a)·b of two vectors (Julia `dot`). */
export function dot(a: MatrixLike, b: MatrixLike): number | Complex | Quantity {
  const A = nd(a);
  const B = nd(b);
  if (A.size !== B.size) throw new Error('dot requires arrays with the same number of elements');
  const ac = A.copy();
  const bc = B.copy();
  let re = 0;
  let im = 0;
  for (let i = 0; i < ac.size; i++) {
    const xr = ac.data[i];
    const xi = ac.imag ? -ac.imag[i] : 0;
    const yr = bc.data[i];
    const yi = bc.imag ? bc.imag[i] : 0;
    re += xr * yr - xi * yi;
    im += xr * yi + xi * yr;
  }
  return scalarOut(re, im, A.isComplex || B.isComplex, mulDims(A.unit, B.unit, 1));
}

/** Outer product a·bᵀ of two vectors (flattened). */
export function outer(a: MatrixLike, b: MatrixLike): any {
  const A = nd(a).copy();
  const B = nd(b).copy();
  const col = new NDArray(A.data, { shape: [A.size, 1], imag: A.imag, unit: A.unit });
  const row = new NDArray(B.data, { shape: [1, B.size], imag: B.imag, unit: B.unit });
  return wrapLike(matmul(col, row), a instanceof NDArray ? b : a);
}

/** Kronecker product of two 2D arrays (1D inputs are treated as row vectors). */
export function kron(a: MatrixLike, b: MatrixLike): any {
  const A = nd(a);
  const B = nd(b);
  const as = A.ndim === 1 ? [1, A.shape[0]] : Array.from(A.shape);
  const bs = B.ndim === 1 ? [1, B.shape[0]] : Array.from(B.shape);
  const ac = A.copy();
  const bc = B.copy();
  const rows = as[0] * bs[0];
  const cols = as[1] * bs[1];
  const complex = A.isComplex || B.isComplex;
  const re = new Float64Array(rows * cols);
  const im = complex ? new Float64Array(rows * cols) : undefined;
  for (let i = 0; i < as[0]; i++)
    for (let j = 0; j < as[1]; j++)
      for (let k = 0; k < bs[0]; k++)
        for (let l = 0; l < bs[1]; l++) {
          const xa = i * as[1] + j;
          const xb = k * bs[1] + l;
          const o = (i * bs[0] + k) * cols + (j * bs[1] + l);
          const ar = ac.data[xa], ai = ac.imag ? ac.imag[xa] : 0;
          const br = bc.data[xb], bi = bc.imag ? bc.imag[xb] : 0;
          re[o] = ar * br - ai * bi;
          if (im) im[o] = ar * bi + ai * br;
        }
  const out = new NDArray(re, { shape: [rows, cols], imag: im, unit: mulDims(A.unit, B.unit, 1) });
  return wrapLike(out, a instanceof NDArray ? b : a);
}

// --- Structure ---

/** Conjugate transpose A' (a 1D vector becomes a 1×n row). */
export function adjoint(x: MatrixLike): any {
  const A = nd(x);
  if (A.ndim > 2) throw new Error('adjoint requires a 1D or 2D array');
  // A 1D vector is a column, so its adjoint is a 1×n row.
  const sr = A.shape[0];
  const sc = A.ndim === 1 ? 1 : A.shape[1];
  const src = A.copy();
  const re = new Float64Array(src.size);
  const im = src.imag ? new Float64Array(src.size) : undefined;
  for (let i = 0; i < sr; i++) {
    for (let j = 0; j < sc; j++) {
      re[j * sr + i] = src.data[i * sc + j];
      if (im) im[j * sr + i] = -src.imag![i * sc + j];
    }
  }
  return wrapLike(new NDArray(re, { shape: [sc, sr], imag: im, unit: A.unit }), x);
}

/** Plain (non-conjugating) transpose copy; 1D arrays become 1×n rows. */
export function transposeCopy(x: MatrixLike): NDArray {
  const T = adjoint(nd(x)) as NDArray;
  if (!T.imag) return T;
  const im = new Float64Array(T.imag.length);
  for (let i = 0; i < im.length; i++) im[i] = -T.imag[i];
  return new NDArray(T.data, { shape: Array.from(T.shape), imag: im, unit: T.unit });
}

export function trace(x: MatrixLike): number | Complex | Quantity {
  const A = nd(x);
  matrix2D(A, 'trace');
  let re = 0;
  let im = 0;
  for (let i = 0; i < Math.min(A.shape[0], A.shape[1]); i++) {
    const k = A.indexOf(i, i);
    re += A.data[k];
    if (A.imag) im += A.imag[k];
  }
  return scalarOut(re, im, A.isComplex, A.unit);
}

/** Vector → diagonal matrix; matrix → k-th diagonal as a vector. */
export function diag(x: MatrixLike, k = 0): any {
  const A = nd(x);
  if (A.ndim === 1) {
    const n = A.shape[0] + Math.abs(k);
    const re = new Float64Array(n * n);
    const im = A.imag ? new Float64Array(n * n) : undefined;
    for (let i = 0; i < A.shape[0]; i++) {
      const r = k >= 0 ? i : i - k;
      const c = k >= 0 ? i + k : i;
      const src = A.indexOf(i);
      re[r * n + c] = A.data[src];
      if (im) im[r * n + c] = A.imag![src];
    }
    return wrapLike(new NDArray(re, { shape: [n, n], imag: im, unit: A.unit }), x);
  }
  matrix2D(A, 'diag');
  const r0 = k >= 0 ? 0 : -k;
  const c0 = k >= 0 ? k : 0;
  const len = Math.max(0, Math.min(A.shape[0] - r0, A.shape[1] - c0));
  const re = new Float64Array(len);
  const im = A.imag ? new Float64Array(len) : undefined;
  for (let i = 0; i < len; i++) {
    const src = A.indexOf(r0 + i, c0 + i);
    re[i] = A.data[src];
    if (im) im[i] = A.imag![src];
  }
  return wrapLike(new NDArray(re, { shape: [len], imag: im, unit: A.unit }), x);
}

function triangle(x: MatrixLike, k: number, keep: (i: number, j: number, k: number) => boolean): any {
  const A = nd(x);
  matrix2D(A, 'triu/tril');
  const out = A.copy();
  for (let i = 0; i < A.shape[0]; i++) {
    for (let j = 0; j < A.shape[1]; j++) {
      if (!keep(i, j, k)) {
        (out.data as Float64Array)[i * A.shape[1] + j] = 0;
        if (out.imag) (out.imag as Float64Array)[i * A.shape[1] + j] = 0;
      }
    }
  }
  return wrapLike(out, x);
}

/** Upper triangle (elements with j - i >= k). */
export const triu = (x: MatrixLike, k = 0): any => triangle(x, k, (i, j, kk) => j - i >= kk);
/** Lower triangle (elements with j - i <= k). */
export const tril = (x: MatrixLike, k = 0): any => triangle(x, k, (i, j, kk) => j - i <= kk);

// --- Norms ---

/** Vectorised p-norm of all elements (Julia `norm`); p = 2 gives the Euclidean / Frobenius norm. */
export function norm(x: MatrixLike, p: number = 2): number | Quantity {
  const A = nd(x);
  const c = A.copy();
  let acc = 0;
  for (let i = 0; i < c.size; i++) {
    const m = Math.hypot(c.data[i], c.imag ? c.imag[i] : 0);
    if (p === Infinity) acc = Math.max(acc, m);
    else if (p === 1) acc += m;
    else if (p === 2) acc += m * m;
    else acc += Math.pow(m, p);
  }
  const value = p === Infinity || p === 1 ? acc : p === 2 ? Math.sqrt(acc) : Math.pow(acc, 1 / p);
  return A.unit ? new Quantity(value, A.unit) : value;
}

/** Singular values in descending order (real matrices). */
export function singularValues(x: MatrixLike): Float64Array {
  const A = nd(x);
  plain(A, 'singularValues');
  matrix2D(A, 'singularValues');
  const T = A.shape[0] < A.shape[1] ? transposeCopy(A) : A;
  return Float64Array.from(svd(T).S.data).sort((a, b) => b - a);
}

/** Operator norm of a matrix: p = 1 (max column sum), Infinity (max row sum) or 2 (largest singular value). */
export function opnorm(x: MatrixLike, p: number = 2): number | Quantity {
  const A = nd(x);
  matrix2D(A, 'opnorm');
  const c = A.copy();
  const [m, n] = [A.shape[0], A.shape[1]];
  const mag = (i: number, j: number) => Math.hypot(c.data[i * n + j], c.imag ? c.imag[i * n + j] : 0);
  let value: number;
  if (p === 1) {
    value = 0;
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += mag(i, j);
      value = Math.max(value, s);
    }
  } else if (p === Infinity) {
    value = 0;
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += mag(i, j);
      value = Math.max(value, s);
    }
  } else if (p === 2) {
    plain(A, 'opnorm(A, 2)');
    value = singularValues(A)[0] ?? 0;
  } else {
    throw new Error('opnorm supports p = 1, 2 or Infinity');
  }
  return A.unit ? new Quantity(value, A.unit) : value;
}

// --- Decompositions ---

/** Cholesky factor L (lower triangular, A = L·Lᵀ) of a symmetric positive-definite matrix. */
export function cholesky(x: MatrixLike): any {
  const A = nd(x);
  plain(A, 'cholesky');
  matrix2D(A, 'cholesky');
  const n = A.shape[0];
  if (A.shape[1] !== n) throw new Error('cholesky requires a square matrix');
  const a = A.copy().data as Float64Array;
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      if (Math.abs(a[i * n + j] - a[j * n + i]) > 1e-10 * (1 + Math.abs(a[i * n + j]))) {
        throw new Error('cholesky requires a symmetric matrix');
      }
      let s = a[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) throw new Error('Matrix is not positive definite');
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return wrapLike(new NDArray(L, { shape: [n, n] }), x);
}

export interface EigenResult {
  values: NDArray;
  vectors: NDArray;
}

/**
 * Eigen-decomposition of a real symmetric matrix by cyclic Jacobi rotations.
 * Eigenvalues are sorted ascending; eigenvectors are the columns of `vectors`.
 */
export function eigen(x: MatrixLike): EigenResult {
  const A = nd(x);
  plain(A, 'eigen');
  matrix2D(A, 'eigen');
  const n = A.shape[0];
  if (A.shape[1] !== n) throw new Error('eigen requires a square matrix');
  const a = A.copy().data as Float64Array;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (Math.abs(a[i * n + j] - a[j * n + i]) > 1e-10 * (1 + Math.abs(a[i * n + j]))) {
        throw new Error('eigen currently supports symmetric matrices only');
      }

  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i * n + j] * a[i * n + j];
    if (off < 1e-30) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k];
          const aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p];
          const vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i * n + i] - a[j * n + j]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  order.forEach((src, dst) => {
    values[dst] = a[src * n + src];
    for (let k = 0; k < n; k++) vectors[k * n + dst] = V[k * n + src];
  });
  return {
    values: wrapLike(new NDArray(values, { shape: [n] }), x),
    vectors: wrapLike(new NDArray(vectors, { shape: [n, n] }), x),
  };
}

/** Moore–Penrose pseudo-inverse via the SVD (real matrices). */
export function pinv(x: MatrixLike, tol?: number): any {
  const A = nd(x);
  plain(A, 'pinv');
  matrix2D(A, 'pinv');
  const [m, n] = [A.shape[0], A.shape[1]];
  if (m < n) {
    return wrapLike(transposeCopy(pinv(transposeCopy(A), tol) as NDArray), x);
  }
  const { U, S, V } = svd(A);
  const smax = Math.max(0, ...Array.from(S.data));
  const cutoff = tol ?? Math.max(m, n) * Number.EPSILON * smax;
  const out = new Float64Array(n * m);
  for (let k = 0; k < n; k++) {
    const s = S.data[k];
    if (s <= cutoff) continue;
    for (let i = 0; i < n; i++) {
      const vik = V.data[i * n + k] / s;
      for (let j = 0; j < m; j++) out[i * m + j] += vik * U.data[j * n + k];
    }
  }
  return wrapLike(new NDArray(out, { shape: [n, m] }), x);
}

/** Least-squares (minimum-norm) solution of A·x ≈ b for any matrix shape. */
export function lstsq(A: MatrixLike, b: MatrixLike): any {
  return matmul(pinv(A), b);
}

/** Numerical rank from the singular values. */
export function rank(x: MatrixLike, tol?: number): number {
  const s = singularValues(x);
  const A = nd(x);
  const cutoff = tol ?? Math.max(A.shape[0], A.shape[1]) * Number.EPSILON * (s[0] ?? 0);
  return s.filter((v) => v > cutoff).length;
}

/** 2-norm condition number σmax / σmin (Infinity for singular matrices). */
export function cond(x: MatrixLike): number {
  const s = singularValues(x);
  const smin = s[s.length - 1] ?? 0;
  return smin === 0 ? Infinity : s[0] / smin;
}

// --- Powers and division ---

/** Integer matrix power A^k by repeated squaring; negative k uses the inverse. */
export function matpow(x: MatrixLike, k: number): any {
  const A = nd(x);
  matrix2D(A, 'matpow');
  const n = A.shape[0];
  if (A.shape[1] !== n) throw new Error('matpow requires a square matrix');
  if (!Number.isInteger(k)) throw new Error('Matrix powers are only supported for integer exponents');

  let base: NDArray = k < 0 ? gaussInv(A) : A;
  let e = Math.abs(k);
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  let result = new NDArray(I, { shape: [n, n] });
  while (e > 0) {
    if (e & 1) result = matmul(result, base) as NDArray;
    e >>= 1;
    if (e) base = matmul(base, base) as NDArray;
  }
  return wrapLike(result, x);
}

/** Left division A\b: square systems are solved exactly, others in the least-squares sense. */
export function ldiv(A: Operand, b: Operand): any {
  if (!isArrayLike(A)) return div(b, A);
  const An = nd(A);
  const bn = nd(b as MatrixLike);
  if (An.ndim === 2 && An.shape[0] === An.shape[1]) {
    return wrapLike(gaussSolve(An, bn), A instanceof NDArray ? undefined : A);
  }
  return lstsq(A as MatrixLike, b as MatrixLike);
}

/** Right division A/B = A·B⁻¹, computed as (Bᵀ\Aᵀ)ᵀ. */
export function rdiv(A: Operand, B: Operand): any {
  if (!isArrayLike(B)) return div(A, B);
  const an = nd(A as MatrixLike);
  const At = an.ndim === 1 ? an : transposeCopy(an);
  const Xt = ldiv(transposeCopy(B as MatrixLike), At) as NDArray;
  const out = an.ndim === 1 ? Xt : transposeCopy(Xt);
  return wrapLike(out, A instanceof NDArray ? (B instanceof NDArray ? undefined : B) : A);
}
