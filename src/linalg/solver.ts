/**
 * @file solver.ts
 * Linear solver by Gaussian elimination with partial pivoting.
 * Works with real or complex strided arrays and propagates physical dimensions.
 */
import { NDArray } from '../core/ndarray.js';
import { Complex } from '../types/complex.js';
import { Quantity } from '../units/units.js';
import { mulDims } from '../ops/elementwise.js';

interface Work {
  re: Float64Array;
  im?: Float64Array;
}

function toWork(A: NDArray, forceComplex: boolean): Work {
  const c = A.copy();
  const re = c.data as Float64Array;
  const im = c.imag ? (c.imag as Float64Array) : forceComplex ? new Float64Array(re.length) : undefined;
  return { re, im };
}

function squareSize(A: NDArray): number {
  if (A.ndim !== 2 || A.shape[0] !== A.shape[1]) {
    throw new Error('Expected a square 2D matrix');
  }
  return A.shape[0];
}

/**
 * Eliminates [A | B] in place and returns the sign of the permutation.
 * `a` is n×n and `b` is n×k, both row-major.
 */
function eliminate(a: Work, b: Work | undefined, n: number, k: number): number {
  const { re: ar, im: ai } = a;
  let sign = 1;

  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = -1;
    for (let r = col; r < n; r++) {
      const mag = Math.hypot(ar[r * n + col], ai ? ai[r * n + col] : 0);
      if (mag > best) {
        best = mag;
        piv = r;
      }
    }
    if (best === 0 || !Number.isFinite(best)) throw new Error('Matrix is singular or near-singular');

    if (piv !== col) {
      sign = -sign;
      for (let j = 0; j < n; j++) {
        let t = ar[col * n + j];
        ar[col * n + j] = ar[piv * n + j];
        ar[piv * n + j] = t;
        if (ai) {
          t = ai[col * n + j];
          ai[col * n + j] = ai[piv * n + j];
          ai[piv * n + j] = t;
        }
      }
      if (b) {
        for (let j = 0; j < k; j++) {
          let t = b.re[col * k + j];
          b.re[col * k + j] = b.re[piv * k + j];
          b.re[piv * k + j] = t;
          if (b.im) {
            t = b.im[col * k + j];
            b.im[col * k + j] = b.im[piv * k + j];
            b.im[piv * k + j] = t;
          }
        }
      }
    }

    const pr = ar[col * n + col];
    const pi = ai ? ai[col * n + col] : 0;
    const den = pr * pr + pi * pi;

    for (let r = col + 1; r < n; r++) {
      const xr = ar[r * n + col];
      const xi = ai ? ai[r * n + col] : 0;
      if (xr === 0 && xi === 0) continue;
      // factor = x / pivot
      const fr = (xr * pr + xi * pi) / den;
      const fi = (xi * pr - xr * pi) / den;

      for (let j = col; j < n; j++) {
        const ur = ar[col * n + j];
        const ui = ai ? ai[col * n + j] : 0;
        ar[r * n + j] -= fr * ur - fi * ui;
        if (ai) ai[r * n + j] -= fr * ui + fi * ur;
      }
      if (b) {
        for (let j = 0; j < k; j++) {
          const ur = b.re[col * k + j];
          const ui = b.im ? b.im[col * k + j] : 0;
          b.re[r * k + j] -= fr * ur - fi * ui;
          if (b.im) b.im[r * k + j] -= fr * ui + fi * ur;
        }
      }
    }
  }
  return sign;
}

/** Solves A·X = B. B may be a vector (n) or a matrix (n×k). */
export function gaussSolve(A: NDArray, B: NDArray): NDArray {
  const n = squareSize(A);
  if (B.ndim !== 1 && B.ndim !== 2) throw new Error('Right-hand side must be 1D or 2D');
  if (B.shape[0] !== n) {
    throw new Error(`Dimension mismatch: A is [${n}x${n}], b has ${B.shape[0]} rows`);
  }
  const k = B.ndim === 1 ? 1 : B.shape[1];
  const complex = A.isComplex || B.isComplex;
  const a = toWork(A, complex);
  const b = toWork(B, complex);

  eliminate(a, b, n, k);

  const xr = new Float64Array(n * k);
  const xi = complex ? new Float64Array(n * k) : undefined;
  for (let j = 0; j < k; j++) {
    for (let i = n - 1; i >= 0; i--) {
      let sr = b.re[i * k + j];
      let si = b.im ? b.im[i * k + j] : 0;
      for (let c = i + 1; c < n; c++) {
        const ur = a.re[i * n + c];
        const ui = a.im ? a.im[i * n + c] : 0;
        const vr = xr[c * k + j];
        const vi = xi ? xi[c * k + j] : 0;
        sr -= ur * vr - ui * vi;
        si -= ur * vi + ui * vr;
      }
      const pr = a.re[i * n + i];
      const pi = a.im ? a.im[i * n + i] : 0;
      const den = pr * pr + pi * pi;
      xr[i * k + j] = (sr * pr + si * pi) / den;
      if (xi) xi[i * k + j] = (si * pr - sr * pi) / den;
    }
  }

  return new NDArray(xr, {
    shape: B.ndim === 1 ? [n] : [n, k],
    imag: xi,
    unit: mulDims(B.unit, A.unit, -1),
  });
}

/** Determinant of a square matrix; returns a Complex for complex input and a Quantity for unit arrays. */
export function gaussDet(A: NDArray): number | Complex | Quantity {
  const n = squareSize(A);
  const a = toWork(A, false);
  let re = 0;
  let im = 0;
  try {
    re = eliminate(a, undefined, n, 0);
    for (let i = 0; i < n; i++) {
      const dr = a.re[i * n + i];
      const di = a.im ? a.im[i * n + i] : 0;
      [re, im] = [re * dr - im * di, re * di + im * dr];
    }
  } catch {
    re = 0;
    im = 0;
  }
  if (A.unit) {
    if (A.isComplex) throw new TypeError('Determinant of complex matrices with units is not supported');
    return new Quantity(re, new Quantity(1, A.unit).pow(n).dims);
  }
  return A.isComplex ? new Complex(re, im) : re;
}

/** Inverse of a square matrix: solves A·X = I. */
export function gaussInv(A: NDArray): NDArray {
  const n = squareSize(A);
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  return gaussSolve(A, new NDArray(I, { shape: [n, n] }));
}
