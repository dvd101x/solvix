/**
 * @file operators.ts
 * Julia-style operator semantics used by the expression parser:
 *   A * B   matrix / matrix-vector product (scalars multiply element-wise)
 *   A .* B  element-wise product with broadcasting
 *   A \ b   left division (linear solve or least squares)
 *   A / B   right division
 *   A ^ n   integer matrix power; .^ is element-wise
 *   A'      conjugate transpose
 */
import { NDArray } from '../core/ndarray.js';
import { Complex } from '../types/complex.js';
import { Quantity } from '../units/units.js';
import { matmul, matpow, ldiv, rdiv, adjoint } from '../linalg/matrix.js';
import { isArrayLike, toNDArray, wrapLike, mul, div, pow, conj } from './elementwise.js';

const isScalarLike = (x: unknown) => !isArrayLike(x);

/** Julia `*`: scalar·array is element-wise, array·array is a matrix product. */
export function mtimes(l: any, r: any): any {
  if (isScalarLike(l) || isScalarLike(r)) return mul(l, r);

  const A = toNDArray(l);
  const B = toNDArray(r);
  if (A.ndim === 1 && B.ndim === 1) {
    throw new TypeError('Cannot multiply two vectors with *; use dot(a, b) or a .* b');
  }

  if (A.ndim === 1 && B.ndim === 2) {
    if (B.shape[0] !== 1) {
      throw new Error(`Dimension mismatch: vector [${A.shape[0]}] * matrix [${Array.from(B.shape)}]`);
    }
    const column = A.view({ shape: [A.shape[0], 1], strides: [A.strides[0], 0] });
    return wrapLike(matmul(column, B), isArrayLike(l) && !(l instanceof NDArray) ? l : r);
  }

  // (1×n)·(n) behaves like an inner product of an adjoint row vector with a vector.
  if (A.ndim === 2 && A.shape[0] === 1 && B.ndim === 1) {
    return matmul(A.view({ shape: [A.shape[1]], strides: [A.strides[1]] }), B);
  }
  return matmul(l, r);
}

/** Julia `^`: scalar power, or integer power of a square matrix. */
export function mpower(l: any, r: any): any {
  if (isArrayLike(l)) {
    if (typeof r !== 'number') throw new TypeError('Matrix powers require a real integer exponent');
    return matpow(l, r);
  }
  if (isArrayLike(r)) throw new TypeError('scalar ^ array is not defined; use .^ for element-wise power');
  return pow(l, r);
}

/** Julia `/`. */
export function mrdiv(l: any, r: any): any {
  if (isScalarLike(l) && isArrayLike(r)) throw new TypeError('scalar / array is not defined; use ./ for element-wise division');
  return rdiv(l, r);
}

/** Julia `\`. */
export function mldiv(l: any, r: any): any {
  return ldiv(l, r);
}

/** Julia postfix `'`. */
export function ctranspose(x: any): any {
  if (isArrayLike(x)) return adjoint(x);
  if (x instanceof Complex) return x.conj();
  return conj(x);
}

