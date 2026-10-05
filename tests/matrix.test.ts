import { describe, expect, it } from 'vitest';
import {
  transpose, NDArray, Complex, matmul, dot, outer, kron, adjoint, trace, diag, triu, tril,
  norm, opnorm, cholesky, eigen, pinv, lstsq, rank, cond, matpow, det, inv, solve,
} from '../src/index.js';

const A = [[1, 2], [3, 4]];
const nd = (x: number[][]) => NDArray.fromArray(x);
const close = (a: any, b: any, d = 8) => {
  const fa = (a instanceof NDArray ? a.toNestedArray() : a) as any;
  expect(JSON.stringify(fa, (_, v) => (typeof v === 'number' ? +v.toFixed(d) : v))).toBe(
    JSON.stringify(b, (_, v) => (typeof v === 'number' ? +v.toFixed(d) : v))
  );
};

describe('matrix fundamentals (strided and nested)', () => {
  it('matmul', () => {
    close(matmul(nd(A), nd(A)), [[7, 10], [15, 22]]);
    expect(matmul(A, A)).toEqual([[7, 10], [15, 22]]);
    // strided (transposed) view
    close(matmul(transpose(nd(A)), nd(A)), [[10, 14], [14, 20]]);
    expect(() => matmul(nd(A), nd([[1, 2, 3]]))).toThrow();
  });

  it('dot, outer, kron', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(dot(NDArray.fromArray([1, 2, 3]), NDArray.fromArray([4, 5, 6]))).toBe(32);
    close(outer(NDArray.fromArray([1, 2]), NDArray.fromArray([3, 4])), [[3, 4], [6, 8]]);
    expect(outer([1, 2], [3, 4])).toEqual([[3, 4], [6, 8]]);
    close(kron(nd([[1, 2]]), nd([[1], [1]])), [[1, 2], [1, 2]]);
  });

  it('adjoint, trace, diag, triu, tril', () => {
    expect(adjoint(A)).toEqual([[1, 3], [2, 4]]);
    close(adjoint(nd(A)), [[1, 3], [2, 4]]);
    expect(trace(A)).toBe(5);
    expect(trace(transpose(nd(A)))).toBe(5);
    close(diag(nd(A)), [1, 4]);
    close(diag(NDArray.fromArray([1, 2])), [[1, 0], [0, 2]]);
    expect(triu(A)).toEqual([[1, 2], [0, 4]]);
    close(tril(nd(A)), [[1, 0], [3, 4]]);
  });

  it('norms', () => {
    expect(norm([3, 4])).toBeCloseTo(5);
    expect(norm(NDArray.fromArray([3, 4]), 1)).toBeCloseTo(7);
    expect(norm(nd(A))).toBeCloseTo(Math.sqrt(30));
    expect(opnorm(A, 1)).toBeCloseTo(6);
    expect(opnorm(nd(A), Infinity)).toBeCloseTo(7);
    expect(opnorm(nd([[3, 0], [0, 1]]), 2)).toBeCloseTo(3);
  });

  it('cholesky, eigen', () => {
    const S = [[4, 2], [2, 3]];
    const L = cholesky(nd(S)) as NDArray;
    close(matmul(L, transpose(L)), S);
    expect(cholesky(S)).toBeInstanceOf(Array);
    expect(() => cholesky([[1, 2], [2, 1]])).toThrow();
    const e: any = eigen(nd([[2, 0], [0, 3]]));
    close(e.values, [2, 3]);
  });

  it('pinv, lstsq, rank, cond, matpow', () => {
    close(matmul(nd(A), pinv(nd(A)) as NDArray), [[1, 0], [0, 1]], 6);
    const x = lstsq(nd([[1, 0], [0, 1], [1, 1]]), NDArray.fromArray([1, 2, 3])) as NDArray;
    close(x, [1, 2], 6);
    expect(rank(nd([[1, 2], [2, 4]]))).toBe(1);
    expect(rank(A)).toBe(2);
    expect(cond(nd([[2, 0], [0, 1]]))).toBeCloseTo(2);
    expect(matpow(A, 2)).toEqual([[7, 10], [15, 22]]);
    close(matpow(nd(A), 0), [[1, 0], [0, 1]]);
  });

  it('det, inv, solve on strided and nested inputs', () => {
    expect(det(nd(A))).toBeCloseTo(-2);
    expect(det(transpose(nd(A)))).toBeCloseTo(-2);
    close(solve(nd(A), NDArray.fromArray([5, 11])), [1, 2], 6);
    close(solve(A as any, [5, 11] as any) as any, [1, 2], 6);
    close(matmul(nd(A), inv(nd(A))), [[1, 0], [0, 1]], 6);
  });

  it('complex matrices', () => {
    const M = NDArray.fromArray([[new Complex(0, 1), 0], [0, new Complex(0, 1)]]);
    const P = matmul(M, M) as NDArray;
    expect(Array.from(P.data)).toEqual([-1, 0, 0, -1]);
    const d: any = det(M);
    expect(d.re).toBeCloseTo(-1);
    const i2 = matmul(M, inv(M)) as NDArray;
    expect(i2.getComplex(0, 0).re).toBeCloseTo(1);
    expect(adjoint(M)).toBeInstanceOf(NDArray);
    expect((adjoint(M) as NDArray).getComplex(0, 0).im).toBeCloseTo(-1);
  });
});
