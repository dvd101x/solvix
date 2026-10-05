import { describe, expect, it } from 'vitest';
import { NDArray, Complex, mtimes, mpower, mrdiv, mldiv, ctranspose, evaluate, meter } from '../src/index.js';

const nest = (x: any) => (x instanceof NDArray ? x.toNestedArray() : x);
const A = NDArray.fromArray([[1, 2], [3, 4]]);

describe('mtimes (Julia *)', () => {
  it('scalar times array is element-wise, in both orders', () => {
    expect(nest(mtimes(2, A))).toEqual([[2, 4], [6, 8]]);
    expect(nest(mtimes(A, 2))).toEqual([[2, 4], [6, 8]]);
    expect(nest(mtimes(new Complex(0, 1), NDArray.fromArray([1]))).length).toBe(1);
  });
  it('matrix · matrix and matrix · vector', () => {
    expect(nest(mtimes(A, A))).toEqual([[7, 10], [15, 22]]);
    expect(nest(mtimes(A, NDArray.fromArray([1, 1])))).toEqual([3, 7]);
  });
  it('vector * vector is an error; column vector times 1×n row is an outer product', () => {
    expect(() => mtimes(NDArray.fromArray([1, 2]), NDArray.fromArray([3, 4]))).toThrow(/dot/);
    expect(nest(mtimes(NDArray.fromArray([1, 2]), NDArray.fromArray([[3, 4]])))).toEqual([[3, 4], [6, 8]]);
    expect(() => mtimes(NDArray.fromArray([1, 2]), A)).toThrow(/mismatch/);
  });
  it('row (1×n) times vector is an inner product', () => {
    expect(mtimes(NDArray.fromArray([[1, 2]]), NDArray.fromArray([3, 4]))).toBe(11);
  });
  it('nested inputs give nested outputs', () => {
    expect(mtimes([[1, 2], [3, 4]], [[1, 0], [0, 1]])).toEqual([[1, 2], [3, 4]]);
    expect(mtimes([1, 2], [[3, 4]])).toEqual([[3, 4], [6, 8]]);
  });
  it('keeps units through products', () => {
    const d = A.withUnit(meter);
    const r = mtimes(d, d) as NDArray;
    expect(r.unit?.m).toBe(2);
    expect(Array.from(r.copy().data)).toEqual([7, 10, 15, 22]);
  });
});

describe('mpower, mrdiv, mldiv, ctranspose', () => {
  it('mpower: matrix power and scalars', () => {
    expect(nest(mpower(A, 2))).toEqual([[7, 10], [15, 22]]);
    expect(mpower(2, 3)).toBe(8);
    expect(() => mpower(2, A)).toThrow(/\.\^/);
    expect(() => mpower(A, new Complex(1, 1) as any)).toThrow(/integer/);
  });
  it('mrdiv: x / A solves from the right; scalar / array is rejected', () => {
    expect(() => mrdiv(2, A)).toThrow(/\.\//);
    const X = NDArray.fromArray([[1, 2]]);
    const r = mrdiv(X, A) as NDArray; // X = r * A
    const back = mtimes(r, A) as NDArray;
    expect(back.get(0, 0)).toBeCloseTo(1);
    expect(back.get(0, 1)).toBeCloseTo(2);
    expect(mrdiv(6, 3)).toBe(2);
    expect(nest(mrdiv(A, 2))).toEqual([[0.5, 1], [1.5, 2]]);
  });
  it('mldiv solves A \\ b, including least squares for tall matrices', () => {
    const x = mldiv(A, NDArray.fromArray([5, 11])) as NDArray;
    expect(x.data[0]).toBeCloseTo(1);
    expect(x.data[1]).toBeCloseTo(2);
    const tall = NDArray.fromArray([[1, 0], [0, 1], [1, 1]]);
    const ls = mldiv(tall, NDArray.fromArray([1, 2, 3])) as NDArray;
    expect(ls.data[0]).toBeCloseTo(1);
    expect(ls.data[1]).toBeCloseTo(2);
  });
  it('ctranspose conjugates complex matrices and scalars', () => {
    const M = NDArray.fromArray([[new Complex(1, 1), new Complex(2, 2)]]);
    const T = ctranspose(M) as NDArray;
    expect(Array.from(T.shape)).toEqual([2, 1]);
    expect(T.getComplex(1, 0)).toEqual(new Complex(2, -2));
    expect(ctranspose(new Complex(1, 2))).toEqual(new Complex(1, -2));
    expect(ctranspose(3)).toBe(3);
    expect(nest(ctranspose(NDArray.fromArray([1, 2])))).toEqual([[1, 2]]);
  });
  it('are reachable from the parser', () => {
    expect(nest(evaluate('A^3', { A }))[0][0]).toBe(37);
    expect(nest(evaluate("A'*A", { A }))).toEqual([[10, 14], [14, 20]]);
    expect(nest(evaluate('A/2', { A }))).toEqual([[0.5, 1], [1.5, 2]]);
    expect(() => evaluate('2/A', { A })).toThrow();
    expect(() => evaluate('A%2', { A })).toThrow(/real numbers/);
  });
});
