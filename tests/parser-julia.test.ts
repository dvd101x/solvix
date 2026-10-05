import { describe, expect, it } from 'vitest';
import { evaluate, NDArray, Complex } from '../src/index.js';

const A = NDArray.fromArray([[1, 2], [3, 4]]);
const b = NDArray.fromArray([1, 1]);
const nested = (x: any) => (x instanceof NDArray ? x.toNestedArray() : x);

describe('Julia-style parser', () => {
  it('A*b is the matrix-vector product', () => {
    expect(nested(evaluate('A*b', { A, b }))).toEqual([3, 7]);
    expect(nested(evaluate('A*b', { A: [[1, 2], [3, 4]], b: [1, 1] }))).toEqual([3, 7]);
    expect(nested(evaluate('A*A', { A }))).toEqual([[7, 10], [15, 22]]);
  });
  it('A.*b is element-wise with broadcasting', () => {
    expect(nested(evaluate('A.*b', { A, b }))).toEqual([[1, 2], [3, 4]]);
    expect(nested(evaluate('A.*A', { A }))).toEqual([[1, 4], [9, 16]]);
    expect(nested(evaluate('A.^2', { A }))).toEqual([[1, 4], [9, 16]]);
    expect(nested(evaluate('A./A', { A }))).toEqual([[1, 1], [1, 1]]);
  });
  it('scalars, transpose, power, solve', () => {
    expect(nested(evaluate('2*A', { A }))).toEqual([[2, 4], [6, 8]]);
    expect(nested(evaluate("A'", { A }))).toEqual([[1, 3], [2, 4]]);
    expect(nested(evaluate('A^2', { A }))).toEqual([[7, 10], [15, 22]]);
    const x = nested(evaluate('A\\b', { A, b: NDArray.fromArray([5, 11]) }));
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
  });
  it('broadcast calls', () => {
    const r = nested(evaluate('sqrt.(x)', { x: NDArray.fromArray([4, 9]) }));
    expect(r).toEqual([2, 3]);
    expect(() => evaluate('sqrt(x)', { x: NDArray.fromArray([4, 9]) })).toThrow(/sqrt\.\(x\)/);
  });
  it('complex unit im', () => {
    const z: Complex = evaluate('(1+2*im)*(3-im)');
    expect(z.re).toBe(5);
    expect(z.im).toBe(5);
    expect(evaluate('abs(3+4*im)')).toBeCloseTo(5);
  });
  it('precedence and numbers', () => {
    expect(evaluate('-2^2')).toBe(-4);
    expect(evaluate('2*3+4')).toBe(10);
    expect(evaluate('1e3+2.5')).toBe(1002.5);
    expect(evaluate('2.*3')).toBe(6);
  });
  it('dimension errors', () => {
    expect(() => evaluate('A*b', { A, b: NDArray.fromArray([1, 2, 3]) })).toThrow();
  });
  it('linear algebra builtins', () => {
    expect(evaluate('det(A)', { A })).toBeCloseTo(-2);
    expect(evaluate('trace(A)', { A })).toBe(5);
    expect(evaluate('norm(x)', { x: NDArray.fromArray([3, 4]) })).toBeCloseTo(5);
  });
});
