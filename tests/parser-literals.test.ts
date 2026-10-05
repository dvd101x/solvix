import { describe, expect, it } from 'vitest';
import { evaluate, NDArray, Complex, mapElements, mapIndexed, toLaTeX } from '../src/index.js';

const nest = (x: any) => (x instanceof NDArray ? x.toNestedArray() : x);
const ev = (e: string, scope: Record<string, any> = {}) => nest(evaluate(e, scope));

describe('ranges', () => {
  it('start:stop and start:step:stop are inclusive', () => {
    expect(ev('1:5')).toEqual([1, 2, 3, 4, 5]);
    expect(ev('1:2:10')).toEqual([1, 3, 5, 7, 9]);
    expect(ev('5:-2:1')).toEqual([5, 3, 1]);
    expect(ev('0:0.1:0.3')).toHaveLength(4);
    expect(ev('3:1')).toEqual([]);
  });
  it('has lower precedence than arithmetic', () => {
    expect(ev('1:n+1', { n: 2 })).toEqual([1, 2, 3]);
    expect(ev('(1:3).*2')).toEqual([2, 4, 6]);
  });
  it('rejects a zero step', () => {
    expect(() => evaluate('1:0:5')).toThrow();
  });
});

describe('array literals', () => {
  it('commas and semicolons build vectors', () => {
    expect(ev('[1, 2, 3]')).toEqual([1, 2, 3]);
    expect(ev('[1; 2; 3]')).toEqual([1, 2, 3]);
  });
  it('spaces and semicolons build matrices', () => {
    expect(ev('[1 2; 3 4]')).toEqual([[1, 2], [3, 4]]);
    expect(ev('[1 2 3]')).toEqual([[1, 2, 3]]);
    expect(ev('[1 2;\n 3 4;]'.replace('\n', ' '))).toEqual([[1, 2], [3, 4]]);
  });
  it('treats "[1 -2]" as two elements and "[1 - 2]" / "[1-2]" as one', () => {
    expect(ev('[1 -2]')).toEqual([[1, -2]]);
    expect(ev('[1 - 2]')).toEqual([-1]);
    expect(ev('[1-2 3]')).toEqual([[-1, 3]]);
  });
  it('evaluates expressions, calls and implicit products inside', () => {
    expect(ev('[x*2 sqrt(9); 2x x^2]', { x: 3 })).toEqual([[6, 3], [6, 9]]);
    expect(ev('[sqrt(4) (1+1)]')).toEqual([[2, 2]]);
  });
  it('concatenates blocks', () => {
    const A = [[1, 2], [3, 4]];
    expect(ev('[A A]', { A })).toEqual([[1, 2, 1, 2], [3, 4, 3, 4]]);
    expect(ev('[A; A]', { A })).toEqual([[1, 2], [3, 4], [1, 2], [3, 4]]);
    expect(ev('[A [5; 6]]', { A })).toEqual([[1, 2, 5], [3, 4, 6]]);
    expect(() => evaluate('[A [1 2 3]]', { A })).toThrow(/heights/);
    expect(() => evaluate('[1 2; 3]')).toThrow(/widths/);
  });
  it('supports complex elements and ranges', () => {
    const z: NDArray = evaluate('[1+2*im, 3]');
    expect(z.isComplex).toBe(true);
    expect(z.getComplex(0).im).toBe(2);
    expect(ev('[1:3]')).toEqual([1, 2, 3]);
    expect(ev('[1:2; 5]')).toEqual([1, 2, 5]);
  });
  it('works with Julia operators', () => {
    expect(ev('[1 2; 3 4]*[1; 1]')).toEqual([3, 7]);
    expect(ev("[1 2; 3 4]'")).toEqual([[1, 3], [2, 4]]);
    expect(ev('[1 2; 3 4].*[1 2]')).toEqual([[1, 4], [3, 8]]);
  });
});

describe('1-based indexing', () => {
  const A = NDArray.fromArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  const v = NDArray.fromArray([10, 20, 30, 40]);
  it('scalar indices', () => {
    expect(evaluate('A[1, 1]', { A })).toBe(1);
    expect(evaluate('A[3,2]', { A })).toBe(8);
    expect(evaluate('v[2]', { v })).toBe(20);
    expect(evaluate('v[end]', { v })).toBe(40);
    expect(evaluate('A[end, end]', { A })).toBe(9);
    expect(evaluate('v[end-1]', { v })).toBe(30);
  });
  it('slices with ":" and ranges', () => {
    expect(ev('A[:, 2]', { A })).toEqual([2, 5, 8]);
    expect(ev('A[2, :]', { A })).toEqual([4, 5, 6]);
    expect(ev('A[1:2, 2:3]', { A })).toEqual([[2, 3], [5, 6]]);
    expect(ev('A[2:end, 1]', { A })).toEqual([4, 7]);
    expect(ev('v[2:3]', { v })).toEqual([20, 30]);
    expect(ev('v[[1, 3]]', { v })).toEqual([10, 30]);
    expect(ev('A[:, :]', { A })).toEqual(A.toNestedArray());
  });
  it('works on strided views and nested arrays', () => {
    const T = evaluate("A'", { A });
    expect(evaluate('T[1, 2]', { T })).toBe(4);
    expect(evaluate('M[2, 1]', { M: [[1, 2], [3, 4]] })).toBe(3);
    expect(evaluate('M[:, 1]', { M: [[1, 2], [3, 4]] })).toEqual([1, 3]);
  });
  it('indexes inside bigger expressions', () => {
    expect(evaluate('A[1, 2] + 2*A[2, 2]', { A })).toBe(12);
    expect(evaluate('v[1]*v[2]', { v })).toBe(200);
    expect(ev('[v[1] v[2]; v[3] v[4]]', { v })).toEqual([[10, 20], [30, 40]]);
    expect(ev('[v[1] v[2]]', { v })).toEqual([[10, 20]]);
  });
  it('keeps complex values', () => {
    const z = NDArray.fromArray([new Complex(1, 2), new Complex(3, 4)]);
    const r: Complex = evaluate('z[2]', { z });
    expect(r.re).toBe(3);
    expect(r.im).toBe(4);
  });
  it('reports errors', () => {
    expect(() => evaluate('v[0]', { v })).toThrow(/out of bounds/);
    expect(() => evaluate('v[5]', { v })).toThrow(/out of bounds/);
    expect(() => evaluate('A[1]', { A })).toThrow(/indices/);
    expect(() => evaluate('A[1.5, 1]', { A })).toThrow();
    expect(() => evaluate('3[1]')).toThrow();
    expect(() => evaluate(':')).toThrow();
  });
});

describe('mapping functions', () => {
  const x = NDArray.fromArray([1, 4, 9]);
  it('mapElements calls the function with the value only', () => {
    const seen: any[][] = [];
    const out = mapElements(x, function (...args: any[]) {
      seen.push(args);
      return Math.sqrt(args[0]);
    } as any);
    expect(out.toNestedArray()).toEqual([1, 2, 3]);
    expect(seen.every((a) => a.length === 1)).toBe(true);
  });
  it('mapIndexed passes (value, index, array)', () => {
    const out = mapIndexed([[1, 2], [3, 4]], (v, idx, arr) => {
      expect(Array.isArray(arr)).toBe(true);
      return (v as number) * 10 + idx[0] + idx[1];
    });
    expect(out).toEqual([[10, 21], [31, 42]]);
  });
  it('f.(x) and map(f, x) are unary', () => {
    const calls: any[][] = [];
    const f = (...a: any[]) => (calls.push(a), a[0] + 1);
    expect(ev('f.(x)', { f, x })).toEqual([2, 5, 10]);
    expect(ev('map(f, x)', { f, x })).toEqual([2, 5, 10]);
    expect(calls.every((a) => a.length === 1)).toBe(true);
    expect(ev('map(sqrt, x)', { x })).toEqual([1, 2, 3]);
  });
  it('mapIndexed in expressions receives value, index and array', () => {
    expect(ev('mapIndexed(f, x)', { f: (v: number, i: number[]) => v + i[0], x })).toEqual([1, 5, 11]);
  });
  it('size and length', () => {
    expect(ev('size(A)', { A: [[1, 2, 3], [4, 5, 6]] })).toEqual([2, 3]);
    expect(evaluate('size(A, 2)', { A: [[1, 2, 3], [4, 5, 6]] })).toBe(3);
    expect(evaluate('length(1:7)')).toBe(7);
  });
});

describe('LaTeX of new nodes', () => {
  it('renders matrices, ranges and indexing', () => {
    expect(toLaTeX('[1 2; 3 4]')).toBe('\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}');
    expect(toLaTeX('1:2:9')).toBe('1:2:9');
    expect(toLaTeX('A[1, 2]')).toBe('A_{1, 2}');
  });
});
