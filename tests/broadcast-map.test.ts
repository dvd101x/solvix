import { describe, expect, it } from 'vitest';
import { broadcastMap, meter, evaluate, NDArray, NestedArray, Complex, transpose } from '../src/index.js';

const nest = (x: any) => (x instanceof NDArray ? x.toNestedArray() : x);

describe('broadcastMap', () => {
  it('maps two arrays of equal shape', () => {
    const out = broadcastMap((a, b) => a + b * 10, NDArray.fromArray([1, 2]), NDArray.fromArray([3, 4]));
    expect(nest(out)).toEqual([31, 42]);
  });
  it('broadcasts a column against a row', () => {
    const col = NDArray.fromArray([[1], [2]]);
    const row = NDArray.fromArray([[10, 20, 30]]);
    expect(nest(broadcastMap((a, b) => a * b, col, row))).toEqual([[10, 20, 30], [20, 40, 60]]);
  });
  it('broadcasts lower-rank arrays and scalars', () => {
    const M = NDArray.fromArray([[1, 2], [3, 4]]);
    expect(nest(broadcastMap((m, v, k) => m + v + k, M, NDArray.fromArray([10, 20]), 100))).toEqual([[111, 122], [113, 124]]);
  });
  it('supports three arrays', () => {
    const x = NDArray.fromArray([-5, 0.5, 9]);
    expect(nest(broadcastMap((v, lo, hi) => Math.min(Math.max(v, lo), hi), x, 0, 1))).toEqual([0, 0.5, 1]);
  });
  it('reads strided views correctly', () => {
    const A = NDArray.fromArray([[1, 2], [3, 4]]);
    expect(nest(broadcastMap((a, b) => a - b, transpose(A), A))).toEqual([[0, 1], [-1, 0]]);
  });
  it('keeps nested containers when no NDArray is involved', () => {
    expect(broadcastMap((a, b) => a + b, [[1, 2], [3, 4]], [10, 20])).toEqual([[11, 22], [13, 24]]);
    expect(broadcastMap((a, b) => a + b, new NestedArray([1, 2]), 1)).toBeInstanceOf(NestedArray);
    expect(broadcastMap((a, b) => a + b, NDArray.fromArray([1]), [1, 2])).toBeInstanceOf(NDArray);
  });
  it('passes only values to fn', () => {
    const calls: any[][] = [];
    broadcastMap((...v) => (calls.push(v), 0), NDArray.fromArray([1, 2]), NDArray.fromArray([3, 4]));
    expect(calls).toEqual([[1, 3], [2, 4]]);
  });
  it('handles complex inputs and outputs', () => {
    const z = NDArray.fromArray([new Complex(1, 1), new Complex(2, 0)]);
    const out: NDArray = broadcastMap((a, b) => (a as Complex).mul(b as Complex), z, new Complex(0, 1));
    expect(out.isComplex).toBe(true);
    expect(out.getComplex(0).re).toBe(-1);
    expect(out.getComplex(0).im).toBe(1);
  });
  it('stores booleans as 1/0', () => {
    expect(nest(broadcastMap((a, b) => a > b, NDArray.fromArray([1, 5]), 3))).toEqual([0, 1]);
  });
  it('calls fn once when there are no arrays', () => {
    expect(broadcastMap((a, b) => a + b, 1, 2)).toBe(3);
  });
  it('rejects incompatible shapes, units and bad results', () => {
    expect(() => broadcastMap((a, b) => a + b, NDArray.fromArray([1, 2]), NDArray.fromArray([1, 2, 3]))).toThrow();
    expect(() => broadcastMap((a) => a, NDArray.fromArray([1]).withUnit(meter))).toThrow();
    expect(() => broadcastMap(() => 'x' as any, NDArray.fromArray([1]))).toThrow();
  });
});

describe('multi-array calls in the parser', () => {
  const y = NDArray.fromArray([[1], [2]]);
  const x = NDArray.fromArray([[1, 2, 3]]);
  it('atan2.(y, x) and hypot.(a, b) broadcast', () => {
    const r = nest(evaluate('hypot.(y, x)', { y, x }));
    expect(r[1][2]).toBeCloseTo(Math.hypot(2, 3));
    expect(r).toHaveLength(2);
    expect(nest(evaluate('atan2.(y, 1)', { y }))[0][0]).toBeCloseTo(Math.PI / 4);
  });
  it('clamp.(x, lo, hi) and max.(a, b)', () => {
    expect(nest(evaluate('clamp.(v, 0, 1)', { v: [-1, 0.5, 2] }))).toEqual([0, 0.5, 1]);
    expect(nest(evaluate('max.(a, b)', { a: [1, 5], b: [3, 2] }))).toEqual([3, 5]);
  });
  it('map(f, a, b) and user functions with f.(a, b)', () => {
    const f = (a: number, b: number) => a * 10 + b;
    expect(nest(evaluate('map(f, y, x)', { f, y, x }))).toEqual([[11, 12, 13], [21, 22, 23]]);
    expect(nest(evaluate('f.(y, x)', { f, y, x }))).toEqual([[11, 12, 13], [21, 22, 23]]);
    expect(nest(evaluate('f.(3, x)', { f, x }))).toEqual([[31, 32, 33]]);
  });
  it('still requires the dot for built-ins on arrays', () => {
    expect(() => evaluate('hypot(y, x)', { y, x })).toThrow(/hypot\./);
  });
  it('rejects complex values in real-only functions', () => {
    expect(() => evaluate('atan2.(z, 1)', { z: [new Complex(1, 1)] })).toThrow(/complex/);
  });
});

import { broadcastInto, addInPlace, mulInPlace, where, clip } from '../src/index.js';

describe('collection functions built on broadcasting', () => {
  it('broadcastInto writes into strided outputs without allocating', () => {
    const base = NDArray.zeros([2, 2]);
    const out = transpose(base); // strided view over base.data
    broadcastInto(out, (a, b) => a + b, NDArray.fromArray([[1], [2]]), NDArray.fromArray([[10, 20]]));
    expect(nest(out)).toEqual([[11, 21], [12, 22]]);
    expect(Array.from(base.data)).toEqual([11, 12, 21, 22]);
    expect(() => broadcastInto(NDArray.zeros([3]), (a) => a, NDArray.fromArray([[1, 2]]))).toThrow(/shape/);
  });
  it('addInPlace / mulInPlace broadcast, handle views, scalars and aliasing', () => {
    const A = NDArray.fromArray([[1, 2], [3, 4]]);
    const out = NDArray.zeros([2, 2]);
    addInPlace(out, transpose(A), NDArray.fromArray([10, 20]));
    expect(nest(out)).toEqual([[11, 23], [12, 24]]);
    mulInPlace(out, out, 2);
    expect(nest(out)).toEqual([[22, 46], [24, 48]]);
    addInPlace(out, 1, 1);
    expect(nest(out)).toEqual([[2, 2], [2, 2]]);
  });
  it('in-place operations support complex outputs only when the output is complex', () => {
    const z = NDArray.fromArray([new Complex(1, 1), new Complex(2, 2)]);
    const out = NDArray.fromArray([new Complex(0, 0), new Complex(0, 0)]);
    mulInPlace(out, z, new Complex(0, 1));
    expect(out.getComplex(0).re).toBe(-1);
    expect(out.getComplex(1).im).toBe(2);
    expect(() => addInPlace(NDArray.zeros([2]), z, 1)).toThrow(/complex/);
  });
  it('where broadcasts condition, and both branches', () => {
    const cond = NDArray.fromArray([[1], [0]]);
    expect(nest(where(cond, NDArray.fromArray([[1, 2, 3]]), -1))).toEqual([[1, 2, 3], [-1, -1, -1]]);
  });
  it('clip accepts scalar or per-column bounds, and keeps units', () => {
    const A = NDArray.fromArray([[0, 5, 10], [20, 5, -3]]);
    expect(nest(clip(A, 0, 10))).toEqual([[0, 5, 10], [10, 5, 0]]);
    expect(nest(clip(A, NDArray.fromArray([[1, 2, 3]]), NDArray.fromArray([[4, 4, 4]])))).toEqual([[1, 4, 4], [4, 4, 3]]);
    const u = clip(NDArray.fromArray([1, 9]).withUnit(meter), 2, 5);
    expect(Array.from(u.data)).toEqual([2, 5]);
    expect(u.unit?.m).toBe(1);
  });
});
