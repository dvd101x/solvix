import { describe, expect, it } from 'vitest';
import { NDArray, DataFrame, globalPool, tidy, evaluate } from '../src/index.js';

describe('buffer pool and tidy', () => {
  it('reuses released buffers of the same size', () => {
    const buf = globalPool.acquire(7);
    expect(buf.length).toBe(7);
    globalPool.release(buf);
    expect(globalPool.acquire(7)).toBe(buf);
    expect(globalPool.acquire(7)).not.toBe(buf);
  });
  it('tidy returns the callback result', () => {
    expect(tidy(() => 42)).toBe(42);
  });
});

describe('DataFrame', () => {
  const df = new DataFrame(NDArray.fromArray([[1, 2], [3, 4], [5, 6]]), { columns: ['a', 'b'] });
  it('validates construction', () => {
    expect(() => new DataFrame(NDArray.fromArray([1, 2]))).toThrow(/2D/);
    expect(() => new DataFrame(NDArray.fromArray([[1, 2]]), { columns: ['a'] })).toThrow(/Columns length/);
    expect(() => new DataFrame(NDArray.fromArray([[1, 2]]), { index: ['x', 'y'] })).toThrow(/Index length/);
    expect(new DataFrame([[1, 2]]).columns.length).toBe(2);
  });
  it('column access and errors', () => {
    expect(Array.from(df.col('b').copy().data)).toEqual([2, 4, 6]);
    expect(() => df.col('zzz')).toThrow(ReferenceError);
  });
});

describe('parser errors', () => {
  it.each(['[1 2', '(1+2', '1 + $', '1 +', ':', ')'])('rejects %s', (src) => {
    expect(() => evaluate(src, { a: 1, b: 2 })).toThrow();
  });
});
