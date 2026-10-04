import { describe, it, expect } from 'vitest';
import { NDArray } from '../../src/ix/core/ndarray.js';
import {
  linspace,
  arange,
  logspace,
  eye,
  meshgrid,
} from '../../src/ix/generators/ranges.js';
import {
  reshape,
  transpose,
  flatten,
  expandDims,
  squeeze,
  clip,
} from '../../src/ix/manipulation/manipulation.js';
import {
  sum,
  mean,
  std,
  variance,
  median,
  quantile,
  sumProduct,
  all,
  any,
  countNonzero,
  skew,
  kurtosis,
  describe as statsDescribe,
} from '../../src/ix/stats/reductions.js';

describe('ix / generators / ranges', () => {
  it('generates linspace correctly', () => {
    const arr = linspace(0, 10, 5);
    expect(Array.from(arr.data)).toEqual([0, 2.5, 5, 7.5, 10]);
  });

  it('handles singleton linspace and descending or empty ranges', () => {
    expect(Array.from(linspace(3, 9, 1).data)).toEqual([3]);
    expect(() => linspace(3, 9, 0)).toThrow(RangeError);
    expect(Array.from(arange(5, 0, -2).data)).toEqual([5, 3, 1]);
    expect(Array.from(arange(5, 5).data)).toEqual([]);
  });

  it('generates arange correctly', () => {
    const a1 = arange(5);
    expect(Array.from(a1.data)).toEqual([0, 1, 2, 3, 4]);

    const a2 = arange(1, 10, 3);
    expect(Array.from(a2.data)).toEqual([1, 4, 7]);
  });

  it('generates logspace and eye', () => {
    const logVals = logspace(1, 3, 3); // 10^1, 10^2, 10^3
    expect(Array.from(logVals.data)).toEqual([10, 100, 1000]);

    const identity = eye(2);
    expect(Array.from(identity.data)).toEqual([1, 0, 0, 1]);
  });

  it('creates rectangular identity matrices', () => {
    const identity = eye(2, 3);
    expect(Array.from(identity.shape)).toEqual([2, 3]);
    expect(Array.from(identity.data)).toEqual([1, 0, 0, 0, 1, 0]);
  });

  it('creates 2D meshgrid', () => {
    const x = new NDArray(new Float64Array([1, 2]), { shape: [2] });
    const y = new NDArray(new Float64Array([10, 20, 30]), { shape: [3] });
    const [X, Y] = meshgrid(x, y);

    expect(Array.from(X.shape)).toEqual([3, 2]);
    expect(Array.from(Y.shape)).toEqual([3, 2]);
    expect(X.get(0, 0)).toBe(1);
    expect(X.get(0, 1)).toBe(2);
    expect(Y.get(0, 0)).toBe(10);
    expect(Y.get(2, 0)).toBe(30);
  });
});

describe('ix / manipulation / tensor views in O(1)', () => {
  it('reshapes arrays with dimension inference (-1)', () => {
    const a = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [6] });
    const r = reshape(a, [2, -1]);

    expect(Array.from(r.shape)).toEqual([2, 3]);
    expect(r.get(1, 2)).toBe(6);
  });

  it('transposes 2D matrices in O(1) by swapping strides', () => {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    const a = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });
    const t = transpose(a);

    expect(Array.from(t.shape)).toEqual([3, 2]);
    expect(t.get(0, 0)).toBe(1);
    expect(t.get(0, 1)).toBe(4);
    expect(t.get(2, 1)).toBe(6);
  });

  it('flattens, expands, squeezes and clips tensors', () => {
    const a = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [2, 2] });
    expect(Array.from(flatten(a).data)).toEqual([1, 2, 3, 4]);

    const exp = expandDims(a, 1);
    expect(Array.from(exp.shape)).toEqual([2, 1, 2]);

    const sq = squeeze(exp, 1);
    expect(Array.from(sq.shape)).toEqual([2, 2]);

    const cl = clip(new NDArray(new Float64Array([-5, 2, 15]), { shape: [3] }), 0, 10);
    expect(Array.from(cl.data)).toEqual([0, 2, 10]);
  });
});

describe('ix / stats / reductions with axis and pandas statistics', () => {
  // Matriz 2x3:
  // [[1, 2, 3],
  //  [4, 5, 6]]
  const M = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });

  it('computes global and axis reductions', () => {
    // Suma global
    expect(sum(M)).toBe(21);

    // Suma por axis=0 (colapsa filas -> [1+4, 2+5, 3+6] = [5, 7, 9])
    const sum0 = sum(M, { axis: 0 }) as NDArray;
    expect(Array.from(sum0.shape)).toEqual([3]);
    expect(Array.from(sum0.data)).toEqual([5, 7, 9]);

    // Suma por axis=1 (colapsa columnas -> [1+2+3, 4+5+6] = [6, 15])
    const sum1 = sum(M, { axis: 1 }) as NDArray;
    expect(Array.from(sum1.shape)).toEqual([2]);
    expect(Array.from(sum1.data)).toEqual([6, 15]);

    // Media por axis=0: [2.5, 3.5, 4.5]
    const m0 = mean(M, { axis: 0 }) as NDArray;
    expect(Array.from(m0.data)).toEqual([2.5, 3.5, 4.5]);
  });

  it('computes sum-products over strided arrays and validates their shapes', () => {
    const source = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });
    const strided = source.slice([0, 2], [0, 3, 2]);
    const weights = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [2, 2] });

    expect(sumProduct(strided, weights)).toBe(43);
    expect(sumProduct(strided, weights, NDArray.ones([2, 2]))).toBe(43);
    expect(sumProduct(NDArray.zeros([0]))).toBe(0);
    expect(() => sumProduct()).toThrow(RangeError);
    expect(() => sumProduct(strided, NDArray.zeros([4]))).toThrowError(/identical shapes/);
  });

  it('reduces numeric truth values and counts NaN as nonzero', () => {
    const values = new NDArray(new Float64Array([0, -2, NaN]), { shape: [3] });
    const zeros = NDArray.zeros([2]);
    const empty = NDArray.zeros([0]);

    expect(all(values)).toBe(false);
    expect(any(values)).toBe(true);
    expect(countNonzero(values)).toBe(2);
    expect(all(zeros)).toBe(false);
    expect(any(zeros)).toBe(false);
    expect(all(empty)).toBe(true);
    expect(any(empty)).toBe(false);
    expect(countNonzero(empty)).toBe(0);
  });

  it('computes median and quantiles', () => {
    const data = new NDArray(new Float64Array([10, 20, 30, 40, 50]), { shape: [5] });
    expect(median(data)).toBe(30);
    expect(quantile(data, 0.5)).toBe(30);
    expect(quantile(data, 0.25)).toBe(20);
    expect(quantile(data, 0.75)).toBe(40);
    expect(quantile(data, 0)).toBe(10);
    expect(quantile(data, 1)).toBe(50);
  });

  it('distinguishes population and sample variance', () => {
    const data = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });
    expect(variance(data)).toBeCloseTo(2 / 3, 12);
    expect(variance(data, { ddof: 1 })).toBe(1);
    expect(std(data, { ddof: 1 })).toBe(1);
  });

  it('computes descriptive statistics summary (pandas describe)', () => {
    const v = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), { shape: [10] });
    const desc = statsDescribe(v);

    expect(desc.count).toBe(10);
    expect(desc.mean).toBe(5.5);
    expect(desc.min).toBe(1);
    expect(desc.max).toBe(10);
    expect(desc.median).toBe(5.5);
    expect(desc.std).toBeCloseTo(3.02765, 4);
    expect(desc.skew).toBeCloseTo(0.0, 4); // simétrica
  });
});
