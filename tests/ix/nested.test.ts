import { describe, it, expect } from 'vitest';
import { NDArray } from '../../src/ix/core/ndarray.js';
import { NestedArray } from '../../src/ix/core/nested-array.js';
import { add } from '../../src/ix/ops/math-ops.js';

describe('ix / core / nested-array', () => {
  it('infers shape and regularity of nested arrays correctly', () => {
    const regular = new NestedArray([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(regular.shape).toEqual([2, 3]);
    expect(regular.ndim).toBe(2);
    expect(regular.isRegular).toBe(true);

    const irregular = new NestedArray([
      [1, 2],
      [3, 4, 5],
    ]);
    expect(irregular.isRegular).toBe(false);
  });

  it('gets and sets elements directly in nested structures without flat buffer conversions', () => {
    const tensor = new NestedArray([
      [
        [10, 20],
        [30, 40],
      ],
      [
        [50, 60],
        [70, 80],
      ],
    ]);

    expect(tensor.get(0, 1, 1)).toBe(40);
    expect(tensor.get(1, 0, 0)).toBe(50);

    // Negative indices count backward from the end.
    expect(tensor.get(-1, -1, -1)).toBe(80);

    // In-place set
    tensor.set(0, 1, 1, 999);
    expect(tensor.get(0, 1, 1)).toBe(999);
    expect(tensor.data[0][1][1]).toBe(999);
  });

  it('iterates through leaf elements seamlessly', () => {
    const arr = new NestedArray([
      [1, 2],
      [3, 4],
    ]);
    const items = Array.from(arr);
    expect(items).toEqual([1, 2, 3, 4]);
  });

  it('converts strided views and offset arrays into nested arrays', () => {
    const base = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });
    const strided = base.slice([0, 2], [0, 3, 2]);
    const offset = new NDArray(base.data, { shape: [2], strides: [2], offset: 1 });
    const reversed = new NDArray(base.data, { shape: [3], strides: [-1], offset: 2 });
    const columnMajor = new NDArray(new Float64Array([1, 3, 2, 4]), {
      shape: [2, 2],
      strides: [1, 2],
      order: 'F',
    });
    const scalar = new NDArray(new Float64Array([42]), { shape: [] });

    expect(strided.toNestedArray()).toEqual([[1, 3], [4, 6]]);
    expect(offset.toNestedArray()).toEqual([2, 4]);
    expect(reversed.toNestedArray()).toEqual([3, 2, 1]);
    expect(columnMajor.toNestedArray()).toEqual([[1, 2], [3, 4]]);
    expect(scalar.toNestedArray()).toBe(42);

    const nestedCopy = strided.toNestedArray() as number[][];
    nestedCopy[0][0] = 99;
    expect(base.get(0, 0)).toBe(1);
  });

  it('works transparently with Multiple Dispatch for add operations', () => {
    const a = new NestedArray([
      [1, 2],
      [3, 4],
    ]);

    // NestedArray + Scalar
    const resScalar = add(a, 10);
    expect(resScalar.data).toEqual([
      [11, 12],
      [13, 14],
    ]);

    // NestedArray + NestedArray
    const b = new NestedArray([
      [10, 20],
      [30, 40],
    ]);
    const resMatrix = add(a, b);
    expect(resMatrix.data).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });
});
