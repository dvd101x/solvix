import { describe, it, expect } from 'vitest';
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

    // Negative indices (Python/Julia style)
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
