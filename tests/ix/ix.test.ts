import { describe, it, expect } from 'vitest';
import {
  NDArray,
  createGeneric,
  sub2ind1D,
  sub2ind2D,
  sub2ind3D,
  computeStridesRowMajor,
  computeStridesColMajor,
} from '../../src/ix/core/index.js';
import { add } from '../../src/ix/ops/math-ops.js';

describe('ix / core / strides', () => {
  it('calculates unrolled 1D-3D strides accurately', () => {
    // 1D: offset=0, s0=1, i0=5 -> 5
    expect(sub2ind1D(0, 1, 5)).toBe(5);

    // 2D: offset=10, shape=[3, 4], strides=[4, 1], idx=(1, 2) -> 10 + 1*4 + 2*1 = 16
    expect(sub2ind2D(10, 4, 1, 1, 2)).toBe(16);

    // 3D: offset=0, s=[12, 4, 1], idx=(1, 1, 2) -> 12 + 4 + 2 = 18
    expect(sub2ind3D(0, 12, 4, 1, 1, 1, 2)).toBe(18);
  });

  it('computes correct strides for Row-Major and Column-Major layouts', () => {
    const shape = [2, 3, 4];
    const rowStrides = computeStridesRowMajor(shape);
    expect(Array.from(rowStrides)).toEqual([12, 4, 1]);

    const colStrides = computeStridesColMajor(shape);
    expect(Array.from(colStrides)).toEqual([1, 2, 6]);
  });
});

describe('ix / core / ndarray', () => {
  it('initializes and provides fast get/set access', () => {
    const arr = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [2, 2] });
    expect(arr.get(0, 0)).toBe(1);
    expect(arr.get(0, 1)).toBe(2);
    expect(arr.get(1, 0)).toBe(3);
    expect(arr.get(1, 1)).toBe(4);

    arr.set(0, 1, 99);
    expect(arr.get(0, 1)).toBe(99);
  });

  it('supports lazy slicing without copying buffers', () => {
    // 3x3 Matrix:
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const arr = new NDArray(data, { shape: [3, 3] });

    // Submatrix rows 0..2, cols 1..3:
    // [[2, 3],
    //  [5, 6]]
    const view = arr.slice([0, 2, 1], [1, 3, 1]);
    expect(Array.from(view.shape)).toEqual([2, 2]);
    expect(view.get(0, 0)).toBe(2);
    expect(view.get(0, 1)).toBe(3);
    expect(view.get(1, 0)).toBe(5);
    expect(view.get(1, 1)).toBe(6);

    // Modifying the view modifies the original buffer (shared memory)
    view.set(0, 0, 42);
    expect(arr.get(0, 1)).toBe(42);
  });

  it('implements zero-allocation Symbol.iterator', () => {
    const arr = new NDArray(new Float64Array([10, 20, 30, 40]), { shape: [2, 2] });
    const collected = Array.from(arr);
    expect(collected).toEqual([10, 20, 30, 40]);
  });
});

describe('ix / core / dispatcher & math-ops', () => {
  it('dispatches dynamically and utilizes cache for subsequent calls', () => {
    const customOp = createGeneric('customOp');
    let invocations = 0;

    customOp.add(['number', 'number'], (a: number, b: number) => {
      invocations++;
      return a * b;
    });

    expect(customOp(4, 5)).toBe(20);
    expect(customOp.cache.size).toBe(1);

    expect(customOp(10, 2)).toBe(20);
    expect(invocations).toBe(2);
  });

  it('performs NDArray + Scalar addition', () => {
    const A = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [2, 2] });
    const res = add(A, 10);
    expect(Array.from(res.data)).toEqual([11, 12, 13, 14]);

    // Commutative scalar + NDArray
    const resCommutative = add(10, A);
    expect(Array.from(resCommutative.data)).toEqual([11, 12, 13, 14]);
  });

  it('performs NDArray + NDArray broadcasting (2x2 + 1x2)', () => {
    // [[1, 2],
    //  [3, 4]]
    const A = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [2, 2] });
    // [[10, 20]]
    const rowVec = new NDArray(new Float64Array([10, 20]), { shape: [1, 2] });

    const C = add(A, rowVec);
    expect(Array.from(C.shape)).toEqual([2, 2]);
    // [[1+10, 2+20],
    //  [3+10, 4+20]] -> [11, 22, 13, 24]
    expect(Array.from(C.data)).toEqual([11, 22, 13, 24]);
  });
});
