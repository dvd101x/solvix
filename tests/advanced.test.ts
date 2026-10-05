import { describe, it, expect } from 'vitest';
import {
  NDArray,
  booleanMask,
  where,
  take,
  putMask,
  sliceWithEllipsis,
  ELLIPSIS,
  createSharedNDArray,
  partitionWork,
  prepareTransfer,
  reconstructFromTransfer,
  serializeScope,
  deserializeScope,
  evaluateExpressionInScope,
  ExpressionDAG,
} from '../src/index.js';
import { add } from '../src/ops/math-ops.js';

describe('ix / indexing / advanced-indexing', () => {
  it('performs booleanMask (A[mask])', () => {
    const arr = new NDArray(new Float64Array([1, -2, 3, -4, 5]), { shape: [5] });
    const mask = [true, false, true, false, true];

    const filtered = booleanMask(arr, mask);
    expect(Array.from(filtered.shape)).toEqual([3]);
    expect(Array.from(filtered.data)).toEqual([1, 3, 5]);
  });

  it('accepts Uint8Array masks and returns an empty result when nothing matches', () => {
    const arr = new NDArray(new Float64Array([4, 5, 6]), { shape: [3] });

    expect(Array.from(booleanMask(arr, new Uint8Array([0, 1, 1])).data)).toEqual([5, 6]);
    expect(Array.from(booleanMask(arr, [false, false, false]).shape)).toEqual([0]);
  });

  it('selects values with where using scalars or matching arrays', () => {
    const condition = new NDArray(new Float64Array([0, 1, NaN]), { shape: [3] });
    const values = new NDArray(new Float64Array([10, 20, 30]), { shape: [3] });
    const fallback = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });

    expect(Array.from(where(condition, 5, -1).data)).toEqual([-1, 5, 5]);
    expect(Array.from(where(condition, values, fallback).data)).toEqual([1, 20, 30]);
    expect(() => where(condition, NDArray.zeros([2]), 0)).toThrow();
  });

  it('performs take / fancy indexing along an axis', () => {
    // 3x2 matrix:
    // [[10, 11],
    //  [20, 21],
    //  [30, 31]]
    const arr = new NDArray(new Float64Array([10, 11, 20, 21, 30, 31]), { shape: [3, 2] });
    // Tomar filas 2 y 0 -> [[30, 31], [10, 11]]
    const res = take(arr, [2, 0], 0);

    expect(Array.from(res.shape)).toEqual([2, 2]);
    expect(res.get(0, 0)).toBe(30);
    expect(res.get(0, 1)).toBe(31);
    expect(res.get(1, 0)).toBe(10);
    expect(res.get(1, 1)).toBe(11);
  });

  it('supports negative axes, typed indices, and empty selections in take', () => {
    const arr = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });
    const columns = take(arr, new Int32Array([2, 0]), -1);
    const empty = take(arr, [], 0);

    expect(Array.from(columns.shape)).toEqual([2, 2]);
    expect(Array.from(columns.data)).toEqual([3, 1, 6, 4]);
    expect(Array.from(empty.shape)).toEqual([0, 3]);
    expect(empty.size).toBe(0);
    expect(() => take(arr, [0], 2)).toThrowError(/Axis 2 out of bounds/);
  });

  it('performs putMask in-place update', () => {
    const arr = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [4] });
    const mask = [false, true, false, true];

    putMask(arr, mask, 999);
    expect(Array.from(arr.data)).toEqual([1, 999, 3, 999]);
  });

  it('assigns successive NDArray values to selected mask positions', () => {
    const arr = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [4] });
    const values = new NDArray(new Float64Array([8, 9]), { shape: [2] });

    putMask(arr, new Uint8Array([0, 1, 0, 1]), values);

    expect(Array.from(arr.data)).toEqual([1, 8, 3, 9]);
  });

  it('supports slicing with ellipsis (...)', () => {
    // 3D tensor: 2 x 2 x 2
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const arr = new NDArray(data, { shape: [2, 2, 2] });

    // arr[..., 1] -> todos los primeros ejes, y coordenada 1 en el último eje
    const sliced = sliceWithEllipsis(arr, ELLIPSIS, 1);
    expect(Array.from(sliced.shape)).toEqual([2, 2, 1]);
    expect(sliced.get(0, 0, 0)).toBe(2);
    expect(sliced.get(0, 1, 0)).toBe(4);
    expect(sliced.get(1, 0, 0)).toBe(6);
    expect(sliced.get(1, 1, 0)).toBe(8);
  });

  it('rejects multiple ellipses in one index', () => {
    const arr = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [2, 2] });
    expect(() => sliceWithEllipsis(arr, ELLIPSIS, ELLIPSIS)).toThrowError(/single ellipsis/);
  });
});

describe('ix / parallel / workers & memory sharing', () => {
  it('creates SharedNDArray backed by SharedArrayBuffer', () => {
    const shared = createSharedNDArray([2, 2]);
    expect(shared.data.buffer instanceof SharedArrayBuffer).toBe(true);

    shared.set(0, 0, 123.45);
    expect(shared.get(0, 0)).toBe(123.45);
  });

  it('partitions work chunks evenly', () => {
    const chunks = partitionWork(100, 3);
    expect(chunks.length).toBe(3);
    expect(chunks[0]).toEqual({ start: 0, end: 34, length: 34 });
    expect(chunks[1]).toEqual({ start: 34, end: 67, length: 33 });
    expect(chunks[2]).toEqual({ start: 67, end: 100, length: 33 });
  });

  it('partitions empty work and work into more workers than items', () => {
    expect(partitionWork(0, 2)).toEqual([
      { start: 0, end: 0, length: 0 },
      { start: 0, end: 0, length: 0 },
    ]);
    expect(partitionWork(2, 4)).toEqual([
      { start: 0, end: 1, length: 1 },
      { start: 1, end: 2, length: 1 },
      { start: 2, end: 2, length: 0 },
      { start: 2, end: 2, length: 0 },
    ]);
  });

  it('supports shared buffers with alternate typed-array constructors', () => {
    const shared = createSharedNDArray([2], Int32Array);
    shared.set(0, 12);
    shared.set(1, 34);

    expect(shared.data).toBeInstanceOf(Int32Array);
    expect(shared.data.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(Array.from(shared.data)).toEqual([12, 34]);
  });

  it('prepares zero-copy transfers and reconstructs arrays', () => {
    const original = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });
    const { message, transferables } = prepareTransfer(original);

    expect(transferables.length).toBe(1);
    expect(transferables[0]).toBe(original.data.buffer);

    const reconstructed = reconstructFromTransfer(message);
    expect(Array.from(reconstructed.shape)).toEqual([3]);
    expect(reconstructed.get(1)).toBe(2);
  });

  it('does not transfer shared buffers and preserves shared data on reconstruction', () => {
    const original = createSharedNDArray([2]);
    original.set(0, 7);
    original.set(1, 11);

    const { message, transferables } = prepareTransfer(original);
    const reconstructed = reconstructFromTransfer(message);

    expect(message.isShared).toBe(true);
    expect(transferables).toEqual([]);
    expect(reconstructed.data.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(Array.from(reconstructed.data)).toEqual([7, 11]);
  });

  it('evaluates expressions in worker scope with Math built-ins', () => {
    const scope = {
      x: 10,
      y: 5,
    };
    const res = evaluateExpressionInScope('sqrt(x * x + y * y)', scope);
    expect(res).toBeCloseTo(11.18, 2);
  });

  it('serializes and deserializes scope with NDArrays', () => {
    const tensor = new NDArray(new Float64Array([10, 20]), { shape: [2] });
    const originalScope = { alpha: 0.5, vec: tensor };

    const serialized = serializeScope(originalScope);
    const restored = deserializeScope(serialized);

    expect(restored.alpha).toBe(0.5);
    expect(restored.vec).toBeInstanceOf(NDArray);
    expect(restored.vec.get(1)).toBe(20);
  });
});

describe('ix / dag / computation graph', () => {
  it('builds DAG, topologically sorts, and evaluates math expressions', () => {
    const dag = new ExpressionDAG();

    // Grafo:
    // a = 2, b = 3
    // c = a + b = 5
    // d = c * 10 = 50
    dag.variable('a', 2);
    dag.variable('b', 3);
    dag.op('c', ['a', 'b'], (x, y) => x + y);
    dag.op('d', ['c'], (cVal) => cVal * 10);

    const sorted = dag.topologicalSort();
    expect(sorted.indexOf('c')).toBeGreaterThan(sorted.indexOf('a'));
    expect(sorted.indexOf('c')).toBeGreaterThan(sorted.indexOf('b'));
    expect(sorted.indexOf('d')).toBeGreaterThan(sorted.indexOf('c'));

    expect(dag.evaluate('d')).toBe(50);

    // Invalida y re-evalúa al cambiar variables
    dag.setVariable('a', 10); // c = 10 + 3 = 13, d = 13 * 10 = 130
    expect(dag.evaluate('d')).toBe(130);
  });

  it('detects cycles in DAG and throws error', () => {
    const dag = new ExpressionDAG();
    dag.variable('x', 1);
    dag.op('y', ['x'], (x) => x + 1);

    // Forzar ciclo introduciendo dependencia circular
    expect(() => {
      dag.op('x', ['y'], (y) => y * 2);
    }).toThrowError(/Cycle detected/);
  });

  it('evaluates DAG with NDArrays and operations', () => {
    const dag = new ExpressionDAG();
    const arrA = new NDArray(new Float64Array([1, 2]), { shape: [2] });
    const arrB = new NDArray(new Float64Array([10, 20]), { shape: [2] });

    dag.variable('A', arrA);
    dag.variable('B', arrB);
    dag.op('C', ['A', 'B'], (t1, t2) => add(t1, t2));

    const result = dag.evaluate('C') as NDArray;
    expect(Array.from(result.data)).toEqual([11, 22]);
  });

  it('memoizes computed nodes and invalidates them when dependencies change', () => {
    const dag = new ExpressionDAG();
    let calls = 0;
    dag.variable('input', 2);
    dag.op('result', ['input'], (value) => {
      calls++;
      return value * 3;
    });

    expect(dag.evaluate('result')).toBe(6);
    expect(dag.evaluate('result')).toBe(6);
    expect(calls).toBe(1);

    dag.setVariable('input', 4);
    expect(dag.evaluate('result')).toBe(12);
    expect(calls).toBe(2);
  });

  it('reports unregistered dependencies and unknown evaluation targets', () => {
    const dag = new ExpressionDAG();
    expect(() => dag.op('result', ['missing'], (value) => value)).toThrowError(/Dependency missing/);
    expect(() => dag.evaluate('missing')).toThrowError(/Target node missing/);
  });
});
