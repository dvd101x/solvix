import { describe, it, expect } from 'vitest';
import { defmulti, defmethod, getType } from '../src/core/dispatch.js';
import { NdArray } from '../src/core/ndarray.js';
import { Matrix } from 'ml-matrix';
import * as tf from '@tensorflow/tfjs';
import { createMathSupersetEnvironment } from '../src/environment/dispatcher.js';

describe('Multiple Dispatch Core', () => {
  it('correctly infers argument types', () => {
    expect(getType(42)).toBe('Number');
    expect(getType(NdArray.zeros([2, 2]))).toBe('NdArray');
    expect(getType(new Matrix(2, 2))).toBe('MlMatrix');
    expect(getType(tf.tensor([1, 2]))).toBe('TfTensor');
  });

  it('resolves overloaded methods by specificity', () => {
    const add = defmulti('add');

    defmethod(add, ['Number', 'Number'], (a: number, b: number) => `num+num: ${a + b}`);
    defmethod(add, ['Number', 'Any'], (a: number, b: any) => `num+any`);
    defmethod(add, ['Any', 'Any'], (a: any, b: any) => `any+any`);

    expect(add(2, 3)).toBe('num+num: 5');
    expect(add(2, 'hello')).toBe('num+any');
    expect(add('hello', 'world')).toBe('any+any');
  });

  it('fails with informative MethodError when no match exists', () => {
    const fn = defmulti('strictFn');
    defmethod(fn, ['Number'], (x: number) => x);

    expect(() => fn('test')).toThrowError(/MethodError: no method matching strictFn\(Any\)/);
  });
});

describe('MathSuperset Environment & Adapters', () => {
  const env = createMathSupersetEnvironment();

  it('dispatches inv() dynamically to numbers and matrices', () => {
    // Number
    expect(env.dispatchers.inv(4)).toBe(0.25);

    // NdArray
    const arr = NdArray.fromArray([
      [4, 7],
      [2, 6],
    ]);
    const invArr = env.dispatchers.inv(arr) as NdArray;
    expect(invArr.shape).toEqual([2, 2]);
    // [4 7; 2 6]^-1 = [0.6 -0.7; -0.2 0.4]
    expect(invArr.get(0, 0)).toBeCloseTo(0.6);
    expect(invArr.get(0, 1)).toBeCloseTo(-0.7);
    expect(invArr.get(1, 0)).toBeCloseTo(-0.2);
    expect(invArr.get(1, 1)).toBeCloseTo(0.4);
  });

  it('evaluates SVD decomposition on NdArray', () => {
    const arr = NdArray.fromArray([
      [1, 2],
      [3, 4],
    ]);
    const svdRes = env.dispatchers.svd(arr) as any;
    expect(svdRes.U).toBeInstanceOf(NdArray);
    expect(svdRes.s).toBeInstanceOf(NdArray);
    expect(svdRes.V).toBeInstanceOf(NdArray);
  });

  it('evaluates @stdlib besselj function on numbers and NdArray', () => {
    const resNum = env.dispatchers.besselj(0, 0);
    expect(resNum).toBe(1);

    const arr = NdArray.fromArray([0, 1, 2]);
    const resArr = env.dispatchers.besselj(0, arr) as NdArray;
    expect(resArr.get(0)).toBe(1);
    expect(resArr.get(1)).toBeCloseTo(0.7651976865, 4);
  });

  it('evaluates expressions seamlessly in mathjs scope', () => {
    const res = env.evaluate('inv(2)');
    expect(res).toBe(0.5);

    const detRes = env.evaluate('det(ndarray([[2, 0], [0, 5]]))');
    expect(detRes).toBe(10);
  });
});
