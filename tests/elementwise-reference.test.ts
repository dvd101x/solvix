import { describe, expect, it } from 'vitest';
import {
  NDArray, Complex, Quantity, meter, second, add, sub, mul, div, pow, neg, conj, real, imag, abs, angle,
  addElementwise, transpose, mapReal, mapComplex, mapElements, mapIndexed, isArrayLike,
} from '../src/index.js';

// Deterministic pseudo-random numbers so failures are reproducible.
let seed = 12345;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 4 - 2;

const shapes: [number[], number[]][] = [
  [[3], [3]], [[3], [1]], [[1], [4]], [[2, 3], [3]], [[2, 3], [2, 1]], [[2, 1], [1, 3]], [[1], [2, 3]],
  [[2, 3, 4], [4]], [[2, 1, 4], [3, 1]], [[1, 1], [1, 1]], [[5, 1, 1], [1, 4, 3]], [[4, 3], [4, 3]],
];

const make = (shape: number[], complex = false): NDArray => {
  const n = shape.reduce((p, q) => p * q, 1);
  const re = Float64Array.from({ length: n }, rnd);
  return new NDArray(re, { shape, imag: complex ? Float64Array.from({ length: n }, rnd) : undefined });
};

/** Naive reference: index every element with explicit broadcasting rules. */
function reference(a: NDArray, b: NDArray, fn: (x: Complex, y: Complex) => Complex): { shape: number[]; values: Complex[] } {
  const nd = Math.max(a.ndim, b.ndim);
  const sa = [...Array(nd - a.ndim).fill(1), ...a.shape];
  const sb = [...Array(nd - b.ndim).fill(1), ...b.shape];
  const shape = sa.map((x, i) => Math.max(x, sb[i]));
  const total = shape.reduce((p, q) => p * q, 1);
  const values: Complex[] = [];
  for (let k = 0; k < total; k++) {
    let rem = k;
    const idx = new Array(nd);
    for (let d = nd - 1; d >= 0; d--) {
      idx[d] = rem % shape[d];
      rem = Math.floor(rem / shape[d]);
    }
    const pick = (arr: NDArray, s: number[]) => {
      const local = idx.slice(nd - arr.ndim).map((v, d) => (s[nd - arr.ndim + d] === 1 ? 0 : v));
      return arr.getComplex(...local);
    };
    values.push(fn(pick(a, sa), pick(b, sb)));
  }
  return { shape, values };
}

const ops: [string, (a: any, b: any) => any, (x: Complex, y: Complex) => Complex][] = [
  ['add', add, (x, y) => x.add(y)],
  ['sub', sub, (x, y) => x.sub(y)],
  ['mul', mul, (x, y) => x.mul(y)],
  ['div', div, (x, y) => x.div(y)],
];

describe('broadcasting matches a naive reference implementation', () => {
  for (const complex of [false, true]) {
    for (const [name, op, ref] of ops) {
      it(`${name} (${complex ? 'complex' : 'real'}) over many shape pairs`, () => {
        for (const [sa, sb] of shapes) {
          const a = make(sa, complex);
          const b = make(sb, false);
          const got = op(a, b) as NDArray;
          const want = reference(a, b, ref);
          expect(Array.from(got.shape)).toEqual(want.shape);
          const flat = got.copy();
          want.values.forEach((w, k) => {
            expect(flat.data[k]).toBeCloseTo(w.re, 9);
            if (complex) expect(flat.imag![k]).toBeCloseTo(w.im, 9);
          });
        }
      });
    }
  }

  it('gives the same results for transposed (strided) operands', () => {
    const a = make([3, 4]);
    const b = make([4, 3]);
    const viaView = add(a, transpose(b)) as NDArray;
    const viaCopy = add(a, transpose(b).copy()) as NDArray;
    expect(Array.from(viaView.copy().data)).toEqual(Array.from(viaCopy.copy().data));
    const scalarLeft = sub(2, transpose(b)) as NDArray;
    expect(scalarLeft.get(1, 2)).toBeCloseTo(2 - b.get(2, 1));
  });

  it('handles empty arrays, 0-d results and single elements', () => {
    expect(Array.from((add(NDArray.zeros([0]), 1) as NDArray).data)).toEqual([]);
    expect(Array.from((add(NDArray.zeros([0, 3]), NDArray.zeros([3])) as NDArray).shape)).toEqual([0, 3]);
    expect(Array.from((mul(NDArray.fromArray([5]), 3) as NDArray).data)).toEqual([15]);
  });

  it('pow with real and complex exponents', () => {
    expect(Array.from((pow(NDArray.fromArray([1, 2, 3]), 2) as NDArray).data)).toEqual([1, 4, 9]);
    const z = pow(NDArray.fromArray([new Complex(0, 1)]), 2) as NDArray;
    expect(z.getComplex(0).re).toBeCloseTo(-1);
    const zero = pow(NDArray.fromArray([new Complex(0, 0)]), new Complex(0, 0)) as NDArray;
    expect(zero.getComplex(0).re).toBe(1);
  });

  it('rejects mismatched shapes', () => {
    expect(() => add(NDArray.zeros([2, 3]), NDArray.zeros([2]))).toThrow();
  });
});

describe('scalar operations through the generic operators', () => {
  it('number, Complex and Quantity scalars', () => {
    expect(add(1, 2)).toBe(3);
    expect(mul(new Complex(1, 1), new Complex(1, -1))).toEqual(new Complex(2, 0));
    expect((add(meter, meter) as Quantity).value).toBe(2);
    expect((mul(meter, second) as Quantity).dims).toMatchObject({ m: 1, s: 1 });
    expect(() => add(meter, second)).toThrow();
    expect(() => add(meter, 1)).toThrow();
    expect(sub(5, 3)).toBe(2);
    expect(div(6, 3)).toBe(2);
    expect(pow(2, 10)).toBe(1024);
  });
  it('unit arrays: exponents must be dimensionless real scalars', () => {
    const d = NDArray.fromArray([1, 2]).withUnit(meter);
    expect(() => pow(d, NDArray.fromArray([1, 2]))).toThrow();
    expect(() => pow(d, meter)).toThrow();
    expect((pow(d, 0.5) as NDArray).unit?.m).toBeCloseTo(0.5);
    expect((div(d, d) as NDArray).unit).toBeUndefined();
  });
  it('addElementwise matches add', () => {
    const a = NDArray.fromArray([1, 2]);
    expect(Array.from((addElementwise(a, 1) as NDArray).data)).toEqual([2, 3]);
  });
});

describe('unary element-wise functions', () => {
  const z = NDArray.fromArray([new Complex(3, 4), new Complex(0, -2)]);
  it('neg, conj, real, imag, abs, angle', () => {
    expect((neg(z) as NDArray).getComplex(0)).toEqual(new Complex(-3, -4));
    expect((conj(z) as NDArray).getComplex(1)).toEqual(new Complex(0, 2));
    expect(Array.from((real(z) as NDArray).data)).toEqual([3, 0]);
    expect(Array.from((imag(z) as NDArray).data)).toEqual([4, -2]);
    expect(Array.from((abs(z) as NDArray).data)).toEqual([5, 2]);
    expect((angle(z) as NDArray).data[1]).toBeCloseTo(-Math.PI / 2);
  });
  it('work on nested input and return nested output', () => {
    expect(neg([[1, -2]])).toEqual([[-1, 2]]);
    expect(abs([[-1, 2]])).toEqual([[1, 2]]);
  });
  it('on scalars', () => {
    expect(neg(3)).toBe(-3);
    expect(abs(-3)).toBe(3);
    expect(conj(new Complex(1, 2))).toEqual(new Complex(1, -2));
    expect(abs(new Complex(3, 4))).toBe(5);
  });
  it('keep units on neg and abs', () => {
    const d = NDArray.fromArray([-1, 2]).withUnit(meter);
    expect((neg(d) as NDArray).unit?.m).toBe(1);
    expect((abs(d) as NDArray).unit?.m).toBe(1);
  });
  it('mapReal / mapComplex / mapElements / mapIndexed reject what they cannot handle', () => {
    expect(() => mapReal(z, (v) => v)).toThrow(/complex/);
    expect(() => mapReal(NDArray.fromArray([1]).withUnit(meter), (v) => v)).toThrow(/dimensionless/);
    expect(() => mapComplex(NDArray.fromArray([1]).withUnit(meter), (a, b) => [a, b])).toThrow();
    expect(() => mapElements(NDArray.fromArray([1]).withUnit(meter), (v) => v)).toThrow();
    expect(() => mapElements([1], (() => 'x') as any)).toThrow();
    expect(mapElements([[1, 2]], (v) => (v as number) + 1)).toEqual([[2, 3]]);
    expect(mapElements(z, (v) => (v as Complex).conj()).getComplex(0)).toEqual(new Complex(3, -4));
    expect(mapIndexed(NDArray.fromArray([5, 6]), (v, i) => (v as number) + i[0]).data).toEqual(new Float64Array([5, 7]));
  });
  it('isArrayLike', () => {
    expect(isArrayLike([1])).toBe(true);
    expect(isArrayLike(NDArray.zeros([1]))).toBe(true);
    expect(isArrayLike(3)).toBe(false);
  });
});
