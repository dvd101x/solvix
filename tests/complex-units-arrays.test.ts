import { describe, expect, it } from 'vitest';
import {
  transpose, NDArray, Complex, add, sub, mul, div, pow, conj, real, imag, abs, sum, mean, kilometer, meter, second, Quantity,
} from '../src/index.js';

describe('NDArray with complex numbers', () => {
  const z = NDArray.fromArray([new Complex(1, 2), new Complex(3, -1)]);
  it('detects and stores complex', () => {
    expect(z.isComplex).toBe(true);
    expect(z.getComplex(1).im).toBe(-1);
    expect(z.toNestedArray()[0]).toBeInstanceOf(Complex);
  });
  it('arithmetic and broadcasting', () => {
    const s = add(z, 1) as NDArray;
    expect(s.getComplex(0).re).toBe(2);
    const m = mul(z, new Complex(0, 1)) as NDArray;
    expect(m.getComplex(0).re).toBe(-2);
    expect(m.getComplex(0).im).toBe(1);
    const col = NDArray.fromArray([[1], [2]]);
    const b = mul(col, z) as NDArray;
    expect(Array.from(b.shape)).toEqual([2, 2]);
    expect(b.getComplex(1, 0).re).toBe(2);
    expect(b.getComplex(1, 0).im).toBe(4);
  });
  it('conj, real, imag, abs', () => {
    expect((conj(z) as NDArray).getComplex(0).im).toBe(-2);
    expect(Array.from((real(z) as NDArray).data)).toEqual([1, 3]);
    expect(Array.from((imag(z) as NDArray).data)).toEqual([2, -1]);
    expect((abs(z) as NDArray).data[0]).toBeCloseTo(Math.sqrt(5));
  });
  it('is preserved by views', () => {
    const M = NDArray.fromArray([[new Complex(1, 1), new Complex(2, 2)], [new Complex(3, 3), new Complex(4, 4)]]);
    expect(transpose(M).getComplex(0, 1).im).toBe(3);
    expect(M.slice([1, 2], null).getComplex(0, 0).im).toBe(3);
  });
  it('rejects unsupported reductions', () => {
    expect(() => sum(z)).toThrow();
  });
});

describe('NDArray with units', () => {
  const d = NDArray.fromArray([1, 2, 3]).withUnit(kilometer);
  it('stores SI values and unit', () => {
    expect(Array.from(d.data)).toEqual([1000, 2000, 3000]);
    expect(d.unit?.m).toBe(1);
    expect(d.getQuantity(1).value).toBe(2000);
  });
  it('converts back', () => {
    expect(Array.from(d.to(kilometer).data)).toEqual([1, 2, 3]);
    expect(() => d.to(second)).toThrow();
  });
  it('dimensional arithmetic', () => {
    const t = NDArray.fromArray([1, 2, 4]).withUnit(second);
    const v = div(d, t) as NDArray;
    expect(v.unit?.m).toBe(1);
    expect(v.unit?.s).toBe(-1);
    expect(Array.from(v.data)).toEqual([1000, 1000, 750]);
    const a = mul(d, d) as NDArray;
    expect(a.unit?.m).toBe(2);
    const sq = pow(d, 2) as NDArray;
    expect(sq.unit?.m).toBe(2);
  });
  it('addition requires equal dimensions', () => {
    expect(() => add(d, NDArray.fromArray([1, 2, 3]))).toThrow();
    expect(() => add(d, NDArray.fromArray([1, 2, 3]).withUnit(second))).toThrow();
    const s = add(d, NDArray.fromArray([1, 1, 1]).withUnit(meter)) as NDArray;
    expect(Array.from(s.data)).toEqual([1001, 2001, 3001]);
    expect((sub(d, d) as NDArray).unit?.m).toBe(1);
  });
  it('reductions return Quantity', () => {
    const s = sum(d) as unknown as Quantity;
    expect(s).toBeInstanceOf(Quantity);
    expect(s.value).toBe(6000);
    expect((mean(d) as unknown as Quantity).value).toBe(2000);
  });
  it('is preserved by views', () => {
    expect(d.slice([1, 3]).unit?.m).toBe(1);
    expect(transpose(NDArray.fromArray([[1, 2], [3, 4]]).withUnit(meter)).unit?.m).toBe(1);
  });
});
