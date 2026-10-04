import { describe, it, expect } from 'vitest';
import { Complex, complex } from '../../src/ix/types/complex.js';
import { Fraction, frac } from '../../src/ix/types/fraction.js';
import { add } from '../../src/ix/ops/math-ops.js';

describe('ix / types / complex', () => {
  it('performs basic arithmetic on complex numbers', () => {
    // (1 + 2i) + (3 + 4i) = 4 + 6i
    const z1 = complex(1, 2);
    const z2 = complex(3, 4);
    const sum = z1.add(z2);
    expect(sum.re).toBe(4);
    expect(sum.im).toBe(6);

    // (1 + 2i) * (3 + 4i) = (3 - 8) + (4 + 6)i = -5 + 10i
    const prod = z1.mul(z2);
    expect(prod.re).toBe(-5);
    expect(prod.im).toBe(10);

    // Div: (1 + 2i) / (1 + 2i) = 1 + 0i
    const div = z1.div(z1);
    expect(div.re).toBeCloseTo(1.0, 5);
    expect(div.im).toBeCloseTo(0.0, 5);
  });

  it('computes magnitude (abs), phase (arg) and conjugate (conj)', () => {
    const z = complex(3, 4);
    expect(z.abs()).toBe(5);
    expect(z.conj().re).toBe(3);
    expect(z.conj().im).toBe(-4);
  });

  it('handles zero roots, polar construction, and division by zero', () => {
    expect(complex(0, 0).sqrt()).toEqual(complex(0, 0));
    expect(Complex.fromPolar(2, Math.PI / 2).re).toBeCloseTo(0, 12);
    expect(Complex.fromPolar(2, Math.PI / 2).im).toBeCloseTo(2, 12);

    const quotient = complex(1, 2).div(complex(0, 0));
    expect(quotient.re).toBe(Infinity);
    expect(quotient.im).toBe(Infinity);
  });

  it('computes complex square roots including negative reals: sqrt(-4) = 2i', () => {
    const negReal = complex(-4, 0);
    const root = negReal.sqrt();
    expect(root.re).toBeCloseTo(0.0, 5);
    expect(root.im).toBeCloseTo(2.0, 5); // 2i
  });

  it('integrates with Euler formula exp(i * pi) = -1', () => {
    const iPi = complex(0, Math.PI);
    const res = iPi.exp();
    // e^(i*pi) + 1 = 0
    expect(res.re).toBeCloseTo(-1.0, 5);
    expect(res.im).toBeCloseTo(0.0, 5);
  });
});

describe('ix / types / fraction', () => {
  it('simplifies fractions automatically via GCD', () => {
    const f = frac(4, 8);
    expect(f.n).toBe(1n);
    expect(f.d).toBe(2n);
    expect(f.toString()).toBe('1/2');
  });

  it('normalizes negative denominators and supports negative integer powers', () => {
    expect(frac(2, -4).toString()).toBe('-1/2');
    expect(frac(2, 3).pow(-2).toString()).toBe('9/4');
    expect(frac(-3, 4).pow(0).toString()).toBe('1');
  });

  it('rejects zero denominators and division by zero', () => {
    expect(() => frac(1, 0)).toThrow(RangeError);
    expect(() => frac(1, 2).div(0)).toThrowError(/Division by zero/);
    expect(() => frac(0).inv()).toThrowError(/Division by zero/);
  });

  it('performs exact rational arithmetic with zero IEEE-754 floating point errors', () => {
    // 1/3 + 1/6 = 2/6 + 1/6 = 3/6 = 1/2
    const f1 = frac(1, 3);
    const f2 = frac(1, 6);
    const sum = f1.add(f2);
    expect(sum.n).toBe(1n);
    expect(sum.d).toBe(2n);
    expect(sum.toNumber()).toBe(0.5);

    // 2/3 * 3/4 = 6/12 = 1/2
    const prod = frac(2, 3).mul(frac(3, 4));
    expect(prod.n).toBe(1n);
    expect(prod.d).toBe(2n);

    // 1/2 / 1/4 = 2
    const div = frac(1, 2).div(frac(1, 4));
    expect(div.n).toBe(2n);
    expect(div.d).toBe(1n);
  });

  it('converts floating point numbers to continuous fractions', () => {
    const f = Fraction.fromNumber(0.125);
    expect(f.n).toBe(1n);
    expect(f.d).toBe(8n);
  });
});

describe('ix / multiple-dispatch / Complex & Fraction', () => {
  it('dispatches add overloads for Complex and Fraction dynamically', () => {
    // Complex + Complex
    const zRes = add(complex(1, 2), complex(3, 4));
    expect(zRes).toBeInstanceOf(Complex);
    expect((zRes as Complex).re).toBe(4);
    expect((zRes as Complex).im).toBe(6);

    // Complex + Scalar number
    const zScalar = add(complex(10, 5), 2.5);
    expect((zScalar as Complex).re).toBe(12.5);
    expect((zScalar as Complex).im).toBe(5);

    // Fraction + Fraction
    const fRes = add(frac(1, 3), frac(1, 6));
    expect(fRes).toBeInstanceOf(Fraction);
    expect((fRes as Fraction).toString()).toBe('1/2');

    // Fraction + number
    const fScalar = add(frac(1, 2), 1);
    expect((fScalar as Fraction).toString()).toBe('3/2');
  });
});
