import { describe, expect, it } from 'vitest';
import {
  Quantity,
  celsius,
  fahrenheit,
  kelvin,
  meter,
  qty,
  second,
} from '../../src/ix/units/units.js';

describe('Quantity', () => {
  it('normalizes missing dimensions and identifies dimensionless quantities', () => {
    const scalar = new Quantity(3);

    expect(scalar.dims).toEqual({
      m: 0,
      kg: 0,
      s: 0,
      A: 0,
      K: 0,
      mol: 0,
      cd: 0,
    });
    expect(scalar.isDimensionless()).toBe(true);
    expect(qty(3, meter).isDimensionless()).toBe(false);
  });

  it('compares dimensions and exposes the base value', () => {
    expect(qty(1, meter).hasSameDimensions(qty(2, meter))).toBe(true);
    expect(qty(1, meter).hasSameDimensions(qty(1, second))).toBe(false);
    expect(qty(20, celsius).getBaseValue()).toBeCloseTo(293.15);
  });

  it('adds and subtracts compatible quantities', () => {
    const distance = qty(8, meter);
    const otherDistance = qty(3, meter);

    expect(distance.add(otherDistance).value).toBe(11);
    expect(distance.sub(otherDistance).value).toBe(5);
    expect(distance.add(otherDistance).dims).toEqual(distance.dims);
  });

  it('allows scalar addition and subtraction only for dimensionless quantities', () => {
    const scalar = new Quantity(7);

    expect(scalar.add(2).value).toBe(9);
    expect(scalar.sub(2).value).toBe(5);
    expect(() => qty(1, meter).add(2)).toThrowError(/Cannot add scalar number/);
    expect(() => qty(1, meter).sub(2)).toThrowError(/Cannot subtract scalar/);
  });

  it('rejects addition and subtraction between incompatible dimensions', () => {
    const distance = qty(1, meter);
    const duration = qty(1, second);

    expect(() => distance.add(duration)).toThrowError(/Dimensional mismatch in addition/);
    expect(() => distance.sub(duration)).toThrowError(/Dimensional mismatch in subtraction/);
  });

  it('combines dimensions when multiplying and dividing quantities', () => {
    const speed = qty(12, meter).div(qty(3, second));
    const distance = speed.mul(qty(2, second));

    expect(speed.value).toBe(4);
    expect(speed.dims).toMatchObject({ m: 1, s: -1 });
    expect(distance.value).toBe(8);
    expect(distance.dims).toMatchObject({ m: 1, s: 0 });
  });

  it('preserves dimensions when multiplying or dividing by a scalar', () => {
    const distance = qty(6, meter);

    expect(distance.mul(2).value).toBe(12);
    expect(distance.div(3).value).toBe(2);
    expect(distance.mul(2).dims).toEqual(distance.dims);
    expect(distance.div(3).dims).toEqual(distance.dims);
  });

  it('raises both the value and dimensions to the given power', () => {
    const area = qty(3, meter).pow(2);

    expect(area.value).toBe(9);
    expect(area.dims.m).toBe(2);
    expect(area.dims.kg).toBe(0);
  });

  it('converts between compatible units and applies temperature offsets', () => {
    expect(qty(2, meter).to(new Quantity(100, { m: 1 }))).toBeCloseTo(0.02);
    expect(qty(0, celsius).to(kelvin)).toBeCloseTo(273.15);
    expect(qty(100, celsius).to(fahrenheit)).toBeCloseTo(212);
    expect(qty(68, fahrenheit).to(celsius)).toBeCloseTo(20);
  });

  it('rejects conversion to an incompatible unit', () => {
    expect(() => qty(1, meter).to(second)).toThrowError(/incompatible dimensions/);
  });

  it('formats dimensions and quantities as readable strings', () => {
    expect(qty(2, meter).formatDimensions()).toBe('m^1');
    expect(new Quantity(2).formatDimensions()).toBe('dimensionless');
    expect(qty(2, meter).toString()).toBe('2 [m^1]');
  });
});
