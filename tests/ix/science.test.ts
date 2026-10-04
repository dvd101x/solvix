import { describe, it, expect } from 'vitest';
import { NDArray } from '../../src/ix/core/ndarray.js';
import { rk4, ode45 } from '../../src/ix/ode/ode45.js';
import { fzero, fsolve } from '../../src/ix/optimize/roots.js';
import {
  Quantity,
  meter,
  kilometer,
  second,
  hour,
  kilogram,
  newton,
  joule,
  qty,
} from '../../src/ix/units/units.js';
import { addInPlace, mulInPlace } from '../../src/ix/memory/in-place.js';

describe('ix / ode / solvers (rk4 & ode45)', () => {
  it('solves simple exponential decay dy/dt = -y with rk4', () => {
    // Solución analítica: y(t) = y0 * exp(-t)
    // Para t=1, y0=1 -> y(1) = 1/e ≈ 0.367879
    const f = (t: number, y: NDArray) => {
      return new NDArray(new Float64Array([-y.data[0]]), { shape: [1] });
    };

    const y0 = new NDArray(new Float64Array([1.0]), { shape: [1] });
    const sol = rk4(f, [0, 1], y0, 100);

    const finalY = sol.y[sol.y.length - 1].data[0];
    expect(finalY).toBeCloseTo(1 / Math.E, 3);
  });

  it('solves harmonic oscillator with adaptive ode45 (Dormand-Prince)', () => {
    // y'' + y = 0  =>  y1' = y2, y2' = -y1
    // Con y(0) = 0, y'(0) = 1  => Solución analítica y(t) = sin(t)
    // A t = pi/2 ≈ 1.570796, y(pi/2) = 1.0, y'(pi/2) = 0.0
    const f = (t: number, y: NDArray) => {
      const y1 = y.data[0];
      const y2 = y.data[1];
      return new NDArray(new Float64Array([y2, -y1]), { shape: [2] });
    };

    const y0 = new NDArray(new Float64Array([0.0, 1.0]), { shape: [2] });
    const sol = ode45(f, [0, Math.PI / 2], y0, { rtol: 1e-5, atol: 1e-7 });

    const finalY = sol.y[sol.y.length - 1];
    expect(finalY.data[0]).toBeCloseTo(1.0, 3); // sin(pi/2)
    expect(finalY.data[1]).toBeCloseTo(0.0, 3); // cos(pi/2)
  });

  it('returns the initial state without evaluating a zero-length interval', () => {
    const y0 = new NDArray(new Float64Array([2, -1]), { shape: [2] });
    let evaluations = 0;
    const sol = ode45((t, y) => {
      evaluations++;
      return y;
    }, [4, 4], y0);

    expect(sol.t).toEqual([4]);
    expect(sol.y).toEqual([y0]);
    expect(evaluations).toBe(0);
  });
});

describe('ix / optimize / root finding (fzero & fsolve)', () => {
  it('finds scalar roots with fzero (Brent)', () => {
    // f(x) = x^2 - 2  => raíz en sqrt(2) ≈ 1.41421356
    const res = fzero((x) => x * x - 2, [1, 2]);
    expect(res.converged).toBe(true);
    expect(res.root).toBeCloseTo(Math.SQRT2, 8);
  });

  it('accepts a root at a bracket endpoint and rejects intervals without a sign change', () => {
    expect(fzero((x) => x, [0, 5]).root).toBe(0);
    expect(() => fzero((x) => x * x + 1, [-1, 1])).toThrowError(/does not bracket a root/);
  });

  it('solves non-linear systems F(x) = 0 with fsolve (Newton-Raphson)', () => {
    // Sistema:
    // f1(x1, x2) = x1^2 + x2^2 - 1  (círculo unitario)
    // f2(x1, x2) = x1 - x2          (diagonal)
    // Solución positiva: x1 = x2 = 1/sqrt(2) ≈ 0.7071
    const F = (x: NDArray) => {
      const x1 = x.data[0];
      const x2 = x.data[1];
      return new NDArray(
        new Float64Array([x1 * x1 + x2 * x2 - 1, x1 - x2]),
        { shape: [2] }
      );
    };

    const res = fsolve(F, [0.5, 0.5]);
    expect(res.converged).toBe(true);
    expect(res.x.data[0]).toBeCloseTo(Math.SQRT1_2, 5);
    expect(res.x.data[1]).toBeCloseTo(Math.SQRT1_2, 5);
  });

  it('returns immediately when the initial guess is already a root', () => {
    const res = fsolve(
      (x) => new NDArray(new Float64Array([x.data[0] - 3]), { shape: [1] }),
      [3]
    );

    expect(res.converged).toBe(true);
    expect(res.iterations).toBe(1);
    expect(res.x.get(0)).toBe(3);
    expect(res.fval.get(0)).toBe(0);
  });
});

describe('ix / units / physical quantities & dimensional analysis', () => {
  it('enforces dimensional consistency in addition and subtraction', () => {
    const d1 = qty(5, meter);
    const d2 = qty(3, meter);
    const sum = d1.add(d2);
    expect(sum.value).toBe(8);

    // Sumar longitud con tiempo debe arrojar TypeError
    const t = qty(10, second);
    expect(() => d1.add(t)).toThrowError(/Dimensional mismatch/);
  });

  it('combines dimensions algebraically on multiplication and division', () => {
    // Fuerza = masa * aceleración = kg * (m / s^2) = N
    const mass = qty(10, kilogram);
    const accel = qty(9.8, meter).div(qty(1, second).pow(2));
    const force = mass.mul(accel);

    expect(force.dims.kg).toBe(1);
    expect(force.dims.m).toBe(1);
    expect(force.dims.s).toBe(-2);
    expect(force.to(newton)).toBeCloseTo(98.0, 5);

    // Energía / Trabajo = Fuerza * Distancia = Joules
    const dist = qty(2, meter);
    const work = force.mul(dist);
    expect(work.to(joule)).toBeCloseTo(196.0, 5);
  });

  it('converts units transparently (km/h <-> m/s)', () => {
    const speedKmh = qty(72, kilometer).div(qty(1, hour));
    const speedMs = speedKmh.to(meter.div(second));
    // 72 km/h = 20 m/s
    expect(speedMs).toBeCloseTo(20.0, 5);
  });
});

describe('ix / memory / in-place ops (!)', () => {
  it('modifies destination array in-place without memory allocation', () => {
    const a = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });
    const b = new NDArray(new Float64Array([10, 20, 30]), { shape: [3] });
    const out = new NDArray(new Float64Array(3), { shape: [3] });

    addInPlace(out, a, b);
    expect(Array.from(out.data)).toEqual([11, 22, 33]);

    // Reutilizar 'out' para multiplicar por un escalar in-place
    mulInPlace(out, out, 2);
    expect(Array.from(out.data)).toEqual([22, 44, 66]);
  });
});
