/**
 * @file ode45.ts
 * Solvers de Ecuaciones Diferenciales Ordinarias (ODE) para problemas de valor inicial:
 * - Algoritmo RK4 (Runge-Kutta clásico de 4to orden de paso fijo)
 * - Algoritmo ODE45 (Dormand-Prince Runge-Kutta de orden 4(5) con control de paso adaptativo)
 * dy/dt = f(t, y)
 */
import { NDArray } from '../core/ndarray.js';

export interface ODESolution {
  t: number[];
  y: NDArray[]; // Solución vectorial en cada instante t
}

export interface ODEOptions {
  rtol?: number; // Tolerancia relativa (por defecto 1e-4)
  atol?: number; // Tolerancia absoluta (por defecto 1e-6)
  dtInit?: number; // Paso inicial sugerido
  dtMin?: number;
  dtMax?: number;
  maxSteps?: number;
}

export type ODEFunction = (t: number, y: NDArray) => NDArray;

function vectorNorm(arr: NDArray): number {
  let sum = 0;
  for (let i = 0; i < arr.size; i++) {
    const v = arr.data[i];
    sum += v * v;
  }
  return Math.sqrt(sum);
}

function vectorAxpy(alpha: number, x: NDArray, y: NDArray): NDArray {
  const out = new Float64Array(x.size);
  for (let i = 0; i < x.size; i++) {
    out[i] = alpha * x.data[i] + y.data[i];
  }
  return new NDArray(out, { shape: Array.from(x.shape) });
}

function vectorLincomb(coeffs: number[], vecs: NDArray[]): NDArray {
  const size = vecs[0].size;
  const out = new Float64Array(size);
  for (let k = 0; k < vecs.length; k++) {
    const c = coeffs[k];
    const data = vecs[k].data;
    for (let i = 0; i < size; i++) {
      out[i] += c * data[i];
    }
  }
  return new NDArray(out, { shape: Array.from(vecs[0].shape) });
}

/**
 * Solucionador Runge-Kutta 4 (RK4) de paso fijo
 */
export function rk4(
  f: ODEFunction,
  tSpan: [number, number],
  y0: NDArray,
  numSteps: number = 100
): ODESolution {
  const [t0, tf] = tSpan;
  const dt = (tf - t0) / numSteps;

  const tArr: number[] = [t0];
  const yArr: NDArray[] = [y0];

  let curT = t0;
  let curY = y0;

  for (let step = 0; step < numSteps; step++) {
    const k1 = f(curT, curY);
    const k2 = f(curT + 0.5 * dt, vectorAxpy(0.5 * dt, k1, curY));
    const k3 = f(curT + 0.5 * dt, vectorAxpy(0.5 * dt, k2, curY));
    const k4 = f(curT + dt, vectorAxpy(dt, k3, curY));

    curY = vectorLincomb([1.0, dt / 6, dt / 3, dt / 3, dt / 6], [curY, k1, k2, k3, k4]);
    curT += dt;

    tArr.push(curT);
    yArr.push(curY);
  }

  return { t: tArr, y: yArr };
}

/**
 * Solucionador ODE45 con control adaptativo de paso (Dormand-Prince / RKDP)
 * Coeficientes estándar Butcher Tableau para orden 4(5) embebido.
 */
export function ode45(
  f: ODEFunction,
  tSpan: [number, number],
  y0: NDArray,
  opts: ODEOptions = {}
): ODESolution {
  const [t0, tf] = tSpan;
  const rtol = opts.rtol ?? 1e-4;
  const atol = opts.atol ?? 1e-6;
  const dtMin = opts.dtMin ?? 1e-10;
  const dtMax = opts.dtMax ?? Math.abs(tf - t0) / 10;
  const maxSteps = opts.maxSteps ?? 50000;

  let dt = opts.dtInit ?? Math.min(dtMax, Math.max(dtMin, (tf - t0) / 100));

  const tArr: number[] = [t0];
  const yArr: NDArray[] = [y0];

  let t = t0;
  let y = y0;
  let stepCount = 0;

  while (t < tf && stepCount < maxSteps) {
    stepCount++;
    if (t + dt > tf) {
      dt = tf - t;
    }

    // Coeficientes Butcher Dormand-Prince 5(4)
    const k1 = f(t, y);
    const k2 = f(t + (1 / 5) * dt, vectorAxpy((1 / 5) * dt, k1, y));
    const k3 = f(
      t + (3 / 10) * dt,
      vectorLincomb([1, (3 / 40) * dt, (9 / 40) * dt], [y, k1, k2])
    );
    const k4 = f(
      t + (4 / 5) * dt,
      vectorLincomb([1, (44 / 45) * dt, (-56 / 15) * dt, (32 / 9) * dt], [y, k1, k2, k3])
    );
    const k5 = f(
      t + (8 / 9) * dt,
      vectorLincomb(
        [1, (19372 / 6561) * dt, (-25360 / 2187) * dt, (64448 / 6561) * dt, (-212 / 729) * dt],
        [y, k1, k2, k3, k4]
      )
    );
    const k6 = f(
      t + dt,
      vectorLincomb(
        [
          1,
          (9017 / 3168) * dt,
          (-355 / 33) * dt,
          (46732 / 5247) * dt,
          (49 / 176) * dt,
          (-5103 / 18656) * dt,
        ],
        [y, k1, k2, k3, k4, k5]
      )
    );

    // Solución de 5to orden: y5
    const y5 = vectorLincomb(
      [
        1,
        (35 / 384) * dt,
        0,
        (500 / 1113) * dt,
        (125 / 192) * dt,
        (-2187 / 6784) * dt,
        (11 / 84) * dt,
      ],
      [y, k1, k2, k3, k4, k5, k6]
    );

    // Estimador de error local: |y5 - y4|
    const k7 = f(t + dt, y5);
    const errVec = vectorLincomb(
      [
        (71 / 57600) * dt,
        0,
        (-71 / 16695) * dt,
        (71 / 1920) * dt,
        (-17253 / 339200) * dt,
        (22 / 525) * dt,
        (-1 / 40) * dt,
      ],
      [k1, k2, k3, k4, k5, k6, k7]
    );

    const err = vectorNorm(errVec);
    const tol = atol + rtol * Math.max(vectorNorm(y), vectorNorm(y5));

    // Factor de escalamiento adaptativo
    const safety = 0.9;
    let factor = safety * Math.pow(Math.max(1e-10, tol / (err + 1e-12)), 0.2);
    factor = Math.min(5.0, Math.max(0.1, factor));

    if (err <= tol || dt <= dtMin) {
      // Paso aceptado
      t += dt;
      y = y5;
      tArr.push(t);
      yArr.push(y);
      dt = Math.min(dtMax, Math.max(dtMin, dt * factor));
    } else {
      // Paso rechazado: reducir dt
      dt = Math.max(dtMin, dt * factor);
    }
  }

  return { t: tArr, y: yArr };
}
