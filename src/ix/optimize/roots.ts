/**
 * @file roots.ts
 * Algoritmos de búsqueda de raíces y resolución de ecuaciones no lineales:
 * - fzero: Búsqueda de raíz unidimensional usando el método de Brent (híbrido Bisección + Secante + Interpolación Inversa)
 * - fsolve: Sistema multidimensional no lineal F(x) = 0 usando Newton-Raphson con Jacobiano numérico y amortiguamiento (Damping)
 */
import { NDArray } from '../core/ndarray.js';
import { solve } from '../linalg/factorizations.js';

export interface RootScalarResult {
  root: number;
  fval: number;
  converged: boolean;
  iterations: number;
}

export interface FsolveResult {
  x: NDArray;
  fval: NDArray;
  converged: boolean;
  iterations: number;
}

export interface RootOptions {
  tol?: number;
  maxIter?: number;
}

/**
 * Método de Brent para hallar raíces f(x) = 0 en un intervalo [a, b] (estilo scipy.optimize.brentq o MATLAB fzero).
 */
export function fzero(
  f: (x: number) => number,
  bracket: [number, number],
  opts: RootOptions = {}
): RootScalarResult {
  const tol = opts.tol ?? 1e-12;
  const maxIter = opts.maxIter ?? 100;

  let [a, b] = bracket;
  let fa = f(a);
  let fb = f(b);

  if (fa * fb > 0) {
    throw new Error(`The interval [${a}, ${b}] does not bracket a root: f(a)=${fa}, f(b)=${fb}`);
  }

  if (Math.abs(fa) < Math.abs(fb)) {
    const tmp = a; a = b; b = tmp;
    const ftmp = fa; fa = fb; fb = ftmp;
  }

  let c = a;
  let fc = fa;
  let mflag = true;
  let d = 0;

  for (let iter = 1; iter <= maxIter; iter++) {
    if (Math.abs(fb) < tol) {
      return { root: b, fval: fb, converged: true, iterations: iter };
    }

    let s: number;
    if (fa !== fc && fb !== fc) {
      // Interpolación cuadrática inversa
      s =
        (a * fb * fc) / ((fa - fb) * (fa - fc)) +
        (b * fa * fc) / ((fb - fa) * (fb - fc)) +
        (c * fa * fb) / ((fc - fa) * (fc - fb));
    } else {
      // Método de la secante
      s = b - fb * ((b - a) / (fb - fa));
    }

    // Condiciones de Brent para forzar bisección
    const cond1 = (s < (3 * a + b) / 4 && s > b) || (s > (3 * a + b) / 4 && s < b);
    const cond2 = mflag && Math.abs(s - b) >= Math.abs(b - c) / 2;
    const cond3 = !mflag && Math.abs(s - b) >= Math.abs(c - d) / 2;
    const cond4 = mflag && Math.abs(b - c) < tol;
    const cond5 = !mflag && Math.abs(c - d) < tol;

    if (cond1 || cond2 || cond3 || cond4 || cond5) {
      s = (a + b) / 2; // Bisección
      mflag = true;
    } else {
      mflag = false;
    }

    const fs = f(s);
    d = c;
    c = b;
    fc = fb;

    if (fa * fs < 0) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }

    if (Math.abs(fa) < Math.abs(fb)) {
      const tmp = a; a = b; b = tmp;
      const ftmp = fa; fa = fb; fb = ftmp;
    }

    if (Math.abs(b - a) < tol) {
      return { root: b, fval: fb, converged: true, iterations: iter };
    }
  }

  return { root: b, fval: fb, converged: false, iterations: maxIter };
}

/**
 * Calcula el Jacobiano numérico J de un sistema vectorial F: R^n -> R^n por diferencias finitas centrales.
 */
function computeJacobian(F: (x: NDArray) => NDArray, x: NDArray, eps = 1e-7): NDArray {
  const n = x.size;
  const JData = new Float64Array(n * n);
  const xData = new Float64Array(x.data);

  for (let j = 0; j < n; j++) {
    const orig = xData[j];

    xData[j] = orig + eps;
    const fPlus = F(new NDArray(xData, { shape: [n] }));

    xData[j] = orig - eps;
    const fMinus = F(new NDArray(xData, { shape: [n] }));

    xData[j] = orig; // restaurar

    for (let i = 0; i < n; i++) {
      JData[i * n + j] = (fPlus.data[i] - fMinus.data[i]) / (2 * eps);
    }
  }

  return new NDArray(JData, { shape: [n, n] });
}

/**
 * Resuelve sistemas de ecuaciones no lineales F(x) = 0 (estilo MATLAB fsolve o scipy.optimize.root).
 * Algoritmo: Newton-Raphson amortiguado con resolución lineal J * delta = -F(x).
 */
export function fsolve(
  F: (x: NDArray) => NDArray,
  x0: NDArray | number[],
  opts: RootOptions = {}
): FsolveResult {
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? 100;

  let x = x0 instanceof NDArray ? x0 : new NDArray(new Float64Array(x0), { shape: [x0.length] });
  const n = x.size;

  for (let iter = 1; iter <= maxIter; iter++) {
    const fx = F(x);

    // Calcular norma residual ||F(x)||
    let residualNorm = 0;
    for (let i = 0; i < n; i++) residualNorm += fx.data[i] * fx.data[i];
    residualNorm = Math.sqrt(residualNorm);

    if (residualNorm < tol) {
      return { x, fval: fx, converged: true, iterations: iter };
    }

    // Jacobiano en el punto actual
    const J = computeJacobian(F, x);

    // Construir vector -F(x)
    const negFx = new Float64Array(n);
    for (let i = 0; i < n; i++) negFx[i] = -fx.data[i];

    // Resolver paso de Newton: J * delta = -F(x)
    const delta = solve(J, negFx);

    // Actualización de x amortiguada (x_new = x + lambda * delta)
    let lambda = 1.0;
    let stepAccepted = false;
    let xNew: NDArray = x;

    for (let lineSearch = 0; lineSearch < 5; lineSearch++) {
      const candidateData = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        candidateData[i] = x.data[i] + lambda * delta.data[i];
      }
      const candidateX = new NDArray(candidateData, { shape: [n] });
      const candidateFx = F(candidateX);

      let candidateNorm = 0;
      for (let i = 0; i < n; i++) candidateNorm += candidateFx.data[i] * candidateFx.data[i];
      candidateNorm = Math.sqrt(candidateNorm);

      if (candidateNorm < residualNorm) {
        xNew = candidateX;
        stepAccepted = true;
        break;
      }
      lambda *= 0.5; // reducir paso
    }

    if (!stepAccepted) {
      // Aceptar paso mínimo para escapar de mínimos locales
      const fallbackData = new Float64Array(n);
      for (let i = 0; i < n; i++) fallbackData[i] = x.data[i] + 0.1 * delta.data[i];
      x = new NDArray(fallbackData, { shape: [n] });
    } else {
      x = xNew;
    }
  }

  const finalFx = F(x);
  return { x, fval: finalFx, converged: false, iterations: maxIter };
}
