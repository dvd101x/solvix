/**
 * @file minimize.ts
 * Algoritmos de optimización y minimización no lineal:
 * - fminbnd: Búsqueda del mínimo escalar 1D en [a, b] mediante Sección Áurea / Brent
 * - nelderMead: Optimización multivariable sin derivadas (Simplex) para R^n
 * - curveFit: Ajuste de curvas no lineales y calibración de modelos
 */
import { NDArray, toFloat64 } from '../core/ndarray.js';

export interface MinScalarResult {
  x: number;
  fval: number;
  converged: boolean;
  iterations: number;
}

export interface MinMultivarResult {
  x: NDArray;
  fval: number;
  converged: boolean;
  iterations: number;
}

/**
 * Minimización escalar en un intervalo cerrado [a, b] (Sección Áurea / Golden Section Search)
 */
export function fminbnd(
  f: (x: number) => number,
  bracket: [number, number],
  opts: { tol?: number; maxIter?: number } = {}
): MinScalarResult {
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? 100;
  const phi = (1 + Math.sqrt(5)) / 2;
  const resphi = 2 - phi;

  let [a, b] = bracket;
  let x1 = a + resphi * (b - a);
  let x2 = b - resphi * (b - a);
  let f1 = f(x1);
  let f2 = f(x2);

  for (let iter = 1; iter <= maxIter; iter++) {
    if (Math.abs(b - a) < tol) {
      const minX = (a + b) / 2;
      return { x: minX, fval: f(minX), converged: true, iterations: iter };
    }

    if (f1 < f2) {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = a + resphi * (b - a);
      f1 = f(x1);
    } else {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = b - resphi * (b - a);
      f2 = f(x2);
    }
  }

  const minX = (a + b) / 2;
  return { x: minX, fval: f(minX), converged: false, iterations: maxIter };
}

/**
 * Optimización multivariable sin derivadas mediante el método Simplex de Nelder-Mead (Downhill Simplex)
 */
export function nelderMead(
  f: (x: NDArray) => number,
  x0: NDArray | number[],
  opts: { tol?: number; maxIter?: number } = {}
): MinMultivarResult {
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? 500;

  const startVec = Array.from(toFloat64(x0));
  const n = startVec.length;

  // Construir simplex de n + 1 vértices
  const simplex: number[][] = [startVec];
  for (let i = 0; i < n; i++) {
    const p = [...startVec];
    p[i] = p[i] !== 0 ? p[i] * 1.05 + 0.00025 : 0.00025;
    simplex.push(p);
  }

  let fVals = simplex.map((p) => f(new NDArray(new Float64Array(p), { shape: [n] })));

  // Parámetros canónicos de Nelder-Mead
  const alpha = 1.0; // reflexión
  const gamma = 2.0; // expansión
  const rho = 0.5;   // contracción
  const sigma = 0.5; // reducción

  for (let iter = 1; iter <= maxIter; iter++) {
    // 1. Ordenar vértices según fVal
    const indices = Array.from({ length: n + 1 }, (_, i) => i);
    indices.sort((i1, i2) => fVals[i1] - fVals[i2]);

    const best = simplex[indices[0]];
    const worst = simplex[indices[n]];
    const secondWorst = simplex[indices[n - 1]];

    // Criterio de parada: desviación de valores en el simplex
    let diff = 0;
    for (let i = 1; i <= n; i++) diff += Math.abs(fVals[indices[i]] - fVals[indices[0]]);
    if (diff / n < tol) {
      return {
        x: new NDArray(new Float64Array(best), { shape: [n] }),
        fval: fVals[indices[0]],
        converged: true,
        iterations: iter,
      };
    }

    // 2. Centroide de los mejores n vértices
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const p = simplex[indices[i]];
      for (let d = 0; d < n; d++) centroid[d] += p[d] / n;
    }

    // 3. Reflexión
    const reflected = centroid.map((c, d) => c + alpha * (c - worst[d]));
    const fReflected = f(new NDArray(new Float64Array(reflected), { shape: [n] }));

    if (fVals[indices[0]] <= fReflected && fReflected < fVals[indices[n - 1]]) {
      simplex[indices[n]] = reflected;
      fVals[indices[n]] = fReflected;
      continue;
    }

    // 4. Expansión
    if (fReflected < fVals[indices[0]]) {
      const expanded = centroid.map((c, d) => c + gamma * (reflected[d] - c));
      const fExpanded = f(new NDArray(new Float64Array(expanded), { shape: [n] }));
      if (fExpanded < fReflected) {
        simplex[indices[n]] = expanded;
        fVals[indices[n]] = fExpanded;
      } else {
        simplex[indices[n]] = reflected;
        fVals[indices[n]] = fReflected;
      }
      continue;
    }

    // 5. Contracción
    const contracted = centroid.map((c, d) => c + rho * (worst[d] - c));
    const fContracted = f(new NDArray(new Float64Array(contracted), { shape: [n] }));
    if (fContracted < fVals[indices[n]]) {
      simplex[indices[n]] = contracted;
      fVals[indices[n]] = fContracted;
      continue;
    }

    // 6. Reducción hacia el mejor punto
    for (let i = 1; i <= n; i++) {
      const idx = indices[i];
      simplex[idx] = simplex[idx].map((coord, d) => best[d] + sigma * (coord - best[d]));
      fVals[idx] = f(new NDArray(new Float64Array(simplex[idx]), { shape: [n] }));
    }
  }

  const bestIdx = fVals.indexOf(Math.min(...fVals));
  return {
    x: new NDArray(new Float64Array(simplex[bestIdx]), { shape: [n] }),
    fval: fVals[bestIdx],
    converged: false,
    iterations: maxIter,
  };
}

/**
 * Ajuste no lineal de curvas por mínimos cuadrados mediante Nelder-Mead:
 * min sum_i (f(x_i, p) - y_i)^2
 */
export function curveFit(
  model: (x: number, p: NDArray) => number,
  xData: NDArray | number[],
  yData: NDArray | number[],
  p0: NDArray | number[]
): MinMultivarResult {
  const xs = toFloat64(xData);
  const ys = toFloat64(yData);
  const n = xs.length;

  const loss = (params: NDArray) => {
    let residual = 0;
    for (let i = 0; i < n; i++) {
      const diff = model(xs[i], params) - ys[i];
      residual += diff * diff;
    }
    return residual;
  };

  return nelderMead(loss, p0, { tol: 1e-7, maxIter: 1000 });
}
