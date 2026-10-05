/**
 * @file factorizations.ts
 * Álgebra lineal numérica de alto rendimiento para NDArray:
 * - Descomposición LU con pivoteo parcial (PA = LU)
 * - Descomposición QR mediante reflexiones de Householder (A = QR)
 * - Descomposición en Valores Singulares (SVD: A = U * S * V^T) mediante algoritmo de Golub-Reinsch
 * - Solución de sistemas lineales Ax = b
 * - Inversión de matrices (inv) y cálculo del determinante (det)
 */
import { NDArray } from '../core/ndarray.js';
import { Complex } from '../types/complex.js';
import { Quantity } from '../units/units.js';
import { isArrayLike, toNDArray, wrapLike } from '../ops/elementwise.js';
import { gaussSolve, gaussInv, gaussDet } from './solver.js';

type MatrixInput = NDArray | any[];

/** True when the plain real LU path cannot be used (complex, units or non-array input). */
function needsGeneral(...xs: unknown[]): boolean {
  return xs.some((x) => !(x instanceof NDArray) || x.isComplex || x.unit !== undefined);
}

export interface LUResult {
  L: NDArray;
  U: NDArray;
  P: Int32Array; // Vector de permutación
  sign: number;  // Signo de la permutación (+1 o -1)
}

export interface QRResult {
  Q: NDArray;
  R: NDArray;
}

export interface SVDResult {
  U: NDArray;
  S: NDArray; // Vector 1D con valores singulares
  V: NDArray;
}

/**
 * Descomposición LU con pivoteo parcial (PA = LU)
 * Algoritmo Doolittle optimizado en memoria plana Float64Array.
 */
export function lu(A: NDArray): LUResult {
  if (A.ndim !== 2 || A.shape[0] !== A.shape[1]) {
    throw new Error(`LU decomposition requires a square 2D matrix, got shape [${Array.from(A.shape)}]`);
  }
  const n = A.shape[0];

  // Copiar datos para trabajar in-place en la descomposición
  const aData = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      aData[r * n + c] = A.get(r, c);
    }
  }

  const P = new Int32Array(n);
  for (let i = 0; i < n; i++) P[i] = i;
  let sign = 1;

  for (let i = 0; i < n; i++) {
    // Pivoteo parcial: encontrar el elemento máximo en la columna i
    let maxVal = Math.abs(aData[i * n + i]);
    let pivotRow = i;
    for (let k = i + 1; k < n; k++) {
      const val = Math.abs(aData[k * n + i]);
      if (val > maxVal) {
        maxVal = val;
        pivotRow = k;
      }
    }

    if (maxVal < 1e-15) {
      throw new Error('Matrix is singular or near-singular, LU factorization failed.');
    }

    // Intercambiar filas si es necesario
    if (pivotRow !== i) {
      for (let k = 0; k < n; k++) {
        const tmp = aData[i * n + k];
        aData[i * n + k] = aData[pivotRow * n + k];
        aData[pivotRow * n + k] = tmp;
      }
      const tmpP = P[i];
      P[i] = P[pivotRow];
      P[pivotRow] = tmpP;
      sign = -sign;
    }

    // Eliminación gaussiana
    const diag = aData[i * n + i];
    for (let j = i + 1; j < n; j++) {
      aData[j * n + i] /= diag;
      const mult = aData[j * n + i];
      for (let k = i + 1; k < n; k++) {
        aData[j * n + k] -= mult * aData[i * n + k];
      }
    }
  }

  // Separar L y U
  const lData = new Float64Array(n * n);
  const uData = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    lData[i * n + i] = 1.0;
    for (let j = 0; j < i; j++) {
      lData[i * n + j] = aData[i * n + j];
    }
    for (let j = i; j < n; j++) {
      uData[i * n + j] = aData[i * n + j];
    }
  }

  return {
    L: new NDArray(lData, { shape: [n, n] }),
    U: new NDArray(uData, { shape: [n, n] }),
    P,
    sign,
  };
}

/**
 * Resuelve el sistema lineal Ax = b usando la descomposición LU (PA = LU)
 */
export function solve(A: MatrixInput, b: NDArray | Float64Array | number[] | any[]): NDArray;
export function solve(A: any, b: any): any {
  const An: NDArray = isArrayLike(A) ? toNDArray(A) : A;
  const bN: NDArray = b instanceof NDArray ? b : isArrayLike(b) ? toNDArray(b) : new NDArray(Float64Array.from(b));
  if (!(A instanceof NDArray) || needsGeneral(An, bN)) {
    return wrapLike(gaussSolve(An, bN), A instanceof NDArray ? undefined : A);
  }
  A = An;
  const { L, U, P } = lu(A);
  const n = A.shape[0];

  const bArr = bN.copy().data;
  if (bArr.length !== n) {
    throw new Error(`Dimension mismatch: A is [${n}x${n}], b has length ${bArr.length}`);
  }

  // 1. Aplicar permutación P * b
  const pb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    pb[i] = bArr[P[i]];
  }

  // 2. Sustitución hacia adelante: L * y = P * b
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = pb[i];
    for (let j = 0; j < i; j++) {
      sum -= L.get(i, j) * y[j];
    }
    y[i] = sum / L.get(i, i);
  }

  // 3. Sustitución hacia atrás: U * x = y
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let j = i + 1; j < n; j++) {
      sum -= U.get(i, j) * x[j];
    }
    x[i] = sum / U.get(i, i);
  }

  return new NDArray(x, { shape: [n] });
}

/**
 * Inversa de una matriz cuadrada A^-1 calculada resolviendo A * X = I
 */
export function inv(A: MatrixInput): NDArray;
export function inv(A: any): any {
  if (needsGeneral(A)) {
    return wrapLike(gaussInv(isArrayLike(A) ? toNDArray(A) : A), A);
  }
  if (A.ndim !== 2 || A.shape[0] !== A.shape[1]) {
    throw new Error('Matrix inversion requires a square 2D matrix');
  }
  const n = A.shape[0];
  const outData = new Float64Array(n * n);

  // Resolver para cada vector unitario canónico e_j
  const ej = new Float64Array(n);
  for (let col = 0; col < n; col++) {
    ej.fill(0);
    ej[col] = 1.0;
    const colSol = solve(A, ej);
    for (let row = 0; row < n; row++) {
      outData[row * n + col] = colSol.data[row];
    }
  }

  return new NDArray(outData, { shape: [n, n] });
}

/**
 * Determinante de una matriz cuadrada vía descomposición LU: det(A) = sign(P) * prod(diag(U))
 */
export function det(A: NDArray): number;
export function det(A: MatrixInput): number | Complex | Quantity;
export function det(A: any): any {
  if (needsGeneral(A)) return gaussDet(isArrayLike(A) ? toNDArray(A) : A);
  const { U, sign } = lu(A);
  const n = A.shape[0];
  let d = sign;
  for (let i = 0; i < n; i++) {
    d *= U.get(i, i);
  }
  return d;
}

/**
 * Descomposición QR mediante transformaciones ortogonales de Householder (A = QR).
 * Válida para matrices m x n con m >= n.
 */
export function qr(A: NDArray): QRResult {
  if (A.ndim !== 2) {
    throw new Error('QR decomposition requires a 2D matrix');
  }
  const m = A.shape[0];
  const n = A.shape[1];
  if (m < n) {
    throw new Error(`QR requires rows >= cols, got [${m}x${n}]`);
  }

  // Copia de trabajo para R
  const rData = new Float64Array(m * n);
  for (let r = 0; r < m; r++) {
    for (let c = 0; c < n; c++) {
      rData[r * n + c] = A.get(r, c);
    }
  }

  // Matriz Q inicializada como la identidad m x m
  const qData = new Float64Array(m * m);
  for (let i = 0; i < m; i++) qData[i * m + i] = 1.0;

  const minMN = Math.min(m - 1, n);

  for (let k = 0; k < minMN; k++) {
    // Vector subdiagonal para Householder
    let normX = 0;
    for (let i = k; i < m; i++) {
      const val = rData[i * n + k];
      normX += val * val;
    }
    normX = Math.sqrt(normX);

    if (normX < 1e-15) continue;

    const alpha = (rData[k * n + k] >= 0 ? 1 : -1) * normX;
    const v = new Float64Array(m - k);
    v[0] = rData[k * n + k] + alpha;
    for (let i = k + 1; i < m; i++) {
      v[i - k] = rData[i * n + k];
    }

    let normV = 0;
    for (let i = 0; i < v.length; i++) normV += v[i] * v[i];
    normV = Math.sqrt(normV);

    if (normV < 1e-15) continue;
    for (let i = 0; i < v.length; i++) v[i] /= normV;

    // Aplicar reflector H = I - 2 * v * v^T a R (desde la izquierda)
    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < v.length; i++) {
        dot += v[i] * rData[(k + i) * n + j];
      }
      for (let i = 0; i < v.length; i++) {
        rData[(k + i) * n + j] -= 2 * v[i] * dot;
      }
    }

    // Aplicar reflector H a Q (acumular Q = Q * H)
    for (let i = 0; i < m; i++) {
      let dot = 0;
      for (let j = 0; j < v.length; j++) {
        dot += qData[i * m + (k + j)] * v[j];
      }
      for (let j = 0; j < v.length; j++) {
        qData[i * m + (k + j)] -= 2 * dot * v[j];
      }
    }
  }

  // Extraer submatriz cuadrada superior R (n x n) o completa (m x n)
  const Q = new NDArray(qData, { shape: [m, m] });
  const R = new NDArray(rData, { shape: [m, n] });
  return { Q, R };
}

/**
 * Descomposición en Valores Singulares (SVD) mediante bidiagonalización de Golub-Kahan.
 * A = U * S * V^T
 */
export function svd(A: NDArray, maxIter = 100): SVDResult {
  if (A.ndim !== 2) throw new Error('SVD requires a 2D matrix');
  const m = A.shape[0];
  const n = A.shape[1];

  // Algoritmo basado en rotaciones de Jacobi / One-Sided Jacobi para máxima estabilidad
  const VData = new Float64Array(n * n);
  for (let i = 0; i < n; i++) VData[i * n + i] = 1.0;

  const UData = new Float64Array(m * n);
  for (let r = 0; r < m; r++) {
    for (let c = 0; c < n; c++) {
      UData[r * n + c] = A.get(r, c);
    }
  }

  // Iteraciones de Jacobi para ortogonalizar columnas de U
  for (let iter = 0; iter < maxIter; iter++) {
    let converged = true;

    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;

        for (let k = 0; k < m; k++) {
          const u_ki = UData[k * n + i];
          const u_kj = UData[k * n + j];
          alpha += u_ki * u_ki;
          beta += u_kj * u_kj;
          gamma += u_ki * u_kj;
        }

        if (Math.abs(gamma) > 1e-14 * Math.sqrt(alpha * beta)) {
          converged = false;
          const zeta = (beta - alpha) / (2 * gamma);
          const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          const c = 1 / Math.sqrt(1 + t * t);
          const s = t * c;

          // Actualizar U
          for (let k = 0; k < m; k++) {
            const u_ki = UData[k * n + i];
            const u_kj = UData[k * n + j];
            UData[k * n + i] = c * u_ki - s * u_kj;
            UData[k * n + j] = s * u_ki + c * u_kj;
          }

          // Actualizar V
          for (let k = 0; k < n; k++) {
            const v_ki = VData[k * n + i];
            const v_kj = VData[k * n + j];
            VData[k * n + i] = c * v_ki - s * v_kj;
            VData[k * n + j] = s * v_ki + c * v_kj;
          }
        }
      }
    }

    if (converged) break;
  }

  // Normalizar columnas de U para obtener vectores singulares ortonormales y valores S
  const SData = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = 0; i < m; i++) norm += UData[i * n + j] * UData[i * n + j];
    norm = Math.sqrt(norm);
    SData[j] = norm;
    if (norm > 1e-15) {
      for (let i = 0; i < m; i++) UData[i * n + j] /= norm;
    }
  }

  return {
    U: new NDArray(UData, { shape: [m, n] }),
    S: new NDArray(SData, { shape: [n] }),
    V: new NDArray(VData, { shape: [n, n] }),
  };
}
