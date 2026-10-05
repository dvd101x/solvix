/**
 * @file series-ops.ts
 * Operaciones de series temporales y transformaciones estadísticas:
 * - rolling (ventanas móviles con mean, std, min, max, sum)
 * - ewm (Exponential Weighted Moving Average)
 * - diff (diferenciación discreta / derivadas)
 * - pct_change (tasa de cambio porcentual)
 * - shift (desplazamientos con lag/lead)
 * - cov (matriz de covarianza) & corr (matriz de correlación de Pearson)
 * - nanmean, nansum, nanstd, isnan, fillna, dropna
 */
import { NDArray } from '../core/ndarray.js';

export interface RollingWindow {
  mean(): NDArray;
  std(ddof?: number): NDArray;
  sum(): NDArray;
  min(): NDArray;
  max(): NDArray;
}

/**
 * Ventana móvil deslizante sobre un vector o array 1D.
 */
export function rolling(arr: NDArray, windowSize: number): RollingWindow {
  if (windowSize <= 0) throw new RangeError(`Window size must be positive, got ${windowSize}`);
  const n = arr.size;
  const data = arr.contiguous().data;

  return {
    mean(): NDArray {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (i < windowSize - 1) {
          out[i] = NaN;
          continue;
        }
        let s = 0;
        for (let k = 0; k < windowSize; k++) {
          s += data[i - k];
        }
        out[i] = s / windowSize;
      }
      return new NDArray(out, { shape: [n] });
    },

    std(ddof = 1): NDArray {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (i < windowSize - 1) {
          out[i] = NaN;
          continue;
        }
        let s = 0;
        for (let k = 0; k < windowSize; k++) s += data[i - k];
        const m = s / windowSize;

        let ss = 0;
        for (let k = 0; k < windowSize; k++) {
          const diff = data[i - k] - m;
          ss += diff * diff;
        }
        out[i] = Math.sqrt(ss / (windowSize - ddof));
      }
      return new NDArray(out, { shape: [n] });
    },

    sum(): NDArray {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (i < windowSize - 1) {
          out[i] = NaN;
          continue;
        }
        let s = 0;
        for (let k = 0; k < windowSize; k++) s += data[i - k];
        out[i] = s;
      }
      return new NDArray(out, { shape: [n] });
    },

    min(): NDArray {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (i < windowSize - 1) {
          out[i] = NaN;
          continue;
        }
        let m = data[i];
        for (let k = 1; k < windowSize; k++) {
          if (data[i - k] < m) m = data[i - k];
        }
        out[i] = m;
      }
      return new NDArray(out, { shape: [n] });
    },

    max(): NDArray {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (i < windowSize - 1) {
          out[i] = NaN;
          continue;
        }
        let m = data[i];
        for (let k = 1; k < windowSize; k++) {
          if (data[i - k] > m) m = data[i - k];
        }
        out[i] = m;
      }
      return new NDArray(out, { shape: [n] });
    },
  };
}

/**
 * Media móvil ponderada exponencialmente (Exponential Weighted Moving Average - EWMA)
 */
export function ewm(arr: NDArray, opts: { alpha?: number; span?: number }): NDArray {
  const alpha = opts.alpha ?? (opts.span ? 2 / (opts.span + 1) : 0.2);
  const n = arr.size;
  const out = new Float64Array(n);
  const data = arr.contiguous().data;

  if (n === 0) return new NDArray(out, { shape: [0] });

  out[0] = data[0];
  for (let i = 1; i < n; i++) {
    out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
  }

  return new NDArray(out, { shape: [n] });
}

/**
 * Diferencia discreta de primer o enésimo orden: y[i] - y[i - periods]
 */
export function diff(arr: NDArray, periods: number = 1): NDArray {
  const n = arr.size;
  const out = new Float64Array(n);
  const data = arr.contiguous().data;

  for (let i = 0; i < n; i++) {
    if (i < periods) {
      out[i] = NaN;
    } else {
      out[i] = data[i] - data[i - periods];
    }
  }

  return new NDArray(out, { shape: [n] });
}

/**
 * Cambio porcentual: (y[i] - y[i - periods]) / y[i - periods]
 */
export function pctChange(arr: NDArray, periods: number = 1): NDArray {
  const n = arr.size;
  const out = new Float64Array(n);
  const data = arr.contiguous().data;

  for (let i = 0; i < n; i++) {
    if (i < periods || data[i - periods] === 0) {
      out[i] = NaN;
    } else {
      out[i] = (data[i] - data[i - periods]) / data[i - periods];
    }
  }

  return new NDArray(out, { shape: [n] });
}

/**
 * Desplaza elementos en 'periods' posiciones hacia adelante o atrás insertando fillValue.
 */
export function shift(arr: NDArray, periods: number = 1, fillValue: number = NaN): NDArray {
  const n = arr.size;
  const out = new Float64Array(n);
  const data = arr.contiguous().data;

  for (let i = 0; i < n; i++) {
    const src = i - periods;
    if (src >= 0 && src < n) {
      out[i] = data[src];
    } else {
      out[i] = fillValue;
    }
  }

  return new NDArray(out, { shape: [n] });
}

// --- Manejo de Datos Faltantes (NaNs) ---

export function isnan(arr: NDArray): NDArray {
  const out = new Uint8Array(arr.size);
  let i = 0;
  for (const v of arr) {
    out[i++] = Number.isNaN(v) ? 1 : 0;
  }
  return new NDArray(out, { shape: Array.from(arr.shape) });
}

export function fillna(arr: NDArray, method: number | 'ffill' | 'bfill'): NDArray {
  const out = new Float64Array(arr.size);
  const data = arr.contiguous().data;
  const n = arr.size;

  if (typeof method === 'number') {
    for (let i = 0; i < n; i++) {
      out[i] = Number.isNaN(data[i]) ? method : data[i];
    }
  } else if (method === 'ffill') {
    let lastValid = NaN;
    for (let i = 0; i < n; i++) {
      if (!Number.isNaN(data[i])) {
        lastValid = data[i];
      }
      out[i] = Number.isNaN(data[i]) ? lastValid : data[i];
    }
  } else if (method === 'bfill') {
    let nextValid = NaN;
    for (let i = n - 1; i >= 0; i--) {
      if (!Number.isNaN(data[i])) {
        nextValid = data[i];
      }
      out[i] = Number.isNaN(data[i]) ? nextValid : data[i];
    }
  }

  return new NDArray(out, { shape: Array.from(arr.shape) });
}

export function dropna(arr: NDArray): NDArray {
  const valid: number[] = [];
  for (const v of arr) {
    if (!Number.isNaN(v)) valid.push(v);
  }
  return new NDArray(new Float64Array(valid), { shape: [valid.length] });
}

export function nanmean(arr: NDArray): number {
  let sum = 0;
  let count = 0;
  for (const v of arr) {
    if (!Number.isNaN(v)) {
      sum += v;
      count++;
    }
  }
  return count > 0 ? sum / count : NaN;
}

export function nansum(arr: NDArray): number {
  let sum = 0;
  for (const v of arr) {
    if (!Number.isNaN(v)) sum += v;
  }
  return sum;
}

// --- Matrices de Covarianza y Correlación ---

/**
 * Matriz de Covarianza muestral de un conjunto de variables columna en una matriz 2D (N filas x M columnas).
 */
export function cov(arr: NDArray, ddof = 1): NDArray {
  if (arr.ndim !== 2) throw new Error('Covariance requires a 2D matrix (samples x variables)');
  const rows = arr.shape[0];
  const cols = arr.shape[1];

  // 1. Calcular media por columna
  const means = new Float64Array(cols);
  for (let c = 0; c < cols; c++) {
    let s = 0;
    for (let r = 0; r < rows; r++) s += arr.get(r, c);
    means[c] = s / rows;
  }

  // 2. Matriz de covarianza cols x cols
  const covData = new Float64Array(cols * cols);
  const factor = 1 / (rows - ddof);

  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) {
        sum += (arr.get(r, i) - means[i]) * (arr.get(r, j) - means[j]);
      }
      const val = sum * factor;
      covData[i * cols + j] = val;
      covData[j * cols + i] = val;
    }
  }

  return new NDArray(covData, { shape: [cols, cols] });
}

/**
 * Matriz de Correlación de Pearson normalizada en [-1, 1] a partir de una matriz de datos 2D.
 */
export function corr(arr: NDArray): NDArray {
  const covMat = cov(arr, 1);
  const cols = covMat.shape[0];
  const corrData = new Float64Array(cols * cols);

  const stds = new Float64Array(cols);
  for (let i = 0; i < cols; i++) {
    stds[i] = Math.sqrt(covMat.get(i, i));
  }

  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < cols; j++) {
      if (i === j) {
        corrData[i * cols + j] = 1.0;
      } else {
        const denom = stds[i] * stds[j];
        corrData[i * cols + j] = denom === 0 ? NaN : covMat.get(i, j) / denom;
      }
    }
  }

  return new NDArray(corrData, { shape: [cols, cols] });
}
