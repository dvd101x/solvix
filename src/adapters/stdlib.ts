import { gamma } from 'mathjs';
import { NdArray } from '../core/ndarray.js';

function besselJ(order: number, x: number): number {
  if (Number.isInteger(order) && order < 0) {
    const positiveOrder = -order;
    return (positiveOrder % 2 === 0 ? 1 : -1) * besselJ(positiveOrder, x);
  }
  if (x === 0) return order === 0 ? 1 : order > 0 ? 0 : Infinity;
  if (x < 0 && !Number.isInteger(order)) return NaN;

  const halfX = x / 2;
  let term = Math.pow(halfX, order) / gamma(order + 1);
  let sum = term;
  for (let k = 1; k <= 200; k++) {
    term *= -(halfX * halfX) / (k * (k + order));
    sum += term;
    if (Math.abs(term) <= Number.EPSILON * Math.max(1, Math.abs(sum))) break;
  }
  return sum;
}

export class StdlibAdapter {
  /**
   * Vectorized evaluation of Bessel function of the first kind J_v(x)
   */
  public static besselj(v: number, x: number | NdArray): number | NdArray {
    if (typeof x === 'number') {
      return besselJ(v, x);
    }
    const outData = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      outData[i] = besselJ(v, x.data[i]);
    }
    return new NdArray(outData, [...x.shape]);
  }
}
