import besselj from '@stdlib/math-base-special-besselj';
import { NdArray } from '../core/ndarray.js';

export class StdlibAdapter {
  /**
   * Vectorized evaluation of Bessel function of the first kind J_v(x)
   */
  public static besselj(v: number, x: number | NdArray): number | NdArray {
    if (typeof x === 'number') {
      return (besselj as any)(v, x);
    }
    const outData = new Float64Array(x.size);
    for (let i = 0; i < x.size; i++) {
      outData[i] = (besselj as any)(v, x.data[i]);
    }
    return new NdArray(outData, [...x.shape]);
  }
}
