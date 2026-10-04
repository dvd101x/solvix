import nj from 'numjs';
import { NdArray } from '../core/ndarray.js';

export class NumJsAdapter {
  public static toNumJs(arr: NdArray): any {
    const nested = arr.toArray();
    return nj.array(nested);
  }

  public static fromNumJs(njArr: any): NdArray {
    const shape = [...njArr.shape];
    const rawData = njArr.tolist();
    return NdArray.fromArray(rawData);
  }
}
