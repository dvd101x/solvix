import { NDArray } from '../core/ndarray.js';

export function reduceTensorAxis(
  arr: NDArray,
  axis: number | undefined,
  keepdims: boolean,
  reducer: (values: number[]) => number
): NDArray | number {
  if (axis === undefined) {
    const values: number[] = [];
    for (const value of arr) values.push(value);
    const result = reducer(values);
    if (keepdims) {
      return new NDArray(new Float64Array([result]), { shape: new Array(arr.ndim).fill(1) });
    }
    return result;
  }

  const normAxis = axis < 0 ? arr.ndim + axis : axis;
  if (normAxis < 0 || normAxis >= arr.ndim) {
    throw new RangeError(`Axis ${axis} is out of bounds for ndim ${arr.ndim}`);
  }

  const outShape: number[] = [];
  for (let i = 0; i < arr.ndim; i++) {
    if (i === normAxis) {
      if (keepdims) outShape.push(1);
    } else {
      outShape.push(arr.shape[i]);
    }
  }

  let outSize = 1;
  for (let d = 0; d < outShape.length; d++) outSize *= outShape[d];

  const outData = new Float64Array(outSize);
  const axisLen = arr.shape[normAxis];
  const coords = new Int32Array(outShape.length);
  for (let outIdx = 0; outIdx < outSize; outIdx++) {
    const srcCoords = new Int32Array(arr.ndim);
    let coordIndex = 0;
    for (let d = 0; d < arr.ndim; d++) {
      if (d === normAxis) {
        if (keepdims) coordIndex++;
      } else {
        srcCoords[d] = coords[coordIndex++];
      }
    }

    const values = new Array<number>(axisLen);
    for (let k = 0; k < axisLen; k++) {
      srcCoords[normAxis] = k;
      values[k] = arr.get(...Array.from(srcCoords));
    }
    outData[outIdx] = reducer(values);

    for (let d = outShape.length - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < outShape[d]) break;
      coords[d] = 0;
    }
  }

  return new NDArray(outData, { shape: outShape.length ? outShape : [1] });
}
