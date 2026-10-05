import { NDArray } from '../core/ndarray.js';

export function scanTensorAxis(
  arr: NDArray,
  axis: number | undefined,
  combine: (accumulated: number, value: number) => number,
  preserveUnit: boolean
): NDArray {
  if (arr.isComplex) {
    throw new TypeError('Cumulative scans are not defined for complex arrays');
  }
  if (arr.unit && !preserveUnit) {
    throw new TypeError('Cumulative product is not supported for arrays with units');
  }

  if (axis === undefined) {
    const out = new Float64Array(arr.size);
    let index = 0;
    let accumulated = 0;
    for (const value of arr) {
      accumulated = index === 0 ? value : combine(accumulated, value);
      out[index++] = accumulated;
    }
    return new NDArray(out, { shape: [arr.size], unit: preserveUnit ? arr.unit : undefined });
  }

  const normAxis = axis < 0 ? arr.ndim + axis : axis;
  if (normAxis < 0 || normAxis >= arr.ndim) {
    throw new RangeError(`Axis ${axis} is out of bounds for ndim ${arr.ndim}`);
  }
  const out = new Float64Array(arr.size);
  const coords = new Int32Array(arr.ndim);
  let axisStride = 1;
  for (let dim = normAxis + 1; dim < arr.ndim; dim++) axisStride *= arr.shape[dim];

  for (let index = 0; index < arr.size; index++) {
    const value = arr.get(...Array.from(coords));
    out[index] = coords[normAxis] === 0
      ? value
      : combine(out[index - axisStride], value);

    for (let dim = arr.ndim - 1; dim >= 0; dim--) {
      coords[dim]++;
      if (coords[dim] < arr.shape[dim]) break;
      coords[dim] = 0;
    }
  }

  return new NDArray(out, {
    shape: arr.shape,
    unit: preserveUnit ? arr.unit : undefined,
  });
}
