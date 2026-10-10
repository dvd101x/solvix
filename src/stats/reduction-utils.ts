import { NDArray } from '../core/ndarray.js';

export type Axis = number | readonly number[];

export function normalizeAxes(axis: Axis, ndim: number): number[] {
  const axes = typeof axis === 'number' ? [axis] : Array.from(axis);
  if (axes.length === 0) {
    throw new RangeError('Axis list must contain at least one axis');
  }

  const normalized = new Set<number>();
  for (const value of axes) {
    if (!Number.isInteger(value)) {
      throw new TypeError(`Axis must be an integer, got ${value}`);
    }
    const normalizedAxis = value < 0 ? ndim + value : value;
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new RangeError(`Axis ${value} is out of bounds for ndim ${ndim}`);
    }
    if (normalized.has(normalizedAxis)) {
      throw new RangeError(`Axis ${value} is specified more than once`);
    }
    normalized.add(normalizedAxis);
  }
  return Array.from(normalized);
}

export function reduceTensorAxis(
  arr: NDArray,
  axis: Axis | undefined,
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

  const axes = normalizeAxes(axis, arr.ndim);
  const reducedAxes = new Set(axes);

  const outShape: number[] = [];
  for (let i = 0; i < arr.ndim; i++) {
    if (reducedAxes.has(i)) {
      if (keepdims) outShape.push(1);
    } else {
      outShape.push(arr.shape[i]);
    }
  }

  let outSize = 1;
  for (let d = 0; d < outShape.length; d++) outSize *= outShape[d];

  const outData = new Float64Array(outSize);
  let reducedSize = 1;
  for (const reducedAxis of axes) reducedSize *= arr.shape[reducedAxis];
  const coords = new Int32Array(outShape.length);
  for (let outIdx = 0; outIdx < outSize; outIdx++) {
    const srcCoords = new Int32Array(arr.ndim);
    let coordIndex = 0;
    for (let d = 0; d < arr.ndim; d++) {
      if (reducedAxes.has(d)) {
        if (keepdims) coordIndex++;
      } else {
        srcCoords[d] = coords[coordIndex++];
      }
    }

    const values = new Array<number>(reducedSize);
    const reducedCoords = new Int32Array(axes.length);
    for (let valueIndex = 0; valueIndex < reducedSize; valueIndex++) {
      for (let axisIndex = 0; axisIndex < axes.length; axisIndex++) {
        srcCoords[axes[axisIndex]] = reducedCoords[axisIndex];
      }
      values[valueIndex] = arr.get(...Array.from(srcCoords));

      for (let axisIndex = axes.length - 1; axisIndex >= 0; axisIndex--) {
        reducedCoords[axisIndex]++;
        if (reducedCoords[axisIndex] < arr.shape[axes[axisIndex]]) break;
        reducedCoords[axisIndex] = 0;
      }
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
