/**
 * @file manipulation.ts
 * Vistas y manipulación de tensores en O(1) modificando strides y shapes sin copiar buffers:
 * - reshape
 * - transpose / swapaxes
 * - flatten / ravel
 * - squeeze / expandDims
 * - clip
 */
import { NDArray } from '../core/ndarray.js';
import { computeStridesRowMajor } from '../core/strides.js';

/**
 * Modifica la forma de un tensor sin copiar datos si es contiguo.
 * Soporta inferencia de dimensión con -1.
 */
export function reshape(arr: NDArray, newShape: number[]): NDArray {
  const totalElements = arr.size;
  const inferredShape = [...newShape];
  let minusOneIdx = -1;
  let knownProduct = 1;

  for (let i = 0; i < inferredShape.length; i++) {
    if (inferredShape[i] === -1) {
      if (minusOneIdx !== -1) {
        throw new Error('Can only specify one unknown dimension (-1)');
      }
      minusOneIdx = i;
    } else {
      knownProduct *= inferredShape[i];
    }
  }

  if (minusOneIdx !== -1) {
    if (knownProduct === 0 || totalElements % knownProduct !== 0) {
      throw new Error(`Cannot infer dimension for shape [${newShape}] with total size ${totalElements}`);
    }
    inferredShape[minusOneIdx] = totalElements / knownProduct;
  } else if (knownProduct !== totalElements) {
    throw new Error(`Cannot reshape array of size ${totalElements} into shape [${newShape}]`);
  }

  // Si es C-contiguo y offset es 0, simplemente creamos una vista directa
  // Non row-major layouts (e.g. transposed views) cannot be reinterpreted without a copy.
  const own = computeStridesRowMajor(Array.from(arr.shape));
  const contiguous = arr.shape.every((n, i) => n === 1 || arr.strides[i] === own[i]);
  const src = contiguous ? arr : arr.copy();
  return src.view({ shape: inferredShape, strides: computeStridesRowMajor(inferredShape) });
}

/**
 * Transpone los ejes de un tensor en O(1) permutando las formas y los strides.
 * Por defecto invierte el orden de todos los ejes (estilo np.transpose / A.T).
 */
export function transpose(arr: NDArray, axes?: number[]): NDArray {
  const ndim = arr.ndim;
  let perm = axes;

  if (!perm) {
    perm = [];
    for (let i = ndim - 1; i >= 0; i--) perm.push(i);
  }

  if (perm.length !== ndim) {
    throw new Error(`Axes permutation must have length ${ndim}, got ${perm.length}`);
  }

  const newShape = new Int32Array(ndim);
  const newStrides = new Int32Array(ndim);

  for (let i = 0; i < ndim; i++) {
    const axis = perm[i] < 0 ? ndim + perm[i] : perm[i];
    newShape[i] = arr.shape[axis];
    newStrides[i] = arr.strides[axis];
  }

  return arr.view({ shape: newShape, strides: newStrides });
}

/**
 * Aplana un tensor multidimensional a un vector 1D.
 */
export function flatten(arr: NDArray): NDArray {
  const flat = arr.copy();
  return flat.view({ shape: [arr.size], strides: [1], offset: 0, order: arr.order });
}

/**
 * Expande las dimensiones de un tensor insertando un nuevo eje de tamaño 1 en 'axis'.
 */
export function expandDims(arr: NDArray, axis: number): NDArray {
  const ndim = arr.ndim;
  const normAxis = axis < 0 ? ndim + 1 + axis : axis;
  const newShape: number[] = [];
  const newStrides: number[] = [];

  let oldDim = 0;
  for (let i = 0; i < ndim + 1; i++) {
    if (i === normAxis) {
      newShape.push(1);
      newStrides.push(arr.strides[oldDim] ?? 1);
    } else {
      newShape.push(arr.shape[oldDim]);
      newStrides.push(arr.strides[oldDim]);
      oldDim++;
    }
  }

  return arr.view({ shape: newShape, strides: new Int32Array(newStrides) });
}

/**
 * Elimina dimensiones de longitud 1 (ejes unitarios).
 */
export function squeeze(arr: NDArray, axis?: number): NDArray {
  const newShape: number[] = [];
  const newStrides: number[] = [];

  for (let i = 0; i < arr.ndim; i++) {
    if (axis !== undefined) {
      const normAxis = axis < 0 ? arr.ndim + axis : axis;
      if (i === normAxis) {
        if (arr.shape[i] !== 1) throw new Error(`Cannot select an axis with size ${arr.shape[i]} to squeeze`);
        continue;
      }
    } else if (arr.shape[i] === 1) {
      continue;
    }
    newShape.push(arr.shape[i]);
    newStrides.push(arr.strides[i]);
  }

  return arr.view({
    shape: newShape.length ? newShape : [1],
    strides: new Int32Array(newStrides.length ? newStrides : [1]),
  });
}

/**
 * Trunca los valores de un tensor al intervalo [min, max].
 */
export function clip(arr: NDArray, minVal: number, maxVal: number): NDArray {
  if (arr.isComplex) throw new TypeError('clip is not defined for complex arrays');
  const outData = new Float64Array(arr.size);
  let idx = 0;
  for (const v of arr) {
    outData[idx++] = Math.min(maxVal, Math.max(minVal, v));
  }
  return new NDArray(outData, { shape: Array.from(arr.shape), order: arr.order, unit: arr.unit });
}
