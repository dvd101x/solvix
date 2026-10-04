/**
 * @file advanced-indexing.ts
 * Indexación avanzada estilo NumPy y Julia para NDArray:
 * - Fancy indexing por índices enteros (take / put)
 * - Indexación booleana condicional (masking)
 * - Slicing multidimensional con soporte de elipsis (...)
 */
import { NDArray, TypedArray } from '../core/ndarray.js';

export const ELLIPSIS = Symbol('ELLIPSIS');

export type IndexSliceArg =
  | number
  | [start?: number, stop?: number, step?: number]
  | typeof ELLIPSIS
  | null
  | undefined;

/**
 * Filtra los elementos de un NDArray según una máscara booleana (NDArray de Uint8Array o boolean[]).
 * Devuelve un NDArray 1D con los elementos que cumplen la condición (estilo A[mask]).
 */
export function booleanMask(arr: NDArray, mask: NDArray | Uint8Array | boolean[]): NDArray {
  const matches: number[] = [];
  let maskIdx = 0;

  const maskGetter: (i: number) => boolean =
    mask instanceof NDArray
      ? (i: number) => Boolean(mask.data[i])
      : (i: number) => Boolean((mask as any)[i]);

  let i = 0;
  for (const val of arr) {
    if (maskGetter(i)) {
      matches.push(val);
    }
    i++;
  }

  const outData = new Float64Array(matches);
  return new NDArray(outData, { shape: [matches.length], order: arr.order });
}

export function where(condition: NDArray, whenTrue: NDArray | number, whenFalse: NDArray | number): NDArray {
  for (const value of [whenTrue, whenFalse]) {
    if (
      value instanceof NDArray &&
      (value.ndim !== condition.ndim || value.shape.some((size, axis) => size !== condition.shape[axis]))
    ) {
      throw new Error('where array arguments must have the same shape as condition');
    }
  }

  const conditionValues = condition[Symbol.iterator]();
  const trueValues = whenTrue instanceof NDArray ? whenTrue[Symbol.iterator]() : undefined;
  const falseValues = whenFalse instanceof NDArray ? whenFalse[Symbol.iterator]() : undefined;
  const outData = new Float64Array(condition.size);

  for (let i = 0; i < condition.size; i++) {
    const selected = conditionValues.next().value !== 0;
    const trueValue = trueValues ? trueValues.next().value! : whenTrue as number;
    const falseValue = falseValues ? falseValues.next().value! : whenFalse as number;
    outData[i] = selected ? trueValue : falseValue;
  }

  return new NDArray(outData, { shape: Array.from(condition.shape), order: condition.order });
}

/**
 * Extrae elementos a lo largo de un eje usando una lista de índices enteros (Fancy indexing / take).
 * @param arr NDArray de origen
 * @param indices Array de enteros con los índices deseados en el eje
 * @param axis Eje sobre el que se aplica la extracción (por defecto 0)
 */
export function take(arr: NDArray, indices: number[] | Int32Array, axis: number = 0): NDArray {
  const normalizedAxis = axis < 0 ? arr.ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= arr.ndim) {
    throw new RangeError(`Axis ${axis} out of bounds for ndim ${arr.ndim}`);
  }

  const newShape = Array.from(arr.shape);
  newShape[normalizedAxis] = indices.length;

  let totalSize = 1;
  for (let d = 0; d < newShape.length; d++) totalSize *= newShape[d];

  const outData = new Float64Array(totalSize);
  const out = new NDArray(outData, { shape: newShape, order: arr.order });

  // Recorrido multidimensional
  const coords = new Int32Array(newShape.length);
  for (let i = 0; i < totalSize; i++) {
    const srcCoords = Array.from(coords);
    srcCoords[normalizedAxis] = indices[coords[normalizedAxis]];

    out.set(...coords, arr.get(...srcCoords));

    for (let d = newShape.length - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < newShape[d]) break;
      coords[d] = 0;
    }
  }

  return out;
}

/**
 * Asigna valores en las posiciones indicadas por la máscara booleana (In-place).
 */
export function putMask(
  target: NDArray,
  mask: NDArray | Uint8Array | boolean[],
  value: number | NDArray
): void {
  const isValNDArray = value instanceof NDArray;
  let valIter: Iterator<number> | null = isValNDArray ? (value as NDArray)[Symbol.iterator]() : null;

  const maskGetter: (i: number) => boolean =
    mask instanceof NDArray
      ? (i: number) => Boolean(mask.data[i])
      : (i: number) => Boolean((mask as any)[i]);

  const coords = new Int32Array(target.ndim);
  for (let i = 0; i < target.size; i++) {
    if (maskGetter(i)) {
      const v = isValNDArray ? valIter!.next().value : (value as number);
      target.set(...coords, v);
    }

    for (let d = target.ndim - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < target.shape[d]) break;
      coords[d] = 0;
    }
  }
}

/**
 * Normaliza cortes incluyendo soporte de Elipsis (...) expandiéndolo a cortes completos.
 */
export function sliceWithEllipsis(arr: NDArray, ...slices: IndexSliceArg[]): NDArray {
  const ellipsisCount = slices.filter((s) => s === ELLIPSIS).length;
  if (ellipsisCount > 1) {
    throw new Error('An index can only have a single ellipsis (...)');
  }

  const normalizedSlices: [number, number, number][] = [];

  if (ellipsisCount === 1) {
    const ellipsisIndex = slices.indexOf(ELLIPSIS);
    const nonEllipsisItems = slices.length - 1;
    const expandCount = Math.max(0, arr.ndim - nonEllipsisItems);

    for (let i = 0; i < slices.length; i++) {
      if (i === ellipsisIndex) {
        for (let e = 0; e < expandCount; e++) {
          normalizedSlices.push([0, arr.shape[normalizedSlices.length], 1]);
        }
      } else {
        const s = slices[i];
        if (typeof s === 'number') {
          normalizedSlices.push([s, s + 1, 1]);
        } else if (Array.isArray(s)) {
          normalizedSlices.push([s[0] ?? 0, s[1] ?? arr.shape[normalizedSlices.length], s[2] ?? 1]);
        } else {
          normalizedSlices.push([0, arr.shape[normalizedSlices.length], 1]);
        }
      }
    }
  } else {
    for (let i = 0; i < arr.ndim; i++) {
      const s = slices[i];
      if (typeof s === 'number') {
        normalizedSlices.push([s, s + 1, 1]);
      } else if (Array.isArray(s)) {
        normalizedSlices.push([s[0] ?? 0, s[1] ?? arr.shape[i], s[2] ?? 1]);
      } else {
        normalizedSlices.push([0, arr.shape[i], 1]);
      }
    }
  }

  return arr.slice(...normalizedSlices);
}
