/**
 * @file math-ops.ts
 * Operaciones matemáticas con despacho múltiple y reglas de broadcasting NumPy/Julia.
 */
import { NDArray, NestedArray, createGeneric, GenericFunction } from '../core/index.js';
import { Complex } from '../types/complex.js';
import { Fraction } from '../types/fraction.js';

export const add: GenericFunction = createGeneric('add');

/**
 * Calcula el shape resultante de dos shapes según reglas de NumPy/Julia.
 */
export function broadcastShapes(
  shapeA: ArrayLike<number>,
  shapeB: ArrayLike<number>
): Int32Array {
  const ndimA = shapeA.length;
  const ndimB = shapeB.length;
  const maxDim = Math.max(ndimA, ndimB);
  const resultShape = new Int32Array(maxDim);

  for (let i = 0; i < maxDim; i++) {
    const dimA = i < ndimA ? shapeA[ndimA - 1 - i] : 1;
    const dimB = i < ndimB ? shapeB[ndimB - 1 - i] : 1;

    if (dimA === dimB || dimA === 1 || dimB === 1) {
      resultShape[maxDim - 1 - i] = Math.max(dimA, dimB);
    } else {
      throw new Error(
        `Incompatible shapes for broadcasting: [${Array.from(shapeA)}] and [${Array.from(shapeB)}]`
      );
    }
  }
  return resultShape;
}

// 1. NDArray + Escalar (Number)
add.add([NDArray, 'number'], (arr: NDArray, scalar: number): NDArray => {
  const outData = new Float64Array(arr.size);
  const out = new NDArray(outData, { shape: Array.from(arr.shape), order: arr.order });

  let i = 0;
  for (const val of arr) {
    outData[i++] = val + scalar;
  }
  return out;
});

// 2. Escalar (Number) + NDArray (Propiedad conmutativa)
add.add(['number', NDArray], (scalar: number, arr: NDArray): NDArray => {
  return add(arr, scalar);
});

// 3. Number + Number
add.add(['number', 'number'], (a: number, b: number): number => a + b);

// 3.1 NestedArray + Number
add.add([NestedArray, 'number'], (arr: NestedArray, scalar: number): NestedArray => {
  return arr.map((x) => x + scalar);
});

// 3.2 Number + NestedArray
add.add(['number', NestedArray], (scalar: number, arr: NestedArray): NestedArray => {
  return arr.map((x) => scalar + x);
});

// 3.3 NestedArray + NestedArray (recursivo elemento a elemento)
add.add([NestedArray, NestedArray], (a: NestedArray, b: NestedArray): NestedArray => {
  const addRec = (nodeA: any, nodeB: any): any => {
    if (Array.isArray(nodeA) && Array.isArray(nodeB)) {
      return nodeA.map((item, idx) => addRec(item, nodeB[idx]));
    }
    return nodeA + nodeB;
  };
  return new NestedArray(addRec(a.data, b.data));
});

// 3.4 Complex + Complex / Complex + number
add.add([Complex, Complex], (z1: Complex, z2: Complex): Complex => z1.add(z2));
add.add([Complex, 'number'], (z: Complex, n: number): Complex => z.add(n));
add.add(['number', Complex], (n: number, z: Complex): Complex => z.add(n));

// 3.5 Fraction + Fraction / Fraction + number
add.add([Fraction, Fraction], (f1: Fraction, f2: Fraction): Fraction => f1.add(f2));
add.add([Fraction, 'number'], (f: Fraction, n: number): Fraction => f.add(n));
add.add(['number', Fraction], (n: number, f: Fraction): Fraction => f.add(n));

// 4. NDArray + NDArray (con Broadcasting general)
add.add([NDArray, NDArray], (a: NDArray, b: NDArray): NDArray => {
  const targetShape = broadcastShapes(a.shape, b.shape);
  let totalSize = 1;
  for (let i = 0; i < targetShape.length; i++) {
    totalSize *= targetShape[i];
  }

  const outData = new Float64Array(totalSize);
  const out = new NDArray(outData, { shape: Array.from(targetShape) });

  // Fast-path: dimensiones idénticas y contiguas
  if (a.size === b.size && a.size === totalSize) {
    const itA = a[Symbol.iterator]();
    const itB = b[Symbol.iterator]();
    for (let i = 0; i < totalSize; i++) {
      outData[i] = itA.next().value! + itB.next().value!;
    }
    return out;
  }

  // Slow-path: Recorrido multidimensional con proyección modular
  const coords = new Int32Array(targetShape.length);
  for (let i = 0; i < totalSize; i++) {
    const coordsA: number[] = [];
    const offsetA = targetShape.length - a.ndim;
    for (let d = 0; d < a.ndim; d++) {
      coordsA.push(a.shape[d] === 1 ? 0 : coords[offsetA + d]);
    }

    const coordsB: number[] = [];
    const offsetB = targetShape.length - b.ndim;
    for (let d = 0; d < b.ndim; d++) {
      coordsB.push(b.shape[d] === 1 ? 0 : coords[offsetB + d]);
    }

    const valA = a.get(...coordsA);
    const valB = b.get(...coordsB);
    outData[i] = valA + valB;

    for (let d = targetShape.length - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < targetShape[d]) break;
      coords[d] = 0;
    }
  }

  return out;
});
