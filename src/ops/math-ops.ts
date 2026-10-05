/**
 * @file math-ops.ts
 * Operaciones matemáticas con despacho múltiple y reglas de broadcasting.
 */
import { NDArray, NestedArray, createGeneric, GenericFunction } from '../core/index.js';
import { Complex } from '../types/complex.js';
import { Fraction } from '../types/fraction.js';
import { Quantity } from '../units/units.js';
import { broadcastShapes } from './broadcast.js';
import { binaryOp, addElementwise } from './elementwise.js';

export { broadcastShapes };

export const add: GenericFunction = createGeneric('add');

// NDArray (+ complex, + units) with scalars or other NDArrays, using broadcasting.
for (const scalarType of ['number', Complex, Quantity] as const) {
  add.add([NDArray, scalarType], (arr: NDArray, s: any) => binaryOp('add', arr, s));
  add.add([scalarType, NDArray], (s: any, arr: NDArray) => binaryOp('add', s, arr));
}
add.add([NDArray, NDArray], (a: NDArray, b: NDArray): NDArray => binaryOp('add', a, b));

// Complex / Quantity scalar pairs (number + number is registered below)
for (const l of ['number', Complex, Quantity] as const) {
  for (const r of ['number', Complex, Quantity] as const) {
    if (l === 'number' && r === 'number') continue;
    add.add([l, r], (a: any, b: any) => addElementwise(a, b));
  }
}

// Number + Number
add.add(['number', 'number'], (a: number, b: number): number => a + b);

// NestedArray + Number
add.add([NestedArray, 'number'], (arr: NestedArray, scalar: number): NestedArray => {
  return arr.map((x) => x + scalar);
});

// Number + NestedArray
add.add(['number', NestedArray], (scalar: number, arr: NestedArray): NestedArray => {
  return arr.map((x) => scalar + x);
});

// NestedArray + NestedArray (recursivo elemento a elemento)
add.add([NestedArray, NestedArray], (a: NestedArray, b: NestedArray): NestedArray => {
  const addRec = (nodeA: any, nodeB: any): any => {
    if (Array.isArray(nodeA) && Array.isArray(nodeB)) {
      return nodeA.map((item, idx) => addRec(item, nodeB[idx]));
    }
    return nodeA + nodeB;
  };
  return new NestedArray(addRec(a.data, b.data));
});

// Complex + Complex / Complex + number
add.add([Complex, Complex], (z1: Complex, z2: Complex): Complex => z1.add(z2));
add.add([Complex, 'number'], (z: Complex, n: number): Complex => z.add(n));
add.add(['number', Complex], (n: number, z: Complex): Complex => z.add(n));

// Fraction + Fraction / Fraction + number
add.add([Fraction, Fraction], (f1: Fraction, f2: Fraction): Fraction => f1.add(f2));
add.add([Fraction, 'number'], (f: Fraction, n: number): Fraction => f.add(n));
add.add(['number', Fraction], (n: number, f: Fraction): Fraction => f.add(n));
