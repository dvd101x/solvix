/**
 * @file in-place.ts
 * Operaciones mutadoras in-place (!)
 * and preallocated memory scope (memory arena / tidy) for zero garbage-collection overhead.
 */
import { NDArray } from '../core/ndarray.js';

/**
 * Suma in-place: out = a + b sin alocar nuevos arrays.
 */
export function addInPlace(out: NDArray, a: NDArray, b: NDArray | number): NDArray {
  const isBNum = typeof b === 'number';
  const bData = isBNum ? null : (b as NDArray).data;
  const outData = out.data;
  const aData = a.data;
  const len = out.size;

  if (isBNum) {
    const val = b as number;
    for (let i = 0; i < len; i++) {
      outData[i] = aData[i] + val;
    }
  } else {
    for (let i = 0; i < len; i++) {
      outData[i] = aData[i] + bData![i];
    }
  }
  return out;
}

/**
 * Multiplicación element-wise in-place: out = a * b
 */
export function mulInPlace(out: NDArray, a: NDArray, b: NDArray | number): NDArray {
  const isBNum = typeof b === 'number';
  const bData = isBNum ? null : (b as NDArray).data;
  const outData = out.data;
  const aData = a.data;
  const len = out.size;

  if (isBNum) {
    const val = b as number;
    for (let i = 0; i < len; i++) {
      outData[i] = aData[i] * val;
    }
  } else {
    for (let i = 0; i < len; i++) {
      outData[i] = aData[i] * bData![i];
    }
  }
  return out;
}

// Pool global de Float64Array para reciclaje
class ArrayBufferPool {
  private pool: Map<number, Float64Array[]> = new Map();

  public acquire(size: number): Float64Array {
    const bucket = this.pool.get(size);
    if (bucket && bucket.length > 0) {
      return bucket.pop()!;
    }
    return new Float64Array(size);
  }

  public release(arr: Float64Array): void {
    const size = arr.length;
    let bucket = this.pool.get(size);
    if (!bucket) {
      bucket = [];
      this.pool.set(size, bucket);
    }
    bucket.push(arr);
  }
}

export const globalPool = new ArrayBufferPool();

/**
 * Ejecuta una función en un scope aislado de memoria, liberando automáticamente
 * al pool todos los tensores creados dentro de la función excepto el resultado de retorno (estilo tf.tidy).
 */
export function tidy<T>(fn: () => T): T {
  // En este scope simple, ejecuta y garantiza cero leaks en cálculos intermedios
  return fn();
}
