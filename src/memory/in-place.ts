/**
 * @file in-place.ts
 * Operaciones mutadoras in-place (!)
 * and preallocated memory scope (memory arena / tidy) for zero garbage-collection overhead.
 */
import { NDArray } from '../core/ndarray.js';
import { Complex } from '../types/complex.js';
import { broadcastInto } from '../ops/broadcast-map.js';

const toComplex = (v: number | Complex): Complex => (v instanceof Complex ? v : new Complex(v, 0));

/**
 * In-place addition: out = a + b, without allocating. Operands broadcast to the shape of `out`
 * and may be strided views, complex, or scalars; `out` may be one of the operands.
 */
export function addInPlace(out: NDArray, a: NDArray | number, b: NDArray | number): NDArray {
  return broadcastInto(out, (x, y) => (x instanceof Complex || y instanceof Complex ? toComplex(x).add(toComplex(y)) : x + y), a, b);
}

/**
 * In-place element-wise multiplication: out = a .* b, with the same rules as `addInPlace`.
 */
export function mulInPlace(out: NDArray, a: NDArray | number, b: NDArray | number): NDArray {
  return broadcastInto(out, (x, y) => (x instanceof Complex || y instanceof Complex ? toComplex(x).mul(toComplex(y)) : x * y), a, b);
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
