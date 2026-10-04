/**
 * @file shared-memory.ts
 * Utilidades para computación paralela en workers:
 * - SharedArrayBuffer (memoria compartida zero-copy entre hilos)
 * - ArrayBuffer Transfer (transferencia sin clonación de buffers estándar)
 * - Particionamiento de trabajo (chunks) para paralelizar bucles
 */
import { NDArray, NDArrayOptions } from '../core/ndarray.js';

export interface ChunkRange {
  start: number;
  end: number;
  length: number;
}

/**
 * Crea un NDArray respaldado por un SharedArrayBuffer para acceso concurrente en Web Workers / Worker Threads.
 */
export function createSharedNDArray(
  shape: number[],
  type: typeof Float64Array | typeof Float32Array | typeof Int32Array = Float64Array
): NDArray {
  let size = 1;
  for (let i = 0; i < shape.length; i++) size *= shape[i];

  const bytesPerElement = type.BYTES_PER_ELEMENT;
  const sab = new SharedArrayBuffer(size * bytesPerElement);
  const data = new type(sab as unknown as ArrayBuffer);

  return new NDArray(data, { shape });
}

/**
 * Divide el tamaño total de un array en rangos contiguos equilibrados para 'numWorkers'.
 */
export function partitionWork(totalSize: number, numWorkers: number): ChunkRange[] {
  const chunks: ChunkRange[] = [];
  const baseChunkSize = Math.floor(totalSize / numWorkers);
  let remainder = totalSize % numWorkers;
  let currentStart = 0;

  for (let i = 0; i < numWorkers; i++) {
    const chunkSize = baseChunkSize + (remainder > 0 ? 1 : 0);
    if (remainder > 0) remainder--;

    const end = currentStart + chunkSize;
    chunks.push({
      start: currentStart,
      end,
      length: chunkSize,
    });
    currentStart = end;
  }

  return chunks;
}

/**
 * Prepara un NDArray estándar para ser transferido (Transferable Objects) a un Worker sin copia profunda.
 */
export function prepareTransfer(arr: NDArray): { message: any; transferables: Transferable[] } {
  const buffer = arr.data.buffer;
  return {
    message: {
      data: arr.data,
      shape: Array.from(arr.shape),
      strides: Array.from(arr.strides),
      offset: arr.offset,
      order: arr.order,
      isShared: buffer instanceof SharedArrayBuffer,
    },
    transferables: buffer instanceof SharedArrayBuffer ? [] : [buffer],
  };
}

/**
 * Reconstruye un NDArray a partir de los datos recibidos en el Worker.
 */
export function reconstructFromTransfer(msg: any): NDArray {
  return new NDArray(msg.data, {
    shape: msg.shape,
    strides: new Int32Array(msg.strides),
    offset: msg.offset,
    order: msg.order,
  });
}
