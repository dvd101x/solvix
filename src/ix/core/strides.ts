/**
 * @file strides.ts
 * Sistema de mapeo de índices multidimensionales a lineales contiguos (sub2ind).
 * Funciones desenrolladas (unrolled) para 1D a 6D optimizadas para V8 JIT inlining y zero-allocations.
 */

export type StrideArray = Int32Array | number[];
export type ShapeArray = Int32Array | number[];

// --- Funciones Inlined Monomórficas (1D a 6D) ---

export const sub2ind1D = (offset: number, s0: number, i0: number): number =>
  (offset + i0 * s0) | 0;

export const sub2ind2D = (
  offset: number,
  s0: number,
  s1: number,
  i0: number,
  i1: number
): number => (offset + i0 * s0 + i1 * s1) | 0;

export const sub2ind3D = (
  offset: number,
  s0: number,
  s1: number,
  s2: number,
  i0: number,
  i1: number,
  i2: number
): number => (offset + i0 * s0 + i1 * s1 + i2 * s2) | 0;

export const sub2ind4D = (
  offset: number,
  s0: number,
  s1: number,
  s2: number,
  s3: number,
  i0: number,
  i1: number,
  i2: number,
  i3: number
): number => (offset + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3) | 0;

export const sub2ind5D = (
  offset: number,
  s0: number,
  s1: number,
  s2: number,
  s3: number,
  s4: number,
  i0: number,
  i1: number,
  i2: number,
  i3: number,
  i4: number
): number => (offset + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4) | 0;

export const sub2ind6D = (
  offset: number,
  s0: number,
  s1: number,
  s2: number,
  s3: number,
  s4: number,
  s5: number,
  i0: number,
  i1: number,
  i2: number,
  i3: number,
  i4: number,
  i5: number
): number => (offset + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5) | 0;

/**
 * Fallback genérico para tensores con N > 6 dimensiones.
 */
export function sub2indND(
  offset: number,
  strides: ArrayLike<number>,
  indices: ArrayLike<number>
): number {
  let idx = offset | 0;
  const len = strides.length;
  for (let i = 0; i < len; i = (i + 1) | 0) {
    idx = (idx + indices[i] * strides[i]) | 0;
  }
  return idx;
}

/**
 * Tabla de despacho indexada por ndim (1 a 6)
 */
export const SUB2IND_TABLE = [
  null,
  sub2ind1D,
  sub2ind2D,
  sub2ind3D,
  sub2ind4D,
  sub2ind5D,
  sub2ind6D,
] as const;

/**
 * Calcula los pasos (strides) para formato Row-Major.
 */
export function computeStridesRowMajor(shape: ArrayLike<number>): Int32Array {
  const ndim = shape.length;
  const strides = new Int32Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Calcula los pasos (strides) para formato Column-Major.
 */
export function computeStridesColMajor(shape: ArrayLike<number>): Int32Array {
  const ndim = shape.length;
  const strides = new Int32Array(ndim);
  let stride = 1;
  for (let i = 0; i < ndim; i++) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}
