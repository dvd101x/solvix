/**
 * @file ndarray.ts
 * Clase base primordial NDArray con acceso ultra rápido, slicing lazy y Symbol.iterator.
 */
import {
  sub2indND,
  computeStridesRowMajor,
  computeStridesColMajor,
} from './strides.js';

export type TypedArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Int16Array
  | Int8Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Uint8ClampedArray;

export type NestedNumberArray = number | NestedNumberArray[];

export interface NDArrayOptions {
  shape?: number[] | Int32Array;
  strides?: number[] | Int32Array;
  offset?: number;
  order?: 'C' | 'F';
}

export type SliceRange = [start?: number, stop?: number, step?: number];

export class NDArray {
  public readonly data: TypedArray;
  public readonly shape: Int32Array;
  public readonly strides: Int32Array;
  public readonly ndim: number;
  public readonly offset: number;
  public readonly order: 'C' | 'F';
  public readonly size: number;

  // Cache explícito de strides para 1D-6D para evitar overhead de indexación en hot-paths
  public readonly _s0: number;
  public readonly _s1: number;
  public readonly _s2: number;
  public readonly _s3: number;
  public readonly _s4: number;
  public readonly _s5: number;

  constructor(data: TypedArray, options: NDArrayOptions = {}) {
    this.data = data;
    this.shape = options.shape instanceof Int32Array ? options.shape : new Int32Array(options.shape || [data.length]);
    this.ndim = this.shape.length;
    this.offset = (options.offset || 0) | 0;
    this.order = options.order || 'C';

    if (options.strides) {
      this.strides = options.strides instanceof Int32Array ? options.strides : new Int32Array(options.strides);
    } else {
      this.strides = this.order === 'F'
        ? computeStridesColMajor(this.shape)
        : computeStridesRowMajor(this.shape);
    }

    let size = 1;
    for (let i = 0; i < this.ndim; i++) {
      size = (size * this.shape[i]) | 0;
    }
    this.size = size;

    this._s0 = this.strides[0] | 0;
    this._s1 = this.strides[1] | 0;
    this._s2 = this.strides[2] | 0;
    this._s3 = this.strides[3] | 0;
    this._s4 = this.strides[4] | 0;
    this._s5 = this.strides[5] | 0;
  }

  // --- Accesos de Máximo Rendimiento Monomórficos ---

  public get1D(i0: number): number {
    return this.data[(this.offset + i0 * this._s0) | 0];
  }

  public set1D(i0: number, val: number): void {
    this.data[(this.offset + i0 * this._s0) | 0] = val;
  }

  public get2D(i0: number, i1: number): number {
    return this.data[(this.offset + i0 * this._s0 + i1 * this._s1) | 0];
  }

  public set2D(i0: number, i1: number, val: number): void {
    this.data[(this.offset + i0 * this._s0 + i1 * this._s1) | 0] = val;
  }

  /**
   * get optimizado para V8 con switch inline monomórfico (1D a 6D) y fallback N > 6
   */
  public get(
    i0: number = 0,
    i1: number = 0,
    i2: number = 0,
    i3: number = 0,
    i4: number = 0,
    i5: number = 0,
    ...rest: number[]
  ): number {
    switch (this.ndim) {
      case 1:
        return this.data[(this.offset + i0 * this._s0) | 0];
      case 2:
        return this.data[(this.offset + i0 * this._s0 + i1 * this._s1) | 0];
      case 3:
        return this.data[(this.offset + i0 * this._s0 + i1 * this._s1 + i2 * this._s2) | 0];
      case 4:
        return this.data[(this.offset + i0 * this._s0 + i1 * this._s1 + i2 * this._s2 + i3 * this._s3) | 0];
      case 5:
        return this.data[(this.offset + i0 * this._s0 + i1 * this._s1 + i2 * this._s2 + i3 * this._s3 + i4 * this._s4) | 0];
      case 6:
        return this.data[(this.offset + i0 * this._s0 + i1 * this._s1 + i2 * this._s2 + i3 * this._s3 + i4 * this._s4 + i5 * this._s5) | 0];
      default: {
        const indices = [i0, i1, i2, i3, i4, i5, ...rest];
        return this.data[sub2indND(this.offset, this.strides, indices)];
      }
    }
  }

  /**
   * set optimizado para V8
   */
  public set(...args: number[]): void {
    const val = args[args.length - 1];
    switch (this.ndim) {
      case 1:
        this.data[(this.offset + args[0] * this._s0) | 0] = val;
        break;
      case 2:
        this.data[(this.offset + args[0] * this._s0 + args[1] * this._s1) | 0] = val;
        break;
      case 3:
        this.data[(this.offset + args[0] * this._s0 + args[1] * this._s1 + args[2] * this._s2) | 0] = val;
        break;
      case 4:
        this.data[(this.offset + args[0] * this._s0 + args[1] * this._s1 + args[2] * this._s2 + args[3] * this._s3) | 0] = val;
        break;
      case 5:
        this.data[(this.offset + args[0] * this._s0 + args[1] * this._s1 + args[2] * this._s2 + args[3] * this._s3 + args[4] * this._s4) | 0] = val;
        break;
      case 6:
        this.data[(this.offset + args[0] * this._s0 + args[1] * this._s1 + args[2] * this._s2 + args[3] * this._s3 + args[4] * this._s4 + args[5] * this._s5) | 0] = val;
        break;
      default:
        this.data[sub2indND(this.offset, this.strides, args.slice(0, -1))] = val;
    }
  }

  /**
   * Slicing Lazy: Crea una vista (view) sin duplicar memoria subyacente.
   */
  public slice(...ranges: (SliceRange | null | undefined)[]): NDArray {
    let newOffset = this.offset;
    const newShape: number[] = [];
    const newStrides: number[] = [];

    for (let d = 0; d < this.ndim; d++) {
      const range = ranges[d];
      if (!range) {
        newShape.push(this.shape[d]);
        newStrides.push(this.strides[d]);
        continue;
      }

      const start = range[0] !== undefined ? range[0] : 0;
      const stop = range[1] !== undefined ? range[1] : this.shape[d];
      const step = range[2] !== undefined ? range[2] : 1;

      newOffset = (newOffset + start * this.strides[d]) | 0;
      const dimLen = Math.max(0, Math.ceil((stop - start) / step));
      newShape.push(dimLen);
      newStrides.push(this.strides[d] * step);
    }

    return new NDArray(this.data, {
      shape: newShape,
      strides: new Int32Array(newStrides),
      offset: newOffset,
      order: this.order,
    });
  }

  /**
   * Symbol.iterator: Recorre los elementos del array strided con zero allocation por valor.
   */
  *[Symbol.iterator](): IterableIterator<number> {
    const coords = new Int32Array(this.ndim);
    const total = this.size;

    for (let count = 0; count < total; count++) {
      let idx = this.offset;
      for (let d = 0; d < this.ndim; d++) {
        idx = (idx + coords[d] * this.strides[d]) | 0;
      }
      yield this.data[idx];

      for (let d = this.ndim - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < this.shape[d]) break;
        coords[d] = 0;
      }
    }
  }

  public toNestedArray(): NestedNumberArray {
    const build = (dimension: number, offset: number): NestedNumberArray => {
      if (dimension === this.ndim) return this.data[offset];

      const values: NestedNumberArray[] = new Array(this.shape[dimension]);
      for (let i = 0; i < values.length; i++) {
        values[i] = build(dimension + 1, offset + i * this.strides[dimension]);
      }
      return values;
    };

    return build(0, this.offset);
  }

  // --- Factory methods ---

  public static zeros(shape: number[] | Int32Array, order: 'C' | 'F' = 'C'): NDArray {
    let size = 1;
    for (let i = 0; i < shape.length; i++) size *= shape[i];
    return new NDArray(new Float64Array(size), { shape, order });
  }

  public static ones(shape: number[] | Int32Array, order: 'C' | 'F' = 'C'): NDArray {
    let size = 1;
    for (let i = 0; i < shape.length; i++) size *= shape[i];
    const data = new Float64Array(size);
    data.fill(1.0);
    return new NDArray(data, { shape, order });
  }

  public static fromArray(nested: any[], order: 'C' | 'F' = 'C'): NDArray {
    const shape: number[] = [];
    let cur: any = nested;
    while (Array.isArray(cur)) {
      shape.push(cur.length);
      cur = cur[0];
    }

    const flat: number[] = [];
    const flatten = (arr: any[]) => {
      for (const item of arr) {
        if (Array.isArray(item)) flatten(item);
        else flat.push(Number(item));
      }
    };
    flatten(nested);

    return new NDArray(new Float64Array(flat), { shape, order });
  }
}
