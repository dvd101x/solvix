/**
 * @file ndarray.ts
 * Clase base primordial NDArray con acceso ultra rápido, slicing lazy y Symbol.iterator.
 */
import { Quantity, type Dimensions } from '../units/units.js';
import { Complex } from '../types/complex.js';
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
  /** Imaginary parts; must have the same length and layout as `data`. */
  imag?: TypedArray;
  /** Physical dimensions of every element; values are stored in SI base units. */
  unit?: Dimensions;
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
  /** Imaginary parts (same layout as `data`); undefined for real arrays. */
  public readonly imag?: TypedArray;
  /** Physical dimensions of the elements (values are stored in SI base units). */
  public readonly unit?: Dimensions;

  // Cache explícito de strides para 1D-6D para evitar overhead de indexación en hot-paths
  public readonly _s0: number;
  public readonly _s1: number;
  public readonly _s2: number;
  public readonly _s3: number;
  public readonly _s4: number;
  public readonly _s5: number;

  constructor(data: TypedArray, options: NDArrayOptions = {}) {
    this.data = data;
    if (options.imag && options.imag.length !== data.length) {
      throw new Error('imag must have the same length as data');
    }
    this.imag = options.imag;
    this.unit = options.unit && Object.values(options.unit).some((d) => d !== 0) ? options.unit : undefined;
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

  public get isComplex(): boolean {
    return this.imag !== undefined;
  }

  /** Linear buffer index of a multi-index. */
  public indexOf(...idx: number[]): number {
    return sub2indND(this.offset, this.strides, idx);
  }

  /** Complex element at the given index (imaginary part is 0 for real arrays). */
  public getComplex(...idx: number[]): Complex {
    const k = this.indexOf(...idx);
    return new Complex(this.data[k], this.imag ? this.imag[k] : 0);
  }

  /** Element at the given index as a Quantity (requires a real array with units). */
  public getQuantity(...idx: number[]): Quantity {
    if (this.isComplex) throw new TypeError('Complex elements with units cannot be represented as a Quantity');
    return new Quantity(this.data[this.indexOf(...idx)], this.unit ?? {});
  }

  /**
   * Attaches a unit: values are interpreted in `unit` (e.g. `arr.withUnit(kilometer)`)
   * and stored in SI base units. Temperature offsets (°C, °F) are applied to real values.
   */
  public withUnit(unit: Quantity): NDArray {
    const c = this.copy();
    const re = c.data as Float64Array;
    for (let i = 0; i < re.length; i++) re[i] = re[i] * unit.value + unit.offset;
    if (c.imag) {
      const im = c.imag as Float64Array;
      for (let i = 0; i < im.length; i++) im[i] *= unit.value;
    }
    const dims = new Quantity(1, this.unit ?? {}).mul(new Quantity(1, unit.dims)).dims;
    return new NDArray(re, { shape: Array.from(c.shape), imag: c.imag, unit: dims });
  }

  /** Values expressed in `target` units, as a plain array without unit information. */
  public to(target: Quantity): NDArray {
    if (!new Quantity(1, this.unit ?? {}).hasSameDimensions(target)) {
      throw new TypeError(
        `Cannot convert [${new Quantity(1, this.unit ?? {}).formatDimensions()}] to [${target.formatDimensions()}]: incompatible dimensions`
      );
    }
    const c = this.copy();
    const re = c.data as Float64Array;
    for (let i = 0; i < re.length; i++) re[i] = (re[i] - target.offset) / target.value;
    if (c.imag) {
      const im = c.imag as Float64Array;
      for (let i = 0; i < im.length; i++) im[i] /= target.value;
    }
    return new NDArray(re, { shape: Array.from(c.shape), imag: c.imag });
  }

  /** New array sharing the same buffers (and unit) with a different layout. */
  public view(opts: { shape: number[] | Int32Array; strides?: number[] | Int32Array; offset?: number; order?: 'C' | 'F' }): NDArray {
    return new NDArray(this.data, {
      shape: opts.shape,
      strides: opts.strides,
      offset: opts.offset ?? this.offset,
      order: opts.order ?? this.order,
      imag: this.imag,
      unit: this.unit,
    });
  }

  /** True when elements are laid out row-major from offset 0 with no gaps. */
  public get isContiguous(): boolean {
    if (this.offset !== 0 || this.data.length !== this.size) return false;
    let expected = 1;
    for (let d = this.ndim - 1; d >= 0; d--) {
      if (this.shape[d] !== 1 && this.strides[d] !== expected) return false;
      expected *= this.shape[d];
    }
    return true;
  }

  /** This array when already contiguous, otherwise a contiguous row-major copy (never mutate the result). */
  public contiguous(): NDArray {
    return this.isContiguous ? this : this.copy();
  }

  /** Contiguous row-major copy that preserves the imaginary part and the unit. */
  public copy(): NDArray {
    const out = new Float64Array(this.size);
    const outIm = this.imag ? new Float64Array(this.size) : undefined;
    const coords = new Int32Array(this.ndim);
    for (let n = 0; n < this.size; n++) {
      let idx = this.offset;
      for (let d = 0; d < this.ndim; d++) idx += coords[d] * this.strides[d];
      out[n] = this.data[idx];
      if (outIm) outIm[n] = this.imag![idx];
      for (let d = this.ndim - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < this.shape[d]) break;
        coords[d] = 0;
      }
    }
    return new NDArray(out, { shape: Array.from(this.shape), imag: outIm, unit: this.unit });
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

    return this.view({ shape: newShape, strides: new Int32Array(newStrides), offset: newOffset });
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
      if (dimension === this.ndim) {
        return (this.imag ? new Complex(this.data[offset], this.imag[offset]) : this.data[offset]) as NestedNumberArray;
      }

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
    const flatIm: number[] = [];
    let complex = false;
    const flatten = (arr: any[]) => {
      for (const item of arr) {
        if (Array.isArray(item)) flatten(item);
        else if (item instanceof Complex) {
          complex = true;
          flat.push(item.re);
          flatIm.push(item.im);
        } else {
          flat.push(Number(item));
          flatIm.push(0);
        }
      }
    };
    flatten(nested);

    return new NDArray(new Float64Array(flat), {
      shape,
      order,
      imag: complex ? new Float64Array(flatIm) : undefined,
    });
  }

  /** Complex array from separate real and imaginary parts (row-major, same shape). */
  public static fromComplex(re: ArrayLike<number>, im: ArrayLike<number>, shape?: number[]): NDArray {
    if (re.length !== im.length) throw new Error('re and im must have the same length');
    return new NDArray(Float64Array.from(re), {
      shape: shape ?? [re.length],
      imag: Float64Array.from(im),
    });
  }
}

/** Real values of an array (any strides/offset) or plain sequence as a contiguous buffer. */
export function toFloat64(x: NDArray | ArrayLike<number>): ArrayLike<number> {
  return x instanceof NDArray ? x.contiguous().data : Float64Array.from(x);
}
