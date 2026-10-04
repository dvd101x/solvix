/**
 * Canonical NdArray structure for math-superset
 * Strict 0-based indexing, C-contiguous (row-major) layout with strides and offset.
 */
export class NdArray {
  public readonly data: Float64Array;
  public readonly shape: readonly number[];
  public readonly strides: readonly number[];
  public readonly offset: number;
  public readonly size: number;

  constructor(
    data: Float64Array | number[],
    shape: number[],
    strides?: number[],
    offset: number = 0
  ) {
    this.data = data instanceof Float64Array ? data : new Float64Array(data);
    this.shape = Object.freeze([...shape]);
    this.offset = offset;

    const totalElements = shape.reduce((acc, dim) => acc * dim, 1);
    this.size = totalElements;

    if (strides) {
      this.strides = Object.freeze([...strides]);
    } else {
      // Compute default row-major (C-contiguous) strides
      const computedStrides = new Array(shape.length);
      let stride = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        computedStrides[i] = stride;
        stride *= shape[i];
      }
      this.strides = Object.freeze(computedStrides);
    }
  }

  /**
   * Translates multi-dimensional coordinates to internal 1D flat buffer index (0-based)
   */
  public getFlatIndex(indices: number[]): number {
    if (indices.length !== this.shape.length) {
      throw new Error(`Dimension mismatch: expected ${this.shape.length} indices, got ${indices.length}`);
    }
    let idx = this.offset;
    for (let i = 0; i < indices.length; i++) {
      const coord = indices[i];
      if (coord < 0 || coord >= this.shape[i]) {
        throw new RangeError(`Index ${coord} out of bounds for axis ${i} with size ${this.shape[i]}`);
      }
      idx += coord * this.strides[i];
    }
    return idx;
  }

  public get(...indices: number[]): number {
    return this.data[this.getFlatIndex(indices)];
  }

  public set(value: number, ...indices: number[]): void {
    this.data[this.getFlatIndex(indices)] = value;
  }

  public isContiguous(): boolean {
    let expectedStride = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      if (this.strides[i] !== expectedStride) return false;
      expectedStride *= this.shape[i];
    }
    return true;
  }

  public toArray(): any {
    const unpack = (indices: number[], dim: number): any => {
      if (dim === this.shape.length - 1) {
        const arr = new Array(this.shape[dim]);
        for (let i = 0; i < this.shape[dim]; i++) {
          arr[i] = this.get(...indices, i);
        }
        return arr;
      }
      const arr = new Array(this.shape[dim]);
      for (let i = 0; i < this.shape[dim]; i++) {
        arr[i] = unpack([...indices, i], dim + 1);
      }
      return arr;
    };
    return unpack([], 0);
  }

  public clone(): NdArray {
    const copy = new Float64Array(this.data);
    return new NdArray(copy, [...this.shape], [...this.strides], this.offset);
  }

  public static zeros(shape: number[]): NdArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new NdArray(new Float64Array(size), shape);
  }

  public static ones(shape: number[]): NdArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    data.fill(1.0);
    return new NdArray(data, shape);
  }

  public static fromArray(nested: any[]): NdArray {
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

    return new NdArray(new Float64Array(flat), shape);
  }
}
