import { Matrix } from 'ml-matrix';
import { NdArray } from '../core/ndarray.js';

/**
 * Adapter between NdArray and ml-matrix Matrix
 * Supports high-performance linear algebra (SVD, Cholesky, Eigenvalues, Inversion)
 */
export class MlMatrixAdapter {
  /**
   * Converts a 2D NdArray to ml-matrix Matrix
   */
  public static toMatrix(arr: NdArray): Matrix {
    if (arr.shape.length !== 2) {
      throw new Error(`ml-matrix requires 2D arrays, got shape [${arr.shape.join(', ')}]`);
    }

    const [rows, cols] = arr.shape;
    const matrix = new Matrix(rows, cols);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        matrix.set(r, c, arr.get(r, c));
      }
    }

    return matrix;
  }

  /**
   * Converts an ml-matrix Matrix to NdArray
   */
  public static fromMatrix(matrix: Matrix): NdArray {
    const rows = matrix.rows;
    const cols = matrix.columns;
    const data = new Float64Array(rows * cols);

    let idx = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        data[idx++] = matrix.get(r, c);
      }
    }

    return new NdArray(data, [rows, cols]);
  }
}
