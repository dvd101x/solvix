import * as tf from '@tensorflow/tfjs';
import { NdArray } from '../core/ndarray.js';

/**
 * Adapter between internal NdArray and TensorFlow.js tf.Tensor
 * Manages memory and data synchronization.
 */
export class TfjsAdapter {
  /**
   * Converts an NdArray to a tf.Tensor
   */
  public static toTensor(arr: NdArray, dtype: tf.NumericDataType = 'float32'): tf.Tensor {
    // If not contiguous, create contiguous copy
    let buffer: Float64Array;
    if (arr.isContiguous() && arr.offset === 0) {
      buffer = arr.data;
    } else {
      buffer = new Float64Array(arr.size);
      const unpack = (indices: number[], dim: number, writeIdx: { val: number }) => {
        if (dim === arr.shape.length - 1) {
          for (let i = 0; i < arr.shape[dim]; i++) {
            buffer[writeIdx.val++] = arr.get(...indices, i);
          }
          return;
        }
        for (let i = 0; i < arr.shape[dim]; i++) {
          unpack([...indices, i], dim + 1, writeIdx);
        }
      };
      unpack([], 0, { val: 0 });
    }

    return tf.tensor(buffer, [...arr.shape], dtype);
  }

  /**
   * Converts a tf.Tensor to an NdArray and disposes the tensor if requested
   */
  public static fromTensor(tensor: tf.Tensor, disposeInput: boolean = false): NdArray {
    const rawData = tensor.dataSync(); // TypedArray
    const shape = [...tensor.shape];
    const float64Data = new Float64Array(rawData);

    if (disposeInput) {
      tensor.dispose();
    }

    return new NdArray(float64Data, shape);
  }

  /**
   * Scoped execution helper for TensorFlow operations to avoid GPU memory leaks
   */
  public static tidy<T>(fn: () => T): T {
    return tf.tidy(fn);
  }
}
