/**
 * @file broadcast.ts
 * Broadcasting shape rules shared by all element-wise operations.
 */

/**
 * Calcula el shape resultante de dos shapes según reglas de broadcasting.
 */
export function broadcastShapes(
  shapeA: ArrayLike<number>,
  shapeB: ArrayLike<number>
): Int32Array {
  const ndimA = shapeA.length;
  const ndimB = shapeB.length;
  const maxDim = Math.max(ndimA, ndimB);
  const resultShape = new Int32Array(maxDim);

  for (let i = 0; i < maxDim; i++) {
    const dimA = i < ndimA ? shapeA[ndimA - 1 - i] : 1;
    const dimB = i < ndimB ? shapeB[ndimB - 1 - i] : 1;

    if (dimA === dimB || dimA === 1 || dimB === 1) {
      resultShape[maxDim - 1 - i] = Math.max(dimA, dimB);
    } else {
      throw new Error(
        `Incompatible shapes for broadcasting: [${Array.from(shapeA)}] and [${Array.from(shapeB)}]`
      );
    }
  }
  return resultShape;
}
