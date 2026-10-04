import { all, create } from 'mathjs';
import { SingularValueDecomposition, inverse, determinant } from 'ml-matrix';
import { defmethod, defmulti, Dispatcher } from '../core/dispatch.js';
import { NdArray } from '../core/ndarray.js';
import { MlMatrixAdapter } from '../adapters/mljs.js';
import { TfjsAdapter } from '../adapters/tfjs.js';
import { StdlibAdapter } from '../adapters/stdlib.js';

/**
 * Creates and configures the math-superset execution environment.
 * Extends MathJS with Julia-style multiple dispatch and scientific adapters.
 */
export function createMathSupersetEnvironment() {
  const math = create(all);

  // 1. Dispatcher: inv (matrix inversion)
  const invMulti = defmulti('inv');

  defmethod(invMulti, ['Number'], (x: number) => 1 / x);

  defmethod(invMulti, ['MlMatrix'], (m: any) => {
    return inverse(m);
  });

  defmethod(invMulti, ['NdArray'], (arr: NdArray) => {
    const mat = MlMatrixAdapter.toMatrix(arr);
    const invMat = inverse(mat);
    return MlMatrixAdapter.fromMatrix(invMat);
  });

  defmethod(invMulti, ['MathJsMatrix'], (m: any) => {
    const arr = NdArray.fromArray(m.toArray());
    const invArr = invMulti(arr);
    return math.matrix(invArr.toArray());
  });

  // 2. Dispatcher: svd (singular value decomposition)
  const svdMulti = defmulti('svd');

  defmethod(svdMulti, ['MlMatrix'], (m: any) => {
    const svd = new SingularValueDecomposition(m);
    return {
      U: svd.leftSingularVectors,
      s: svd.diagonal,
      V: svd.rightSingularVectors,
    };
  });

  defmethod(svdMulti, ['NdArray'], (arr: NdArray) => {
    const mat = MlMatrixAdapter.toMatrix(arr);
    const svd = new SingularValueDecomposition(mat);
    return {
      U: MlMatrixAdapter.fromMatrix(svd.leftSingularVectors),
      s: new NdArray(new Float64Array(svd.diagonal), [svd.diagonal.length]),
      V: MlMatrixAdapter.fromMatrix(svd.rightSingularVectors),
    };
  });

  // 3. Dispatcher: det (determinant)
  const detMulti = defmulti('det');

  defmethod(detMulti, ['Number'], (x: number) => x);

  defmethod(detMulti, ['MlMatrix'], (m: any) => determinant(m));

  defmethod(detMulti, ['NdArray'], (arr: NdArray) => {
    const mat = MlMatrixAdapter.toMatrix(arr);
    return determinant(mat);
  });

  // 4. Dispatcher: besselj (Bessel function from @stdlib)
  const besseljMulti = defmulti('besselj');

  defmethod(besseljMulti, ['Number', 'Number'], (v: number, x: number) => {
    return StdlibAdapter.besselj(v, x);
  });

  defmethod(besseljMulti, ['Number', 'NdArray'], (v: number, x: NdArray) => {
    return StdlibAdapter.besselj(v, x);
  });

  // 5. Dispatcher: matmul / multiply
  const matmulMulti = defmulti('matmul');

  defmethod(matmulMulti, ['Number', 'Number'], (a: number, b: number) => a * b);

  defmethod(matmulMulti, ['TfTensor', 'TfTensor'], (a: any, b: any) => {
    return TfjsAdapter.tidy(() => a.matMul(b));
  });

  defmethod(matmulMulti, ['NdArray', 'NdArray'], (a: NdArray, b: NdArray) => {
    // Delegate to tfjs for hardware-accelerated / optimized matmul
    return TfjsAdapter.tidy(() => {
      const tA = TfjsAdapter.toTensor(a);
      const tB = TfjsAdapter.toTensor(b);
      const res = tA.matMul(tB);
      return TfjsAdapter.fromTensor(res);
    });
  });

  // Register multi-dispatch functions into mathjs scope
  math.import({
    inv: invMulti,
    svd: svdMulti,
    det: detMulti,
    besselj: besseljMulti,
    matmul: matmulMulti,
    ndarray: (data: any[], shape?: number[]) => {
      if (shape) {
        return new NdArray(data, shape);
      }
      return NdArray.fromArray(data);
    },
  }, { override: true });

  return {
    math,
    dispatchers: {
      inv: invMulti,
      svd: svdMulti,
      det: detMulti,
      besselj: besseljMulti,
      matmul: matmulMulti,
    } as Record<string, Dispatcher>,
    evaluate: (expr: string, scope: Record<string, any> = {}) => {
      return math.evaluate(expr, scope);
    },
  };
}
