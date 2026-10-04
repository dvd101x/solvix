import { bench, run } from 'mitata';
import { NDArray } from '../src/ix/core/ndarray.js';
import { sub2ind2D, sub2indND } from '../src/ix/core/strides.js';
import { add } from '../src/ix/ops/math-ops.js';

// Setup de datos
const N = 1000;
const A = new NDArray(new Float64Array(N * N), { shape: [N, N] });
const B = new NDArray(new Float64Array(N * N), { shape: [N, N] });
const s0 = N;
const s1 = 1;

bench('sub2ind2D (inlined unrolled)', () => {
  return sub2ind2D(0, s0, s1, 500, 500);
});

bench('sub2indND (dynamic fallback)', () => {
  return sub2indND(0, [s0, s1], [500, 500]);
});

bench('NDArray.get2D', () => {
  return A.get2D(500, 500);
});

bench('NDArray.get (switch unrolled)', () => {
  return A.get(500, 500);
});

bench('Multiple Dispatch Cached (add NDArray + scalar)', () => {
  return add(A, 1.0);
});

await run();
