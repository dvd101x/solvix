import { bench, run } from 'mitata';
import { NDArray } from '../src/ix/core/ndarray.js';
import { sub2ind2D, sub2indND } from '../src/ix/core/strides.js';
import { add } from '../src/ix/ops/math-ops.js';

// Benchmark fixtures
const N = 1000;
const A = new NDArray(new Float64Array(N * N), { shape: [N, N] });
const s0 = N;
const s1 = 1;
let checksum = 0;

bench('sub2ind2D (inlined unrolled)', () => {
  checksum += sub2ind2D(0, s0, s1, 500, 500);
});

bench('sub2indND (dynamic fallback)', () => {
  checksum += sub2indND(0, [s0, s1], [500, 500]);
});

bench('NDArray.get2D', () => {
  checksum += A.get2D(500, 500);
});

bench('NDArray.get (switch unrolled)', () => {
  checksum += A.get(500, 500);
});

bench('Multiple Dispatch Cached (add NDArray + scalar)', () => {
  checksum += add(A, 1.0).data[0];
});

await run();
console.log('Benchmark checksum:', checksum);
