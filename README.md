# Solvix — High-Performance Scientific Computing for Modern JavaScript / TypeScript

> **Solvix** is a lightweight scientific computing library for modern JavaScript and TypeScript.

[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)

---

## Installation

```bash
npm install solvix
```

```typescript
import { NDArray, add } from 'solvix';

const values = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });
const result = add(values, 2);
```

To work from a local checkout, run `npm install`, then `npm run build` or `npm run example`. The public package entry point is `solvix`; its source lives in `src/` and it has no runtime dependencies.

## ⚡ Design Principles

1. **API priorities**: clear, predictable numerical operations.
2. **Zero allocation**: Loop-unrolled indexing algorithms for dimensions $1\text{D}$ through $6\text{D}$ minimize garbage-collector pressure.
3. **Monomorphism and inlining in V8**: Direct access to flat (`TypedArray`) buffers helps avoid deoptimizations such as megamorphic call sites.
4. **Dynamic multiple dispatch**: signature-based polymorphism with caching for fast hot paths.
5. **Lazy, zero-copy slicing**: Strided views share the same `ArrayBuffer`.
6. **Worker parallelism**: `SharedArrayBuffer` access, zero-copy tensor transfer, and remotely evaluable mathematical scopes.
7. **Computation graphs (DAGs)**: Expressions support topological dependency resolution, memoization, and cycle detection.

---

## 📁 Repository Layout

```
solvix/
├── src/                  # Library source; src/index.ts is the public entry point
│   ├── core/             # NDArray (strided, real/complex/units), NestedArray, strides, dispatcher
│   ├── ops/              # Broadcasting arithmetic, Julia-style operators (*, \, ^, ')
│   ├── linalg/           # LU/QR/SVD, solve/inv/det, matmul, norms, cholesky, eigen, pinv, lstsq
│   ├── parser/           # Julia-like expression parser (AST, ranges, literals, indexing)
│   ├── indexing/         # Masks, fancy indexing, ellipsis, and 1-based Julia indexing
│   ├── manipulation/     # reshape, transpose, stack, ...
│   ├── stats/ signal/ integrate/ interpolate/ optimize/ ode/   # Numerical modules
│   ├── types/ units/ constants/                                 # Complex, Fraction, Quantity
│   ├── dataframe/ plot/ latex/ agent/ ai/                       # Data and tooling helpers
│   └── parallel/ dag/ memory/                                   # Workers, expression graphs, in-place ops
├── tests/                # Vitest suites (`npm test`)
├── examples/             # `npm run example`
└── benchmarks/           # `npm run bench`
```

---

## 🛠️ Features

### 1. Advanced Indexing

#### A. Boolean Masks (`booleanMask` and `putMask`)
```typescript
import { NDArray, booleanMask, putMask } from 'solvix';

const A = new NDArray(new Float64Array([1, -2, 3, -4, 5]), { shape: [5] });
const mask = [true, false, true, false, true];

// Filter values (A[mask]) -> [1, 3, 5]
const pos = booleanMask(A, mask);

// Assign in place using a mask
putMask(A, mask, 0); // [0, -2, 0, -4, 0]
```

#### B. Axis-Based Fancy Indexing (`take`)
```typescript
import { NDArray, take } from 'solvix';

const matrix = new NDArray(new Float64Array([10, 11, 20, 21, 30, 31]), { shape: [3, 2] });
// Select rows 2 and 0
const sub = take(matrix, [2, 0], 0);
```

#### C. Ellipsis Slicing (`...`)
```typescript
import { NDArray, sliceWithEllipsis, ELLIPSIS } from 'solvix';

const tensor = new NDArray(data, { shape: [4, 5, 6, 2] });
// tensor[..., 1] -> select coordinate 1 on the last axis, preserving intermediate dimensions
const slice = sliceWithEllipsis(tensor, ELLIPSIS, 1);
```

#### D. Conditional Selection and Mask Reductions
```typescript
import { NDArray, where, all, any, countNonzero } from 'solvix';

const mask = NDArray.fromArray([1, 0, 1]);
const values = NDArray.fromArray([10, 20, 30]);
const selected = where(mask, values, 0); // [10, 0, 30]; all arguments broadcast

all(mask);          // false
any(mask);          // true
countNonzero(mask); // 2

const matrixMask = NDArray.fromArray([[1, 0, 1], [0, 1, 0]]);
countNonzero(matrixMask, { axis: 0 }); // [1, 1, 1]
any(matrixMask, { axis: 1 });         // [1, 1]
```

---

### 2. Parallelism and Concurrency with Workers

#### A. Shared Memory (`SharedArrayBuffer`)
Multiple worker threads can read and write the same memory without serialization or copies:
```typescript
import { createSharedNDArray, partitionWork } from 'solvix';

// Tensor shared across workers
const sharedTensor = createSharedNDArray([1000, 1000]);

// Partition work across four workers
const chunks = partitionWork(1_000_000, 4);
// chunks = [{start: 0, end: 250000}, {start: 250000, end: 500000}, ...]
```

#### B. Zero-Copy Transfer of Standard Arrays
```typescript
import { prepareTransfer, reconstructFromTransfer } from 'solvix';

const A = NDArray.zeros([500, 500]);
const { message, transferables } = prepareTransfer(A);

// Transfer ownership of the ArrayBuffer to the worker (zero-copy)
worker.postMessage(message, transferables);
```

#### C. Remote Mathematical Expressions and Scopes
Send a mathematical expression string together with its variable context:
```typescript
import { serializeScope, evaluateExpressionInScope } from 'solvix';

// On the main thread:
const scope = serializeScope({
  mass: 10.5,
  velocity: 20.0
});

// In the worker:
const energy = evaluateExpressionInScope('0.5 * mass * pow(velocity, 2)', scope);
```

---

### 3. Mathematical Expression DAGs

Compose formulas and tensors in a directed acyclic graph with:
- **Topological sorting (Kahn's algorithm)**.
- **Automatic cycle detection**.
- **Memoized lazy evaluation**.
- **Automatic Mermaid visualization**.

```typescript
import { ExpressionDAG, NDArray, add } from 'solvix';

const dag = new ExpressionDAG();

// Define variables
dag.variable('A', new NDArray(new Float64Array([1, 2]), { shape: [2] }));
dag.variable('B', new NDArray(new Float64Array([10, 20]), { shape: [2] }));

// Define dependent operations
dag.op('C', ['A', 'B'], (a, b) => add(a, b));
dag.op('D', ['C'], (c) => add(c, 100));

// Evaluate on demand
const result = dag.evaluate('D'); // [111, 122]

// Only affected branches are invalidated when a variable changes
dag.setVariable('A', new NDArray(new Float64Array([0, 0]), { shape: [2] }));
const updated = dag.evaluate('D'); // [110, 120]

// Export a Mermaid diagram
console.log(dag.toMermaid());
```

---

### 4. Numerical Linear Algebra (Factorizations and Solvers)

Pure TypeScript implementation operating on `Float64Array`:

```typescript
import { NDArray, solve, lu, qr, svd, inv, det } from 'solvix';

const A = new NDArray(new Float64Array([4, 3, 6, 3]), { shape: [2, 2] });
const b = new Float64Array([10, 12]);

// 1. Solve the linear system Ax = b
const x = solve(A, b); // [1.0, 2.0]

// 2. Determinant and inverse
console.log(det(A));   // -6.0
const invA = inv(A);   // Matrix inverse

// 3. LU decomposition with partial pivoting (PA = LU)
const { L, U, P } = lu(A);

// 4. QR decomposition (Householder reflections)
const { Q, R } = qr(A);

// 5. Singular value decomposition (SVD: A = U * S * V^T)
const { U: uVec, S: sVals, V: vVec } = svd(A);
```

---

### 5. Native Mathematical Parser and AST Compiler

The syntax follows Julia: `A*b` is the matrix product, `A.*b` is element-wise, `A\\b` solves a system, `A'` is the conjugate transpose, `f.(x)` broadcasts, and `im` is the imaginary unit. Array literals (`[1 2; 3 4]`, `[1, 2, 3]`), ranges (`1:2:9`) and **1-based** indexing (`A[2, :]`, `v[end]`) are supported. `f.(x)` and `map(f, x)` call `f` with the value only; use `mapIndexed(f, x)` for `(value, index, array)`.

Tokenizer and parser based on **Shunting-Yard (Dijkstra)** with a native abstract syntax tree (**AST**) and no external dependencies:

```typescript
import { evaluate, compile, parseExpression } from 'solvix';

// Evaluate immediately using standard precedence and associativity
const val = evaluate('3 + 4 * 2 / (1 - 5)^2'); // 3.5

// Evaluate using variables and functions in a scope
const res = evaluate('sin(pi / 2) + sqrt(x^2 + y^2)', { x: 3, y: 4 }); // 6.0
const magnitude = evaluate('hypot(3, 4) + log10(100)'); // 7

// Compile a reusable high-performance function
const kineticEnergy = compile('0.5 * m * v^2');
console.log(kineticEnergy({ m: 10, v: 20 })); // 2000
const bounded = compile('clamp(x, 0, 1)');
console.log(bounded({ x: 2 })); // 1
```

---

### 6. Ordinary Differential Equation Solvers (ODE45 and RK4)

Solve initial-value problems of the form $\frac{dy}{dt} = f(t, y)$:
- **RK4**: Classical fourth-order Runge-Kutta with a fixed step size.
- **ODE45**: Embedded Dormand-Prince 5(4) with adaptive step-size control.

```typescript
import { NDArray, ode45 } from 'solvix';

// Simple harmonic oscillator: y'' + y = 0  =>  y1' = y2, y2' = -y1
const f = (t: number, y: NDArray) => {
  return new NDArray(new Float64Array([y.data[1], -y.data[0]]), { shape: [2] });
};

const y0 = new NDArray(new Float64Array([0.0, 1.0]), { shape: [2] });
const sol = ode45(f, [0, Math.PI / 2], y0, { rtol: 1e-5 });

console.log(sol.y[sol.y.length - 1].data); // [1.0, 0.0] -> [sin(pi/2), cos(pi/2)]
```

---

### 7. Root Finding and Nonlinear Systems (`fzero` and `fsolve`)

- **`fzero`**: Hybrid Brent algorithm for scalar roots $f(x) = 0$ with superlinear convergence.
- **`fsolve`**: Damped multidimensional Newton-Raphson with a numerical Jacobian for coupled systems $F(x) = 0$.

```typescript
import { fzero, fsolve, NDArray } from 'solvix';

// Scalar root: x^2 - 2 = 0
const { root } = fzero((x) => x * x - 2, [1, 2]);
console.log(root); // 1.41421356 (sqrt(2))

// Nonlinear system: intersection of a circle and a diagonal
const F = (x: NDArray) => new NDArray(new Float64Array([
  x.data[0]**2 + x.data[1]**2 - 1,
  x.data[0] - x.data[1]
]), { shape: [2] });

const { x } = fsolve(F, [0.5, 0.5]);
console.log(x.data); // [0.7071, 0.7071]
```

---

### 8. Physical Units and Dimensional Analysis (SI)

Enforce dimensional consistency at runtime:

```typescript
import { qty, meter, kilometer, second, hour, kilogram, newton, joule } from 'solvix';

const distance = qty(5, meter);
const time = qty(2, second);

// 1. Detect dimensional mismatches
// distance.add(time); // Throws TypeError: Dimensional mismatch [m^1] vs [s^1]

// 2. Compose derived quantities
const velocity = distance.div(time); // 2.5 m/s

// 3. Convert between units
const speedKmh = qty(72, kilometer).div(qty(1, hour));
console.log(speedKmh.to(meter.div(second))); // 20.0 m/s

// 4. Dynamics and energy
const mass = qty(10, kilogram);
const acceleration = qty(9.8, meter).div(qty(1, second).pow(2));
const force = mass.mul(acceleration); // 98 N (kg*m/s^2)
const work = force.mul(distance); // 490 J (kg*m^2/s^2)
```

---

### 9. In-Place Operations (`!`) and Reusable Memory

Avoid garbage-collection pauses in high-frequency loops:

```typescript
import { NDArray, addInPlace, mulInPlace } from 'solvix';

const A = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });
const B = new NDArray(new Float64Array([10, 20, 30]), { shape: [3] });
const out = new NDArray(new Float64Array(3), { shape: [3] });

// Write directly to the 'out' buffer without allocating a new array
addInPlace(out, A, B); // out = [11, 22, 33]
mulInPlace(out, out, 2); // out = [22, 44, 66]
```

---

### 10. Numeric Generators and Grids (`linspace`, `arange`, `meshgrid`, `eye`)

```typescript
import { linspace, arange, logspace, eye, meshgrid } from 'solvix';

// Linear spacing and strided ranges
const lin = linspace(0, 10, 5); // [0, 2.5, 5, 7.5, 10]
const rng = arange(1, 10, 2);   // [1, 3, 5, 7, 9]

// 2D coordinate grid (meshgrid) for evaluating surfaces
const x = linspace(-1, 1, 100);
const y = linspace(-1, 1, 100);
const [X, Y] = meshgrid(x, y);

// Identity matrix
const I = eye(3);
```

---

### 11. $\mathcal{O}(1)$ Views and Tensor Manipulation

Change shapes and swap axes by modifying strides without copying data buffers:

```typescript
import { NDArray, reshape, transpose, expandDims, squeeze, clip } from 'solvix';

const A = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });

// 1. Reshape with inferred dimension (-1) in O(1)
const B = reshape(A, [3, -1]); // shape: [3, 2]

// 2. Transpose in O(1) by swapping strides
const At = transpose(A); // shape: [3, 2]
const nestedRows = At.toNestedArray(); // [[1, 4], [2, 5], [3, 6]]

// 3. Add and remove singleton axes
const expanded = expandDims(A, 1); // shape: [2, 1, 3]
const squeezed = squeeze(expanded); // shape: [2, 3]

// 4. Clip values to a range
const limited = clip(A, 2, 5); // [2, 2, 3, 4, 5, 5]; bounds may also be arrays (broadcast)
```

---

### 12. Axis Reductions and Descriptive Statistics

Global and axis-based (`axis`) reductions support degrees-of-freedom correction (`ddof`):

```typescript
import {
  NDArray, sum, mean, std, variance, median, quantile, percentile, sumProduct, describe,
  cumsum, cumprod, cummin, cummax
} from 'solvix';

// 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
const M = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });

// 1. Reductions across axes
const colSum = sum(M, { axis: 0 }); // [5, 7, 9] (sum across rows)
const rowMean = mean(M, { axis: 1 }); // [2, 5] (mean per row)
const devStd = std(M, { axis: 0, ddof: 1 }); // Sample standard deviation
const partialSums = cumsum(M, { axis: 1 }); // [[1, 3, 6], [4, 9, 15]]
const partialProducts = cumprod(M, { axis: 1 }); // [[1, 2, 6], [4, 20, 120]]
const runningMin = cummin(M, { axis: 0 }); // [[1, 2, 3], [1, 2, 3]]
const runningMax = cummax(M, { axis: 0 }); // [[1, 2, 3], [4, 5, 6]]
const dot = sumProduct(
  new NDArray(new Float64Array([1, 2, 3]), { shape: [3] }),
  new NDArray(new Float64Array([4, 5, 6]), { shape: [3] })
); // 32
const weightedColumns = sumProduct(
  new NDArray(new Float64Array([2, 3]), { shape: [2, 1] }),
  new NDArray(new Float64Array([10, 20]), { shape: [1, 2] }),
  { axis: 0 }
); // [50, 100]; elementwise multiplication broadcasts first

// 2. Measures of position
const med = median(M); // 3.5
const q75 = quantile(M, 0.75); // 75th percentile
const p25 = percentile(M, 25); // 25th percentile

// With no axis, cumulative scans flatten the array in iteration order.
const allPartialSums = cumsum(M); // [1, 3, 6, 10, 15, 21]

// 3. Full descriptive summary
const stats = describe(M);
const columnStats = describe(M, { axis: 0 }); // each statistic is an NDArray with one value per column
console.log(stats);
// {
//   count: 6,
//   mean: 3.5,
//   std: 1.8708,
//   min: 1,
//   p25: 2.25,
//   median: 3.5,
//   p75: 4.75,
//   max: 6,
//   skew: 0,      // Skewness
//   kurtosis: -1.2 // Excess kurtosis
// }
```

---

### 13. Time Series, Rolling Windows, and NaNs (`rolling`, `ewm`, `cov`, `corr`)

```typescript
import {
  NDArray,
  rolling,
  ewm,
  diff,
  pctChange,
  shift,
  fillna,
  dropna,
  nanmean,
  nansum,
  nanstd,
  cov,
  corr
} from 'solvix';

const s = new NDArray(new Float64Array([10, 20, 30, 40, 50]), { shape: [5] });

// 1. Rolling windows (SMA, volatility, etc.)
const sma3 = rolling(s, 3).mean(); // [NaN, NaN, 20, 30, 40]
const stdRoll = rolling(s, 3).std(); // Rolling standard deviation

// 2. Exponentially weighted moving average (EMA)
const ema = ewm(s, { alpha: 0.5 });

// 3. Differencing and rates of return
const d = diff(s);       // [NaN, 10, 10, 10, 10]
const ret = pctChange(s); // Retornos porcentuales
const lag = shift(s, 1);  // Desplazamiento temporal

// NaN-ignoring reductions also support axis and keepdims options.
const withMissing = NDArray.fromArray([[1, NaN, 3], [NaN, 5, NaN]]);
const columnMeans = nanmean(withMissing, { axis: 0 }); // [1, 5, 3]
const columnSums = nansum(withMissing, { axis: 0 });   // [1, 5, 3]
const columnStd = nanstd(withMissing, { axis: 0 });    // [0, 0, 0]

// 4. Missing-data cleanup
const conNaN = new NDArray(new Float64Array([1, NaN, 3]), { shape: [3] });
const filled = fillna(conNaN, 'ffill'); // [1, 1, 3] (forward fill)
const clean = dropna(conNaN);            // [1, 3]

// 5. Multivariate covariance and correlation matrices
const data = new NDArray(new Float64Array([1, 2, 2, 4, 3, 6, 4, 8]), { shape: [4, 2] });
const covariance = cov(data);
const correlation = corr(data); // Pearson correlation matrix [-1, 1]
```

---

### 14. Lightweight `DataFrame` with Named Indices

Zero-copy tabular structure backed by a contiguous 2D `NDArray`:

```typescript
import { DataFrame } from 'solvix';

const df = new DataFrame(
  [
    [20.5, 1013.2],
    [21.0, 1012.8],
    [22.4, 1011.5],
  ],
  {
    columns: ['temperature', 'pressure'],
    index: ['t0', 't1', 't2']
  }
);

// 1. Access columns by name as NDArrays
const temp = df.col('temperature'); // NDArray [20.5, 21.0, 22.4]

// 2. Select a subset
const subDf = df.select('pressure');

// 3. Correlation matrix between columns
const dfCorr = df.corr();

// 4. Export rows as JSON records
const jsonRows = df.toRecords();

// 5. Cell-wise map and computed columns (built on broadcastMap)
const scaled = df.map((v) => v / 100);
const withK = df.withColumn('kelvin', (t) => t + 273.15, 'temperature'); // adds or replaces a column
```

---

### 15. Complex Numbers and Exact Fractions

#### A. Complex Numbers ($z = a + bi$)
Support for imaginary numbers, Euler's formula, and complex roots:
```typescript
import { complex, add } from 'solvix';

const z1 = complex(1, 2); // 1 + 2i
const z2 = complex(3, 4); // 3 + 4i

// Complex arithmetic
const sum = add(z1, z2);  // 4 + 6i
const prod = z1.mul(z2);  // -5 + 10i

// Square root of a negative real: sqrt(-4) = 2i
const rootNeg = complex(-4, 0).sqrt(); // 0 + 2i

// Euler's formula: e^(i * pi) = -1
const euler = complex(0, Math.PI).exp(); // -1 + 0i
```

#### B. Exact Rational Fractions (`BigInt`)
Avoid IEEE-754 floating-point rounding errors ($1/3 + 1/6 = 1/2$ exactly):
```typescript
import { frac, add } from 'solvix';

const f1 = frac(1, 3);
const f2 = frac(1, 6);

const total = add(f1, f2); // Fraction 1/2
console.log(total.toString()); // '1/2'
console.log(total.toNumber()); // 0.5

// Convert a decimal to an exact fraction
const fromDec = Fraction.fromNumber(0.125); // 1/8
```

---

### 16. Digital Signal Processing (DSP and FFT)

Implemented in TypeScript without WebAssembly binaries:

```typescript
import { sinewave, fft, convolve, lowpassFilter } from 'solvix';

// 1. Generate a 50 Hz sine wave sampled at 500 Hz
const signal = sinewave(50, 1.0, 500, 2.0);

// 2. One-dimensional Cooley-Tukey FFT
const { frequencies, magnitude } = fft(signal, 500);

// 3. IIR low-pass filtering
const filtered = lowpassFilter(signal, 30, 500);
```

---

### 17. Numerical Quadrature

```typescript
import { trapz, simpson, quad, cumulativeIntegrate } from 'solvix';

// Adaptive continuous quadrature (Gauss-Kronrod / adaptive Simpson)
const area = quad((x) => Math.sin(x), 0, Math.PI); // 2.0

// Integrate discrete sampled data
const t = trapz([0, 1, 4, 9], [0, 1, 2, 3]);
const s = simpson([0, 1, 4, 9], [0, 1, 2, 3]);

// Array limits are broadcast: one integral per (a, b) pair, same shape as the limits
const areas = quad((x) => x * x, [0, 1, 2], 3); // NDArray-like [9, 8.67, 6.33]
```

---

### 18. One-Dimensional Interpolation and Cubic Splines

```typescript
import { interp1d, cubicSpline } from 'solvix';

// Linear interpolation with O(log N) binary search
const fLin = interp1d([0, 10], [0, 100], { method: 'linear' });
console.log(fLin(5)); // 50

// C2-continuous natural cubic spline
const fSpline = cubicSpline([0, 1, 2, 3], [0, 1, 8, 27]);
console.log(fSpline(1.5)); // 3.15 (natural boundary conditions)
```

---

### 19. Numerical Optimization and Curve Fitting

```typescript
import { fminbnd, nelderMead, curveFit, NDArray } from 'solvix';

// 1. Scalar minimum (golden-section search)
const min1D = fminbnd((x) => (x - 3)**2 + 5, [0, 10]); // x = 3, fval = 5

// 2. Derivative-free multivariable minimum (Nelder-Mead simplex)
const minND = nelderMead((v) => (v.get(0) - 2)**2 + (v.get(1) + 4)**2, [0, 0]); // [2, -4]

// 3. Curve fitting (model calibration)
const model = (x: number, p: NDArray) => p.get(0) * x + p.get(1);
const fit = curveFit(model, [1, 2, 3], [3, 5, 7], [1, 0]); // p = [2, 1]
```

---

### 20. Standalone SVG Plot

```typescript
import { plotSVG } from 'solvix';

const x = [0, 1, 2, 3];
const y = [0, 1, 4, 9];

// Standalone SVG renderer with no dependencies
const svgString = plotSVG(x, y, { color: '#38bdf8' });
```

---

### 21. AI-Ready Helpers, Prompting, and Offset Units

Tools for interactive use in chats and editors:

```typescript
import {
  help,
  summary,
  fromCSV,
  fromMatrixString,
  unitsHelp,
  units,
  qty,
  celsius,
  fahrenheit,
  kelvin,
  meter,
  SPEED_OF_LIGHT,
  STANDARD_GRAVITY
} from 'solvix';

// 1. Immediate agent assistance and introspection
console.log(help('linspace'));
console.log(unitsHelp());

// 2. Compact summary for LLM contexts
const A = new NDArray(new Float64Array([1, 2, 3, 4, 5]), { shape: [5] });
console.log(summary(A)); // NDArray shape=[5], min=1.0000, max=5.0000, mean=3.0000

// 3. Quickly load common formats
const df = fromCSV("x,y\n1,10\n2,20");
const mat = fromMatrixString("1 2 3; 4 5 6");

// 4. Units with relative temperature offsets
const tC = qty(100, celsius);
console.log(tC.to(fahrenheit)); // 212 °F
console.log(tC.to(kelvin));     // 373.15 K

// 5. Custom units (with removal and reset)
units.defineUnit('parsec', 'pc', qty(3.0857e16, meter), 'Astronomical parsec');
units.removeUnit('pc'); // Remove the added unit
units.reset();          // Restore the canonical unit catalog

// 6. Fundamental physical constants (CODATA)
console.log(SPEED_OF_LIGHT.value);   // 299792458 m/s
console.log(STANDARD_GRAVITY.value); // 9.80665 m/s^2
```

---

### 22. $\LaTeX$ Input (Wikipedia and Papers) and Implicit Multiplication

Paste formulas copied from Wikipedia or papers directly into the parser:

```typescript
import { fromLaTeX, toLaTeX, quickCalc, evaluate } from 'solvix';

// 1. Natural implicit multiplication: 2x instead of 2*x
console.log(evaluate('2x + 3y', { x: 4, y: 5 })); // 23
console.log(evaluate('3(x + 2)', { x: 10 }));      // 36

// 2. Parse LaTeX directly with a variable scope
const E = fromLaTeX('m c^2', { m: 2, c: 3e8 }); // 1.8e17
const Ek = fromLaTeX('\\frac{1}{2} m v^2', { m: 10, v: 20 }); // 2000
const h = fromLaTeX('\\sqrt{x^2 + y^2}', { x: 3, y: 4 }); // 5

// 3. Quick calculator (auto-detects plain text or LaTeX)
const r1 = quickCalc('10 * (2 + 3)');
const r2 = quickCalc('\\frac{10}{2}');

// 4. Export the AST back to LaTeX for rendering in a UI (KaTeX / MathJax)
console.log(toLaTeX('x / y + sqrt(z)')); // \frac{x}{y} + \sqrt{z}
```

---

### 23. AI Agent Tools (JSON Schema, MCP, and Self-Repair)

Connect the library as a tool-calling server for models such as GPT-4, Claude, or Gemini:

```typescript
import { getToolDefinitions, tryEval } from 'solvix';

// 1. Standard JSON Schema definitions for function calling / MCP
const tools = getToolDefinitions();
// [
//   { name: 'evaluateExpression', description: '...', parameters: { ... } },
//   { name: 'solveODE', ... },
//   { name: 'optimizeFunction', ... },
//   { name: 'convertUnits', ... }
// ]

// 2. Execution sandbox with diagnostics and self-repair hints
const diag = tryEval(() => {
  // Code generated by the agent
  return evaluate('2 * variable_no_declarada');
});

if (!diag.success) {
  console.log(diag.code);      // e.g. "UNDEFINED_VARIABLE"
  console.log(diag.message);   // "Undefined variable in scope: variable_no_declarada"
  console.log(diag.agentHint); // "A variable was not declared in the scope. Pass a scope object..."
}
```

---

## 🧪 Development and Validation

```bash
npm install
npm run build
npm test
npm run example
npm run bench
```

`npm run example` runs a short local example of the public API. `npm test` runs tests for that API. `npm run bench` runs the numerical benchmarks, and `npm run dev` starts the TypeScript compiler in watch mode.

### Mapping over several arrays

```typescript
import { broadcastMap, broadcastInto, NDArray } from 'solvix';

const y = NDArray.fromArray([[1], [2]]);
const x = NDArray.fromArray([[1, 2, 3]]);
broadcastMap((a, b) => Math.hypot(a, b), y, x); // 2x3 result; scalars and nested arrays also accepted
broadcastInto(out, (a, b) => a + b, y, x);      // same, writing into `out` (any strides)
```

`fn` receives only element values. In expressions: `hypot.(y, x)`, `map(f, a, b)`.

Everything below shares this kernel and works on strided views, complex values and nested arrays:

| Function | Purpose |
|---|---|
| `mapElements(x, f)` | unary map, `f(value)` |
| `mapIndexed(x, f)` | `f(value, index, array)` with a 0-based multi-index |
| `where(cond, a, b)`, `clip(x, lo, hi)` | broadcast all arguments (array bounds allowed) |
| `addInPlace`, `mulInPlace` | write into any strided `out`, with broadcasting |
| `quad(f, a, b)`, `fzeroMap(f, a, b)` | one result per broadcast limit/bracket |
| `DataFrame.map`, `DataFrame.withColumn` | cell-wise and column-wise computation |
| `diff`, `pctChange` | computed from `shift` |

Helpers: `NDArray.contiguous()` returns the array itself when it is already row-major and gap-free, otherwise a copy; `toFloat64(x)` does the same for plain sequences.
