/**
 * @file helpers.ts
 * AI-ready utilities and interactive help for users and LLM agents:
 * - help(fn): Function documentation and signatures with executable examples
 * - summary(tensor): Compact tensor summary for LLM contexts
 * - fromCSV: Parse CSV tables directly into a DataFrame or NDArray
 * - fromMatrixString: Parse compact matrix notation such as "1 2; 3 4"
 * - unitsHelp: Unit catalog and interactive quantity guide
 */
import { NDArray } from '../core/ndarray.js';
import { DataFrame } from '../dataframe/dataframe.js';
import { units } from '../units/units.js';

/**
 * Interactive documentation catalog for users and agents
 */
const DOCS_REGISTRY: Record<string, { desc: string; signature: string; example: string }> = {
  linspace: {
    desc: 'Generates a 1D vector with N values linearly spaced between start and stop.',
    signature: 'linspace(start: number, stop: number, num?: number): NDArray',
    example: 'const x = solvix.linspace(0, 10, 5); // [0, 2.5, 5, 7.5, 10]',
  },
  fft: {
    desc: 'Computes the one-dimensional Cooley-Tukey Fast Fourier Transform (FFT).',
    signature: 'fft(signal: NDArray | number[], sampleRate?: number): FFTResult',
    example: 'const { frequencies, magnitude } = solvix.fft(signal, 1000);',
  },
  ode45: {
    desc: 'Adaptive Dormand-Prince Runge-Kutta solver for ordinary differential equations.',
    signature: 'ode45(f: (t, y) => NDArray, tSpan: [t0, tf], y0: NDArray, opts?): ODESolution',
    example: 'const f = (t, y) => new solvix.NDArray(new Float64Array([-y.get(0)]), { shape: [1] }); solvix.ode45(f, [0, 5], solvix.NDArray.ones([1]));',
  },
  fsolve: {
    desc: 'Solves nonlinear systems F(x) = 0 using Newton-Raphson.',
    signature: 'fsolve(F: (x: NDArray) => NDArray, x0: NDArray | number[], opts?): FsolveResult',
    example: 'const F = (x) => new solvix.NDArray(new Float64Array([x.get(0)**2 - 2]), { shape: [1] }); solvix.fsolve(F, [1]);',
  },
  nelderMead: {
    desc: 'Derivative-free multivariable minimization (simplex) for finding local minima.',
    signature: 'nelderMead(f: (x: NDArray) => number, x0: NDArray | number[], opts?): MinMultivarResult',
    example: 'const res = solvix.nelderMead((v) => v.get(0)**2 + v.get(1)**2, [2, 3]);',
  },
  quad: {
    desc: 'Adaptive Simpson quadrature for continuous functions on [a, b].',
    signature: 'quad(f: (x: number) => number, a: number, b: number, tol?): number',
    example: 'const area = solvix.quad((x) => Math.sin(x), 0, Math.PI); // ≈ 2.0',
  },
  evaluate: {
    desc: 'Evaluates a mathematical expression with optional variable and function bindings.',
    signature: 'evaluate(expression: string, scope?: Record<string, unknown>): unknown',
    example: 'solvix.evaluate("hypot(x, y)", { x: 3, y: 4 }); // 5',
  },
  compile: {
    desc: 'Compiles an expression once and returns a reusable evaluator.',
    signature: 'compile(expression: string): (scope?: Record<string, unknown>) => unknown',
    example: 'const f = solvix.compile("x^2"); f({ x: 4 }); // 16',
  },
  where: {
    desc: 'Selects values element-wise where the condition is nonzero; array arguments broadcast against the condition.',
    signature: 'where(condition: NDArray, whenTrue: NDArray | number, whenFalse: NDArray | number): NDArray',
    example: 'const mask = solvix.NDArray.fromArray([1, 0, 1]); const values = solvix.NDArray.fromArray([3, 4, 5]); solvix.where(mask, values, 0);',
  },
  all: {
    desc: 'Returns true when every element is nonzero; optionally reduces along an axis.',
    signature: 'all(arr: NDArray, opts?: ReductionOptions): boolean | NDArray',
    example: 'solvix.all(mask); solvix.all(mask, { axis: 0 });',
  },
  any: {
    desc: 'Returns true when at least one element is nonzero; optionally reduces along an axis.',
    signature: 'any(arr: NDArray, opts?: ReductionOptions): boolean | NDArray',
    example: 'solvix.any(mask); solvix.any(mask, { axis: 0 });',
  },
  countNonzero: {
    desc: 'Counts nonzero elements, including NaN; optionally counts along an axis.',
    signature: 'countNonzero(arr: NDArray, opts?: ReductionOptions): number | NDArray',
    example: 'solvix.countNonzero(values); solvix.countNonzero(values, { axis: 0 });',
  },
  percentile: {
    desc: 'Computes a percentile using linear interpolation, optionally along an axis.',
    signature: 'percentile(arr: NDArray, p: number, opts?: ReductionOptions): number | NDArray | Quantity',
    example: 'solvix.percentile(values, 25); // 25th percentile',
  },
  sumProduct: {
    desc: 'Sums element-wise products of one or more NDArrays with identical shapes.',
    signature: 'sumProduct(...arrays: NDArray[]): number',
    example: 'solvix.sumProduct(solvix.NDArray.fromArray([1, 2]), solvix.NDArray.fromArray([3, 4])); // 11',
  },
  toNestedArray: {
    desc: 'Copies an NDArray view into nested JavaScript arrays while respecting its shape, strides, and offset.',
    signature: 'NDArray.toNestedArray(): number | NestedNumberArray',
    example: 'const rows = solvix.transpose(solvix.NDArray.fromArray([[1, 2], [3, 4]])).toNestedArray();',
  },
  fromLaTeX: {
    desc: 'Parses and evaluates a LaTeX expression with an optional variable scope.',
    signature: 'fromLaTeX(expression: string, scope?: Record<string, unknown>): unknown',
    example: 'solvix.fromLaTeX("x^2 + y^2", { x: 3, y: 4 }); // 25',
  },
  quickCalc: {
    desc: 'Evaluates plain-text or LaTeX expressions, detecting the input format automatically.',
    signature: 'quickCalc(expression: string, scope?: Record<string, unknown>): unknown',
    example: 'solvix.quickCalc("10 / 2"); // 5',
  },
};

/**
 * Return structured help and a function signature as text
 */
export function help(nameOrFn: string | Function): string {
  const name = typeof nameOrFn === 'function' ? nameOrFn.name : nameOrFn;
  const doc = DOCS_REGISTRY[name];
  if (!doc) {
    return `[solvix.help] No documentation found for "${name}". Available topics: ${Object.keys(DOCS_REGISTRY).join(', ')}`;
  }
  return `=== ${name} ===\n${doc.desc}\n\nSignature: ${doc.signature}\nExample:\n  ${doc.example}`;
}

/**
 * Generate a compact Markdown summary of a tensor for LLM context windows
 */
export function summary(arr: NDArray): string {
  const shapeStr = `[${Array.from(arr.shape).join(' x ')}]`;
  const n = arr.size;
  if (n === 0) return `NDArray(empty) shape: ${shapeStr}`;

  let min = arr.data[0], max = arr.data[0];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = arr.data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / n;

  const previewLen = Math.min(3, n);
  const head = Array.from(arr.data.slice(0, previewLen)).map((v) => v.toFixed(3));
  const tail = Array.from(arr.data.slice(-previewLen)).map((v) => v.toFixed(3));

  return `NDArray shape=${shapeStr}, size=${n}, min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)} | head=[${head.join(', ')}] ... tail=[${tail.join(', ')}]`;
}

/**
 * Parse a CSV string into a DataFrame
 */
export function fromCSV(csvString: string, opts: { hasHeader?: boolean; delimiter?: string } = {}): DataFrame {
  const delimiter = opts.delimiter ?? ',';
  const hasHeader = opts.hasHeader ?? true;

  const lines = csvString
    .trim()
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) throw new Error('Empty CSV string provided');

  let columns: string[] = [];
  let startRow = 0;

  const firstTokens = lines[0].split(delimiter).map((t) => t.trim());
  if (hasHeader) {
    columns = firstTokens;
    startRow = 1;
  } else {
    columns = firstTokens.map((_, i) => `col_${i}`);
  }

  const rows = lines.length - startRow;
  const cols = columns.length;
  const data = new Float64Array(rows * cols);

  for (let r = 0; r < rows; r++) {
    const tokens = lines[startRow + r].split(delimiter).map((t) => t.trim());
    for (let c = 0; c < cols; c++) {
      data[r * cols + c] = parseFloat(tokens[c]) || 0;
    }
  }

  return new DataFrame(new NDArray(data, { shape: [rows, cols] }), { columns });
}

/**
 * Parse compact matrix notation, such as "1 2 3; 4 5 6"
 */
export function fromMatrixString(matStr: string): NDArray {
  const rowStrings = matStr.trim().replace(/^\[|\]$/g, '').split(';');
  const rows = rowStrings.length;
  let cols = -1;
  const elements: number[] = [];

  for (let r = 0; r < rows; r++) {
    const tokens = rowStrings[r].trim().split(/[\s,]+/).filter(Boolean);
    if (cols === -1) {
      cols = tokens.length;
    } else if (tokens.length !== cols) {
      throw new Error(`Inconsistent row length in matrix string: expected ${cols}, got ${tokens.length}`);
    }
    for (const tok of tokens) {
      elements.push(parseFloat(tok));
    }
  }

  return new NDArray(new Float64Array(elements), { shape: [rows, cols] });
}

/**
 * Unit catalog and introspection for users and agents
 */
export function unitsHelp(): string {
  const list = units.listUnits();
  let out = '=== Physical Unit Catalog (solvix.units) ===\n';
  out += 'Name'.padEnd(16) + 'Symbol'.padEnd(10) + 'Description\n';
  out += '-'.repeat(55) + '\n';
  for (const u of list) {
    out += `${u.name.padEnd(16)}${u.symbol.padEnd(10)}${u.description || ''}\n`;
  }
  return out;
}
