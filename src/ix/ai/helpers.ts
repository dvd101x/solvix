/**
 * @file helpers.ts
 * Utilidades AI-Ready y ayudas interactivas para personas y agentes LLM:
 * - help(fn): Documentación y firmas de funciones con ejemplos ejecutables
 * - summary(tensor): Vista resumida compacta para no saturar contextos de LLM
 * - fromCSV: Parser de tablas y strings CSV directo a DataFrame o NDArray
 * - fromMatrixString: Parser de notación estilo MATLAB "1 2; 3 4"
 * - unitsHelp: Catálogo de unidades y guía interactiva de magnitudes
 */
import { NDArray } from '../core/ndarray.js';
import { DataFrame } from '../dataframe/dataframe.js';
import { units } from '../units/units.js';

/**
 * Catálogo interactivo de documentación para humanos y agentes
 */
const DOCS_REGISTRY: Record<string, { desc: string; signature: string; example: string }> = {
  linspace: {
    desc: 'Genera un vector 1D con N valores linealmente espaciados entre start y stop.',
    signature: 'linspace(start: number, stop: number, num?: number): NDArray',
    example: 'const x = ix.linspace(0, 10, 5); // [0, 2.5, 5, 7.5, 10]',
  },
  fft: {
    desc: 'Calcula la Transformada Rápida de Fourier (FFT) 1D mediante Cooley-Tukey.',
    signature: 'fft(signal: NDArray | number[], sampleRate?: number): FFTResult',
    example: 'const { frequencies, magnitude } = ix.fft(signal, 1000);',
  },
  ode45: {
    desc: 'Solucionador de ecuaciones diferenciales ordinarias adaptativo Runge-Kutta Dormand-Prince.',
    signature: 'ode45(f: (t, y) => NDArray, tSpan: [t0, tf], y0: NDArray, opts?): ODESolution',
    example: 'const sol = ix.ode45((t, y) => ix.mul(y, -1), [0, 5], ix.NDArray.ones([1]));',
  },
  fsolve: {
    desc: 'Resuelve sistemas de ecuaciones no lineales F(x) = 0 usando Newton-Raphson.',
    signature: 'fsolve(F: (x: NDArray) => NDArray, x0: NDArray | number[], opts?): FsolveResult',
    example: 'const { x } = ix.fsolve((v) => ix.sub(ix.pow(v, 2), 2), [1.0]);',
  },
  nelderMead: {
    desc: 'Minimización multivariable sin derivadas (Simplex) para encontrar mínimos locales.',
    signature: 'nelderMead(f: (x: NDArray) => number, x0: NDArray | number[], opts?): MinMultivarResult',
    example: 'const res = ix.nelderMead((v) => v.get(0)**2 + v.get(1)**2, [2, 3]);',
  },
  quad: {
    desc: 'Integración numérica adaptativa de Simpson para funciones continuas en [a, b].',
    signature: 'quad(f: (x: number) => number, a: number, b: number, tol?): number',
    example: 'const area = ix.quad((x) => Math.sin(x), 0, Math.PI); // ≈ 2.0',
  },
};

/**
 * Consulta la ayuda y firma de una función en texto estructurado
 */
export function help(nameOrFn: string | Function): string {
  const name = typeof nameOrFn === 'function' ? nameOrFn.name : nameOrFn;
  const doc = DOCS_REGISTRY[name];
  if (!doc) {
    return `[ix.help] No documentation found for "${name}". Available topics: ${Object.keys(DOCS_REGISTRY).join(', ')}`;
  }
  return `=== ${name} ===\n${doc.desc}\n\nFirma: ${doc.signature}\nEjemplo:\n  ${doc.example}`;
}

/**
 * Genera un resumen compacto en Markdown de un tensor para no desbordar ventanas de contexto de LLMs
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
 * Parsea una cadena de texto en formato CSV a un DataFrame
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
 * Parsea notación matricial compacta estilo MATLAB/Octave: "1 2 3; 4 5 6"
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
 * Catálogo e introspección de unidades para personas y agentes
 */
export function unitsHelp(): string {
  const list = units.listUnits();
  let out = '=== Catálogo de Unidades Físicas (ix.units) ===\n';
  out += 'Nombre'.padEnd(16) + 'Símbolo'.padEnd(10) + 'Descripción\n';
  out += '-'.repeat(55) + '\n';
  for (const u of list) {
    out += `${u.name.padEnd(16)}${u.symbol.padEnd(10)}${u.description || ''}\n`;
  }
  return out;
}
