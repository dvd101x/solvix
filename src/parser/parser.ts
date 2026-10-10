/**
 * @file parser.ts
 * Parser matemático y evaluador de expresiones (AST) para ix, con sintaxis tipo Julia:
 * - Operadores: + - * / \ ^ %, elemento a elemento (.* ./ .^ .+ .-), transpuesta conjugada (')
 * - Llamadas con broadcast f.(x) y unidad imaginaria `im`
 * - Tokenizador de operadores, funciones y variables
 * - Algoritmo Shunting-Yard (Dijkstra) con precedencia y asociatividad estándar
 * - Compilación a función evaluable rápida o evaluación directa contra un Scope
 * - Soporte de funciones nativas (sin, cos, exp, log, sqrt, det, inv, etc.)
 */

import { NDArray } from '../core/ndarray.js';
import { Complex } from '../types/complex.js';
import { eye } from '../generators/ranges.js';
import {
  abs, conj, real, imag, angle, sub, mul, div, pow, neg,
  addElementwise, isArrayLike, toNDArray, mapReal, mapComplex,
} from '../ops/elementwise.js';
import { mtimes, mpower, mrdiv, mldiv, ctranspose } from '../ops/operators.js';
import {
  matmul, dot, outer, kron, trace, diag, triu, tril, norm, opnorm, pinv, lstsq,
  rank, cond, cholesky, eigen, adjoint, transposeCopy,
} from '../linalg/matrix.js';
import { det, inv, solve } from '../linalg/factorizations.js';
import { broadcastMap, mapElements, mapIndexed } from '../ops/broadcast-map.js';
import { colonRange, buildArray, indexOneBased, type AxisIndex } from '../indexing/one-based.js';
import {
  all, any, argmax, argmin, countNonzero, cummax, cummin, cumprod, cumsum,
  max, mean, median, min, percentile, prod, quantile, std, sum, variance,
  type Axis,
} from '../stats/reductions.js';
import { nanmean, nansum, nanstd } from '../stats/series-ops.js';

export interface Token {
  type:
    | 'NUMBER' | 'IDENTIFIER' | 'OPERATOR' | 'LPAREN' | 'RPAREN' | 'COMMA'
    | 'LBRACKET' | 'RBRACKET' | 'SEMICOLON' | 'COLON';
  value: string;
  /** Whitespace precedes this token; it separates elements inside `[...]`. */
  spaceBefore?: boolean;
  /** Julia-style broadcast call `f.(x)`; only set on identifiers. */
  broadcast?: boolean;
  /** Zero-based character range in the original expression. */
  start?: number;
  end?: number;
}

export type ASTNode =
  | { type: 'NUMBER'; value: number }
  | { type: 'VARIABLE'; name: string }
  | { type: 'BINARY_OP'; op: string; left: ASTNode; right: ASTNode }
  | { type: 'UNARY_OP'; op: string; expr: ASTNode }
  | { type: 'POSTFIX_OP'; op: string; expr: ASTNode }
  | { type: 'FUNCTION_CALL'; name: string; args: ASTNode[]; broadcast?: boolean }
  | { type: 'MATRIX'; rows: ASTNode[][]; commaSeparated: boolean }
  | { type: 'RANGE'; start: ASTNode; step?: ASTNode; stop: ASTNode }
  | { type: 'INDEX'; target: ASTNode; indices: ASTNode[] }
  | { type: 'COLON_ALL' };

const OPERATOR_PRECEDENCE: Record<string, { prec: number; assoc: 'L' | 'R' }> = {
  '+': { prec: 2, assoc: 'L' },
  '-': { prec: 2, assoc: 'L' },
  '.+': { prec: 2, assoc: 'L' },
  '.-': { prec: 2, assoc: 'L' },
  '*': { prec: 3, assoc: 'L' },
  '/': { prec: 3, assoc: 'L' },
  '\\': { prec: 3, assoc: 'L' },
  '.*': { prec: 3, assoc: 'L' },
  './': { prec: 3, assoc: 'L' },
  '%': { prec: 3, assoc: 'L' },
  '^': { prec: 4, assoc: 'R' },
  '.^': { prec: 4, assoc: 'R' },
  'unary-': { prec: 5, assoc: 'R' },
};

function syntaxErrorAt(expr: string, index: number, message: string): SyntaxError {
  const safeIndex = Math.max(0, Math.min(index, expr.length));
  const lineStart = expr.lastIndexOf('\n', safeIndex - 1) + 1;
  const lineEnd = expr.indexOf('\n', safeIndex);
  const line = expr.slice(lineStart, lineEnd === -1 ? expr.length : lineEnd);
  const lineNumber = expr.slice(0, lineStart).split('\n').length;
  const column = safeIndex - lineStart + 1;
  return new SyntaxError(`${message} at line ${lineNumber}, column ${column}\n${line}\n${' '.repeat(column - 1)}^`);
}

/**
 * Tokeniza una cadena de texto en un flujo de tokens léxicos.
 */
export function tokenize(expr: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const len = expr.length;

  let space = false;
  let tokenStart = 0;
  const brackets: string[] = [];
  const push = (t: Token) => {
    if (space) t.spaceBefore = true;
    space = false;
    t.start = tokenStart;
    t.end = i;
    tokens.push(t);
  };

  while (i < len) {
    tokenStart = i;
    const ch = expr[i];

    if (/\s/.test(ch)) {
      space = true;
      i++;
      continue;
    }

    if (/\d/.test(ch) || (ch === '.' && /\d/.test(expr[i + 1] || ''))) {
      let numStr = '';
      while (i < len) {
        const c = expr[i];
        if (/\d/.test(c)) {
          numStr += expr[i++];
        } else if (c === '.' && !numStr.includes('.') && !/[*/^]/.test(expr[i + 1] || '')) {
          // A dot followed by * / ^ belongs to a broadcast operator (2 .* x, 2.^x).
          numStr += expr[i++];
        } else if ((c === 'e' || c === 'E') && /^[+-]?\d/.test(expr.slice(i + 1, i + 3))) {
          numStr += expr[i++];
          if (expr[i] === '+' || expr[i] === '-') numStr += expr[i++];
        } else {
          break;
        }
      }
      push({ type: 'NUMBER', value: numStr });
      continue;
    }

    if (/[a-zA-Z_]/.test(ch)) {
      let ident = '';
      while (i < len && /[a-zA-Z0-9_]/.test(expr[i])) {
        ident += expr[i++];
      }
      if (expr[i] === '.' && expr[i + 1] === '(') {
        i++; // consume '.' of a broadcast call f.(x)
        push({ type: 'IDENTIFIER', value: ident, broadcast: true });
      } else {
        push({ type: 'IDENTIFIER', value: ident });
      }
      continue;
    }

    if (ch === '(' || ch === '[') {
      brackets.push(ch);
      push({ type: ch === '(' ? 'LPAREN' : 'LBRACKET', value: ch });
      i++;
      continue;
    }

    if (ch === ')' || ch === ']') {
      brackets.pop();
      push({ type: ch === ')' ? 'RPAREN' : 'RBRACKET', value: ch });
      i++;
      continue;
    }

    if (ch === ';' || ch === ':') {
      push({ type: ch === ';' ? 'SEMICOLON' : 'COLON', value: ch });
      i++;
      continue;
    }

    if (ch === ',') {
      push({ type: 'COMMA', value: ',' });
      i++;
      continue;
    }

    if (ch === '.' && ['*', '/', '^', '+', '-'].includes(expr[i + 1])) {
      push({ type: 'OPERATOR', value: '.' + expr[i + 1] });
      i += 2;
      continue;
    }

    if (['+', '-', '*', '/', '^', '%', '\\', "'"].includes(ch)) {
      push({ type: 'OPERATOR', value: ch });
      i++;
      continue;
    }

    throw syntaxErrorAt(expr, i, `Unexpected character "${ch}"`);
  }

  // Implicit multiplication (2x -> 2*x, 3(x+1) -> 3*(x+1), (a)(b) -> (a)*(b)).
  // Inside [...] whitespace separates elements instead, and `x[` with no space is indexing.
  const withImplicitMul: Token[] = [];
  const stack: string[] = [];
  for (let idx = 0; idx < tokens.length; idx++) {
    const cur = tokens[idx];
    const next = tokens[idx + 1];
    if (cur.type === 'LPAREN' || cur.type === 'LBRACKET') stack.push(cur.value);
    if (cur.type === 'RPAREN' || cur.type === 'RBRACKET') stack.pop();
    withImplicitMul.push(cur);

    if (next) {
      const isCurEnd =
        cur.type === 'NUMBER' ||
        cur.type === 'IDENTIFIER' ||
        cur.type === 'RPAREN' ||
        cur.type === 'RBRACKET' ||
        (cur.type === 'OPERATOR' && cur.value === "'");
      const isNextStart = next.type === 'IDENTIFIER' || next.type === 'LPAREN' || next.type === 'LBRACKET';

      const isFunctionCall = cur.type === 'IDENTIFIER' && next.type === 'LPAREN' && !next.spaceBefore;
      const isIndexing = next.type === 'LBRACKET' && !next.spaceBefore;
      const separatesElements = stack[stack.length - 1] === '[' && next.spaceBefore;
      if (isCurEnd && isNextStart && !isFunctionCall && !isIndexing && !separatesElements) {
        withImplicitMul.push({ type: 'OPERATOR', value: '*' });
      }
    }
  }

  return withImplicitMul;
}

/**
 * Parsea una expresión matemática convirtiéndola en un AST mediante Shunting-Yard.
 */
export function parseExpression(expr: string): ASTNode {
  const tokens = tokenize(expr);
  let pos = 0;
  const syntaxError = (message: string, token: Token | undefined = tokens[pos]) =>
    syntaxErrorAt(expr, token?.start ?? expr.length, message);
  // True while parsing the elements of a [...] literal, where spaces separate elements.
  let inMatrix = false;

  function nested<T>(matrix: boolean, fn: () => T): T {
    const saved = inMatrix;
    inMatrix = matrix;
    try {
      return fn();
    } finally {
      inMatrix = saved;
    }
  }

  function parseRange(): ASTNode {
    const first = parseBinary(2);
    if (tokens[pos]?.type !== 'COLON') return first;
    pos++;
    const second = parseBinary(2);
    if (tokens[pos]?.type !== 'COLON') return { type: 'RANGE', start: first, stop: second };
    pos++;
    const third = parseBinary(2);
    return { type: 'RANGE', start: first, step: second, stop: third };
  }

  function parseMatrix(): ASTNode {
    pos++; // consume '['
    const rows: ASTNode[][] = [[]];
    let commaSeparated = false;
    nested(true, () => {
      while (true) {
        const t = tokens[pos];
        if (!t) throw syntaxError('Mismatched bracket: expected "]"');
        if (t.type === 'RBRACKET') break;
        if (t.type === 'SEMICOLON') {
          rows.push([]);
          pos++;
        } else if (t.type === 'COMMA') {
          commaSeparated = true;
          pos++;
        } else {
          rows[rows.length - 1].push(parseRange());
        }
      }
    });
    pos++; // consume ']'
    while (rows.length > 1 && rows[rows.length - 1].length === 0) rows.pop();
    return { type: 'MATRIX', rows, commaSeparated };
  }

  function parseIndex(target: ASTNode): ASTNode {
    pos++; // consume '['
    const indices: ASTNode[] = [];
    nested(false, () => {
      while (true) {
        const t = tokens[pos];
        const after = tokens[pos + 1];
        if (t?.type === 'COLON' && (after?.type === 'COMMA' || after?.type === 'RBRACKET')) {
          pos++;
          indices.push({ type: 'COLON_ALL' });
        } else {
          indices.push(parseRange());
        }
        if (tokens[pos]?.type === 'COMMA') pos++;
        else break;
      }
    });
    if (tokens[pos]?.type !== 'RBRACKET') throw syntaxError('Mismatched bracket: expected "]"');
    pos++;
    return { type: 'INDEX', target, indices };
  }

  function parsePrimary(): ASTNode {
    const token = tokens[pos];
    if (!token) throw syntaxError('Unexpected end of expression');

    // Unary minus binds tighter than * but looser than ^, so -2^2 === -(2^2).
    if (token.type === 'OPERATOR' && (token.value === '-' || token.value === '+')) {
      pos++;
      const sub = parseBinary(OPERATOR_PRECEDENCE['^'].prec);
      return token.value === '-' ? { type: 'UNARY_OP', op: '-', expr: sub } : sub;
    }

    let node = parseAtom();
    while (true) {
      const t = tokens[pos];
      if (t?.type === 'OPERATOR' && t.value === "'") {
        pos++;
        node = { type: 'POSTFIX_OP', op: "'", expr: node };
      } else if (t?.type === 'LBRACKET' && !t.spaceBefore) {
        node = parseIndex(node);
      } else {
        return node;
      }
    }
  }

  function parseAtom(): ASTNode {
    const token = tokens[pos];
    if (!token) throw syntaxError('Unexpected end of expression');

    if (token.type === 'NUMBER') {
      pos++;
      return { type: 'NUMBER', value: parseFloat(token.value) };
    }

    if (token.type === 'IDENTIFIER') {
      const name = token.value;
      pos++;
      if (tokens[pos]?.type === 'LPAREN' && !(inMatrix && tokens[pos].spaceBefore)) {
        pos++; // consume '('
        const args: ASTNode[] = [];
        if (tokens[pos]?.type !== 'RPAREN') {
          nested(false, () => {
            while (true) {
              args.push(parseRange());
              if (tokens[pos]?.type === 'COMMA') {
                pos++;
              } else {
                break;
              }
            }
          });
        }
        if (tokens[pos]?.type !== 'RPAREN') {
          throw syntaxError(`Expected closing parenthesis after arguments in function ${name}`);
        }
        pos++; // consume ')'
        return token.broadcast ? { type: 'FUNCTION_CALL', name, args, broadcast: true } : { type: 'FUNCTION_CALL', name, args };
      }
      return { type: 'VARIABLE', name };
    }

    if (token.type === 'LBRACKET') return parseMatrix();

    if (token.type === 'LPAREN') {
      pos++;
      const node = nested(false, parseRange);
      if (tokens[pos]?.type !== 'RPAREN') {
        throw syntaxError('Mismatched parenthesis: expected ")"');
      }
      pos++;
      return node;
    }

    throw syntaxError(`Unexpected token "${token.value}" of type ${token.type}`, token);
  }

  function parseBinary(minPrec: number): ASTNode {
    let left = parsePrimary();

    while (pos < tokens.length && tokens[pos].type === 'OPERATOR') {
      const opToken = tokens[pos];
      const opInfo = OPERATOR_PRECEDENCE[opToken.value];
      if (!opInfo || opInfo.prec < minPrec) break;
      // In [1 -2] the "-" starts a new element; in [1 - 2] and [1-2] it is a binary operator.
      if (inMatrix && (opToken.value === '+' || opToken.value === '-') && opToken.spaceBefore && !tokens[pos + 1]?.spaceBefore) break;

      pos++;
      const nextMinPrec = opInfo.assoc === 'L' ? opInfo.prec + 1 : opInfo.prec;
      const right = parseBinary(nextMinPrec);
      left = { type: 'BINARY_OP', op: opToken.value, left, right };
    }

    return left;
  }

  const ast = parseRange();
  if (pos < tokens.length) {
    throw syntaxError(`Extra unexpected tokens after valid expression starting at "${tokens[pos].value}"`);
  }
  return ast;
}

const REAL_MATH: Record<string, Function> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sqrt: Math.sqrt,
  cbrt: Math.cbrt,
  exp: Math.exp,
  log: Math.log,
  floor: Math.floor,
  ceil: Math.ceil,
  round: Math.round,
  sign: Math.sign,
  trunc: Math.trunc,
  log2: Math.log2,
  log10: Math.log10,
  log1p: Math.log1p,
  expm1: Math.expm1,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  atan2: Math.atan2,
  hypot: Math.hypot,
  min: Math.min,
  max: Math.max,
  pow: Math.pow,
  clamp: (value: number, min: number, max: number) => Math.min(Math.max(value, min), max),
};

const viaComplex = (fn: (z: Complex) => Complex) => (re: number, im: number): [number, number] => {
  const z = fn(new Complex(re, im));
  return [z.re, z.im];
};

/** Complex-valued versions of the scalar functions that have a natural extension. */
const COMPLEX_MATH: Record<string, (re: number, im: number) => [number, number]> = {
  sqrt: viaComplex((z) => z.sqrt()),
  exp: viaComplex((z) => z.exp()),
  log: viaComplex((z) => z.log()),
  sin: (a, b) => [Math.sin(a) * Math.cosh(b), Math.cos(a) * Math.sinh(b)],
  cos: (a, b) => [Math.cos(a) * Math.cosh(b), -Math.sin(a) * Math.sinh(b)],
};

function parserAxis(axis: unknown): Axis | undefined {
  if (axis === undefined) return undefined;
  const axes = typeof axis === 'number'
    ? [axis]
    : isArrayLike(axis)
      ? Array.from(toNDArray(axis).data)
      : null;
  if (!axes || axes.length === 0 || axes.some((value) => !Number.isInteger(value) || value < 1)) {
    throw new RangeError('Parser axes must be positive 1-based integers');
  }
  const zeroBasedAxes = axes.map((value) => value - 1);
  return typeof axis === 'number' ? zeroBasedAxes[0] : zeroBasedAxes;
}

function reduction(
  fn: (arr: NDArray, opts: { axis?: Axis; ddof?: number }) => unknown,
  arr: unknown,
  axis?: unknown,
  ddof?: unknown,
): unknown {
  if (!isArrayLike(arr)) {
    throw new TypeError('Reduction functions require an array argument');
  }
  if (ddof !== undefined && (!Number.isInteger(ddof) || (ddof as number) < 0)) {
    throw new RangeError('ddof must be a non-negative integer');
  }
  return fn(toNDArray(arr), { axis: parserAxis(axis), ddof: ddof as number | undefined });
}

function scan(
  fn: (arr: NDArray, opts: { axis?: Axis }) => NDArray,
  arr: unknown,
  axis?: unknown,
): NDArray {
  if (!isArrayLike(arr)) {
    throw new TypeError('Cumulative functions require an array argument');
  }
  return fn(toNDArray(arr), { axis: parserAxis(axis) });
}

/** Functions that already understand scalars, complex numbers and whole arrays. */
const ARRAY_FUNCTIONS: Record<string, Function> = {
  abs, conj, real, imag, angle,
  det, inv, solve, matmul, dot, outer, kron, trace, diag, triu, tril,
  norm, opnorm, pinv, lstsq, rank, cond, cholesky, eigen,
  adjoint,
  transpose: transposeCopy,
  size: (x: any, dim?: number) => {
    const shape = Array.from(toNDArray(x).shape);
    return dim === undefined ? NDArray.fromArray(shape) : shape[dim - 1];
  },
  length: (x: any) => (isArrayLike(x) ? toNDArray(x).size : 1),
  // Julia argument order: map(f, x) calls f(value); mapIndexed(f, x) calls f(value, index, array)
  map: (f: Function, ...xs: any[]) =>
    xs.length === 1 ? mapElements(xs[0], (v) => f(v)) : broadcastMap((...v: any[]) => f(...v), ...xs),
  mapIndexed: (f: Function, x: any) => mapIndexed(x, (v, i, a) => f(v, i, a)),
  eye: (n: number, m?: number) => eye(n, m),
  zeros: (...dims: number[]) => NDArray.zeros(dims),
  ones: (...dims: number[]) => NDArray.ones(dims),
  sum: (arr: unknown, axis?: unknown) => reduction(sum, arr, axis),
  prod: (arr: unknown, axis?: unknown) => reduction(prod, arr, axis),
  mean: (arr: unknown, axis?: unknown) => reduction(mean, arr, axis),
  variance: (arr: unknown, axis?: unknown, ddof?: unknown) => reduction(variance, arr, axis, ddof),
  std: (arr: unknown, axis?: unknown, ddof?: unknown) => reduction(std, arr, axis, ddof),
  median: (arr: unknown, axis?: unknown) => reduction(median, arr, axis),
  min: (...args: unknown[]) => isArrayLike(args[0])
    ? reduction(min, args[0], args[1])
    : Math.min(...args as number[]),
  max: (...args: unknown[]) => isArrayLike(args[0])
    ? reduction(max, args[0], args[1])
    : Math.max(...args as number[]),
  argmin: (arr: unknown, axis?: unknown) => reduction(argmin, arr, axis),
  argmax: (arr: unknown, axis?: unknown) => reduction(argmax, arr, axis),
  all: (arr: unknown, axis?: unknown) => reduction(all, arr, axis),
  any: (arr: unknown, axis?: unknown) => reduction(any, arr, axis),
  countNonzero: (arr: unknown, axis?: unknown) => reduction(countNonzero, arr, axis),
  quantile: (arr: unknown, q: number, axis?: unknown) => {
    if (typeof q !== 'number') throw new TypeError('quantile requires a numeric quantile');
    return reduction((value, opts) => quantile(value, q, opts), arr, axis);
  },
  percentile: (arr: unknown, p: number, axis?: unknown) => {
    if (typeof p !== 'number') throw new TypeError('percentile requires a numeric percentile');
    return reduction((value, opts) => percentile(value, p, opts), arr, axis);
  },
  nansum: (arr: unknown, axis?: unknown) => reduction(nansum, arr, axis),
  nanmean: (arr: unknown, axis?: unknown) => reduction(nanmean, arr, axis),
  nanstd: (arr: unknown, axis?: unknown, ddof?: unknown) => reduction(nanstd, arr, axis, ddof),
  cumsum: (arr: unknown, axis?: unknown) => scan(cumsum, arr, axis),
  cumprod: (arr: unknown, axis?: unknown) => scan(cumprod, arr, axis),
  cummin: (arr: unknown, axis?: unknown) => scan(cummin, arr, axis),
  cummax: (arr: unknown, axis?: unknown) => scan(cummax, arr, axis),
};

function callScalarFunction(name: string, fn: Function, args: any[], broadcast?: boolean): any {
  const arrayArgs = args.filter(isArrayLike);
  if (arrayArgs.length === 0) {
    if (args.some((a) => a instanceof Complex)) {
      const cf = COMPLEX_MATH[name];
      if (!cf) throw new TypeError(`${name} is not defined for complex numbers`);
      const [re, im] = cf(args[0].re, args[0].im);
      return new Complex(re, im);
    }
    return fn(...args);
  }
  if (!broadcast) {
    throw new TypeError(`Use ${name}.(x) to apply ${name} element-wise to an array`);
  }
  if (args.length !== 1) {
    return broadcastMap((...v: any[]) => {
      if (v.some((e) => e instanceof Complex)) throw new TypeError(`${name} is not defined for complex values`);
      return fn(...v);
    }, ...args);
  }
  const arr = toNDArray(args[0]);
  if (arr.isComplex) {
    const cf = COMPLEX_MATH[name];
    if (!cf) throw new TypeError(`${name} is not defined for complex arrays`);
    return mapComplex(args[0], cf);
  }
  return mapReal(args[0], (v) => fn(v));
}

function evalBinary(op: string, l: any, r: any): any {
  if (typeof l === 'number' && typeof r === 'number') {
    switch (op) {
      case '+': case '.+': return l + r;
      case '-': case '.-': return l - r;
      case '*': case '.*': return l * r;
      case '/': case './': return l / r;
      case '\\': return r / l;
      case '%': return l % r;
      case '^': case '.^': return Math.pow(l, r);
    }
  }
  switch (op) {
    case '+': case '.+': return addElementwise(l, r);
    case '-': case '.-': return sub(l, r);
    case '*': return mtimes(l, r);
    case '.*': return mul(l, r);
    case '/': return mrdiv(l, r);
    case './': return div(l, r);
    case '\\': return mldiv(l, r);
    case '^': return mpower(l, r);
    case '.^': return pow(l, r);
    case '%': throw new TypeError('The % operator is only defined for real numbers');
    default:
      throw new Error(`Unsupported binary operator: "${op}"`);
  }
}

/**
 * Evaluates an AST against a scope of variables and functions.
 * Operators follow Julia: `*` is the matrix product, `.*` `./` `.^` are element-wise,
 * `\` solves linear systems, `'` is the conjugate transpose, `f.(x)` broadcasts and `im` is the imaginary unit.
 */
export function evaluateAST(ast: ASTNode, scope: Record<string, any> = {}): any {
  switch (ast.type) {
    case 'NUMBER':
      return ast.value;

    case 'VARIABLE': {
      if (ast.name in scope) {
        return scope[ast.name];
      }
      if (ast.name === 'pi' || ast.name === 'PI') return Math.PI;
      if (ast.name === 'e' || ast.name === 'E') return Math.E;
      if (ast.name === 'im') return new Complex(0, 1);
      // Built-in scalar functions are first-class values, e.g. map(sqrt, x)
      if (REAL_MATH[ast.name]) return (v: any) => callScalarFunction(ast.name, REAL_MATH[ast.name], [v]);
      throw new ReferenceError(`Undefined variable or symbol in scope: "${ast.name}"`);
    }

    case 'UNARY_OP': {
      const val = evaluateAST(ast.expr, scope);
      return typeof val === 'number' ? -val : neg(val);
    }

    case 'COLON_ALL':
      throw new SyntaxError('":" can only be used inside an index, as in A[:, 1]');

    case 'RANGE': {
      const start = evaluateAST(ast.start, scope);
      const stop = evaluateAST(ast.stop, scope);
      const step = ast.step ? evaluateAST(ast.step, scope) : 1;
      if (![start, stop, step].every((v) => typeof v === 'number')) {
        throw new TypeError('Range bounds must be real numbers');
      }
      return colonRange(start, stop, step);
    }

    case 'MATRIX':
      return buildArray(
        ast.rows.map((row) => row.map((e) => evaluateAST(e, scope))),
        ast.commaSeparated
      );

    case 'INDEX': {
      const target = evaluateAST(ast.target, scope);
      if (!isArrayLike(target)) throw new TypeError('Only arrays can be indexed');
      const shape = toNDArray(target).shape;
      const indices: AxisIndex[] = ast.indices.map((node, d) => {
        if (node.type === 'COLON_ALL') return ':';
        const v = evaluateAST(node, { ...scope, end: shape[d] });
        if (typeof v === 'number') return v;
        if (isArrayLike(v)) return Array.from(toNDArray(v).data);
        throw new TypeError('Indices must be integers, ranges or ":"');
      });
      return indexOneBased(target, indices);
    }

    case 'POSTFIX_OP':
      return ctranspose(evaluateAST(ast.expr, scope));

    case 'BINARY_OP':
      return evalBinary(ast.op, evaluateAST(ast.left, scope), evaluateAST(ast.right, scope));

    case 'FUNCTION_CALL': {
      const args = ast.args.map((arg) => evaluateAST(arg, scope));
      const userFn = scope[ast.name];
      if (typeof userFn === 'function') {
        if (!ast.broadcast) return userFn(...args);
        return args.length === 1 && isArrayLike(args[0])
          ? mapElements(args[0], (v) => userFn(v))
          : broadcastMap((...v: any[]) => userFn(...v), ...args);
      }
      const scalarFn = REAL_MATH[ast.name];
      if (ast.broadcast && typeof scalarFn === 'function') {
        return callScalarFunction(ast.name, scalarFn, args, true);
      }
      if (ARRAY_FUNCTIONS[ast.name]) return ARRAY_FUNCTIONS[ast.name](...args);
      if (typeof scalarFn !== 'function') {
        throw new ReferenceError(`Unknown function: "${ast.name}"`);
      }
      return callScalarFunction(ast.name, scalarFn, args, ast.broadcast);
    }
  }
}

/**
 * Evalúa directamente una expresión de texto sobre un scope.
 * @example
 * evaluate("3 * x + sin(y)", { x: 5, y: 0 }) // -> 15
 */
export function evaluate(expr: string, scope: Record<string, any> = {}): any {
  const ast = parseExpression(expr);
  return evaluateAST(ast, scope);
}

/**
 * Compila una expresión de texto a una función ejecutable de alta velocidad.
 */
export function compile(expr: string): (scope?: Record<string, any>) => any {
  const ast = parseExpression(expr);
  return (scope: Record<string, any> = {}) => evaluateAST(ast, scope);
}
