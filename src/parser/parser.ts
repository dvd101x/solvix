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
    | 'LBRACKET' | 'RBRACKET' | 'LBRACE' | 'RBRACE' | 'SEMICOLON' | 'COLON' | 'NEWLINE';
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
  | { type: 'COLON_ALL' }
  | { type: 'OBJECT'; properties: { key: string; value: ASTNode }[] }
  | { type: 'PROGRAM'; statements: ASTNode[] }
  | { type: 'ASSIGNMENT'; name: string; value: ASTNode }
  | { type: 'FUNCTION_DECLARATION'; name: string; params: { name: string; defaultValue?: ASTNode }[]; body: ASTNode[] }
  | { type: 'IF'; branches: { condition: ASTNode; body: ASTNode[] }[]; elseBody?: ASTNode[] }
  | { type: 'WHILE'; condition: ASTNode; body: ASTNode[] }
  | { type: 'FOR'; variable: string; iterable: ASTNode; body: ASTNode[] }
  | { type: 'RETURN'; value?: ASTNode }
  | { type: 'BREAK' }
  | { type: 'CONTINUE' };

const OPERATOR_PRECEDENCE: Record<string, { prec: number; assoc: 'L' | 'R' }> = {
  '||': { prec: 1, assoc: 'L' },
  '&&': { prec: 2, assoc: 'L' },
  '==': { prec: 3, assoc: 'L' },
  '!=': { prec: 3, assoc: 'L' },
  '<': { prec: 3, assoc: 'L' },
  '<=': { prec: 3, assoc: 'L' },
  '>': { prec: 3, assoc: 'L' },
  '>=': { prec: 3, assoc: 'L' },
  '+': { prec: 4, assoc: 'L' },
  '-': { prec: 4, assoc: 'L' },
  '.+': { prec: 4, assoc: 'L' },
  '.-': { prec: 4, assoc: 'L' },
  '*': { prec: 5, assoc: 'L' },
  '/': { prec: 5, assoc: 'L' },
  '\\': { prec: 5, assoc: 'L' },
  '.*': { prec: 5, assoc: 'L' },
  './': { prec: 5, assoc: 'L' },
  '%': { prec: 5, assoc: 'L' },
  '^': { prec: 6, assoc: 'R' },
  '.^': { prec: 6, assoc: 'R' },
  'unary-': { prec: 7, assoc: 'R' },
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

    if (ch === '\n') {
      push({ type: 'NEWLINE', value: '\n' });
      i++;
      continue;
    }

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

    if (ch === '(' || ch === '[' || ch === '{') {
      brackets.push(ch);
      push({ type: ch === '(' ? 'LPAREN' : ch === '[' ? 'LBRACKET' : 'LBRACE', value: ch });
      i++;
      continue;
    }

    if (ch === ')' || ch === ']' || ch === '}') {
      brackets.pop();
      push({ type: ch === ')' ? 'RPAREN' : ch === ']' ? 'RBRACKET' : 'RBRACE', value: ch });
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

    const twoCharacterOperator = expr.slice(i, i + 2);
    if (['==', '!=', '<=', '>=', '&&', '||'].includes(twoCharacterOperator)) {
      push({ type: 'OPERATOR', value: twoCharacterOperator });
      i += 2;
      continue;
    }

    if (['+', '-', '*', '/', '^', '%', '\\', "'", '<', '>', '!', '='].includes(ch)) {
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
  const statementKeywords = new Set(['function', 'if', 'elseif', 'else', 'while', 'for', 'return', 'break', 'continue', 'end']);
  for (let idx = 0; idx < tokens.length; idx++) {
    const cur = tokens[idx];
    const next = tokens[idx + 1];
    if (cur.type === 'LPAREN' || cur.type === 'LBRACKET' || cur.type === 'LBRACE') stack.push(cur.value);
    if (cur.type === 'RPAREN' || cur.type === 'RBRACKET' || cur.type === 'RBRACE') stack.pop();
    withImplicitMul.push(cur);

    if (next) {
      const isCurEnd =
        cur.type === 'NUMBER' ||
        cur.type === 'IDENTIFIER' ||
        cur.type === 'RPAREN' ||
        cur.type === 'RBRACKET' ||
        cur.type === 'RBRACE' ||
        (cur.type === 'OPERATOR' && cur.value === "'");
      const isNextStart = next.type === 'IDENTIFIER' || next.type === 'LPAREN' || next.type === 'LBRACKET' || next.type === 'LBRACE';

      const isFunctionCall = cur.type === 'IDENTIFIER' && next.type === 'LPAREN' && !next.spaceBefore;
      const isIndexing = next.type === 'LBRACKET' && !next.spaceBefore;
      const separatesElements = stack[stack.length - 1] === '[' && next.spaceBefore;
      const isStatementKeyword = cur.type === 'IDENTIFIER' && statementKeywords.has(cur.value);
      const isForIn = next.type === 'IDENTIFIER' && next.value === 'in';
      if (isCurEnd && isNextStart && !isFunctionCall && !isIndexing && !separatesElements && !isStatementKeyword && !isForIn) {
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
  let inMatrix = false;
  let loopDepth = 0;

  function nested<T>(matrix: boolean, fn: () => T): T {
    const saved = inMatrix;
    inMatrix = matrix;
    try {
      return fn();
    } finally {
      inMatrix = saved;
    }
  }

  function skipSeparators(): void {
    while (tokens[pos]?.type === 'NEWLINE' || tokens[pos]?.type === 'SEMICOLON') pos++;
  }

  function requireBlockSeparator(): void {
    if (tokens[pos]?.type !== 'NEWLINE' && tokens[pos]?.type !== 'SEMICOLON') {
      throw syntaxError('Expected a newline or ";" before block body');
    }
    skipSeparators();
  }

  function parseRange(): ASTNode {
    const first = parseBinary(1);
    if (tokens[pos]?.type !== 'COLON') return first;
    pos++;
    const second = parseBinary(1);
    if (tokens[pos]?.type !== 'COLON') return { type: 'RANGE', start: first, stop: second };
    pos++;
    const third = parseBinary(1);
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
        if (t.type === 'SEMICOLON' || t.type === 'NEWLINE') {
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

  function parseObject(): ASTNode {
    pos++;
    const properties: { key: string; value: ASTNode }[] = [];
    while (true) {
      while (tokens[pos]?.type === 'NEWLINE') pos++;
      const token = tokens[pos];
      if (token?.type === 'RBRACE') {
        pos++;
        return { type: 'OBJECT', properties };
      }
      if (token?.type !== 'IDENTIFIER') {
        throw syntaxError('Expected object property name');
      }
      pos++;
      let value: ASTNode = { type: 'VARIABLE', name: token.value };
      if (tokens[pos]?.type === 'COLON') {
        pos++;
        value = parseRange();
      }
      properties.push({ key: token.value, value });
      while (tokens[pos]?.type === 'NEWLINE') pos++;
      if (tokens[pos]?.type === 'COMMA') {
        pos++;
        continue;
      }
      if (tokens[pos]?.type !== 'RBRACE') {
        throw syntaxError('Expected "," or "}" after object property');
      }
    }
  }

  function parseIndex(target: ASTNode): ASTNode {
    pos++; // consume '['
    const indices: ASTNode[] = [];
    nested(false, () => {
      while (true) {
        while (tokens[pos]?.type === 'NEWLINE') pos++;
        const t = tokens[pos];
        const after = tokens[pos + 1];
        if (t?.type === 'COLON' && (after?.type === 'COMMA' || after?.type === 'RBRACKET')) {
          pos++;
          indices.push({ type: 'COLON_ALL' });
        } else {
          indices.push(parseRange());
        }
        if (tokens[pos]?.type === 'COMMA') {
          pos++;
          while (tokens[pos]?.type === 'NEWLINE') pos++;
        }
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
    if (token.type === 'OPERATOR' && (token.value === '-' || token.value === '+' || token.value === '!')) {
      pos++;
      const sub = parseBinary(OPERATOR_PRECEDENCE['^'].prec);
      return token.value === '+' ? sub : { type: 'UNARY_OP', op: token.value, expr: sub };
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
        while (tokens[pos]?.type === 'NEWLINE') pos++;
        if (tokens[pos]?.type !== 'RPAREN') {
          nested(false, () => {
            while (true) {
              args.push(parseRange());
              if (tokens[pos]?.type === 'COMMA') {
                pos++;
                while (tokens[pos]?.type === 'NEWLINE') pos++;
              } else {
                break;
              }
            }
          });
        }
        while (tokens[pos]?.type === 'NEWLINE') pos++;
        if (tokens[pos]?.type !== 'RPAREN') {
          throw syntaxError(`Expected closing parenthesis after arguments in function ${name}`);
        }
        pos++; // consume ')'
        return token.broadcast ? { type: 'FUNCTION_CALL', name, args, broadcast: true } : { type: 'FUNCTION_CALL', name, args };
      }
      return { type: 'VARIABLE', name };
    }

    if (token.type === 'LBRACKET') return parseMatrix();
    if (token.type === 'LBRACE') return parseObject();

    if (token.type === 'LPAREN') {
      pos++;
      while (tokens[pos]?.type === 'NEWLINE') pos++;
      const node = nested(false, parseRange);
      while (tokens[pos]?.type === 'NEWLINE') pos++;
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

  function parseFunction(): ASTNode {
    pos++;
    const name = tokens[pos];
    if (name?.type !== 'IDENTIFIER') throw syntaxError('Expected function name');
    pos++;
    if (tokens[pos]?.type !== 'LPAREN') throw syntaxError(`Expected "(" after function name ${name.value}`);
    pos++;
    const params: { name: string; defaultValue?: ASTNode }[] = [];
    let sawDefault = false;
    if (tokens[pos]?.type !== 'RPAREN') {
      while (true) {
        const param = tokens[pos];
        if (param?.type !== 'IDENTIFIER') throw syntaxError('Expected parameter name');
        pos++;
        let defaultValue: ASTNode | undefined;
        if (tokens[pos]?.type === 'OPERATOR' && tokens[pos].value === '=') {
          pos++;
          defaultValue = parseRange();
          sawDefault = true;
        } else if (sawDefault) {
          throw syntaxError('Required parameters cannot follow parameters with defaults', param);
        }
        if (params.some(({ name: existingName }) => existingName === param.value)) {
          throw syntaxError(`Duplicate parameter name "${param.value}"`, param);
        }
        params.push({ name: param.value, defaultValue });
        if (tokens[pos]?.type !== 'COMMA') break;
        pos++;
      }
    }
    if (tokens[pos]?.type !== 'RPAREN') throw syntaxError(`Expected ")" after parameters for ${name.value}`);
    pos++;
    requireBlockSeparator();
    const enclosingLoopDepth = loopDepth;
    loopDepth = 0;
    const body = parseStatements(new Set(['end']));
    loopDepth = enclosingLoopDepth;
    if (tokens[pos]?.value !== 'end') throw syntaxError(`Expected "end" for function ${name.value}`);
    pos++;
    return { type: 'FUNCTION_DECLARATION', name: name.value, params, body };
  }

  function parseIf(): ASTNode {
    pos++;
    const condition = parseRange();
    requireBlockSeparator();
    const branches = [{ condition, body: parseStatements(new Set(['elseif', 'else', 'end'])) }];
    while (tokens[pos]?.value === 'elseif') {
      pos++;
      const elseifCondition = parseRange();
      requireBlockSeparator();
      branches.push({ condition: elseifCondition, body: parseStatements(new Set(['elseif', 'else', 'end'])) });
    }
    let elseBody: ASTNode[] | undefined;
    if (tokens[pos]?.value === 'else') {
      pos++;
      requireBlockSeparator();
      elseBody = parseStatements(new Set(['end']));
    }
    if (tokens[pos]?.value !== 'end') throw syntaxError('Expected "end" for if block');
    pos++;
    return { type: 'IF', branches, elseBody };
  }

  function parseWhile(): ASTNode {
    pos++;
    const condition = parseRange();
    requireBlockSeparator();
    loopDepth++;
    const body = parseStatements(new Set(['end']));
    loopDepth--;
    if (tokens[pos]?.value !== 'end') throw syntaxError('Expected "end" for while block');
    pos++;
    return { type: 'WHILE', condition, body };
  }

  function parseFor(): ASTNode {
    pos++;
    const variable = tokens[pos];
    if (variable?.type !== 'IDENTIFIER') throw syntaxError('Expected loop variable after "for"');
    pos++;
    const separator = tokens[pos];
    if (
      !(separator?.type === 'OPERATOR' && separator.value === '=') &&
      !(separator?.type === 'IDENTIFIER' && separator.value === 'in')
    ) {
      throw syntaxError(`Expected "=" or "in" after loop variable ${variable.value}`);
    }
    pos++;
    const iterable = parseRange();
    requireBlockSeparator();
    loopDepth++;
    const body = parseStatements(new Set(['end']));
    loopDepth--;
    if (tokens[pos]?.value !== 'end') throw syntaxError('Expected "end" for for block');
    pos++;
    return { type: 'FOR', variable: variable.value, iterable, body };
  }

  function parseStatement(): ASTNode {
    const token = tokens[pos];
    if (token?.type === 'IDENTIFIER') {
      if (token.value === 'function') return parseFunction();
      if (token.value === 'if') return parseIf();
      if (token.value === 'while') return parseWhile();
      if (token.value === 'for') return parseFor();
      if (token.value === 'return') {
        pos++;
        if (tokens[pos]?.type === 'NEWLINE' || tokens[pos]?.type === 'SEMICOLON' || !tokens[pos]) {
          return { type: 'RETURN' };
        }
        return { type: 'RETURN', value: parseRange() };
      }
      if (token.value === 'break' || token.value === 'continue') {
        if (loopDepth === 0) throw syntaxError(`${token.value} can only be used inside a loop`, token);
        pos++;
        return { type: token.value === 'break' ? 'BREAK' : 'CONTINUE' };
      }
      if (tokens[pos + 1]?.type === 'OPERATOR' && tokens[pos + 1].value === '=') {
        pos += 2;
        return { type: 'ASSIGNMENT', name: token.value, value: parseRange() };
      }
    }
    return parseRange();
  }

  function parseStatements(stopWords: Set<string>): ASTNode[] {
    const statements: ASTNode[] = [];
    skipSeparators();
    while (pos < tokens.length) {
      const token = tokens[pos];
      if (token.type === 'IDENTIFIER' && stopWords.has(token.value)) break;
      statements.push(parseStatement());
      if (pos >= tokens.length) break;
      if (tokens[pos]?.type !== 'NEWLINE' && tokens[pos]?.type !== 'SEMICOLON') {
        throw syntaxError(`Expected a statement separator before "${tokens[pos].value}"`);
      }
      skipSeparators();
    }
    return statements;
  }

  const statements = parseStatements(new Set());
  if (pos < tokens.length) throw syntaxError(`Extra unexpected tokens after valid expression starting at "${tokens[pos].value}"`);
  if (statements.length === 0) throw syntaxError('Expected expression');
  return statements.length === 1 ? statements[0] : { type: 'PROGRAM', statements };
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
      case '==': return l === r;
      case '!=': return l !== r;
      case '<': return l < r;
      case '<=': return l <= r;
      case '>': return l > r;
      case '>=': return l >= r;
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
    case '==': return l === r;
    case '!=': return l !== r;
    case '<': case '<=': case '>': case '>=':
      throw new TypeError(`The ${op} operator is only defined for real numbers`);
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

const MAX_LOOP_ITERATIONS = 1_000_000;

class ReturnSignal {
  constructor(public readonly value: unknown) {}
}

class BreakSignal {}
class ContinueSignal {}

function conditionValue(value: unknown): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  throw new TypeError('Flow-control conditions must evaluate to a number or boolean');
}

function evaluateStatements(statements: ASTNode[], scope: Record<string, any>): unknown {
  let result: unknown;
  for (const statement of statements) result = evaluateAST(statement, scope);
  return result;
}

/**
 * Evaluates an AST against a scope of variables and functions.
 * Operators follow Julia: `*` is the matrix product, `.*` `./` `.^` are element-wise,
 * `\` solves linear systems, `'` is the conjugate transpose, `f.(x)` broadcasts and `im` is the imaginary unit.
 */
export function evaluateAST(ast: ASTNode, scope: Record<string, any> = {}): any {
  switch (ast.type) {
    case 'PROGRAM':
      return evaluateStatements(ast.statements, scope);

    case 'ASSIGNMENT': {
      const value = evaluateAST(ast.value, scope);
      scope[ast.name] = value;
      return value;
    }

    case 'FUNCTION_DECLARATION': {
      const fn = (...args: unknown[]) => {
        const requiredParameterCount = ast.params.filter((param) => param.defaultValue === undefined).length;
        if (args.length < requiredParameterCount || args.length > ast.params.length) {
          const expected = requiredParameterCount === ast.params.length
            ? String(requiredParameterCount)
            : `${requiredParameterCount} to ${ast.params.length}`;
          throw new RangeError(`${ast.name} expects ${expected} arguments, got ${args.length}`);
        }
        const localScope = Object.create(scope) as Record<string, any>;
        for (let i = 0; i < ast.params.length; i++) {
          const param = ast.params[i];
          localScope[param.name] = i < args.length
            ? args[i]
            : evaluateAST(param.defaultValue!, localScope);
        }
        try {
          return evaluateStatements(ast.body, localScope);
        } catch (error) {
          if (error instanceof ReturnSignal) return error.value;
          throw error;
        }
      };
      scope[ast.name] = fn;
      return fn;
    }

    case 'IF':
      for (const branch of ast.branches) {
        if (conditionValue(evaluateAST(branch.condition, scope))) {
          return evaluateStatements(branch.body, scope);
        }
      }
      return ast.elseBody ? evaluateStatements(ast.elseBody, scope) : undefined;

    case 'WHILE': {
      let result: unknown;
      for (let iteration = 0; conditionValue(evaluateAST(ast.condition, scope)); iteration++) {
        if (iteration >= MAX_LOOP_ITERATIONS) {
          throw new RangeError(`while loop exceeded ${MAX_LOOP_ITERATIONS} iterations`);
        }
        try {
          result = evaluateStatements(ast.body, scope);
        } catch (error) {
          if (error instanceof BreakSignal) break;
          if (error instanceof ContinueSignal) continue;
          throw error;
        }
      }
      return result;
    }

    case 'FOR': {
      const iterable = evaluateAST(ast.iterable, scope);
      if (!isArrayLike(iterable)) {
        throw new TypeError('for loop iterable must be an array or range');
      }
      let result: unknown;
      for (const value of toNDArray(iterable)) {
        scope[ast.variable] = value;
        try {
          result = evaluateStatements(ast.body, scope);
        } catch (error) {
          if (error instanceof BreakSignal) break;
          if (error instanceof ContinueSignal) continue;
          throw error;
        }
      }
      return result;
    }

    case 'RETURN':
      throw new ReturnSignal(ast.value === undefined ? undefined : evaluateAST(ast.value, scope));

    case 'BREAK':
      throw new BreakSignal();

    case 'CONTINUE':
      throw new ContinueSignal();

    case 'NUMBER':
      return ast.value;

    case 'VARIABLE': {
      if (ast.name in scope) {
        return scope[ast.name];
      }
      if (ast.name === 'pi' || ast.name === 'PI') return Math.PI;
      if (ast.name === 'e' || ast.name === 'E') return Math.E;
      if (ast.name === 'im') return new Complex(0, 1);
      if (ast.name === 'true') return true;
      if (ast.name === 'false') return false;
      // Built-in scalar functions are first-class values, e.g. map(sqrt, x)
      if (REAL_MATH[ast.name]) return (v: any) => callScalarFunction(ast.name, REAL_MATH[ast.name], [v]);
      throw new ReferenceError(`Undefined variable or symbol in scope: "${ast.name}"`);
    }

    case 'UNARY_OP': {
      const val = evaluateAST(ast.expr, scope);
      if (ast.op === '!') return !conditionValue(val);
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

    case 'OBJECT': {
      const object: Record<string, unknown> = {};
      for (const property of ast.properties) object[property.key] = evaluateAST(property.value, scope);
      return object;
    }

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

    case 'BINARY_OP': {
      const left = evaluateAST(ast.left, scope);
      if (ast.op === '&&') return conditionValue(left) && conditionValue(evaluateAST(ast.right, scope));
      if (ast.op === '||') return conditionValue(left) || conditionValue(evaluateAST(ast.right, scope));
      return evalBinary(ast.op, left, evaluateAST(ast.right, scope));
    }

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
  try {
    return evaluateAST(ast, scope);
  } catch (error) {
    if (error instanceof ReturnSignal) {
      throw new SyntaxError('return can only be used inside a function declaration');
    }
    throw error;
  }
}

/**
 * Compila una expresión de texto a una función ejecutable de alta velocidad.
 */
export function compile(expr: string): (scope?: Record<string, any>) => any {
  const ast = parseExpression(expr);
  return (scope: Record<string, any> = {}) => evaluateAST(ast, scope);
}
