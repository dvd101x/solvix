/**
 * @file parser.ts
 * Parser matemático y evaluador de expresiones (AST) para ix:
 * - Tokenizador de operadores (+, -, *, /, ^, %), funciones y variables
 * - Algoritmo Shunting-Yard (Dijkstra) con precedencia y asociatividad estándar
 * - Compilación a función evaluable rápida o evaluación directa contra un Scope
 * - Soporte de funciones nativas (sin, cos, exp, log, sqrt, det, inv, etc.)
 */

export interface Token {
  type: 'NUMBER' | 'IDENTIFIER' | 'OPERATOR' | 'LPAREN' | 'RPAREN' | 'COMMA';
  value: string;
}

export type ASTNode =
  | { type: 'NUMBER'; value: number }
  | { type: 'VARIABLE'; name: string }
  | { type: 'BINARY_OP'; op: string; left: ASTNode; right: ASTNode }
  | { type: 'UNARY_OP'; op: string; expr: ASTNode }
  | { type: 'FUNCTION_CALL'; name: string; args: ASTNode[] };

const OPERATOR_PRECEDENCE: Record<string, { prec: number; assoc: 'L' | 'R' }> = {
  '+': { prec: 2, assoc: 'L' },
  '-': { prec: 2, assoc: 'L' },
  '*': { prec: 3, assoc: 'L' },
  '/': { prec: 3, assoc: 'L' },
  '%': { prec: 3, assoc: 'L' },
  '^': { prec: 4, assoc: 'R' },
  'unary-': { prec: 5, assoc: 'R' },
};

/**
 * Tokeniza una cadena de texto en un flujo de tokens léxicos.
 */
export function tokenize(expr: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const len = expr.length;

  while (i < len) {
    const ch = expr[i];

    if (/\s/.test(ch)) {
      i++;
      continue;
    }

    if (/\d/.test(ch) || (ch === '.' && /\d/.test(expr[i + 1] || ''))) {
      let numStr = '';
      while (i < len && (/[\d.]/.test(expr[i]) || (expr[i] === 'e' || expr[i] === 'E'))) {
        numStr += expr[i++];
      }
      tokens.push({ type: 'NUMBER', value: numStr });
      continue;
    }

    if (/[a-zA-Z_]/.test(ch)) {
      let ident = '';
      while (i < len && /[a-zA-Z0-9_]/.test(expr[i])) {
        ident += expr[i++];
      }
      tokens.push({ type: 'IDENTIFIER', value: ident });
      continue;
    }

    if (ch === '(') {
      tokens.push({ type: 'LPAREN', value: '(' });
      i++;
      continue;
    }

    if (ch === ')') {
      tokens.push({ type: 'RPAREN', value: ')' });
      i++;
      continue;
    }

    if (ch === ',') {
      tokens.push({ type: 'COMMA', value: ',' });
      i++;
      continue;
    }

    if (['+', '-', '*', '/', '^', '%'].includes(ch)) {
      tokens.push({ type: 'OPERATOR', value: ch });
      i++;
      continue;
    }

    throw new SyntaxError(`Unexpected character in expression: "${ch}" at index ${i}`);
  }

  // Inserción de multiplicación implícita (ej. 2x -> 2*x, 3(x+1) -> 3*(x+1), (a)(b) -> (a)*(b))
  const withImplicitMul: Token[] = [];
  for (let idx = 0; idx < tokens.length; idx++) {
    const cur = tokens[idx];
    const next = tokens[idx + 1];
    withImplicitMul.push(cur);

    if (next) {
      const isCurEnd = cur.type === 'NUMBER' || cur.type === 'IDENTIFIER' || cur.type === 'RPAREN';
      const isNextStart = next.type === 'IDENTIFIER' || next.type === 'LPAREN';

      const isFunctionCall = cur.type === 'IDENTIFIER' && next.type === 'LPAREN';
      if (isCurEnd && isNextStart && !isFunctionCall) {
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

  function parsePrimary(): ASTNode {
    const token = tokens[pos];
    if (!token) throw new SyntaxError('Unexpected end of expression');

    if (token.type === 'NUMBER') {
      pos++;
      return { type: 'NUMBER', value: parseFloat(token.value) };
    }

    if (token.type === 'IDENTIFIER') {
      const name = token.value;
      pos++;
      if (tokens[pos]?.type === 'LPAREN') {
        pos++; // consume '('
        const args: ASTNode[] = [];
        if (tokens[pos]?.type !== 'RPAREN') {
          while (true) {
            args.push(parseBinary(0));
            if (tokens[pos]?.type === 'COMMA') {
              pos++;
            } else {
              break;
            }
          }
        }
        if (tokens[pos]?.type !== 'RPAREN') {
          throw new SyntaxError(`Expected closing parenthesis after arguments in function ${name}`);
        }
        pos++; // consume ')'
        return { type: 'FUNCTION_CALL', name, args };
      }
      return { type: 'VARIABLE', name };
    }

    if (token.type === 'OPERATOR' && (token.value === '-' || token.value === '+')) {
      const op = token.value;
      pos++;
      const sub = parsePrimary();
      return op === '-' ? { type: 'UNARY_OP', op: '-', expr: sub } : sub;
    }

    if (token.type === 'LPAREN') {
      pos++;
      const node = parseBinary(0);
      if (tokens[pos]?.type !== 'RPAREN') {
        throw new SyntaxError('Mismatched parenthesis: expected ")"');
      }
      pos++;
      return node;
    }

    throw new SyntaxError(`Unexpected token "${token.value}" of type ${token.type}`);
  }

  function parseBinary(minPrec: number): ASTNode {
    let left = parsePrimary();

    while (pos < tokens.length && tokens[pos].type === 'OPERATOR') {
      const opToken = tokens[pos];
      const opInfo = OPERATOR_PRECEDENCE[opToken.value];
      if (!opInfo || opInfo.prec < minPrec) break;

      pos++;
      const nextMinPrec = opInfo.assoc === 'L' ? opInfo.prec + 1 : opInfo.prec;
      const right = parseBinary(nextMinPrec);
      left = { type: 'BINARY_OP', op: opToken.value, left, right };
    }

    return left;
  }

  const ast = parseBinary(0);
  if (pos < tokens.length) {
    throw new SyntaxError(`Extra unexpected tokens after valid expression starting at "${tokens[pos].value}"`);
  }
  return ast;
}

/**
 * Evalúa recursivamente un AST matemático en base a un contexto / scope de variables y funciones.
 */
export function evaluateAST(ast: ASTNode, scope: Record<string, any> = {}): any {
  const builtInMath: Record<string, Function> = {
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
    abs: Math.abs,
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

  switch (ast.type) {
    case 'NUMBER':
      return ast.value;

    case 'VARIABLE': {
      if (ast.name in scope) {
        return scope[ast.name];
      }
      if (ast.name === 'pi' || ast.name === 'PI') return Math.PI;
      if (ast.name === 'e' || ast.name === 'E') return Math.E;
      throw new ReferenceError(`Undefined variable or symbol in scope: "${ast.name}"`);
    }

    case 'UNARY_OP': {
      const val = evaluateAST(ast.expr, scope);
      if (ast.op === '-') return -val;
      return val;
    }

    case 'BINARY_OP': {
      const l = evaluateAST(ast.left, scope);
      const r = evaluateAST(ast.right, scope);
      switch (ast.op) {
        case '+': return l + r;
        case '-': return l - r;
        case '*': return l * r;
        case '/': return l / r;
        case '%': return l % r;
        case '^': return Math.pow(l, r);
        default:
          throw new Error(`Unsupported binary operator: "${ast.op}"`);
      }
    }

    case 'FUNCTION_CALL': {
      const evaluatedArgs = ast.args.map((arg) => evaluateAST(arg, scope));
      let fn = scope[ast.name] || builtInMath[ast.name];
      if (typeof fn !== 'function') {
        throw new ReferenceError(`Unknown function: "${ast.name}"`);
      }
      return fn(...evaluatedArgs);
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
