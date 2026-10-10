import { describe, it, expect } from 'vitest';
import { NDArray } from '../src/core/ndarray.js';
import {
  lu,
  solve,
  inv,
  det,
  qr,
  svd,
} from '../src/linalg/factorizations.js';
import {
  tokenize,
  parseExpression,
  evaluate,
  compile,
} from '../src/parser/parser.js';

describe('ix / linalg / factorizations & linear systems', () => {
  it('computes LU decomposition and solves Ax = b', () => {
    // A = [[4, 3],
    //      [6, 3]]
    // b = [10, 12]
    // Solución esperada:
    // 4x + 3y = 10
    // 6x + 3y = 12
    // Restando: 2x = 2 => x = 1, y = 2
    const A = new NDArray(new Float64Array([4, 3, 6, 3]), { shape: [2, 2] });
    const b = new Float64Array([10, 12]);

    const x = solve(A, b);
    expect(x.get(0)).toBeCloseTo(1.0, 5);
    expect(x.get(1)).toBeCloseTo(2.0, 5);

    // Validar determinante: 4*3 - 3*6 = 12 - 18 = -6
    expect(det(A)).toBeCloseTo(-6.0, 5);
  });

  it('inverts matrices (inv)', () => {
    // Matriz 2x2: [[4, 7], [2, 6]]
    // Inversa: 1/(24 - 14) * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
    const A = new NDArray(new Float64Array([4, 7, 2, 6]), { shape: [2, 2] });
    const invA = inv(A);

    expect(invA.get(0, 0)).toBeCloseTo(0.6, 5);
    expect(invA.get(0, 1)).toBeCloseTo(-0.7, 5);
    expect(invA.get(1, 0)).toBeCloseTo(-0.2, 5);
    expect(invA.get(1, 1)).toBeCloseTo(0.4, 5);
  });

  it('rejects singular matrices and right-hand sides with the wrong dimension', () => {
    const singular = new NDArray(new Float64Array([1, 2, 2, 4]), { shape: [2, 2] });
    const identity = new NDArray(new Float64Array([1, 0, 0, 1]), { shape: [2, 2] });

    expect(() => lu(singular)).toThrowError(/singular or near-singular/);
    expect(() => solve(identity, [1])).toThrowError(/Dimension mismatch/);
  });

  it('computes QR decomposition (A = QR)', () => {
    // Matriz 3x2
    const A = new NDArray(new Float64Array([12, -51, 6, 167, -4, 24]), { shape: [3, 2] });
    const { Q, R } = qr(A);

    expect(Array.from(Q.shape)).toEqual([3, 3]);
    expect(Array.from(R.shape)).toEqual([3, 2]);

    // R es triangular superior: R[1, 0] = 0
    expect(R.get(1, 0)).toBeCloseTo(0, 5);
  });

  it('rejects wide matrices for QR decomposition', () => {
    const wide = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });
    expect(() => qr(wide)).toThrowError(/rows >= cols/);
  });

  it('computes SVD decomposition (U, S, V)', () => {
    const A = new NDArray(new Float64Array([3, 2, 2, 3]), { shape: [2, 2] });
    const { U, S, V } = svd(A);

    expect(S.shape[0]).toBe(2);
    // Valores singulares ordenados en magnitud
    expect(S.get(0)).toBeGreaterThan(0);
    expect(S.get(1)).toBeGreaterThan(0);
  });
});

describe('ix / parser / mathematical expression parser & AST', () => {
  it('tokenizes arithmetic expressions', () => {
    const tokens = tokenize('3 + 4 * 2 / (1 - 5)^2');
    expect(tokens.map((t) => t.value)).toEqual([
      '3', '+', '4', '*', '2', '/', '(', '1', '-', '5', ')', '^', '2'
    ]);
  });

  it('evaluates basic arithmetic respecting operator precedence', () => {
    expect(evaluate('2 + 3 * 4')).toBe(14);
    expect(evaluate('(2 + 3) * 4')).toBe(20);
    expect(evaluate('2 ^ 3 ^ 2')).toBe(512); // Right associative: 2^(3^2) = 2^9 = 512
  });

  it('evaluates variables and mathematical functions in scope', () => {
    const res = evaluate('sin(pi / 2) + sqrt(x^2 + y^2)', { x: 3, y: 4 });
    // sin(pi/2) = 1, sqrt(9 + 16) = 5 -> total = 6
    expect(res).toBeCloseTo(6.0, 5);
  });

  it('compiles expressions into reusable high-speed functions', () => {
    const energy = compile('0.5 * m * v^2');
    expect(energy({ m: 2, v: 3 })).toBe(9);
    expect(energy({ m: 10, v: 4 })).toBe(80);
  });

  it('supports custom function calls in scope (e.g. det, custom functions)', () => {
    const customScope = {
      double: (x: number) => x * 2,
      A: new NDArray(new Float64Array([1, 0, 0, 1]), { shape: [2, 2] }),
      det,
    };
    const res = evaluate('double(5) + det(A)', customScope);
    expect(res).toBe(11); // 10 + 1
  });

  it('reports incomplete expressions and unknown function names', () => {
    expect(() => evaluate('2 +')).toThrow(SyntaxError);
    expect(() => evaluate('unknownFunction(2)')).toThrowError(/Unknown function/);
  });

  it('reports syntax errors with source locations', () => {
    expect(() => evaluate('2 +\n$')).toThrow(/Unexpected character "\$" at line 2, column 1\n\$\n\^/);
    expect(() => evaluate('sin(1, 2')).toThrow(/Expected closing parenthesis after arguments in function sin at line 1, column 9/);
    expect(() => evaluate('1 +')).toThrow(/Unexpected end of expression at line 1, column 4/);
  });

  it('supports common Math built-ins in evaluated and compiled expressions', () => {
    expect(evaluate('log10(100) + log2(8)')).toBe(5);
    expect(evaluate('sign(-4) + trunc(2.9)')).toBe(1);
    expect(evaluate('atan2(1, 0)')).toBeCloseTo(Math.PI / 2, 12);
    expect(evaluate('hypot(3, 4)')).toBe(5);
    expect(evaluate('min(3, 1, 2) + max(3, 1, 2)')).toBe(4);
    expect(evaluate('pow(2, 3)')).toBe(8);
    expect(evaluate('clamp(5, 0, 3)')).toBe(3);
    expect(compile('sinh(x) + cosh(x)')({ x: 0 })).toBe(1);
  });

  it('exposes array reductions and cumulative functions with 1-based axes', () => {
    const A = [[1, 2, 3], [4, 5, 6]];
    const nested = (value: unknown) => value instanceof NDArray ? value.toNestedArray() : value;

    expect(evaluate('sum(A)', { A })).toBe(21);
    expect(nested(evaluate('sum(A, 1)', { A }))).toEqual([5, 7, 9]);
    expect(nested(evaluate('mean(A, 2)', { A }))).toEqual([2, 5]);
    expect(nested(evaluate('sum(A, [1, 2])', { A }))).toEqual([21]);
    expect(nested(evaluate('cumsum(A, 2)', { A }))).toEqual([[1, 3, 6], [4, 9, 15]]);
    expect(nested(evaluate('argmax(A, 2)', { A }))).toEqual([2, 2]);
    expect(evaluate('nanmean(x)', { x: [1, NaN, 3] })).toBe(2);
    expect(() => evaluate('sum(A, 0)', { A })).toThrow(/1-based/);
  });

  it('supports Julia-style function declarations, assignments and return', () => {
    const program = `
function factorial(n)
  result = 1
  for i = 1:n
    result = result * i
  end
  return result
end
factorial(5)
`;
    expect(evaluate(program)).toBe(120);
    expect(evaluate('function twice(x); return x * 2; end; twice(4)')).toBe(8);
    expect(() => evaluate('return 1')).toThrow(/only be used inside/);
  });

  it('supports conditionals, loops and logical comparisons', () => {
    const program = `
sum = 0
for i = 1:6
  if i % 2 == 0 && i < 6
    sum = sum + i
  elseif i == 6
    sum = sum + 10
  else
    sum = sum + 1
  end
end
while sum < 25
  sum = sum + 1
end
sum
`;
    expect(evaluate(program)).toBe(25);
    expect(evaluate('if !(2 > 3) || false; 7; else; 9; end')).toBe(7);
  });

  it('supports recursive functions, local scope and compiled programs', () => {
    const scope: Record<string, unknown> = { offset: 2 };
    const program = `
function fibonacci(n)
  if n <= 1
    return n
  end
  temporary = fibonacci(n - 1) + fibonacci(n - 2)
  temporary + offset
end
fibonacci(4)
`;
    expect(evaluate(program, scope)).toBe(11);
    expect(scope).not.toHaveProperty('temporary');

    const compiled = compile('function increment(x); x + 1; end; increment(value)');
    expect(compiled({ value: 4 })).toBe(5);
  });

  it('validates flow-control contracts and malformed blocks', () => {
    expect(() => evaluate('function f(x)\n  x + 1')).toThrow(/Expected "end" for function f/);
    expect(() => evaluate('function f(x)\n  x\nend\nf()')).toThrow(/expects 1 arguments, got 0/);
    expect(() => evaluate('for i = 3\n  i\nend')).toThrow(/iterable must be an array or range/);
    expect(() => evaluate('if [1]\n  1\nend')).toThrow(/conditions must evaluate/);
  });
});
