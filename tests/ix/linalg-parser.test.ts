import { describe, it, expect } from 'vitest';
import { NDArray } from '../../src/ix/core/ndarray.js';
import {
  lu,
  solve,
  inv,
  det,
  qr,
  svd,
} from '../../src/ix/linalg/factorizations.js';
import {
  tokenize,
  parseExpression,
  evaluate,
  compile,
} from '../../src/ix/parser/parser.js';

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

  it('computes QR decomposition (A = QR)', () => {
    // Matriz 3x2
    const A = new NDArray(new Float64Array([12, -51, 6, 167, -4, 24]), { shape: [3, 2] });
    const { Q, R } = qr(A);

    expect(Array.from(Q.shape)).toEqual([3, 3]);
    expect(Array.from(R.shape)).toEqual([3, 2]);

    // R es triangular superior: R[1, 0] = 0
    expect(R.get(1, 0)).toBeCloseTo(0, 5);
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
});
