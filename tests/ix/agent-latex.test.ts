import { describe, it, expect } from 'vitest';
import {
  fromLaTeX,
  toLaTeX,
  latexToMathExpr,
} from '../../src/ix/latex/latex.js';
import {
  evaluate,
  compile,
} from '../../src/ix/parser/parser.js';
import {
  getToolDefinitions,
  tryEval,
  quickCalc,
} from '../../src/ix/agent/agent-tools.js';

describe('ix / parser / implicit multiplication & natural math', () => {
  it('parses implicit multiplication: 2x, 3(x+1), (a)(b)', () => {
    // 2x + 3y con x=4, y=5 -> 2*4 + 3*5 = 8 + 15 = 23
    expect(evaluate('2x + 3y', { x: 4, y: 5 })).toBe(23);

    // 3(x + 2) con x=10 -> 3 * 12 = 36
    expect(evaluate('3(x + 2)', { x: 10 })).toBe(36);

    // (2 + 3)(4 + 1) -> 5 * 5 = 25
    expect(evaluate('(2 + 3)(4 + 1)')).toBe(25);
  });
});

describe('ix / latex / Wikipedia & paper formulas translation', () => {
  it('normalizes common LaTeX patterns to readable math expressions', () => {
    const expr1 = latexToMathExpr('\\frac{1}{2} m v^{2}');
    expect(expr1).toBe('((1) / (2)) * m * v^(2)');

    const expr2 = latexToMathExpr('\\sqrt{a^2 + b^2}');
    expect(expr2).toBe('sqrt(a^2 + b^2)');
  });

  it('evaluates physics formulas directly from LaTeX strings', () => {
    // Energía cinética: E = \frac{1}{2} m v^2
    const Ek = fromLaTeX('\\frac{1}{2} m v^2', { m: 10, v: 20 });
    expect(Ek).toBe(2000);

    // Wikipedia style: E = m c^2
    const c = 3e8;
    const E = fromLaTeX('m c^2', { m: 2, c });
    expect(E).toBe(1.8e17);

    // Hipotenusa: \sqrt{x^2 + y^2}
    const h = fromLaTeX('\\sqrt{x^2 + y^2}', { x: 3, y: 4 });
    expect(h).toBe(5);

    // Euler's identity: e^{i \cdot \pi} + 1 (evaluación trigonométrica)
    const sinHalfPi = fromLaTeX('\\sin(\\frac{\\pi}{2})');
    expect(sinHalfPi).toBeCloseTo(1.0, 5);
  });

  it('converts mathematical AST expressions to formatted LaTeX', () => {
    const latexStr = toLaTeX('x / y + sqrt(z)');
    expect(latexStr).toContain('\\frac{x}{y}');
    expect(latexStr).toContain('\\sqrt{z}');
  });
});

describe('ix / agent / AI tool calling, auto-repair & quickCalc', () => {
  it('provides JSON Schema tool definitions for LLMs', () => {
    const defs = getToolDefinitions();
    expect(defs.length).toBeGreaterThanOrEqual(4);

    const evalTool = defs.find((t) => t.name === 'evaluateExpression');
    expect(evalTool).toBeDefined();
    expect(evalTool?.parameters.properties.expression).toBeDefined();
  });

  it('diagnoses errors and provides hints for agent self-repair', () => {
    // 1. Caso exitoso
    const diagGood = tryEval(() => evaluate('2 + 3'));
    expect(diagGood.success).toBe(true);
    expect(diagGood.result).toBe(5);

    // 2. Variable no declarada
    const diagVar = tryEval(() => evaluate('2 * foo'));
    expect(diagVar.success).toBe(false);
    expect(diagVar.code).toBe('UNDEFINED_VARIABLE');
    expect(diagVar.agentHint).toContain('A variable was not declared');

    const diagFunction = tryEval(() => evaluate('missingFunction(1)'));
    expect(diagFunction.code).toBe('UNKNOWN_FUNCTION');

    const diagSyntax = tryEval(() => evaluate('2 +'));
    expect(diagSyntax.code).toBe('SYNTAX_ERROR');
  });

  it('provides quickCalc for instant calculation of text or LaTeX', () => {
    // Texto plano
    expect(quickCalc('5 * (2 + 3)')).toBe(25);

    // LaTeX
    expect(quickCalc('\\frac{10}{2}')).toBe(5);
  });
});
