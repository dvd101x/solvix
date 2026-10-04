/**
 * @file agent-tools.ts
 * Utilities for LLM agents and the Model Context Protocol (MCP):
 * - getToolDefinitions(): Generate JSON Schema definitions compatible with OpenAI, Anthropic, and MCP
 * - tryEval(codeFn): Execute code in a sandbox and return diagnostics with repair suggestions
 * - quickCalc(formula, inputs): Evaluate an expression from a formula string
 */
import { evaluate } from '../parser/parser.js';
import { fromLaTeX } from '../latex/latex.js';

export interface AgentDiagnostic {
  success: boolean;
  result?: any;
  errorType?: string;
  message?: string;
  agentHint?: string;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, { type: string; description: string }>;
    required: string[];
  };
}

/**
 * Standard JSON Schema definitions for LLM function calling and MCP servers
 */
export function getToolDefinitions(): ToolDefinition[] {
  return [
    {
      name: 'evaluateExpression',
      description: 'Evaluate a mathematical expression in plain text or LaTeX (supports Wikipedia-style syntax and implicit multiplication such as 2x).',
      parameters: {
        type: 'object',
        properties: {
          expression: { type: 'string', description: 'Mathematical expression (e.g. "0.5 * m * v^2" or "\\frac{1}{2} m v^2")' },
          scope: { type: 'object', description: 'Variable bindings (e.g. {"m": 10, "v": 20})' },
        },
        required: ['expression'],
      },
    },
    {
      name: 'solveODE',
      description: 'Solve an ordinary differential equation system dy/dt = f(t, y) with the adaptive Dormand-Prince method (ODE45).',
      parameters: {
        type: 'object',
        properties: {
          tSpan: { type: 'array', description: 'Time interval [t0, tf]' },
          y0: { type: 'array', description: 'Initial state vector y0' },
        },
        required: ['tSpan', 'y0'],
      },
    },
    {
      name: 'optimizeFunction',
      description: 'Find a local minimum of a multivariable function in R^n using derivative-free Nelder-Mead simplex.',
      parameters: {
        type: 'object',
        properties: {
          x0: { type: 'array', description: 'Initial point [x0, y0, ...]' },
        },
        required: ['x0'],
      },
    },
    {
      name: 'convertUnits',
      description: 'Convert a physical quantity between compatible units, including Celsius/Fahrenheit temperatures and SI/imperial units.',
      parameters: {
        type: 'object',
        properties: {
          value: { type: 'number', description: 'Numeric value' },
          fromUnit: { type: 'string', description: 'Source unit (e.g. "km/h", "degC", "psi")' },
          toUnit: { type: 'string', description: 'Target unit (e.g. "m/s", "degF", "bar")' },
        },
        required: ['value', 'fromUnit', 'toUnit'],
      },
    },
  ];
}

/**
 * Run a mathematical operation, returning errors and repair hints for agents
 */
export function tryEval(action: () => any): AgentDiagnostic {
  try {
    const result = action();
    return { success: true, result };
  } catch (err: any) {
    const msg = err?.message || String(err);
    let hint = 'Check the function syntax and the argument types.';

    if (msg.includes('Dimension mismatch') || msg.includes('Incompatible shapes')) {
      hint = 'The tensor dimensions are incompatible for this operation. Consider using transpose(A) or reshape(A, [...]).';
    } else if (msg.includes('Dimensional mismatch')) {
      hint = 'You are trying to add or subtract incompatible physical quantities (for example, length and time).';
    } else if (msg.includes('Undefined variable')) {
      hint = 'A variable was not declared in the scope. Pass a scope object with the required variables (e.g. { x: 5 }).';
    } else if (msg.includes('MethodError')) {
      hint = 'No multiple-dispatch overload exists for this combination of types. Check whether this operation requires an NDArray instead of a regular array.';
    }

    return {
      success: false,
      errorType: err.name || 'Error',
      message: msg,
      agentHint: hint,
    };
  }
}

/**
 * Detect whether the input is plain text or LaTeX and evaluate it immediately
 */
export function quickCalc(formula: string, scope: Record<string, any> = {}): any {
  if (formula.includes('\\') || formula.includes('{') || formula.includes('}')) {
    return fromLaTeX(formula, scope);
  }
  return evaluate(formula, scope);
}
