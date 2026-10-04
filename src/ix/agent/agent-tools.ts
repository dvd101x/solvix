/**
 * @file agent-tools.ts
 * Utilidades de alta eficacia para Agentes LLM y Model Context Protocol (MCP):
 * - getToolDefinitions(): Genera esquemas JSON Schema compatibles con OpenAI/Anthropic/MCP
 * - tryEval(codeFn): Sandbox de ejecución con diagnóstico semántico enriquecido y sugerencias de corrección
 * - quickCalc(formula, inputs): Calculador instantáneo para agentes y usuarios a partir de descripciones
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
 * Esquemas estándar JSON Schema para Function Calling de LLMs y servidores MCP
 */
export function getToolDefinitions(): ToolDefinition[] {
  return [
    {
      name: 'evaluateExpression',
      description: 'Evalúa una expresión matemática en texto o LaTeX (soporta Wikipedia syntax, multiplicación implícita 2x, etc.)',
      parameters: {
        type: 'object',
        properties: {
          expression: { type: 'string', description: 'Fórmula matemática (ej: "0.5 * m * v^2" o "\\frac{1}{2} m v^2")' },
          scope: { type: 'object', description: 'Variables asociadas (ej: {"m": 10, "v": 20})' },
        },
        required: ['expression'],
      },
    },
    {
      name: 'solveODE',
      description: 'Resuelve un sistema de ecuaciones diferenciales dy/dt = f(t, y) con el método adaptativo Dormand-Prince (ODE45)',
      parameters: {
        type: 'object',
        properties: {
          tSpan: { type: 'array', description: 'Intervalo de tiempo [t0, tf]' },
          y0: { type: 'array', description: 'Condición inicial vectorial y0' },
        },
        required: ['tSpan', 'y0'],
      },
    },
    {
      name: 'optimizeFunction',
      description: 'Encuentra el mínimo local de una función multivariable en R^n usando Nelder-Mead Simplex sin derivadas',
      parameters: {
        type: 'object',
        properties: {
          x0: { type: 'array', description: 'Punto de inicio inicial [x0, y0, ...]' },
        },
        required: ['x0'],
      },
    },
    {
      name: 'convertUnits',
      description: 'Convierte una cantidad física entre unidades compatibles (incluyendo temperaturas Celsius/Fahrenheit y unidades SI/imperiales)',
      parameters: {
        type: 'object',
        properties: {
          value: { type: 'number', description: 'Valor numérico' },
          fromUnit: { type: 'string', description: 'Unidad de origen (ej: "km/h", "degC", "psi")' },
          toUnit: { type: 'string', description: 'Unidad de destino (ej: "m/s", "degF", "bar")' },
        },
        required: ['value', 'fromUnit', 'toUnit'],
      },
    },
  ];
}

/**
 * Ejecuta una rutina matemática capturando errores y generando pistas de auto-corrección para agentes
 */
export function tryEval(action: () => any): AgentDiagnostic {
  try {
    const result = action();
    return { success: true, result };
  } catch (err: any) {
    const msg = err?.message || String(err);
    let hint = 'Revisa la sintaxis de la función y los tipos de los argumentos pasados.';

    if (msg.includes('Dimension mismatch') || msg.includes('Incompatible shapes')) {
      hint = 'Las dimensiones de los tensores no son compatibles para esta operación. Considera aplicar transpose(A) o reshape(A, [...]).';
    } else if (msg.includes('Dimensional mismatch')) {
      hint = 'Estás intentando sumar o restar cantidades con magnitudes físicas incompatibles (ej. longitud y tiempo).';
    } else if (msg.includes('Undefined variable')) {
      hint = 'Una variable no fue declarada en el scope. Pasa un objeto scope con las variables requeridas (ej: { x: 5 }).';
    } else if (msg.includes('MethodError')) {
      hint = 'No existe una sobrecarga de despacho múltiple para esa combinación de tipos. Verifica si requieres pasar NDArray en lugar de Array regular.';
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
 * Calculador rápido que detecta si la entrada es texto plano o LaTeX y resuelve inmediatamente
 */
export function quickCalc(formula: string, scope: Record<string, any> = {}): any {
  if (formula.includes('\\') || formula.includes('{') || formula.includes('}')) {
    return fromLaTeX(formula, scope);
  }
  return evaluate(formula, scope);
}
