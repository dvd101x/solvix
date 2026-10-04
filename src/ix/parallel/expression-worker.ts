/**
 * @file expression-worker.ts
 * Ejecución de expresiones matemáticas arbitrarias en Workers con transferencia de Scope.
 */
import { NDArray } from '../core/ndarray.js';

export interface SerializedScope {
  [varName: string]: number | { isNDArray: true; shape: number[]; data: ArrayLike<number> };
}

export interface TaskPayload {
  expression: string;
  scope: SerializedScope;
  range?: [number, number];
}

/**
 * Serializa un scope con variables numéricas y tensores NDArray para enviarlo por postMessage.
 */
export function serializeScope(scope: Record<string, any>): SerializedScope {
  const serialized: SerializedScope = {};

  for (const [key, value] of Object.entries(scope)) {
    if (typeof value === 'number') {
      serialized[key] = value;
    } else if (value instanceof NDArray) {
      serialized[key] = {
        isNDArray: true,
        shape: Array.from(value.shape),
        data: value.data,
      };
    } else {
      serialized[key] = value;
    }
  }

  return serialized;
}

/**
 * Deserializa un scope dentro del Worker.
 */
export function deserializeScope(serialized: SerializedScope): Record<string, any> {
  const scope: Record<string, any> = {};

  for (const [key, val] of Object.entries(serialized)) {
    if (val && typeof val === 'object' && (val as any).isNDArray) {
      const obj = val as any;
      scope[key] = new NDArray(obj.data, { shape: obj.shape });
    } else {
      scope[key] = val;
    }
  }

  return scope;
}

/**
 * Compila y evalúa de manera segura una expresión matemática sobre un scope dado.
 * Soporta funciones matemáticas comunes en el namespace global (sin, cos, exp, etc.).
 */
export function evaluateExpressionInScope(
  expression: string,
  scope: Record<string, any>
): any {
  const keys = Object.keys(scope);
  const values = Object.values(scope);

  // Inyección de Math builtins para conveniencia matemática
  const mathContext = {
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan,
    sqrt: Math.sqrt,
    exp: Math.exp,
    log: Math.log,
    abs: Math.abs,
    pow: Math.pow,
    pi: Math.PI,
    e: Math.E,
  };

  const allKeys = [...Object.keys(mathContext), ...keys];
  const allValues = [...Object.values(mathContext), ...values];

  // Compilador JIT en el worker
  const fn = new Function(...allKeys, `"use strict"; return (${expression});`);
  return fn(...allValues);
}
