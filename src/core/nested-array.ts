/**
 * @file nested-array.ts
 * Estructura para arreglos multidimensionales anidados (Jagged / Native Nested Arrays).
 * Permite manejar tensores directamente sobre estructuras anidadas JS estándar (number[][], etc.)
 * sin requerir aplanamiento ni asignación de buffers continuos planos (Strided TypedArrays).
 */

export type NestedArrayData = any[] | number;

/**
 * Infiere recursivamente la forma (shape) de un arreglo anidado.
 * Detecta si es regular o irregular (jagged array).
 */
export function inferNestedShape(nested: any[]): { shape: number[]; isRegular: boolean } {
  if (!Array.isArray(nested)) {
    return { shape: [], isRegular: true };
  }

  const shape: number[] = [nested.length];
  let isRegular = true;

  if (nested.length > 0 && Array.isArray(nested[0])) {
    const firstSubResult = inferNestedShape(nested[0]);
    const expectedSubShape = firstSubResult.shape;
    if (!firstSubResult.isRegular) isRegular = false;

    for (let i = 1; i < nested.length; i++) {
      if (!Array.isArray(nested[i])) {
        isRegular = false;
        break;
      }
      const sub = inferNestedShape(nested[i]);
      if (sub.shape.length !== expectedSubShape.length) {
        isRegular = false;
        break;
      }
      for (let d = 0; d < expectedSubShape.length; d++) {
        if (sub.shape[d] !== expectedSubShape[d]) {
          isRegular = false;
          break;
        }
      }
    }
    shape.push(...expectedSubShape);
  }

  return { shape, isRegular };
}

export class NestedArray {
  public readonly data: any[];
  public readonly shape: number[];
  public readonly ndim: number;
  public readonly isRegular: boolean;

  constructor(data: any[]) {
    this.data = data;
    const { shape, isRegular } = inferNestedShape(data);
    this.shape = shape;
    this.ndim = shape.length;
    this.isRegular = isRegular;
  }

  /**
   * Accede a un elemento multidimensional por índices arbitrarios directamente sobre el árbol anidado.
   */
  public get(...indices: number[]): any {
    let cur: any = this.data;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (cur === undefined || cur === null || !Array.isArray(cur)) {
        throw new TypeError(`Cannot index into non-array at dimension ${i}`);
      }
      // Support negative indices relative to the end of the array.
      const normalizedIdx = idx < 0 ? cur.length + idx : idx;
      if (normalizedIdx < 0 || normalizedIdx >= cur.length) {
        throw new RangeError(`Index ${idx} out of range for axis ${i} of size ${cur.length}`);
      }
      cur = cur[normalizedIdx];
    }
    return cur;
  }

  /**
   * Modifica un elemento en el árbol de arrays anidados in-place.
   */
  public set(...args: any[]): void {
    const val = args[args.length - 1];
    const indices = args.slice(0, -1) as number[];

    let cur: any = this.data;
    for (let i = 0; i < indices.length - 1; i++) {
      const idx = indices[i];
      const normalizedIdx = idx < 0 ? cur.length + idx : idx;
      cur = cur[normalizedIdx];
      if (!Array.isArray(cur)) {
        throw new TypeError(`Target at dimension ${i} is not an array`);
      }
    }

    const lastIdx = indices[indices.length - 1];
    const normalizedLast = lastIdx < 0 ? cur.length + lastIdx : lastIdx;
    cur[normalizedLast] = val;
  }

  /**
   * Itera linealmente por todos los escalares hojas del arreglo anidado.
   */
  *[Symbol.iterator](): IterableIterator<any> {
    function* traverse(node: any): IterableIterator<any> {
      if (Array.isArray(node)) {
        for (const item of node) {
          yield* traverse(item);
        }
      } else {
        yield node;
      }
    }
    yield* traverse(this.data);
  }

  /**
   * Aplica una función a cada elemento hoja conservando la estructura de arreglo anidado.
   */
  public map(fn: (val: any) => any): NestedArray {
    const mapRecursive = (node: any): any => {
      if (Array.isArray(node)) {
        return node.map(mapRecursive);
      }
      return fn(node);
    };
    return new NestedArray(mapRecursive(this.data));
  }

  /**
   * Convierte a copia plana nativa de JS.
   */
  public toJSON(): any[] {
    return JSON.parse(JSON.stringify(this.data));
  }
}
