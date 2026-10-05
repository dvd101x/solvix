/**
 * @file dispatcher.ts
 * Motor de despacho múltiple con caché de firmas polimórficas.
 */

export const Any = Symbol('Any');

export type TypeIdentifier = string | Function | typeof Any;

export interface MethodEntry {
  types: TypeIdentifier[];
  fn: Function;
}

export interface GenericFunction {
  (...args: any[]): any;
  add(types: TypeIdentifier[], fn: Function): GenericFunction;
  genericName: string;
  methods: MethodEntry[];
  cache: Map<string, Function>;
}

export function getTypeId(val: any): string | Function {
  if (val === null) return 'null';
  if (val === undefined) return 'undefined';
  const type = typeof val;
  if (type === 'number') return 'number';
  if (type === 'string') return 'string';
  if (type === 'boolean') return 'boolean';
  if (type === 'symbol') return 'symbol';
  if (type === 'bigint') return 'bigint';
  return val.constructor || Object;
}

export function getTypeName(t: any): string {
  if (typeof t === 'string') return t;
  if (typeof t === 'function') return t.name || 'AnonymousConstructor';
  if (t === Any) return 'Any';
  return String(t);
}

/**
 * Crea una función genérica polimórfica extensible.
 */
export function createGeneric(name: string): GenericFunction {
  const methods: MethodEntry[] = [];
  const cache = new Map<string, Function>();

  const generic = function (...args: any[]) {
    const arity = args.length;
    let signatureKey = '';

    for (let i = 0; i < arity; i++) {
      const tid = getTypeId(args[i]);
      signatureKey += (i === 0 ? '' : '|') + getTypeName(tid);
    }

    // Fast-path: resolución desde caché
    const cachedFn = cache.get(signatureKey);
    if (cachedFn) {
      return cachedFn(...args);
    }

    // Slow-path: linear dispatch matching
    for (let i = 0; i < methods.length; i++) {
      const entry = methods[i];
      if (entry.types.length !== arity) continue;

      let match = true;
      for (let j = 0; j < arity; j++) {
        const expected = entry.types[j];
        const arg = args[j];

        if (expected === Any) {
          continue;
        } else if (typeof expected === 'string') {
          if (typeof arg !== expected) {
            match = false;
            break;
          }
        } else if (typeof expected === 'function') {
          if (!(arg instanceof expected) && arg?.constructor !== expected) {
            match = false;
            break;
          }
        } else {
          match = false;
          break;
        }
      }

      if (match) {
        cache.set(signatureKey, entry.fn);
        return entry.fn(...args);
      }
    }

    throw new TypeError(
      `[MultipleDispatch] MethodError: no method matching ${name}(${signatureKey})`
    );
  } as GenericFunction;

  generic.add = function (types: TypeIdentifier[], fn: Function) {
    methods.push({ types, fn });
    cache.clear(); // Invalida el caché cuando se registran nuevas especializaciones
    return generic;
  };

  generic.genericName = name;
  generic.methods = methods;
  generic.cache = cache;

  return generic;
}
