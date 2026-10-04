import { NdArray } from './ndarray.js';

export type TypeTag =
  | 'Number'
  | 'Complex'
  | 'NdArray'
  | 'TfTensor'
  | 'NumJsArray'
  | 'MlMatrix'
  | 'MathJsMatrix'
  | 'Any';

/**
 * Type hierarchy definition (child -> parent)
 */
const TYPE_HIERARCHY: Record<TypeTag, TypeTag | null> = {
  Number: 'Any',
  Complex: 'Any',
  NdArray: 'Any',
  TfTensor: 'Any',
  NumJsArray: 'Any',
  MlMatrix: 'Any',
  MathJsMatrix: 'Any',
  Any: null,
};

/**
 * Computes distance in inheritance hierarchy.
 * Returns 0 for exact match, > 0 for ancestor, or Infinity if incompatible.
 */
function getTypeDistance(target: TypeTag, base: TypeTag): number {
  if (target === base) return 0;
  if (base === 'Any') {
    let dist = 0;
    let curr: TypeTag | null = target;
    while (curr && curr !== 'Any') {
      dist++;
      curr = TYPE_HIERARCHY[curr];
    }
    return dist;
  }
  let dist = 0;
  let curr: TypeTag | null = target;
  while (curr) {
    if (curr === base) return dist;
    dist++;
    curr = TYPE_HIERARCHY[curr];
  }
  return Infinity;
}

/**
 * Runtime type inspector to classify values into TypeTag.
 */
export function getType(arg: any): TypeTag {
  if (typeof arg === 'number') return 'Number';
  if (arg === null || arg === undefined) return 'Any';

  // MathJS Complex or similar
  if (typeof arg === 'object' && (arg.isComplex || ('re' in arg && 'im' in arg))) {
    return 'Complex';
  }

  // Canonical NdArray
  if (arg instanceof NdArray || (arg && arg.shape && arg.strides && arg.data instanceof Float64Array)) {
    return 'NdArray';
  }

  // TensorFlow.js Tensor
  if (arg && (arg.isTensor || (typeof arg.dataSync === 'function' && typeof arg.dispose === 'function'))) {
    return 'TfTensor';
  }

  // NumJS NdArray
  if (arg && arg._data && arg.selection && typeof arg.pick === 'function') {
    return 'NumJsArray';
  }

  // ml-matrix Matrix
  if (arg && (arg.isMatrix || (typeof arg.to2DArray === 'function' && 'rows' in arg && 'columns' in arg))) {
    return 'MlMatrix';
  }

  // MathJS DenseMatrix or Matrix
  if (arg && (arg.isDenseMatrix || arg.isMatrix || (typeof arg.subset === 'function' && '_data' in arg))) {
    return 'MathJsMatrix';
  }

  return 'Any';
}

export type MethodFn = (...args: any[]) => any;

export interface MethodEntry {
  types: TypeTag[];
  fn: MethodFn;
  specificity: number; // tie-breaker
}

export interface Dispatcher {
  name: string;
  (...args: any[]): any;
  methods: MethodEntry[];
  cache: Map<string, MethodFn>;
}

/**
 * Creates a generic multi-dispatch function.
 */
export function defmulti(name: string): Dispatcher {
  const methods: MethodEntry[] = [];
  const cache = new Map<string, MethodFn>();

  const dispatcher: any = function (...args: any[]) {
    const argTypes = args.map(getType);
    const cacheKey = argTypes.join(',');

    const cached = cache.get(cacheKey);
    if (cached) {
      return cached(...args);
    }

    // Resolve best matching method
    let bestMethod: MethodEntry | null = null;
    let minDistance = Infinity;

    for (const entry of methods) {
      if (entry.types.length !== argTypes.length) {
        continue;
      }

      let distance = 0;
      let matches = true;

      for (let i = 0; i < argTypes.length; i++) {
        const d = getTypeDistance(argTypes[i], entry.types[i]);
        if (d === Infinity) {
          matches = false;
          break;
        }
        distance += d;
      }

      if (matches) {
        if (distance < minDistance) {
          minDistance = distance;
          bestMethod = entry;
        } else if (distance === minDistance && bestMethod) {
          // If distances equal, select more specific non-Any entries
          if (entry.specificity > bestMethod.specificity) {
            bestMethod = entry;
          }
        }
      }
    }

    if (!bestMethod) {
      throw new Error(
        `MethodError: no method matching ${name}(${argTypes.join(', ')}). Registered signatures:\n` +
        methods.map((m) => `  ${name}(${m.types.join(', ')})`).join('\n')
      );
    }

    cache.set(cacheKey, bestMethod.fn);
    return bestMethod.fn(...args);
  };

  dispatcher.dispatcherName = name;
  dispatcher.methods = methods;
  dispatcher.cache = cache;

  return dispatcher as Dispatcher;
}

/**
 * Registers an implementation method with specific argument types.
 */
export function defmethod(
  dispatcher: Dispatcher,
  types: TypeTag[],
  fn: MethodFn
): void {
  // Clear resolution cache when a new method is registered
  dispatcher.cache.clear();

  // Specificity score: count how many arguments are non-'Any'
  const specificity = types.reduce((acc, t) => acc + (t !== 'Any' ? 10 : 1), 0);

  // If signature matches exactly, overwrite
  const existingIdx = dispatcher.methods.findIndex(
    (m) => m.types.length === types.length && m.types.every((t, i) => t === types[i])
  );

  if (existingIdx >= 0) {
    dispatcher.methods[existingIdx] = { types, fn, specificity };
  } else {
    dispatcher.methods.push({ types, fn, specificity });
  }
}
