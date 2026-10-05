/**
 * @file index.ts
 * Exportaciones principales de ix
 */
export * from './core/index.js';
export * from './ops/math-ops.js';
export { sub, mul, div, pow, neg, conj, real, imag, abs, angle, mapReal, mapComplex, addElementwise, isArrayLike, toNDArray } from './ops/elementwise.js';
export { broadcastMap, broadcastInto, mapElements, mapIndexed, type Element, type BroadcastArg } from './ops/broadcast-map.js';
export { mtimes, mpower, mldiv, mrdiv, ctranspose } from './ops/operators.js';
export * from './indexing/advanced-indexing.js';
export { colonRange, buildArray, indexOneBased, type AxisIndex } from './indexing/one-based.js';
export * from './parallel/shared-memory.js';
export * from './parallel/expression-worker.js';
export * from './dag/dag.js';
export * from './linalg/factorizations.js';
export * from './linalg/matrix.js';
export * from './parser/parser.js';
export * from './ode/ode45.js';
export * from './optimize/roots.js';
export * from './optimize/minimize.js';
export * from './units/units.js';
export * from './constants/constants.js';
export * from './memory/in-place.js';
export * from './generators/ranges.js';
export * from './manipulation/manipulation.js';
export * from './stats/reductions.js';
export * from './stats/series-ops.js';
export * from './dataframe/dataframe.js';
export * from './types/complex.js';
export * from './types/fraction.js';
export * from './signal/transforms.js';
export * from './integrate/quadrature.js';
export * from './interpolate/interpolation.js';
export * from './plot/svg.js';
export * from './ai/helpers.js';
export * from './latex/latex.js';
export * from './agent/agent-tools.js';
