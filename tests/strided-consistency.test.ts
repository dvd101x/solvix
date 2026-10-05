import { describe, expect, it } from 'vitest';
import * as S from '../src/index.js';
const { NDArray } = S;

// A strided view (every second element of a longer buffer, with an offset) and its contiguous copy.
const base = NDArray.fromArray([99, 1, 99, 4, 99, 9, 99, 16, 99, 25, 99, 36]);
const view = base.slice([1, 12, 2]);
const dense = NDArray.fromArray([1, 4, 9, 16, 25, 36]);
const nest = (x: any) => (x instanceof NDArray ? Array.from(x.copy().data) : x);

describe('fixture sanity', () => {
  it('view is strided and equals dense element-wise', () => {
    expect(view.strides[0]).toBe(2);
    expect(nest(view)).toEqual(nest(dense));
  });
});

const unary: [string, (a: any) => any][] = [
  ['sum', (a) => S.sum(a)],
  ['cumsum', (a) => S.cumsum(a)],
  ['cumprod', (a) => S.cumprod(a)],
  ['cummin', (a) => S.cummin(a)],
  ['cummax', (a) => S.cummax(a)],
  ['mean', (a) => S.mean(a)],
  ['min', (a) => S.min(a)],
  ['max', (a) => S.max(a)],
  ['median', (a) => S.median(a)],
  ['std', (a) => S.std(a)],
  ['variance', (a) => S.variance(a)],
  ['prod', (a) => S.prod(a)],
  ['quantile', (a) => S.quantile(a, 0.3)],
  ['skew', (a) => S.skew(a)],
  ['kurtosis', (a) => S.kurtosis(a)],
  ['describe', (a) => S.describe(a)],
  ['countNonzero', (a) => S.countNonzero(a)],
  ['diff', (a) => S.diff(a)],
  ['pctChange', (a) => S.pctChange(a)],
  ['shift', (a) => S.shift(a, 2, 0)],
  ['isnan', (a) => S.isnan(a)],
  ['fillna', (a) => S.fillna(a, 0)],
  ['dropna', (a) => S.dropna(a)],
  ['nanmean', (a) => S.nanmean(a)],
  ['nansum', (a) => S.nansum(a)],
  ['ewm.mean', (a) => S.ewm(a, { alpha: 0.5 })],
  ['rolling.mean', (a) => S.rolling(a, 3).mean()],
  ['rolling.std', (a) => S.rolling(a, 3).std()],
  ['rolling.sum', (a) => S.rolling(a, 3).sum()],
  ['cumulativeIntegrate', (a) => S.cumulativeIntegrate(a)],
  ['trapz', (a) => S.trapz(a)],
  ['simpson', (a) => S.simpson(a)],
  ['trapz with x', (a) => S.trapz(a, a)],
  ['interp1d y', (a) => S.interp1d(NDArray.fromArray([0, 1, 2, 3, 4, 5]), a)(2.5)],
  ['interp1d x', (a) => S.interp1d(a, NDArray.fromArray([0, 1, 2, 3, 4, 5]))(10)],
  ['interp1d query', (a) => S.interp1d(NDArray.fromArray([0, 50]), NDArray.fromArray([0, 50]))(a)],
  ['cubicSpline y', (a) => S.cubicSpline(NDArray.fromArray([0, 1, 2, 3, 4, 5]), a)(2.5)],
  ['fft', (a) => S.fft(a)],
  ['clip', (a) => S.clip(a, 2, 20)],
  ['flatten', (a) => S.flatten(a)],
  ['norm', (a) => S.norm(a)],
  ['cumsum-like mul', (a) => S.mul(a, a)],
];

describe('functions give the same result for strided views and contiguous arrays', () => {
  for (const [name, fn] of unary) {
    it(name, () => {
      const norm = (r: any) => JSON.parse(JSON.stringify(r instanceof NDArray ? { s: Array.from(r.shape), d: Array.from(r.copy().data), i: r.imag && Array.from(r.copy().imag!) } : r, (_, v) => (typeof v === 'number' && Number.isNaN(v) ? 'NaN' : v)));
      expect(norm(fn(view))).toEqual(norm(fn(dense)));
    });
  }
});

describe('matrix functions agree for transposed (strided) and contiguous inputs', () => {
  const M = NDArray.fromArray([[4, 2, 1], [2, 5, 3], [1, 3, 6]]);
  const Mt = S.transpose(S.transpose(M)); // double transpose: strided view of the same values
  const Mf = S.transpose(NDArray.fromArray([[4, 2, 1], [2, 5, 3], [1, 3, 6]])); // symmetric, so equal to M
  const cases: [string, (a: any) => any][] = [
    ['det', (a) => S.det(a)],
    ['inv', (a) => S.inv(a)],
    ['solve', (a) => S.solve(a, NDArray.fromArray([1, 2, 3]))],
    ['lu', (a) => S.lu(a).L],
    ['qr', (a) => S.qr(a).R],
    ['svd', (a) => S.svd(a).S],
    ['cholesky', (a) => S.cholesky(a)],
    ['eigen', (a) => (S.eigen(a) as any).values],
    ['trace', (a) => S.trace(a)],
    ['norm', (a) => S.norm(a)],
    ['opnorm', (a) => S.opnorm(a, 1)],
    ['pinv', (a) => S.pinv(a)],
    ['rank', (a) => S.rank(a)],
    ['cond', (a) => S.cond(a)],
    ['matpow', (a) => S.matpow(a, 3)],
    ['sum axis', (a) => S.sum(a, { axis: 0 })],
    ['reshape', (a) => S.reshape(a, [9])],
    ['flatten', (a) => S.flatten(a)],
    ['kron', (a) => S.kron(a, a)],
    ['diag', (a) => S.diag(a)],
    ['triu', (a) => S.triu(a)],
  ];
  const round = (r: any): any => {
    if (r instanceof NDArray) return Array.from(r.copy().data).map((v) => Math.round(Math.abs(v) * 1e6) / 1e6);
    if (typeof r === 'number') return Math.round(Math.abs(r) * 1e6) / 1e6;
    if (r && typeof r === 'object' && 'toNestedArray' in r) return round(r.toNestedArray());
    return r;
  };
  for (const [name, fn] of cases) {
    it(name, () => {
      expect(round(fn(Mt))).toEqual(round(fn(M)));
      expect(round(fn(Mf))).toEqual(round(fn(M)));
    });
  }
});
