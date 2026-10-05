import { describe, it, expect } from 'vitest';
import { NDArray } from '../src/core/ndarray.js';
import {
  rolling,
  ewm,
  diff,
  pctChange,
  shift,
  isnan,
  fillna,
  dropna,
  nanmean,
  nansum,
  nanstd,
  cov,
  corr,
} from '../src/stats/series-ops.js';
import { DataFrame } from '../src/dataframe/dataframe.js';

describe('ix / stats / series operations', () => {
  it('computes rolling windows (mean, sum, min, max)', () => {
    // Serie: [10, 20, 30, 40, 50]
    const s = new NDArray(new Float64Array([10, 20, 30, 40, 50]), { shape: [5] });
    const r = rolling(s, 3);

    const m = r.mean();
    // [NaN, NaN, 20, 30, 40]
    expect(Number.isNaN(m.get(0))).toBe(true);
    expect(Number.isNaN(m.get(1))).toBe(true);
    expect(m.get(2)).toBe(20);
    expect(m.get(3)).toBe(30);
    expect(m.get(4)).toBe(40);

    const sumRoll = r.sum();
    expect(sumRoll.get(2)).toBe(60); // 10+20+30
    expect(sumRoll.get(4)).toBe(120); // 30+40+50
  });

  it('computes exponential weighted moving average (ewm)', () => {
    const s = new NDArray(new Float64Array([10, 20, 30]), { shape: [3] });
    const res = ewm(s, { alpha: 0.5 });

    expect(res.get(0)).toBe(10);
    // 0.5 * 20 + 0.5 * 10 = 15
    expect(res.get(1)).toBe(15);
    // 0.5 * 30 + 0.5 * 15 = 22.5
    expect(res.get(2)).toBe(22.5);
  });

  it('computes diff, pctChange, and shift', () => {
    const s = new NDArray(new Float64Array([10, 15, 30]), { shape: [3] });

    // diff: [NaN, 5, 15]
    const d = diff(s);
    expect(Number.isNaN(d.get(0))).toBe(true);
    expect(d.get(1)).toBe(5);
    expect(d.get(2)).toBe(15);

    // pctChange: [NaN, 0.5, 1.0]
    const pct = pctChange(s);
    expect(pct.get(1)).toBeCloseTo(0.5, 4);
    expect(pct.get(2)).toBeCloseTo(1.0, 4);

    // shift: [NaN, 10, 15]
    const sh = shift(s, 1);
    expect(Number.isNaN(sh.get(0))).toBe(true);
    expect(sh.get(1)).toBe(10);
    expect(sh.get(2)).toBe(15);
  });

  it('handles missing values (isnan, fillna, dropna, nanmean)', () => {
    const messy = new NDArray(new Float64Array([1, NaN, 3, NaN, 5]), { shape: [5] });

    expect(Array.from(isnan(messy).data)).toEqual([0, 1, 0, 1, 0]);

    // fillna con constante
    const filledConst = fillna(messy, 0);
    expect(Array.from(filledConst.data)).toEqual([1, 0, 3, 0, 5]);

    // fillna con forward-fill
    const filledFfill = fillna(messy, 'ffill');
    expect(Array.from(filledFfill.data)).toEqual([1, 1, 3, 3, 5]);

    // dropna
    const clean = dropna(messy);
    expect(Array.from(clean.data)).toEqual([1, 3, 5]);

    // nanmean (1 + 3 + 5) / 3 = 3
    expect(nanmean(messy)).toBe(3);
    expect(nansum(messy)).toBe(9);
    expect(nanstd(messy)).toBeCloseTo(Math.sqrt(8 / 3), 12);
  });

  it('ignores NaNs when reducing by axis', () => {
    const values = NDArray.fromArray([[1, NaN, 3], [NaN, 5, NaN]]);

    expect(Array.from((nanmean(values, { axis: 0 }) as NDArray).data)).toEqual([1, 5, 3]);
    expect(Array.from((nansum(values, { axis: 0 }) as NDArray).data)).toEqual([1, 5, 3]);
    expect(Array.from((nanmean(values, { axis: 1 }) as NDArray).data)).toEqual([2, 5]);
    expect(Array.from((nanstd(values, { axis: 0, ddof: 1 }) as NDArray).data).every(Number.isNaN)).toBe(true);
    expect(Array.from((nansum(values, { axis: 0, keepdims: true }) as NDArray).shape)).toEqual([1, 3]);
  });

  it('computes covariance and correlation matrices', () => {
    // 4 muestras x 2 variables perfectamente correlacionadas: y = 2*x
    const data = new NDArray(
      new Float64Array([
        1, 2,
        2, 4,
        3, 6,
        4, 8,
      ]),
      { shape: [4, 2] }
    );

    const cv = cov(data);
    expect(cv.shape[0]).toBe(2);

    const cr = corr(data);
    expect(cr.get(0, 0)).toBeCloseTo(1.0, 5);
    expect(cr.get(0, 1)).toBeCloseTo(1.0, 5); // correlación perfecta
    expect(cr.get(1, 0)).toBeCloseTo(1.0, 5);
    expect(cr.get(1, 1)).toBeCloseTo(1.0, 5);
  });
});

describe('ix / dataframe / DataFrame', () => {
  it('instantiates, selects columns and exports to records', () => {
    const df = new DataFrame(
      [
        [20.5, 1013.2],
        [21.0, 1012.8],
        [22.4, 1011.5],
      ],
      {
        columns: ['temp', 'pressure'],
        index: ['t0', 't1', 't2'],
      }
    );

    expect(df.shape).toEqual([3, 2]);

    // Extraer columna individual
    const tempCol = df.col('temp');
    expect(Array.from(tempCol.data)).toEqual([20.5, 21.0, 22.4]);

    // Selección de subconjunto
    const sub = df.select('pressure');
    expect(sub.columns).toEqual(['pressure']);
    expect(sub.shape).toEqual([3, 1]);

    // Matriz de correlación
    const c = df.corr();
    expect(c.columns).toEqual(['temp', 'pressure']);
    expect(c.values.get(0, 0)).toBeCloseTo(1.0, 4);

    // Exportación a JSON records
    const records = df.toRecords();
    expect(records.length).toBe(3);
    expect(records[0]).toEqual({ _index: 't0', temp: 20.5, pressure: 1013.2 });
  });
});
