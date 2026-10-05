import { describe, it, expect } from 'vitest';
import { NDArray } from '../src/core/ndarray.js';
import {
  fft,
  convolve,
  sinewave,
  lowpassFilter,
} from '../src/signal/transforms.js';
import {
  trapz,
  simpson,
  quad,
  cumulativeIntegrate,
} from '../src/integrate/quadrature.js';
import {
  interp1d,
  cubicSpline,
} from '../src/interpolate/interpolation.js';
import {
  fminbnd,
  nelderMead,
  curveFit,
} from '../src/optimize/minimize.js';
import { plotSVG } from '../src/plot/svg.js';
import {
  help,
  summary,
  fromCSV,
  fromMatrixString,
  unitsHelp,
} from '../src/ai/helpers.js';
import {
  units,
  Quantity,
  celsius,
  fahrenheit,
  kelvin,
  meter,
  qty,
} from '../src/units/units.js';
import {
  SPEED_OF_LIGHT,
  STANDARD_GRAVITY,
  PLANCK_CONSTANT,
} from '../src/constants/constants.js';

describe('ix / signal / DSP & FFT', () => {
  it('computes FFT peak of a pure 50Hz sine wave', () => {
    const fs = 512;
    const duration = 1.0;
    const s = sinewave(50, duration, fs, 2.0); // 50 Hz sine wave with amplitude 2.0.

    const { frequencies, magnitude } = fft(s, fs);

    // Find the index of the maximum peak.
    let maxIdx = 0;
    for (let i = 1; i < magnitude.length; i++) {
      if (magnitude[i] > magnitude[maxIdx]) maxIdx = i;
    }

    expect(frequencies[maxIdx]).toBeCloseTo(50, 0);
    expect(magnitude[maxIdx]).toBeCloseTo(2.0, 1);
  });

  it('performs discrete convolution and low-pass filtering', () => {
    const x = new Float64Array([1, 2, 3]);
    const h = new Float64Array([1, 1]);
    const conv = convolve(x, h);
    // [1*1, 1*1+2*1, 2*1+3*1, 3*1] = [1, 3, 5, 3]
    expect(Array.from(conv.data)).toEqual([1, 3, 5, 3]);

    const s = sinewave(10, 0.1, 1000);
    const filtered = lowpassFilter(s, 5, 1000);
    expect(filtered.size).toBe(s.size);
  });
});

describe('ix / integrate / numerical quadrature', () => {
  it('integrates using trapz, simpson and adaptive quad', () => {
    // Integral of sin(x) from 0 to pi = -cos(pi) - (-cos(0)) = 1 - (-1) = 2.0.
    const n = 101;
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    const dx = Math.PI / (n - 1);
    for (let i = 0; i < n; i++) {
      xs[i] = i * dx;
      ys[i] = Math.sin(xs[i]);
    }

    const tRes = trapz(ys, xs);
    expect(tRes).toBeCloseTo(2.0, 3);

    const sRes = simpson(ys, xs);
    expect(sRes).toBeCloseTo(2.0, 5);

    // Continuous adaptive quadrature.
    const qRes = quad((x) => Math.sin(x), 0, Math.PI);
    expect(qRes).toBeCloseTo(2.0, 6);
  });

  it('computes cumulative integration', () => {
    // Constant acceleration a = 2 -> velocity v(t) = 2t -> at t=2, v=4.
    const a = new NDArray(new Float64Array([2, 2, 2]), { shape: [3] });
    const x = new NDArray(new Float64Array([0, 1, 2]), { shape: [3] });
    const v = cumulativeIntegrate(a, x);

    expect(v.get(0)).toBe(0);
    expect(v.get(1)).toBe(2);
    expect(v.get(2)).toBe(4);

    const irregular = cumulativeIntegrate(
      NDArray.fromArray([1, 3, 5]),
      NDArray.fromArray([0, 1, 3])
    );
    expect(Array.from(irregular.data)).toEqual([0, 2, 10]);
    expect(cumulativeIntegrate(NDArray.zeros([0])).size).toBe(0);
    expect(Array.from(cumulativeIntegrate(NDArray.fromArray([7])).data)).toEqual([0]);
    expect(Array.from(cumulativeIntegrate(a, undefined, 0.5).data)).toEqual([0, 1, 2]);
    expect(Array.from(cumulativeIntegrate(a, NDArray.fromArray([2, 1, 0])).data)).toEqual([0, -2, -4]);

    const stridedY = NDArray.fromArray([99, 1, 99, 3, 99, 5]).slice([1, 6, 2]);
    const stridedX = NDArray.fromArray([99, 0, 99, 1, 99, 3]).slice([1, 6, 2]);
    expect(Array.from(cumulativeIntegrate(stridedY, stridedX).data)).toEqual([0, 2, 10]);
  });
});

describe('ix / interpolate / 1D & Splines', () => {
  it('interpolates linearly and nearest neighbor', () => {
    const x = [0, 10];
    const y = [0, 100];
    const interp = interp1d(x, y, { method: 'linear' });

    expect(interp(5)).toBe(50);
    expect(interp(2.5)).toBe(25);
  });

  it('interpolates with Natural Cubic Spline', () => {
    const x = [0, 1, 2, 3];
    const y = [0, 1, 8, 27]; // y = x^3
    const spline = cubicSpline(x, y);

    // Evaluate at the intermediate point x = 1.5.
    const val = spline(1.5);
    expect(val).toBeCloseTo(3.15, 1);
  });
});

describe('ix / optimize / minimize & curveFit', () => {
  it('minimizes 1D scalar function with fminbnd', () => {
    // f(x) = (x - 3)^2 + 5 -> minimum at x = 3, f(x) = 5.
    const res = fminbnd((x) => (x - 3) ** 2 + 5, [0, 10]);
    expect(res.converged).toBe(true);
    expect(res.x).toBeCloseTo(3.0, 5);
    expect(res.fval).toBeCloseTo(5.0, 5);
  });

  it('minimizes multivariable function with nelderMead', () => {
    // 2D parabola: f(x, y) = (x - 2)^2 + (y + 4)^2.
    const f = (v: NDArray) => (v.get(0) - 2) ** 2 + (v.get(1) + 4) ** 2;
    const res = nelderMead(f, [0, 0]);

    expect(res.converged).toBe(true);
    expect(res.x.get(0)).toBeCloseTo(2.0, 3);
    expect(res.x.get(1)).toBeCloseTo(-4.0, 3);
  });

  it('calibrates models with curveFit', () => {
    // Linear model y = a * x + b with a = 2.5 and b = 1.0.
    const x = [1, 2, 3, 4];
    const y = [3.5, 6.0, 8.5, 11.0];
    const model = (xi: number, p: NDArray) => p.get(0) * xi + p.get(1);

    const fit = curveFit(model, x, y, [1.0, 0.0]);
    expect(fit.x.get(0)).toBeCloseTo(2.5, 2);
    expect(fit.x.get(1)).toBeCloseTo(1.0, 2);
  });
});

describe('ix / plot / SVG generator', () => {
  it('generates lightweight standalone SVG plots', () => {
    const svg = plotSVG([0, 1, 2], [0, 1, 4]);
    expect(svg).toContain('<svg');
    expect(svg).toContain('</svg>');
    expect(svg).toContain('<path');
  });
});

describe('ix / ai / helpers & prompts', () => {
  it('provides documentation and introspection via help()', () => {
    const doc = help('linspace');
    expect(doc).toContain('linspace');
    expect(doc).toContain('Signature:');

    expect(help('sumProduct')).toContain('broadcast');
    expect(help('where')).toContain('broadcast');
    expect(help(NDArray.prototype.toNestedArray)).toContain('strides');
    expect(help('compile')).toContain('reusable evaluator');
    expect(help('unknownTopic')).toContain('toNestedArray');
  });

  it('generates concise tensor summaries for LLMs', () => {
    const A = new NDArray(new Float64Array([1, 2, 3, 4, 5]), { shape: [5] });
    const summ = summary(A);
    expect(summ).toContain('shape=[5]');
    expect(summ).toContain('min=1.0000');
    expect(summ).toContain('max=5.0000');
  });

  it('parses CSV strings to DataFrame', () => {
    const csv = `x,y\n1,10\n2,20\n3,30`;
    const df = fromCSV(csv);
    expect(df.shape).toEqual([3, 2]);
    expect(df.columns).toEqual(['x', 'y']);
    expect(df.col('y').get(1)).toBe(20);
  });

  it('parses compact matrix notation "1 2; 3 4"', () => {
    const mat = fromMatrixString('1 2 3; 4 5 6');
    expect(Array.from(mat.shape)).toEqual([2, 3]);
    expect(mat.get(1, 2)).toBe(6);
  });

  it('provides units catalog via unitsHelp()', () => {
    const uHelp = unitsHelp();
    expect(uHelp).toContain('Physical Unit Catalog');
    expect(uHelp).toContain('meter');
  });
});

describe('ix / units & constants / offsets & custom units', () => {
  it('handles relative thermal scales with offset (Celsius <-> Fahrenheit <-> Kelvin)', () => {
    // 0 °C = 273.15 K.
    const t0C = qty(0, celsius);
    expect(t0C.to(kelvin)).toBeCloseTo(273.15, 2);

    // 100 °C = 212 °F.
    const t100C = qty(100, celsius);
    expect(t100C.to(fahrenheit)).toBeCloseTo(212.0, 2);

    // 68 °F = 20 °C.
    const t68F = qty(68, fahrenheit);
    expect(t68F.to(celsius)).toBeCloseTo(20.0, 2);
  });

  it('allows registering, removing and resetting user-defined units', () => {
    // Define the custom unit "parsec" as 3.0857e16 meters.
    const parsecQty = qty(3.0857e16, meter);
    units.defineUnit('parsec', 'pc', parsecQty, 'Astronomical distance unit parsec');

    const pcUnit = units.get('pc');
    expect(pcUnit).toBeDefined();

    // Remove the custom unit.
    expect(units.removeUnit('pc')).toBe(true);
    expect(units.get('pc')).toBeUndefined();

    // Reset the registry.
    units.defineUnit('furlong', 'fur', qty(201.168, meter));
    units.reset();
    expect(units.get('fur')).toBeUndefined();
  });

  it('contains fundamental physical constants', () => {
    expect(SPEED_OF_LIGHT.value).toBe(299792458);
    expect(STANDARD_GRAVITY.value).toBe(9.80665);
    expect(PLANCK_CONSTANT.value).toBeCloseTo(6.62607015e-34, 40);
  });
});
