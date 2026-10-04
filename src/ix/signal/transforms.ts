/**
 * @file transforms.ts
 * Procesamiento digital de señales (DSP) en TypeScript puro (Zero-Wasm):
 * - FFT 1D y IFFT mediante algoritmo Cooley-Tukey Radix-2
 * - Convolución discreta 1D (convolve)
 * - Filtro IIR Pasa-Bajas / Pasa-Altas de 1er orden
 * - Generadores de señales (sinewave, squarewave, sawtooth, chirp)
 */
import { NDArray } from '../core/ndarray.js';
import { Complex, complex } from '../types/complex.js';

export interface FFTResult {
  real: Float64Array;
  imag: Float64Array;
  frequencies: Float64Array; // Eje de frecuencias en Hz
  magnitude: Float64Array;   // Espectro de amplitud |X(f)|
}

/**
 * FFT 1D Cooley-Tukey Radix-2 (Decimation-in-time)
 */
export function fft(signal: NDArray | Float64Array | number[], sampleRate = 1.0): FFTResult {
  const inData = signal instanceof NDArray ? signal.data : new Float64Array(signal);
  const origN = inData.length;

  // Rellenar con ceros hasta la siguiente potencia de 2 (Zero-padding)
  let n = 1;
  while (n < origN) n <<= 1;

  const real = new Float64Array(n);
  const imag = new Float64Array(n);
  for (let i = 0; i < origN; i++) real[i] = inData[i];

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tempR = real[i]; real[i] = real[j]; real[j] = tempR;
      const tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Iteraciones de mariposa (Butterfly stages)
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wR = 1.0;
      let wI = 0.0;
      for (let k = 0; k < half; k++) {
        const uR = real[i + k];
        const uI = imag[i + k];
        const vR = real[i + k + half] * wR - imag[i + k + half] * wI;
        const vI = real[i + k + half] * wI + imag[i + k + half] * wR;

        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;
        real[i + k + half] = uR - vR;
        imag[i + k + half] = uI - vI;

        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }

  // Espectro de magnitud y vector de frecuencias
  const numFreqs = (n >> 1) + 1;
  const mag = new Float64Array(numFreqs);
  const freqs = new Float64Array(numFreqs);
  const freqStep = sampleRate / n;

  for (let i = 0; i < numFreqs; i++) {
    const r = real[i];
    const im = imag[i];
    mag[i] = Math.sqrt(r * r + im * im) / (i === 0 ? n : n / 2);
    freqs[i] = i * freqStep;
  }

  return {
    real,
    imag,
    frequencies: freqs,
    magnitude: mag,
  };
}

/**
 * Convolución discreta lineal 1D: (x * h)[n]
 */
export function convolve(x: NDArray | Float64Array, h: NDArray | Float64Array): NDArray {
  const xData = x instanceof NDArray ? x.data : x;
  const hData = h instanceof NDArray ? h.data : h;
  const nx = xData.length;
  const nh = hData.length;
  const outLen = nx + nh - 1;
  const out = new Float64Array(outLen);

  for (let i = 0; i < nx; i++) {
    const xi = xData[i];
    for (let j = 0; j < nh; j++) {
      out[i + j] += xi * hData[j];
    }
  }

  return new NDArray(out, { shape: [outLen] });
}

/**
 * Filtro paso-bajas IIR de 1er orden en tiempo discreto: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
 */
export function lowpassFilter(signal: NDArray, cutoffFreq: number, sampleRate: number): NDArray {
  const dt = 1 / sampleRate;
  const rc = 1 / (2 * Math.PI * cutoffFreq);
  const alpha = dt / (rc + dt);

  const n = signal.size;
  const out = new Float64Array(n);
  const data = signal.data;

  if (n === 0) return new NDArray(out, { shape: [0] });
  out[0] = data[0];
  for (let i = 1; i < n; i++) {
    out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
  }

  return new NDArray(out, { shape: [n] });
}

/**
 * Generador de onda senoidal pura: A * sin(2*pi*f*t + phase)
 */
export function sinewave(freq: number, duration: number, sampleRate = 1000, amp = 1.0): NDArray {
  const numSamples = Math.floor(duration * sampleRate);
  const out = new Float64Array(numSamples);
  const dt = 1 / sampleRate;
  const omega = 2 * Math.PI * freq;

  for (let i = 0; i < numSamples; i++) {
    out[i] = amp * Math.sin(omega * (i * dt));
  }

  return new NDArray(out, { shape: [numSamples] });
}

/**
 * Generador de barrido de frecuencia lineal (Chirp signal) de f0 a f1 en 'duration'
 */
export function chirp(f0: number, f1: number, duration: number, sampleRate = 1000): NDArray {
  const numSamples = Math.floor(duration * sampleRate);
  const out = new Float64Array(numSamples);
  const dt = 1 / sampleRate;
  const k = (f1 - f0) / duration;

  for (let i = 0; i < numSamples; i++) {
    const t = i * dt;
    const phase = 2 * Math.PI * (f0 * t + 0.5 * k * t * t);
    out[i] = Math.sin(phase);
  }

  return new NDArray(out, { shape: [numSamples] });
}
