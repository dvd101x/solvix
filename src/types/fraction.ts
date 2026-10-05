/**
 * @file fraction.ts
 * Números racionales (fracciones) exactas basadas en BigInt:
 * - Reducción canónica automática mediante Máximo Común Divisor (MCD / GCD euclidiano)
 * - Cero pérdida de precisión por redondeo IEEE-754 (ej. 1/3 + 1/6 = 1/2 exacto)
 * - Conversión bidireccional desde y hacia números decimales
 */

function gcd(a: bigint, b: bigint): bigint {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x;
}

export class Fraction {
  public readonly n: bigint; // Numerador
  public readonly d: bigint; // Denominador (siempre > 0)

  constructor(numerator: bigint | number, denominator: bigint | number = 1n) {
    let n = typeof numerator === 'number' ? BigInt(Math.trunc(numerator)) : numerator;
    let d = typeof denominator === 'number' ? BigInt(Math.trunc(denominator)) : denominator;

    if (d === 0n) {
      throw new RangeError('Denominator cannot be zero in Fraction');
    }

    if (d < 0n) {
      n = -n;
      d = -d;
    }

    const common = gcd(n, d);
    this.n = n / common;
    this.d = d / common;
  }

  // --- Operaciones Aritméticas ---

  public add(other: Fraction | number | bigint): Fraction {
    const b = Fraction.from(other);
    // a/b + c/d = (ad + bc) / bd
    return new Fraction(this.n * b.d + b.n * this.d, this.d * b.d);
  }

  public sub(other: Fraction | number | bigint): Fraction {
    const b = Fraction.from(other);
    return new Fraction(this.n * b.d - b.n * this.d, this.d * b.d);
  }

  public mul(other: Fraction | number | bigint): Fraction {
    const b = Fraction.from(other);
    return new Fraction(this.n * b.n, this.d * b.d);
  }

  public div(other: Fraction | number | bigint): Fraction {
    const b = Fraction.from(other);
    if (b.n === 0n) throw new RangeError('Division by zero in Fraction');
    return new Fraction(this.n * b.d, this.d * b.n);
  }

  public inv(): Fraction {
    if (this.n === 0n) throw new RangeError('Division by zero in Fraction inversion');
    return new Fraction(this.d, this.n);
  }

  public neg(): Fraction {
    return new Fraction(-this.n, this.d);
  }

  public pow(exp: number): Fraction {
    if (exp === 0) return new Fraction(1n, 1n);
    if (exp < 0) return this.inv().pow(-exp);
    const bigExp = BigInt(exp);
    return new Fraction(this.n ** bigExp, this.d ** bigExp);
  }

  // --- Comparación y Conversiones ---

  public toNumber(): number {
    return Number(this.n) / Number(this.d);
  }

  public equals(other: Fraction | number | bigint): boolean {
    const b = Fraction.from(other);
    return this.n === b.n && this.d === b.d;
  }

  public toString(): string {
    if (this.d === 1n) return `${this.n}`;
    return `${this.n}/${this.d}`;
  }

  // --- Constructores auxiliares ---

  public static from(val: Fraction | number | bigint): Fraction {
    if (val instanceof Fraction) return val;
    if (typeof val === 'bigint') return new Fraction(val, 1n);
    return Fraction.fromNumber(val);
  }

  /**
   * Convierte un número en coma flotante a fracción exacta con aproximación continua.
   */
  public static fromNumber(x: number, tolerance = 1e-9): Fraction {
    if (!Number.isFinite(x)) throw new RangeError('Cannot convert NaN or Infinity to Fraction');
    if (Number.isInteger(x)) return new Fraction(BigInt(x), 1n);

    // Algoritmo de fracciones continuas (Farey approximation)
    let h1 = 1n, h2 = 0n;
    let k1 = 0n, k2 = 1n;
    let b = x;

    do {
      const a = Math.floor(b);
      const bigA = BigInt(a);

      let aux = h1;
      h1 = bigA * h1 + h2;
      h2 = aux;

      aux = k1;
      k1 = bigA * k1 + k2;
      k2 = aux;

      b = 1 / (b - a);
    } while (Math.abs(x - Number(h1) / Number(k1)) > x * tolerance && Number(k1) < 1e12);

    return new Fraction(h1, k1);
  }
}

/**
 * Función fábrica cómoda: frac(1, 3) o frac(0.25)
 */
export function frac(n: number | bigint = 0, d: number | bigint = 1): Fraction {
  return new Fraction(n, d);
}
