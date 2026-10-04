/**
 * @file complex.ts
 * Números complejos z = a + bi de alto rendimiento con V8 inlining:
 * - Operaciones aritméticas elementales (+, -, *, /)
 * - Módulo (abs), fase (arg), conjugado (conj)
 * - Exponenciales, logaritmos y raíces cuadradas complejas (sqrt)
 * - Forma fasorial / polar (r, theta)
 */

export class Complex {
  public readonly re: number;
  public readonly im: number;

  constructor(re: number = 0, im: number = 0) {
    this.re = re;
    this.im = im;
  }

  // --- Propiedades y Métricas ---

  /**
   * Módulo o magnitud |z| = sqrt(re^2 + im^2)
   */
  public abs(): number {
    return Math.hypot(this.re, this.im);
  }

  /**
   * Argumento o ángulo de fase en radianes (-pi a pi]
   */
  public arg(): number {
    return Math.atan2(this.im, this.re);
  }

  /**
   * Conjugado complejo z* = a - bi
   */
  public conj(): Complex {
    return new Complex(this.re, -this.im);
  }

  // --- Operaciones Aritméticas ---

  public add(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re + other, this.im);
    }
    return new Complex(this.re + other.re, this.im + other.im);
  }

  public sub(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re - other, this.im);
    }
    return new Complex(this.re - other.re, this.im - other.im);
  }

  public mul(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re * other, this.im * other);
    }
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    return new Complex(
      this.re * other.re - this.im * other.im,
      this.re * other.im + this.im * other.re
    );
  }

  public div(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re / other, this.im / other);
    }
    // (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c^2 + d^2)
    const c = other.re;
    const d = other.im;
    const denom = c * c + d * d;
    if (denom === 0) {
      return new Complex(Infinity, Infinity);
    }
    return new Complex(
      (this.re * c + this.im * d) / denom,
      (this.im * c - this.re * d) / denom
    );
  }

  // --- Funciones Trascendentes Complejas ---

  /**
   * Raíz cuadrada principal compleja: sqrt(z)
   * Resuelve el caso de números negativos reales: sqrt(-4) = 2i
   */
  public sqrt(): Complex {
    const r = this.abs();
    if (r === 0) return new Complex(0, 0);

    const re = Math.sqrt((r + this.re) / 2);
    const im = Math.sign(this.im === 0 ? 1 : this.im) * Math.sqrt((r - this.re) / 2);
    return new Complex(re, im);
  }

  /**
   * Exponencial compleja e^z = e^re * (cos(im) + i*sin(im)) (Fórmula de Euler)
   */
  public exp(): Complex {
    const expRe = Math.exp(this.re);
    return new Complex(expRe * Math.cos(this.im), expRe * Math.sin(this.im));
  }

  /**
   * Logaritmo natural complejo ln(z) = ln(|z|) + i*arg(z)
   */
  public log(): Complex {
    return new Complex(Math.log(this.abs()), this.arg());
  }

  /**
   * Potencia compleja z^w
   */
  public pow(exponent: Complex | number): Complex {
    const w = typeof exponent === 'number' ? new Complex(exponent, 0) : exponent;
    // z^w = exp(w * log(z))
    return this.log().mul(w).exp();
  }

  public toString(): string {
    if (this.im === 0) return `${this.re}`;
    if (this.re === 0) return `${this.im}i`;
    const sign = this.im < 0 ? '-' : '+';
    return `${this.re} ${sign} ${Math.abs(this.im)}i`;
  }

  // --- Constructores auxiliares ---

  public static fromPolar(r: number, theta: number): Complex {
    return new Complex(r * Math.cos(theta), r * Math.sin(theta));
  }

  public static i(): Complex {
    return new Complex(0, 1);
  }
}

/**
 * Función fábrica cómoda: complex(3, 4) o complex(5)
 */
export function complex(re: number = 0, im: number = 0): Complex {
  return new Complex(re, im);
}
