/**
 * @file units.ts
 * Extensible system for physical units, dimensional quantities (SI and imperial),
 * and scales with relative offsets (Celsius, Fahrenheit):
 * - Dynamic registration of user-defined units
 * - Ability to remove units or reset the registry to its canonical state
 * - Bidirectional conversion with offsets (T_C, T_F, T_K)
 */

export interface Dimensions {
  m: number;   // Length (meter)
  kg: number;  // Mass (kilogram)
  s: number;   // Time (second)
  A: number;   // Electric current (ampere)
  K: number;   // Temperature (kelvin)
  mol: number; // Amount of substance (mole)
  cd: number;  // Luminous intensity (candela)
}

export class Quantity {
  public readonly value: number;
  public readonly dims: Dimensions;
  public readonly offset: number; // Offset for relative temperature scales (Celsius, Fahrenheit)

  constructor(value: number, dims: Partial<Dimensions> = {}, offset: number = 0) {
    this.value = value;
    this.dims = {
      m: dims.m || 0,
      kg: dims.kg || 0,
      s: dims.s || 0,
      A: dims.A || 0,
      K: dims.K || 0,
      mol: dims.mol || 0,
      cd: dims.cd || 0,
    };
    this.offset = offset;
  }

  public isDimensionless(): boolean {
    return Object.values(this.dims).every((d) => d === 0);
  }

  public hasSameDimensions(other: Quantity): boolean {
    return (
      this.dims.m === other.dims.m &&
      this.dims.kg === other.dims.kg &&
      this.dims.s === other.dims.s &&
      this.dims.A === other.dims.A &&
      this.dims.K === other.dims.K &&
      this.dims.mol === other.dims.mol &&
      this.dims.cd === other.dims.cd
    );
  }

  public getBaseValue(): number {
    return this.value * (1) + this.offset;
  }

  public add(other: Quantity | number): Quantity {
    if (typeof other === 'number') {
      if (!this.isDimensionless()) {
        throw new TypeError(`Cannot add scalar number to dimensioned quantity [${this.formatDimensions()}]`);
      }
      return new Quantity(this.value + other, this.dims);
    }
    if (!this.hasSameDimensions(other)) {
      throw new TypeError(
        `Dimensional mismatch in addition: [${this.formatDimensions()}] vs [${other.formatDimensions()}]`
      );
    }
    return new Quantity(this.value + other.value, this.dims);
  }

  public sub(other: Quantity | number): Quantity {
    if (typeof other === 'number') {
      if (!this.isDimensionless()) {
        throw new TypeError(`Cannot subtract scalar from dimensioned quantity [${this.formatDimensions()}]`);
      }
      return new Quantity(this.value - other, this.dims);
    }
    if (!this.hasSameDimensions(other)) {
      throw new TypeError(
        `Dimensional mismatch in subtraction: [${this.formatDimensions()}] vs [${other.formatDimensions()}]`
      );
    }
    return new Quantity(this.value - other.value, this.dims);
  }

  public mul(other: Quantity | number): Quantity {
    if (typeof other === 'number') {
      return new Quantity(this.value * other, this.dims);
    }
    return new Quantity(this.value * other.value, {
      m: this.dims.m + other.dims.m,
      kg: this.dims.kg + other.dims.kg,
      s: this.dims.s + other.dims.s,
      A: this.dims.A + other.dims.A,
      K: this.dims.K + other.dims.K,
      mol: this.dims.mol + other.dims.mol,
      cd: this.dims.cd + other.dims.cd,
    });
  }

  public div(other: Quantity | number): Quantity {
    if (typeof other === 'number') {
      return new Quantity(this.value / other, this.dims);
    }
    return new Quantity(this.value / other.value, {
      m: this.dims.m - other.dims.m,
      kg: this.dims.kg - other.dims.kg,
      s: this.dims.s - other.dims.s,
      A: this.dims.A - other.dims.A,
      K: this.dims.K - other.dims.K,
      mol: this.dims.mol - other.dims.mol,
      cd: this.dims.cd - other.dims.cd,
    });
  }

  public pow(exp: number): Quantity {
    return new Quantity(Math.pow(this.value, exp), {
      m: this.dims.m * exp,
      kg: this.dims.kg * exp,
      s: this.dims.s * exp,
      A: this.dims.A * exp,
      K: this.dims.K * exp,
      mol: this.dims.mol * exp,
      cd: this.dims.cd * exp,
    });
  }

  public to(targetUnit: Quantity): number {
    if (!this.hasSameDimensions(targetUnit)) {
      throw new TypeError(
        `Cannot convert [${this.formatDimensions()}] to [${targetUnit.formatDimensions()}]: incompatible dimensions`
      );
    }
    // Handle units with relative offsets (temperatures such as degC and degF).
    const baseVal = this.value * 1 + this.offset;
    return (baseVal - targetUnit.offset) / targetUnit.value;
  }

  public formatDimensions(): string {
    const parts: string[] = [];
    if (this.dims.kg) parts.push(`kg^${this.dims.kg}`);
    if (this.dims.m) parts.push(`m^${this.dims.m}`);
    if (this.dims.s) parts.push(`s^${this.dims.s}`);
    if (this.dims.A) parts.push(`A^${this.dims.A}`);
    if (this.dims.K) parts.push(`K^${this.dims.K}`);
    if (this.dims.mol) parts.push(`mol^${this.dims.mol}`);
    if (this.dims.cd) parts.push(`cd^${this.dims.cd}`);
    return parts.length ? parts.join('*') : 'dimensionless';
  }

  public toString(): string {
    return `${this.value} [${this.formatDimensions()}]`;
  }
}

// --- Canonical Base Units (SI and Imperial) ---

export const meter = new Quantity(1, { m: 1 });
export const kilometer = new Quantity(1000, { m: 1 });
export const centimeter = new Quantity(0.01, { m: 1 });
export const millimeter = new Quantity(0.001, { m: 1 });
export const micrometer = new Quantity(1e-6, { m: 1 });
export const nanometer = new Quantity(1e-9, { m: 1 });
export const inch = new Quantity(0.0254, { m: 1 });
export const foot = new Quantity(0.3048, { m: 1 });
export const yard = new Quantity(0.9144, { m: 1 });
export const mile = new Quantity(1609.344, { m: 1 });

export const second = new Quantity(1, { s: 1 });
export const millisecond = new Quantity(0.001, { s: 1 });
export const microsecond = new Quantity(1e-6, { s: 1 });
export const minute = new Quantity(60, { s: 1 });
export const hour = new Quantity(3600, { s: 1 });
export const day = new Quantity(86400, { s: 1 });

export const kilogram = new Quantity(1, { kg: 1 });
export const gram = new Quantity(0.001, { kg: 1 });
export const milligram = new Quantity(1e-6, { kg: 1 });
export const tonne = new Quantity(1000, { kg: 1 });
export const pound = new Quantity(0.45359237, { kg: 1 });
export const ounce = new Quantity(0.028349523125, { kg: 1 });

export const ampere = new Quantity(1, { A: 1 });
export const kelvin = new Quantity(1, { K: 1 });
export const mol = new Quantity(1, { mol: 1 });
export const candela = new Quantity(1, { cd: 1 });

// Temperature units with offsets (T_K = T_C + 273.15, T_K = (T_F + 459.67) * 5/9).
export const celsius = new Quantity(1, { K: 1 }, 273.15);
export const fahrenheit = new Quantity(5 / 9, { K: 1 }, 255.3722222222222);

// Derived units
export const newton = kilogram.mul(meter).div(second.pow(2));
export const joule = newton.mul(meter);
export const watt = joule.div(second);
export const pascal = newton.div(meter.pow(2));
export const bar = new Quantity(1e5, { kg: 1, m: -1, s: -2 });
export const psi = new Quantity(6894.757, { kg: 1, m: -1, s: -2 });
export const atmosphere = new Quantity(101325, { kg: 1, m: -1, s: -2 });

export const volt = watt.div(ampere);
export const ohm = volt.div(ampere);
export const coulomb = ampere.mul(second);
export const farad = coulomb.div(volt);
export const henry = volt.mul(second).div(ampere);
export const tesla = volt.mul(second).div(meter.pow(2));

export const liter = new Quantity(0.001, { m: 3 });
export const gallon = new Quantity(0.00378541, { m: 3 });
export const kmh = kilometer.div(hour);
export const mph = new Quantity(0.44704, { m: 1, s: -1 });

export function qty(value: number, base: Quantity): Quantity {
  return new Quantity(value * base.value, base.dims, base.offset);
}

// --- Extensible User-Defined Unit Registry ---

export interface UnitRegistryEntry {
  name: string;
  symbol: string;
  quantity: Quantity;
  description?: string;
  isUserDefined?: boolean;
}

class UnitSystemRegistry {
  private registry: Map<string, UnitRegistryEntry> = new Map();
  private canonicalKeys: Set<string> = new Set();

  constructor() {
    this.registerCanonicalDefaults();
  }

  private registerCanonicalDefaults() {
    const defaults: [string, string, Quantity, string][] = [
      ['meter', 'm', meter, 'Longitud base SI'],
      ['kilometer', 'km', kilometer, '1000 metros'],
      ['centimeter', 'cm', centimeter, '0.01 metros'],
      ['millimeter', 'mm', millimeter, '0.001 metros'],
      ['inch', 'in', inch, 'Pulgada (0.0254 m)'],
      ['foot', 'ft', foot, 'Pie (0.3048 m)'],
      ['mile', 'mi', mile, 'Milla (1609.344 m)'],
      ['second', 's', second, 'Tiempo base SI'],
      ['millisecond', 'ms', millisecond, '0.001 segundos'],
      ['minute', 'min', minute, '60 segundos'],
      ['hour', 'h', hour, '3600 segundos'],
      ['day', 'd', day, '86400 segundos'],
      ['kilogram', 'kg', kilogram, 'Masa base SI'],
      ['gram', 'g', gram, '0.001 kg'],
      ['pound', 'lb', pound, 'Libra avoirdupois (0.45359 kg)'],
      ['ounce', 'oz', ounce, 'Onza (28.349 g)'],
      ['kelvin', 'K', kelvin, 'Temperatura termodinámica base SI'],
      ['celsius', 'degC', celsius, 'Grados Celsius (offset 273.15 K)'],
      ['fahrenheit', 'degF', fahrenheit, 'Grados Fahrenheit'],
      ['ampere', 'A', ampere, 'Corriente eléctrica SI'],
      ['newton', 'N', newton, 'Fuerza (kg*m/s^2)'],
      ['joule', 'J', joule, 'Energía / Trabajo (N*m)'],
      ['watt', 'W', watt, 'Potencia (J/s)'],
      ['pascal', 'Pa', pascal, 'Presión (N/m^2)'],
      ['bar', 'bar', bar, 'Presión (100 kPa)'],
      ['psi', 'psi', psi, 'Presión libra por pulgada cuadrada'],
      ['atmosphere', 'atm', atmosphere, 'Presión atmosférica estándar (101.325 kPa)'],
      ['volt', 'V', volt, 'Potencial eléctrico (W/A)'],
      ['ohm', 'ohm', ohm, 'Resistencia eléctrica (V/A)'],
      ['farad', 'F', farad, 'Capacitancia (C/V)'],
      ['liter', 'L', liter, 'Volumen (0.001 m^3)'],
      ['gallon', 'gal', gallon, 'Galón estadounidense líquido (3.785 L)'],
      ['kmh', 'km/h', kmh, 'Velocidad kilómetros por hora'],
      ['mph', 'mph', mph, 'Velocidad millas por hora'],
    ];

    for (const [name, symbol, quantity, desc] of defaults) {
      this.registry.set(name.toLowerCase(), { name, symbol, quantity, description: desc, isUserDefined: false });
      this.registry.set(symbol.toLowerCase(), { name, symbol, quantity, description: desc, isUserDefined: false });
      this.canonicalKeys.add(name.toLowerCase());
      this.canonicalKeys.add(symbol.toLowerCase());
    }
  }

  /**
   * Registers a new user-defined unit.
   */
  public defineUnit(name: string, symbol: string, quantity: Quantity, description?: string): void {
    const entry: UnitRegistryEntry = { name, symbol, quantity, description, isUserDefined: true };
    this.registry.set(name.toLowerCase(), entry);
    this.registry.set(symbol.toLowerCase(), entry);
  }

  /**
   * Removes a specific registered user-defined unit.
   */
  public removeUnit(nameOrSymbol: string): boolean {
    const key = nameOrSymbol.toLowerCase();
    const entry = this.registry.get(key);
    if (!entry) return false;
    if (!entry.isUserDefined) {
      throw new Error(`Cannot delete canonical standard unit "${nameOrSymbol}"`);
    }
    this.registry.delete(entry.name.toLowerCase());
    this.registry.delete(entry.symbol.toLowerCase());
    return true;
  }

  /**
   * Resets the registry to its standard defaults, removing all user-defined units.
   */
  public reset(): void {
    for (const [key, entry] of this.registry.entries()) {
      if (entry.isUserDefined) {
        this.registry.delete(key);
      }
    }
  }

  /**
   * Looks up a unit by name or symbol.
   */
  public get(nameOrSymbol: string): Quantity | undefined {
    return this.registry.get(nameOrSymbol.toLowerCase())?.quantity;
  }

  /**
   * Lists all registered units.
   */
  public listUnits(): UnitRegistryEntry[] {
    const seen = new Set<string>();
    const list: UnitRegistryEntry[] = [];
    for (const entry of this.registry.values()) {
      if (!seen.has(entry.name)) {
        seen.add(entry.name);
        list.push(entry);
      }
    }
    return list;
  }
}

export const units = new UnitSystemRegistry();
