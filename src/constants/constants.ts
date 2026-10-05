/**
 * @file constants.ts
 * Constantes físicas fundamentales (CODATA) integradas con el sistema de unidades y magnitudes exactas.
 */
import {
  meter,
  second,
  kilogram,
  ampere,
  kelvin,
  mol,
  joule,
  Quantity,
  qty,
} from '../units/units.js';

/** Velocidad de la luz en el vacío c = 299,792,458 m/s */
export const SPEED_OF_LIGHT = qty(299792458, meter.div(second));

/** Aceleración de la gravedad estándar terrestre g = 9.80665 m/s^2 */
export const STANDARD_GRAVITY = qty(9.80665, meter.div(second.pow(2)));

/** Constante de gravitación universal G = 6.67430e-11 m^3 / (kg * s^2) */
export const GRAVITATIONAL_CONSTANT = new Quantity(6.6743e-11, { m: 3, kg: -1, s: -2 });

/** Constante de Planck h = 6.62607015e-34 J * s */
export const PLANCK_CONSTANT = qty(6.62607015e-34, joule.mul(second));

/** Constante de Planck reducida ħ = h / (2*pi) */
export const REDUCED_PLANCK_CONSTANT = qty(6.62607015e-34 / (2 * Math.PI), joule.mul(second));

/** Constante de Boltzmann k_B = 1.380649e-23 J / K */
export const BOLTZMANN_CONSTANT = qty(1.380649e-23, joule.div(kelvin));

/** Número de Avogadro N_A = 6.02214076e23 mol^-1 */
export const AVOGADRO_NUMBER = new Quantity(6.02214076e23, { mol: -1 });

/** Carga elemental del electrón e = 1.602176634e-19 C (A * s) */
export const ELEMENTARY_CHARGE = new Quantity(1.602176634e-19, { A: 1, s: 1 });

/** Masa en reposo del electrón m_e = 9.1093837015e-31 kg */
export const ELECTRON_MASS = qty(9.1093837015e-31, kilogram);

/** Masa en reposo del protón m_p = 1.67262192369e-27 kg */
export const PROTON_MASS = qty(1.67262192369e-27, kilogram);

/** Constante universal de los gases R = N_A * k_B = 8.314462618 J / (mol * K) */
export const GAS_CONSTANT = qty(8.314462618, joule.div(mol.mul(kelvin)));

/** Constante de Stefan-Boltzmann sigma = 5.670374419e-8 W / (m^2 * K^4) */
export const STEFAN_BOLTZMANN_CONSTANT = new Quantity(5.670374419e-8, { kg: 1, s: -3, K: -4 });

/** Permitividad eléctrica del vacío epsilon_0 = 8.8541878128e-12 F/m */
export const VACUUM_PERMITTIVITY = new Quantity(8.8541878128e-12, { A: 2, s: 4, kg: -1, m: -3 });

/** Permeabilidad magnética del vacío mu_0 = 1.25663706212e-6 N / A^2 */
export const VACUUM_PERMEABILITY = new Quantity(1.25663706212e-6, { kg: 1, m: 1, s: -2, A: -2 });
