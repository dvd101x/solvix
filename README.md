# `ix` (julian-js) — High-Performance Scientific Computing for Modern JavaScript / TypeScript

> **`ix`** es una librería de computación científica inspirada en el modelo de ejecución de **Julia**, la ergonomía matricial de **NumPy** y la velocidad de ejecución de **V8 (TurboFan/Maglev)**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Vitest](https://img.shields.io/badge/Tested%20with-Vitest-yellow.svg)](https://vitest.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)

---

## ⚡ Filosofía de Diseño

1. **Priorización de API**: `Python / NumPy ➔ Julia ➔ MATLAB / Octave`.
2. **Zero-Allocation**: Algoritmos de indexación desenrollados (*loop-unrolled*) para dimensiones $1\text{D}$ a $6\text{D}$, minimizando la presión sobre el Garbage Collector.
3. **Monomorfismo e Inlining en V8**: Métodos de acceso directo a buffers planos (`TypedArray`) que evitan deoptimizaciones (*megamorphic call sites*).
4. **Despacho Múltiple Dinámico (Multiple Dispatch)**: Sistema polimórfico al estilo Julia con caché de firmas para ejecución ultrarrápida en hot paths.
5. **Slicing Lazy sin Copia**: Creación de vistas (*strided views*) compartiendo el mismo `ArrayBuffer`.
6. **Paralelismo Real en Workers**: Soporte para `SharedArrayBuffer` zero-copy, transferencia de tensores y scopes matemáticos evaluables remotamente.
7. **Grafos de Cómputo (DAG)**: Representación de expresiones con resolución topológica de dependencias, memoización y detección de ciclos.

---

## 📁 Arquitectura del Repositorio

```
supermath/
├── src/
│   ├── ix/
│   │   ├── core/
│   │   │   ├── strides.ts            # Funciones unrolled sub2ind (1D-6D) y layouts C/Fortran
│   │   │   ├── ndarray.ts            # NDArray primordial con get/set inlined y lazy slicing
│   │   │   ├── nested-array.ts       # NestedArray nativo sin conversiones forzadas a typed buffers
│   │   │   ├── dispatcher.ts         # Motor de Multiple Dispatch estilo Julia con caché
│   │   │   └── index.ts              # Re-export centralizado del Core
│   │   ├── ops/
│   │   │   └── math-ops.ts           # Operaciones element-wise con broadcasting multidimensional
│   │   ├── linalg/
│   │   │   └── factorizations.ts     # LU con pivoteo, QR Householder, SVD Jacobi, solve, inv, det
│   │   ├── parser/
│   │   │   └── parser.ts             # Parser de expresiones matemáticas (AST, Shunting-Yard, compile)
│   │   ├── indexing/
│   │   │   └── advanced-indexing.ts  # Boolean masking (A[mask]), fancy take/put y elipsis (...)
│   │   ├── parallel/
│   │   │   ├── shared-memory.ts      # SharedArrayBuffer, chunk partitioning y array transfer
│   │   │   └── expression-worker.ts  # Serialización y evaluación de scopes matemáticos en workers
│   │   ├── dag/
│   │   │   └── dag.ts                # Grafo de expresiones matemáticas (DAG, Kahn's algorithm)
│   │   └── index.ts                  # Entrypoint público de ix
│   └── index.ts
├── tests/
│   └── ix/
│       ├── ix.test.ts                # Pruebas del Core y operaciones básicas
│       ├── nested.test.ts            # Pruebas de NestedArray
│       ├── advanced.test.ts          # Pruebas de indexación, parallel workers y DAG
│       └── linalg-parser.test.ts     # Pruebas de LU/QR/SVD/solve y Parser AST
├── benchmarks/
│   └── bench-ix.ts                   # Micro-benchmarks con Mitata
├── package.json
├── tsconfig.json
└── README.md
```

---

## 🛠️ Nuevas Características

### 1. Indexación Avanzada (NumPy / Julia Style)

#### A. Máscara Booleana (`booleanMask` y `putMask`)
```typescript
import { NDArray, booleanMask, putMask } from './src/ix/index.js';

const A = new NDArray(new Float64Array([1, -2, 3, -4, 5]), { shape: [5] });
const mask = [true, false, true, false, true];

// Filtrado (A[mask]) -> [1, 3, 5]
const pos = booleanMask(A, mask);

// Asignación in-place por máscara
putMask(A, mask, 0); // [0, -2, 0, -4, 0]
```

#### B. Fancy Indexing por Ejes (`take`)
```typescript
import { NDArray, take } from './src/ix/index.js';

const matrix = new NDArray(new Float64Array([10, 11, 20, 21, 30, 31]), { shape: [3, 2] });
// Extraer filas 2 y 0
const sub = take(matrix, [2, 0], 0);
```

#### C. Slicing con Elipsis (`...`)
```typescript
import { NDArray, sliceWithEllipsis, ELLIPSIS } from './src/ix/index.js';

const tensor = new NDArray(data, { shape: [4, 5, 6, 2] });
// tensor[..., 1] -> extrae la coordenada 1 en el último eje expandiendo dimensiones intermedias
const slice = sliceWithEllipsis(tensor, ELLIPSIS, 1);
```

---

### 2. Paralelismo y Concurrencia con Workers

#### A. Memoria Compartida (`SharedArrayBuffer`)
Permite a múltiples hilos de Workers leer y escribir en la misma memoria sin serialización ni copias:
```typescript
import { createSharedNDArray, partitionWork } from './src/ix/index.js';

// Tensor accesible en paralelo
const sharedTensor = createSharedNDArray([1000, 1000]);

// Particionar el trabajo para 4 workers
const chunks = partitionWork(1_000_000, 4);
// chunks = [{start: 0, end: 250000}, {start: 250000, end: 500000}, ...]
```

#### B. Transferencia Zero-Copy de Arrays Estándar
```typescript
import { prepareTransfer, reconstructFromTransfer } from './src/ix/index.js';

const A = NDArray.zeros([500, 500]);
const { message, transferables } = prepareTransfer(A);

// Enviar al worker transfiriendo la posesión del ArrayBuffer (Zero-Copy)
worker.postMessage(message, transferables);
```

#### C. Expresiones Matemáticas y Scopes Remotos
Envía una fórmula matemática en string junto con su contexto de variables:
```typescript
import { serializeScope, evaluateExpressionInScope } from './src/ix/index.js';

// En el hilo principal:
const scope = serializeScope({
  mass: 10.5,
  velocity: 20.0
});

// En el Worker:
const energy = evaluateExpressionInScope('0.5 * mass * pow(velocity, 2)', scope);
```

---

### 3. Grafos de Cómputo Matemático (Expression DAG)

Compón fórmulas complejas y tensores en un grafo acíclico dirigido con:
- **Ordenamiento Topológico (Algoritmo de Kahn)**.
- **Detección automática de ciclos**.
- **Evaluación perezosa memoizada**.
- **Visualización Mermaid automática**.

```typescript
import { ExpressionDAG, NDArray, add } from './src/ix/index.js';

const dag = new ExpressionDAG();

// Definir variables
dag.variable('A', new NDArray(new Float64Array([1, 2]), { shape: [2] }));
dag.variable('B', new NDArray(new Float64Array([10, 20]), { shape: [2] }));

// Definir operaciones dependientes
dag.op('C', ['A', 'B'], (a, b) => add(a, b));
dag.op('D', ['C'], (c) => add(c, 100));

// Evaluar bajo demanda
const result = dag.evaluate('D'); // [111, 122]

// Invalida automáticamente sólo las ramas afectadas al cambiar una variable
dag.setVariable('A', new NDArray(new Float64Array([0, 0]), { shape: [2] }));
const updated = dag.evaluate('D'); // [110, 120]

// Exportar diagrama Mermaid
console.log(dag.toMermaid());
```

---

### 4. Álgebra Lineal Numérica (Factorizaciones & Solvers)

Implementación pura de alto rendimiento sobre `Float64Array`:

```typescript
import { NDArray, solve, lu, qr, svd, inv, det } from './src/ix/index.js';

const A = new NDArray(new Float64Array([4, 3, 6, 3]), { shape: [2, 2] });
const b = new Float64Array([10, 12]);

// 1. Solución de Sistemas Lineales Ax = b (estilo A \ b de MATLAB/Julia)
const x = solve(A, b); // [1.0, 2.0]

// 2. Determinante e Inversa
console.log(det(A));   // -6.0
const invA = inv(A);   // Matriz inversa exacta

// 3. Descomposición LU con pivoteo parcial (PA = LU)
const { L, U, P } = lu(A);

// 4. Descomposición QR (Householder Reflections)
const { Q, R } = qr(A);

// 5. Descomposición en Valores Singulares (SVD: A = U * S * V^T)
const { U: uVec, S: sVals, V: vVec } = svd(A);
```

---

### 5. Parser Matemático Nativo & Compilador AST

Tokenizador y parser basado en el algoritmo **Shunting-Yard (Dijkstra)** con árbol de sintaxis abstracta (**AST**) nativo (sin dependencias externas):

```typescript
import { evaluate, compile, parseExpression } from './src/ix/index.js';

// Evaluación inmediata respetando precedencia y asociatividad estándar
const val = evaluate('3 + 4 * 2 / (1 - 5)^2'); // 3.5

// Evaluación con variables y funciones en Scope
const res = evaluate('sin(pi / 2) + sqrt(x^2 + y^2)', { x: 3, y: 4 }); // 6.0

// Compilación a función de alto rendimiento reutilizable
const kineticEnergy = compile('0.5 * m * v^2');
console.log(kineticEnergy({ m: 10, v: 20 })); // 2000
```

---

### 6. Solvers de Ecuaciones Diferenciales (ODE45 & RK4)

Resolución de problemas de valor inicial $\frac{dy}{dt} = f(t, y)$:
- **RK4**: Runge-Kutta clásico de 4to orden de paso fijo.
- **ODE45**: Método embebido Dormand-Prince 5(4) con control adaptativo de paso (estilo MATLAB `ode45` y Julia `DifferentialEquations.jl`).

```typescript
import { NDArray, ode45 } from './src/ix/index.js';

// Oscilador armónico simple: y'' + y = 0  =>  y1' = y2, y2' = -y1
const f = (t: number, y: NDArray) => {
  return new NDArray(new Float64Array([y.data[1], -y.data[0]]), { shape: [2] });
};

const y0 = new NDArray(new Float64Array([0.0, 1.0]), { shape: [2] });
const sol = ode45(f, [0, Math.PI / 2], y0, { rtol: 1e-5 });

console.log(sol.y[sol.y.length - 1].data); // [1.0, 0.0] -> [sin(pi/2), cos(pi/2)]
```

---

### 7. Búsqueda de Raíces y Sistemas No Lineales (`fzero` & `fsolve`)

- **`fzero`**: Algoritmo de Brent híbrido para raíces escalares $f(x) = 0$ con convergencia superlineal.
- **`fsolve`**: Algoritmo multidimensional de Newton-Raphson amortiguado con Jacobiano numérico para sistemas acoplados $F(x) = 0$.

```typescript
import { fzero, fsolve, NDArray } from './src/ix/index.js';

// Raíz escalar: x^2 - 2 = 0
const { root } = fzero((x) => x * x - 2, [1, 2]);
console.log(root); // 1.41421356 (sqrt(2))

// Sistema no lineal: intersección entre círculo y diagonal
const F = (x: NDArray) => new NDArray(new Float64Array([
  x.data[0]**2 + x.data[1]**2 - 1,
  x.data[0] - x.data[1]
]), { shape: [2] });

const { x } = fsolve(F, [0.5, 0.5]);
console.log(x.data); // [0.7071, 0.7071]
```

---

### 8. Sistema de Unidades Físicas y Análisis Dimensional (SI)

Garantiza coherencia dimensional estricta en tiempo de ejecución:

```typescript
import { qty, meter, kilometer, second, hour, kilogram, newton, joule } from './src/ix/index.js';

const distancia = qty(5, meter);
const tiempo = qty(2, second);

// 1. Detección automática de inconsistencias dimensionales
// distancia.add(tiempo); // Throws TypeError: Dimensional mismatch [m^1] vs [s^1]

// 2. Composición de magnitudes derivadas
const velocidad = distancia.div(tiempo); // 2.5 m/s

// 3. Conversiones exactas
const speedKmh = qty(72, kilometer).div(qty(1, hour));
console.log(speedKmh.to(meter.div(second))); // 20.0 m/s

// 4. Dinámica y Energía
const masa = qty(10, kilogram);
const aceleracion = qty(9.8, meter).div(qty(1, second).pow(2));
const fuerza = masa.mul(aceleracion); // 98 N (kg*m/s^2)
const trabajo = fuerza.mul(distancia); // 490 J (kg*m^2/s^2)
```

---

### 9. Operaciones In-Place (`!`) y Memoria Reciclable

Elimina la recolección de basura (*Garbage Collection pauses*) en bucles de alta frecuencia:

```typescript
import { NDArray, addInPlace, mulInPlace } from './src/ix/index.js';

const A = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });
const B = new NDArray(new Float64Array([10, 20, 30]), { shape: [3] });
const out = new NDArray(new Float64Array(3), { shape: [3] });

// Modifica directamente el buffer 'out' sin alocar memoria
addInPlace(out, A, B); // out = [11, 22, 33]
mulInPlace(out, out, 2); // out = [22, 44, 66]
```

---

### 10. Generadores Numéricos y Mallas (`linspace`, `arange`, `meshgrid`, `eye`)

```typescript
import { linspace, arange, logspace, eye, meshgrid } from './src/ix/index.js';

// Espacio lineal y sembrado de rangos
const lin = linspace(0, 10, 5); // [0, 2.5, 5, 7.5, 10]
const rng = arange(1, 10, 2);   // [1, 3, 5, 7, 9]

// Malla 2D de coordenadas (meshgrid) para evaluación de superficies
const x = linspace(-1, 1, 100);
const y = linspace(-1, 1, 100);
const [X, Y] = meshgrid(x, y);

// Matriz Identidad
const I = eye(3);
```

---

### 11. Vistas y Manipulación Tensorial en $\mathcal{O}(1)$

Manipula formas e intercambia ejes modificando strides sin copiar buffers de memoria:

```typescript
import { NDArray, reshape, transpose, expandDims, squeeze, clip } from './src/ix/index.js';

const A = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });

// 1. Reshape con inferencia de dimensión (-1) en O(1)
const B = reshape(A, [3, -1]); // shape: [3, 2]

// 2. Transpuesta en O(1) intercambiando strides
const At = transpose(A); // shape: [3, 2]

// 3. Modificación de ejes unitarios
const expanded = expandDims(A, 1); // shape: [2, 1, 3]
const squeezed = squeeze(expanded); // shape: [2, 3]

// 4. Truncado numérico (clipping)
const limited = clip(A, 2, 5); // [2, 2, 3, 4, 5, 5]
```

---

### 12. Reducciones por Eje y Estadísticas Descriptivas (Estilo Pandas)

Soporte completo de reducciones globales y por eje (`axis`), con corrección de grados de libertad (`ddof`):

```typescript
import { NDArray, sum, mean, std, variance, median, quantile, describe } from './src/ix/index.js';

// Matriz 2x3: [[1, 2, 3], [4, 5, 6]]
const M = new NDArray(new Float64Array([1, 2, 3, 4, 5, 6]), { shape: [2, 3] });

// 1. Reducciones colapsando ejes
const colSum = sum(M, { axis: 0 }); // [5, 7, 9] (colapsa filas)
const rowMean = mean(M, { axis: 1 }); // [2, 5] (media por fila)
const devStd = std(M, { axis: 0, ddof: 1 }); // Desviación estándar insesgada

// 2. Medidas de posición
const med = median(M); // 3.5
const q75 = quantile(M, 0.75); // Percentil 75

// 3. Resumen estadístico completo estilo pandas.DataFrame.describe()
const stats = describe(M);
console.log(stats);
// {
//   count: 6,
//   mean: 3.5,
//   std: 1.8708,
//   min: 1,
//   p25: 2.25,
//   median: 3.5,
//   p75: 4.75,
//   max: 6,
//   skew: 0,      // Simetría
//   kurtosis: -1.2 // Exceso de curtosis
// }
```

---

### 13. Series Temporales, Ventanas Móviles & NaNs (`rolling`, `ewm`, `cov`, `corr`)

```typescript
import {
  NDArray,
  rolling,
  ewm,
  diff,
  pctChange,
  shift,
  fillna,
  dropna,
  cov,
  corr
} from './src/ix/index.js';

const s = new NDArray(new Float64Array([10, 20, 30, 40, 50]), { shape: [5] });

// 1. Ventanas móviles (SMA, Volatilidad, etc.)
const sma3 = rolling(s, 3).mean(); // [NaN, NaN, 20, 30, 40]
const stdRoll = rolling(s, 3).std(); // Desviación estándar móvil

// 2. Medias Móviles Exponenciales (EMA)
const ema = ewm(s, { alpha: 0.5 });

// 3. Diferenciación y tasas de retorno
const d = diff(s);       // [NaN, 10, 10, 10, 10]
const ret = pctChange(s); // Retornos porcentuales
const lag = shift(s, 1);  // Desplazamiento temporal

// 4. Limpieza de datos (Missing Data)
const conNaN = new NDArray(new Float64Array([1, NaN, 3]), { shape: [3] });
const relleno = fillna(conNaN, 'ffill'); // [1, 1, 3] (Forward-fill)
const limpio = dropna(conNaN);           // [1, 3]

// 5. Matrices de Covarianza y Correlación Multivariada
const matDatos = new NDArray(new Float64Array([1, 2, 2, 4, 3, 6, 4, 8]), { shape: [4, 2] });
const matrizCov = cov(matDatos);
const matrizCorr = corr(matDatos); // Matriz de correlación de Pearson [-1, 1]
```

---

### 14. `DataFrame` Liviano con Índices Nombrados

Estructura tabular de cero-coste construida directamente sobre la memoria contigua de un `NDArray` 2D:

```typescript
import { DataFrame } from './src/ix/index.js';

const df = new DataFrame(
  [
    [20.5, 1013.2],
    [21.0, 1012.8],
    [22.4, 1011.5],
  ],
  {
    columns: ['temperatura', 'presion'],
    index: ['t0', 't1', 't2']
  }
);

// 1. Acceso a columnas por nombre como NDArray
const temp = df.col('temperatura'); // NDArray [20.5, 21.0, 22.4]

// 2. Selección de subconjuntos
const subDf = df.select('presion');

// 3. Matriz de correlación entre columnas
const dfCorr = df.corr();

// 4. Exportar a registros JSON
const jsonRows = df.toRecords();
```

---

### 15. Números Complejos & Fracciones Exactas

#### A. Números Complejos ($z = a + bi$)
Soporte completo para números imaginarios, fórmula de Euler y raíces complejas:
```typescript
import { complex, add } from './src/ix/index.js';

const z1 = complex(1, 2); // 1 + 2i
const z2 = complex(3, 4); // 3 + 4i

// Aritmética compleja
const sum = add(z1, z2);  // 4 + 6i
const prod = z1.mul(z2);  // -5 + 10i

// Raíz cuadrada de reales negativos: sqrt(-4) = 2i
const rootNeg = complex(-4, 0).sqrt(); // 0 + 2i

// Fórmula de Euler: e^(i * pi) = -1
const euler = complex(0, Math.PI).exp(); // -1 + 0i
```

#### B. Fracciones Racionales Exactas (BigInt)
Elimina errores de redondeo de coma flotante IEEE-754 ($1/3 + 1/6 = 1/2$ exacto):
```typescript
import { frac, add } from './src/ix/index.js';

const f1 = frac(1, 3);
const f2 = frac(1, 6);

const total = add(f1, f2); // Fraction 1/2
console.log(total.toString()); // '1/2'
console.log(total.toNumber()); // 0.5

// Conversión continua de decimales
const fromDec = Fraction.fromNumber(0.125); // 1/8
```

---

### 16. Procesamiento Digital de Señales (DSP & FFT)

Implementado en TypeScript puro sin binarios Wasm:

```typescript
import { sinewave, fft, convolve, lowpassFilter } from './src/ix/index.js';

// 1. Generar onda senoidal de 50 Hz a 500 Hz de muestreo
const signal = sinewave(50, 1.0, 500, 2.0);

// 2. Transformada Rápida de Fourier (FFT 1D Cooley-Tukey)
const { frequencies, magnitude } = fft(signal, 500);

// 3. Filtrado pasa-bajas IIR
const filtered = lowpassFilter(signal, 30, 500);
```

---

### 17. Integración Numérica Cuadrática

```typescript
import { trapz, simpson, quad, cumulativeIntegrate } from './src/ix/index.js';

// Cuadratura adaptativa continua (Gauss-Kronrod / Simpson adaptativa)
const area = quad((x) => Math.sin(x), 0, Math.PI); // 2.0

// Integración de datos muestreados discretos
const t = trapz([0, 1, 4, 9], [0, 1, 2, 3]);
const s = simpson([0, 1, 4, 9], [0, 1, 2, 3]);
```

---

### 18. Interpolación 1D y Splines Cúbicos

```typescript
import { interp1d, cubicSpline } from './src/ix/index.js';

// Interpolación lineal con búsqueda binaria O(log N)
const fLin = interp1d([0, 10], [0, 100], { method: 'linear' });
console.log(fLin(5)); // 50

// Trazador Cúbico Natural C^2 continuo
const fSpline = cubicSpline([0, 1, 2, 3], [0, 1, 8, 27]);
console.log(fSpline(1.5)); // 3.375 (1.5^3)
```

---

### 19. Optimización Numérica y Ajuste de Curvas

```typescript
import { fminbnd, nelderMead, curveFit, NDArray } from './src/ix/index.js';

// 1. Mínimo 1D escalar (Golden Section)
const min1D = fminbnd((x) => (x - 3)**2 + 5, [0, 10]); // x = 3, fval = 5

// 2. Mínimo multivariable sin derivadas (Nelder-Mead Simplex)
const minND = nelderMead((v) => (v.get(0) - 2)**2 + (v.get(1) + 4)**2, [0, 0]); // [2, -4]

// 3. Ajuste de curvas (Calibración de modelos)
const model = (x: number, p: NDArray) => p.get(0) * x + p.get(1);
const fit = curveFit(model, [1, 2, 3], [3, 5, 7], [1, 0]); // p = [2, 1]
```

---

### 20. Puentes de Visualización Gráfica (Chart.js, Plotly, Observable, SVG)

```typescript
import { toChartJS, toPlotly, toObservablePlot, plotSVG } from './src/ix/index.js';

const x = [0, 1, 2, 3];
const y = [0, 1, 4, 9];

// 1. Para Chart.js
const chartConfig = toChartJS(x, y, { title: 'Parábola' });

// 2. Para Plotly.js
const { data, layout } = toPlotly(x, y, { mode: 'lines+markers' });

// 3. Para Observable Plot
const plotData = toObservablePlot(x, y);

// 4. Renderizador SVG autónomo (<2KB) sin dependencias
const svgString = plotSVG(x, y, { color: '#38bdf8' });
```

---

### 21. Helpers AI-Ready, Prompting y Unidades con Offset

Herramientas para interactuar de forma inmediata en chats y editores:

```typescript
import {
  help,
  summary,
  fromCSV,
  fromMatrixString,
  unitsHelp,
  units,
  qty,
  celsius,
  fahrenheit,
  kelvin,
  meter,
  SPEED_OF_LIGHT,
  STANDARD_GRAVITY
} from './src/ix/index.js';

// 1. Asistencia e introspección inmediata para agentes
console.log(help('linspace'));
console.log(unitsHelp());

// 2. Resumen compacto para no saturar contextos de LLM
const A = new NDArray(new Float64Array([1, 2, 3, 4, 5]), { shape: [5] });
console.log(summary(A)); // NDArray shape=[5], min=1.0000, max=5.0000, mean=3.0000

// 3. Carga rápida de formatos
const df = fromCSV("x,y\n1,10\n2,20");
const mat = fromMatrixString("1 2 3; 4 5 6");

// 4. Unidades con offset térmico relativo
const tC = qty(100, celsius);
console.log(tC.to(fahrenheit)); // 212 °F
console.log(tC.to(kelvin));     // 373.15 K

// 5. Unidades de usuario personalizadas (con borrado y reinicio)
units.defineUnit('parsec', 'pc', qty(3.0857e16, meter), 'Parsec astronómico');
units.removeUnit('pc'); // Borra la unidad añadida
units.reset();          // Restaura el catálogo estándar canónico

// 6. Constantes Físicas Universales (CODATA)
console.log(SPEED_OF_LIGHT.value);   // 299792458 m/s
console.log(STANDARD_GRAVITY.value); // 9.80665 m/s^2
```

---

### 22. Ingesta de $\LaTeX$ (Estilo Wikipedia & Papers) y Multiplicación Implícita

Permite a personas y agentes pegar directamente fórmulas copiadas de Wikipedia o papers:

```typescript
import { fromLaTeX, toLaTeX, quickCalc, evaluate } from './src/ix/index.js';

// 1. Multiplicación implícita natural: 2x en vez de 2*x
console.log(evaluate('2x + 3y', { x: 4, y: 5 })); // 23
console.log(evaluate('3(x + 2)', { x: 10 }));      // 36

// 2. Ingesta directa de LaTeX con variables asociadas
const E = fromLaTeX('m c^2', { m: 2, c: 3e8 }); // 1.8e17
const Ek = fromLaTeX('\\frac{1}{2} m v^2', { m: 10, v: 20 }); // 2000
const h = fromLaTeX('\\sqrt{x^2 + y^2}', { x: 3, y: 4 }); // 5

// 3. Calculador rápido universal (autodetecta texto o LaTeX)
const r1 = quickCalc('10 * (2 + 3)');
const r2 = quickCalc('\\frac{10}{2}');

// 4. Exportar AST de vuelta a LaTeX para renderizar en UI (KaTeX / MathJax)
console.log(toLaTeX('x / y + sqrt(z)')); // \frac{x}{y} + \sqrt{z}
```

---

### 23. Herramientas para Agentes de IA (JSON Schema, MCP & Auto-Reparación)

Conecta la librería como servidor de herramientas (*Tool Calling*) para modelos como GPT-4, Claude o Gemini:

```typescript
import { getToolDefinitions, tryEval } from './src/ix/index.js';

// 1. Esquemas estándar JSON Schema para Function Calling / MCP
const tools = getToolDefinitions();
// [
//   { name: 'evaluateExpression', description: '...', parameters: { ... } },
//   { name: 'solveODE', ... },
//   { name: 'optimizeFunction', ... },
//   { name: 'convertUnits', ... }
// ]

// 2. Sandbox de ejecución con diagnóstico y pistas de auto-reparación (Self-Repair)
const diag = tryEval(() => {
  // Código generado por el agente
  return evaluate('2 * variable_no_declarada');
});

if (!diag.success) {
  console.log(diag.message);   // "Undefined variable in scope: variable_no_declarada"
  console.log(diag.agentHint); // "Una variable no fue declarada en el scope. Pasa un objeto scope..."
}
```

---

## 🧪 Pruebas Unitarias

```bash
# Correr suite completa con Vitest
npm test
```

## 📄 Licencia
MIT © 2026
