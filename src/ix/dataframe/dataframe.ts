/**
 * @file dataframe.ts
 * DataFrame liviano de alto rendimiento respaldado por NDArray 2D:
 * - Acceso a columnas por nombre
 * - Índices de filas opcionales
 * - Selección de columnas (select)
 * - Estadísticas y transformaciones en bloque (describe, corr, cov, toJSON)
 */
import { NDArray } from '../core/ndarray.js';
import { cov, corr } from '../stats/series-ops.js';
import { describe, StatsSummary } from '../stats/reductions.js';

export interface DataFrameOptions {
  columns?: string[];
  index?: string[];
}

export class DataFrame {
  public readonly values: NDArray;
  public readonly columns: string[];
  public readonly index: string[];
  public readonly shape: [number, number];

  private colIndexMap: Map<string, number> = new Map();

  constructor(data: NDArray | number[][], options: DataFrameOptions = {}) {
    if (data instanceof NDArray) {
      if (data.ndim !== 2) throw new Error(`DataFrame requires a 2D NDArray, got ${data.ndim}D`);
      this.values = data;
    } else {
      this.values = NDArray.fromArray(data);
    }

    const [rows, cols] = [this.values.shape[0], this.values.shape[1]];
    this.shape = [rows, cols];

    this.columns = options.columns ?? Array.from({ length: cols }, (_, i) => `col_${i}`);
    if (this.columns.length !== cols) {
      throw new Error(`Columns length (${this.columns.length}) does not match matrix cols (${cols})`);
    }

    this.index = options.index ?? Array.from({ length: rows }, (_, i) => String(i));
    if (this.index.length !== rows) {
      throw new Error(`Index length (${this.index.length}) does not match matrix rows (${rows})`);
    }

    for (let c = 0; c < cols; c++) {
      this.colIndexMap.set(this.columns[c], c);
    }
  }

  /**
   * Obtiene una columna específica como un NDArray 1D (vista o copia).
   */
  public col(name: string): NDArray {
    const colIdx = this.colIndexMap.get(name);
    if (colIdx === undefined) {
      throw new ReferenceError(`Column "${name}" does not exist in DataFrame [${this.columns.join(', ')}]`);
    }

    const rows = this.shape[0];
    const colData = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      colData[r] = this.values.get(r, colIdx);
    }
    return new NDArray(colData, { shape: [rows] });
  }

  /**
   * Filtra y selecciona un subconjunto de columnas en un nuevo DataFrame.
   */
  public select(...colNames: string[]): DataFrame {
    const rows = this.shape[0];
    const newCols = colNames.length;
    const outData = new Float64Array(rows * newCols);

    for (let c = 0; c < newCols; c++) {
      const colIdx = this.colIndexMap.get(colNames[c]);
      if (colIdx === undefined) throw new ReferenceError(`Column "${colNames[c]}" not found`);
      for (let r = 0; r < rows; r++) {
        outData[r * newCols + c] = this.values.get(r, colIdx);
      }
    }

    return new DataFrame(new NDArray(outData, { shape: [rows, newCols] }), {
      columns: colNames,
      index: [...this.index],
    });
  }

  /**
   * Matriz de correlación entre todas las columnas del DataFrame.
   */
  public corr(): DataFrame {
    const correlationMatrix = corr(this.values);
    return new DataFrame(correlationMatrix, {
      columns: [...this.columns],
      index: [...this.columns],
    });
  }

  /**
   * Matriz de covarianza entre todas las columnas.
   */
  public cov(ddof = 1): DataFrame {
    const covMatrix = cov(this.values, ddof);
    return new DataFrame(covMatrix, {
      columns: [...this.columns],
      index: [...this.columns],
    });
  }

  /**
   * Resumen estadístico por columna.
   */
  public describe(): Record<string, StatsSummary> {
    const result: Record<string, StatsSummary> = {};
    for (const name of this.columns) {
      result[name] = describe(this.col(name));
    }
    return result;
  }

  /**
   * Exporta a formato estructurado de objetos JS (JSON).
   */
  public toRecords(): Record<string, any>[] {
    const records: Record<string, any>[] = [];
    const rows = this.shape[0];
    for (let r = 0; r < rows; r++) {
      const rowObj: Record<string, any> = { _index: this.index[r] };
      for (const col of this.columns) {
        rowObj[col] = this.values.get(r, this.colIndexMap.get(col)!);
      }
      records.push(rowObj);
    }
    return records;
  }
}
