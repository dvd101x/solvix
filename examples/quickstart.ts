import { NDArray, add } from '../src/index.js';

const values = new NDArray(new Float64Array([1, 2, 3]), { shape: [3] });
const result = add(values, 2);

console.log(Array.from(result));
