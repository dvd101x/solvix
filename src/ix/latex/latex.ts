/**
 * @file latex.ts
 * Parser e ingesta de fórmulas en LaTeX (estilo Wikipedia, papers científicos y salidas de LLMs):
 * - Traduce comandos de LaTeX comunes (\frac{a}{b}, \sqrt{x}, \sin, \cos, x^2, \cdot, etc.) a expresiones matemáticas evaluables
 * - fromLaTeX(latexStr, scope): Evalúa directamente una fórmula de LaTeX
 * - toLaTeX(astOrExpr): Exporta un AST o expresión a código LaTeX limpio para renderizar con KaTeX / MathJax
 */
import { evaluate, parseExpression, ASTNode } from '../parser/parser.js';

/**
 * Normaliza y traduce código LaTeX a una expresión en texto plano estándar compatible con el parser de ix.
 * Maneja comandos comunes de Wikipedia y fórmulas científicas.
 */
export function latexToMathExpr(latex: string): string {
  let s = latex.trim();

  // 1. Quitar delimitadores de modo matemático ($...$, $$...$$, \[...\], \(...\))
  s = s.replace(/^\$\$|\$\$$|^\\\[|\\\]$|^\$|\$$|^\\\(|\\\)$/g, '').trim();

  // 2. Normalizar multiplicaciones explícitas de LaTeX (\cdot, \times)
  s = s.replace(/\\cdot|\\times/g, ' * ');

  // 3. Normalizar comandos de funciones trigonométricas y logarítmicas (\sin, \cos, \ln, etc.)
  s = s.replace(/\\(sin|cos|tan|arcsin|arccos|arctan|exp|log|ln|sqrt|abs)/g, '$1');
  s = s.replace(/ln\b/g, 'log'); // ln -> log

  // 4. Normalizar letras griegas comunes a sus nombres de variable
  s = s.replace(/\\(alpha|beta|gamma|theta|lambda|mu|pi|sigma|omega|phi|delta|epsilon)/g, '$1');

  // 5. Expandir fracciones: \frac{numerador}{denominador} -> ((numerador) / (denominador))
  // Se procesa en bucle para manejar fracciones anidadas \frac{\frac{a}{b}}{c}
  const fracRegex = /\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/;
  while (fracRegex.test(s)) {
    s = s.replace(fracRegex, '(($1) / ($2))');
  }

  // 6. Expandir raíces cuadradas: \sqrt{x} o sqrt{x} -> sqrt(x)
  const sqrtRegex = /sqrt\s*\{([^{}]+)\}/;
  while (sqrtRegex.test(s)) {
    s = s.replace(sqrtRegex, 'sqrt($1)');
  }

  // 7. Reemplazar llaves de exponentes y agrupaciones: ^{2} -> ^(2), {x + 1} -> (x + 1)
  s = s.replace(/\^{([^{}]+)}/g, '^($1)');
  s = s.replace(/_\{([^{}]+)}/g, ''); // subíndices se descartan en cálculo escalar si son etiquetas
  s = s.replace(/\{([^{}]+)\}/g, '($1)');

  // 8. Limpiar espacios redundantes
  s = s.replace(/\s+/g, ' ').trim();

  return s;
}

/**
 * Evalúa directamente una fórmula matemática escrita en formato LaTeX sobre un scope dado.
 * @example
 * fromLaTeX('\\frac{1}{2} m v^2', { m: 10, v: 20 }) // -> 2000
 */
export function fromLaTeX(latexStr: string, scope: Record<string, any> = {}): any {
  const expr = latexToMathExpr(latexStr);
  return evaluate(expr, scope);
}

/**
 * Convierte un AST matemático o una expresión de texto a código LaTeX formateado.
 */
export function toLaTeX(astOrExpr: ASTNode | string): string {
  const ast: ASTNode = typeof astOrExpr === 'string' ? parseExpression(astOrExpr) : astOrExpr;

  function render(node: ASTNode): string {
    switch (node.type) {
      case 'NUMBER':
        return String(node.value);

      case 'VARIABLE':
        if (['alpha', 'beta', 'theta', 'omega', 'pi', 'lambda'].includes(node.name)) {
          return `\\${node.name}`;
        }
        return node.name;

      case 'UNARY_OP':
        return `-${render(node.expr)}`;

      case 'BINARY_OP': {
        const left = render(node.left);
        const right = render(node.right);
        switch (node.op) {
          case '/':
            return `\\frac{${left}}{${right}}`;
          case '*':
            return `${left} \\cdot ${right}`;
          case '^':
            return `{${left}}^{${right}}`;
          default:
            return `${left} ${node.op} ${right}`;
        }
      }

      case 'FUNCTION_CALL': {
        const args = node.args.map(render).join(', ');
        if (node.name === 'sqrt') {
          return `\\sqrt{${args}}`;
        }
        return `\\${node.name}\\left(${args}\\right)`;
      }
    }
  }

  return render(ast);
}
