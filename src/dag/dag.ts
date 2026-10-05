/**
 * @file dag.ts
 * Grafo Acíclico Dirigido (DAG) para computación matemática diferida y composición de expresiones:
 * - Nodos de variables y operaciones
 * - Ordenamiento topológico
 * - Detección de ciclos
 * - Evaluación perezosa (Lazy Evaluation) con memoización de resultados intermedios
 */
import { NDArray } from '../core/ndarray.js';

export type NodeId = string;

export interface ComputeNode {
  id: NodeId;
  name: string;
  deps: NodeId[];
  op: (...args: any[]) => any;
  value?: any;
  isEvaluated: boolean;
}

export class ExpressionDAG {
  private nodes: Map<NodeId, ComputeNode> = new Map();

  /**
   * Agrega un nodo constante o variable de entrada al DAG.
   */
  public variable(id: NodeId, initialValue?: any): NodeId {
    this.nodes.set(id, {
      id,
      name: id,
      deps: [],
      op: () => initialValue,
      value: initialValue,
      isEvaluated: initialValue !== undefined,
    });
    return id;
  }

  /**
   * Actualiza el valor de una variable del DAG e invalida sus dependientes.
   */
  public setVariable(id: NodeId, value: any): void {
    const node = this.nodes.get(id);
    if (!node) {
      throw new Error(`Node ${id} does not exist in DAG.`);
    }
    node.value = value;
    node.isEvaluated = true;
    this.invalidateDependents(id);
  }

  /**
   * Agrega una operación dependiente de otros nodos.
   */
  public op(
    id: NodeId,
    deps: NodeId[],
    computeFn: (...args: any[]) => any
  ): NodeId {
    for (const dep of deps) {
      if (!this.nodes.has(dep)) {
        throw new Error(`Dependency ${dep} not registered in DAG before adding node ${id}.`);
      }
    }

    this.nodes.set(id, {
      id,
      name: id,
      deps,
      op: computeFn,
      value: undefined,
      isEvaluated: false,
    });

    // Validar aciclicidad
    this.topologicalSort();

    return id;
  }

  /**
   * Invalida recursivamente la caché de los nodos que dependen directa o indirectamente de 'id'.
   */
  private invalidateDependents(id: NodeId): void {
    for (const [nodeId, node] of this.nodes.entries()) {
      if (node.deps.includes(id)) {
        node.isEvaluated = false;
        node.value = undefined;
        this.invalidateDependents(nodeId);
      }
    }
  }

  /**
   * Calcula el orden topológico de ejecución (Kahn's Algorithm).
   * Lanza un error si detecta ciclos.
   */
  public topologicalSort(): NodeId[] {
    const inDegree: Map<NodeId, number> = new Map();
    const adjList: Map<NodeId, NodeId[]> = new Map();

    for (const id of this.nodes.keys()) {
      inDegree.set(id, 0);
      adjList.set(id, []);
    }

    for (const [id, node] of this.nodes.entries()) {
      for (const dep of node.deps) {
        adjList.get(dep)!.push(id);
        inDegree.set(id, (inDegree.get(id) || 0) + 1);
      }
    }

    const queue: NodeId[] = [];
    for (const [id, deg] of inDegree.entries()) {
      if (deg === 0) queue.push(id);
    }

    const order: NodeId[] = [];
    while (queue.length > 0) {
      const current = queue.shift()!;
      order.push(current);

      for (const neighbor of adjList.get(current)!) {
        const newDeg = inDegree.get(neighbor)! - 1;
        inDegree.set(neighbor, newDeg);
        if (newDeg === 0) queue.push(neighbor);
      }
    }

    if (order.length !== this.nodes.size) {
      throw new Error('Cycle detected in ExpressionDAG! Graph must be acyclic.');
    }

    return order;
  }

  /**
   * Evalúa un nodo objetivo resolviendo sólo los subgrafos requeridos con memoización.
   */
  public evaluate(targetId: NodeId, scopeOverride?: Record<string, any>): any {
    if (scopeOverride) {
      for (const [k, v] of Object.entries(scopeOverride)) {
        if (this.nodes.has(k)) {
          this.setVariable(k, v);
        }
      }
    }

    const targetNode = this.nodes.get(targetId);
    if (!targetNode) {
      throw new Error(`Target node ${targetId} not found in DAG.`);
    }

    if (targetNode.isEvaluated && targetNode.value !== undefined) {
      return targetNode.value;
    }

    const order = this.topologicalSort();
    for (const id of order) {
      const node = this.nodes.get(id)!;
      if (node.isEvaluated && node.value !== undefined) continue;

      const argValues = node.deps.map((depId) => {
        const depNode = this.nodes.get(depId)!;
        if (!depNode.isEvaluated) {
          throw new Error(`Dependency ${depId} was not evaluated in expected topological order.`);
        }
        return depNode.value;
      });

      node.value = node.op(...argValues);
      node.isEvaluated = true;

      if (id === targetId) {
        return node.value;
      }
    }

    return targetNode.value;
  }

  /**
   * Representación textual Mermaid del DAG para visualización y depuración.
   */
  public toMermaid(): string {
    let mermaid = 'graph TD\n';
    for (const [id, node] of this.nodes.entries()) {
      for (const dep of node.deps) {
        mermaid += `  ${dep} --> ${id}\n`;
      }
    }
    return mermaid;
  }
}
