import type { EnergyNetwork } from "./network.js";

export interface PotentialWell {
  readonly wellId: number;
  /** 峰边（不相邻极大值之一）的两个端点与峰值强度 */
  readonly peak: { readonly from: number; readonly to: number; readonly weight: number };
  /** 势阱成员神经元：峰值的 wellRatio 邻域内连通边覆盖的节点 */
  readonly memberNeuronIds: readonly number[];
  /** 构成势阱的边数 */
  readonly edgeCount: number;
}

interface Edge {
  readonly from: number;
  readonly to: number;
  readonly weight: number;
}

// 赫布累加的浮点误差容忍：0.1×8 未必恰好等于 0.8
const EPSILON = 1e-9;

/**
 * 势阱检测（要求 8）：
 * 1. 在连接强度图（边权 W_ij > 0）上找出局部极大边——权值严格大于所有
 *    与其共端点的相邻边；
 * 2. 按权值降序贪心选取互不相邻的极大边（每选一个，排除其所有相邻边）；
 * 3. 每个极大值 P 的势阱 = 从峰边出发、边权 ≥ wellRatio × P（默认 0.8）的
 *    连通分量（BFS），覆盖的神经元即势阱成员。
 * 标签不在此处赋予——由教师程序或模型自身通过 ReadoutModule 赋予。
 */
export function detectWells(network: EnergyNetwork): PotentialWell[] {
  const n = network.neuronCount;
  const ratio = network.config.wellRatio;

  // 收集正权边（无向，i<j 只存一次）
  const edges: Edge[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = network.getWeight(i, j);
      if (w > 0) edges.push({ from: i, to: j, weight: w });
    }
  }
  if (edges.length === 0) return [];

  // 每个节点的邻接边索引
  const nodeEdges: number[][] = Array.from({ length: n }, () => []);
  edges.forEach((e, idx) => {
    nodeEdges[e.from]!.push(idx);
    nodeEdges[e.to]!.push(idx);
  });

  // ① 局部极大边：严格大于所有相邻边
  const localMaxima: number[] = [];
  for (let idx = 0; idx < edges.length; idx++) {
    const e = edges[idx]!;
    let isMax = true;
    for (const node of [e.from, e.to]) {
      for (const other of nodeEdges[node]!) {
        if (other !== idx && edges[other]!.weight >= e.weight) {
          isMax = false;
          break;
        }
      }
      if (!isMax) break;
    }
    if (isMax) localMaxima.push(idx);
  }

  // ② 贪心选不相邻极大值：降序，选中后排除其相邻边
  localMaxima.sort((a, b) => edges[b]!.weight - edges[a]!.weight || a - b);
  const excluded = new Uint8Array(edges.length);
  const selectedPeaks: number[] = [];
  for (const idx of localMaxima) {
    if (excluded[idx] === 1) continue;
    selectedPeaks.push(idx);
    excluded[idx] = 1;
    const e = edges[idx]!;
    for (const node of [e.from, e.to]) {
      for (const other of nodeEdges[node]!) excluded[other] = 1;
    }
  }

  // ③ 0.8 邻域连通分量 → 势阱成员
  const wells: PotentialWell[] = [];
  for (const peakIdx of selectedPeaks) {
    const peak = edges[peakIdx]!;
    const threshold = ratio * peak.weight;
    const visitedNodes = new Set<number>([peak.from, peak.to]);
    const visitedEdges = new Set<number>([peakIdx]);
    const queue: number[] = [peak.from, peak.to];
    while (queue.length > 0) {
      const node = queue.pop()!;
      for (const eIdx of nodeEdges[node]!) {
        if (visitedEdges.has(eIdx)) continue;
        const e = edges[eIdx]!;
        if (e.weight < threshold - EPSILON) continue;
        visitedEdges.add(eIdx);
        for (const endpoint of [e.from, e.to]) {
          if (!visitedNodes.has(endpoint)) {
            visitedNodes.add(endpoint);
            queue.push(endpoint);
          }
        }
      }
    }
    wells.push({
      wellId: wells.length,
      peak: { from: peak.from, to: peak.to, weight: peak.weight },
      memberNeuronIds: [...visitedNodes].sort((a, b) => a - b),
      edgeCount: visitedEdges.size,
    });
  }
  return wells;
}
