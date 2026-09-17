import { EnergyNetwork, hebbianLearn } from "../../index.js";
import type { SensoryEncoder } from "./sensory.js";

/**
 * 概念形成循环（M2）：连续实验 → S0 编码 → 共现成阱 → 按维度聚类出值概念。
 *
 * 形成机理（与"相似输入成阱"同一引擎）：同一值区域的感受野在每次相关
 * 实验中反复共激活、被隐性重复，互连挖深；同一维度的替代值从不共激活
 * （每次实验每维只有一个值），区间互连保持微弱；换对观察再把替代区
 * 之间写上抑制。于是每个维度上，值区域在权重图里自然分成若干强连通的簇。
 * 没有任何"档"被预先承诺——簇就是自形成的值概念。
 */

export interface EmergentConcept {
  readonly conceptId: number;
  readonly dimension: string;
  /** 成员感受野神经元 id */
  readonly memberNeuronIds: readonly number[];
  /** 概念中心值（成员感受野中心的均值） */
  readonly centerValue: number;
  /** 簇内平均边强（形成强度的证据） */
  readonly meanInternalWeight: number;
}

export interface FormationConfig {
  readonly activationEnergy?: number;
  readonly maintenanceEnergy?: number;
  readonly learningRate?: number;
  readonly maxWeight?: number;
  /** 簇检测阈值：边强 ≥ 该比例 × 本维度最大边强，默认 0.5 */
  readonly clusterThresholdRatio?: number;
}

export class ConceptFormation {
  readonly net: EnergyNetwork;
  readonly encoder: SensoryEncoder;

  constructor(encoder: SensoryEncoder, config: FormationConfig = {}) {
    this.encoder = encoder;
    this.net = new EnergyNetwork({
      neuronCount: encoder.neuronCount,
      activationEnergy: config.activationEnergy ?? 1.0,
      maintenanceEnergy: config.maintenanceEnergy ?? 0.5,
      learningRate: config.learningRate ?? 0.1,
      maxWeight: config.maxWeight ?? 3.0,
    });
  }

  /** 呈现一次实验（连续值帧的共激活绑定，封顶 0.4） */
  presentExperiment(values: Record<string, number>, repeats = 4): void {
    hebbianLearn(this.net, this.encoder.encode(values), repeats, undefined, 0.4);
  }

  /** 换对互斥：同维度两个被对换的值区域之间写抑制边 */
  presentSwap(dimension: string, valueA: number, valueB: number, strength = 3.0): void {
    const a = this.encoder.encodeDimension(dimension, valueA);
    const b = this.encoder.encodeDimension(dimension, valueB);
    for (const x of a) {
      for (const y of b) this.net.strengthenInhibitory(x, y, strength);
    }
  }

  /**
   * 按维度提取自形成值概念：在维度内部的权重子图上，
   * 取边强 ≥ ratio × 本维最大边强的边，做连通分量聚类。
   * 成员按感受野中心均值登记概念中心值。
   */
  extractConcepts(ratio = 0.5): EmergentConcept[] {
    const concepts: EmergentConcept[] = [];
    for (const dim of this.encoder.dimensions) {
      const centers = this.encoder.fieldCenters(dim.name);
      const n = centers.length;
      const offset = this.encoder.dimensionOffset(dim.name);
      // 本维最大边强
      let maxW = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          maxW = Math.max(maxW, this.net.getWeight(offset + i, offset + j));
        }
      }
      if (maxW <= 0) continue;
      const threshold = ratio * maxW;
      // 并查集聚类
      const parent = Array.from({ length: n }, (_, i) => i);
      const find = (x: number): number => {
        while (parent[x] !== x) {
          parent[x] = parent[parent[x]!]!;
          x = parent[x]!;
        }
        return x;
      };
      const internalSum = new Map<number, { sum: number; count: number }>();
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const w = this.net.getWeight(offset + i, offset + j);
          if (w >= threshold) {
            const ri = find(i);
            const rj = find(j);
            if (ri !== rj) parent[ri] = rj;
            const s = internalSum.get(find(j)) ?? { sum: 0, count: 0 };
            s.sum += w;
            s.count++;
            internalSum.set(find(j), s);
          }
        }
      }
      const clusters = new Map<number, number[]>();
      for (let i = 0; i < n; i++) {
        const r = find(i);
        const arr = clusters.get(r) ?? [];
        arr.push(i);
        clusters.set(r, arr);
      }
      for (const members of clusters.values()) {
        if (members.length < 2) continue; // 孤立感受野不构成概念
        const key = members[0]!;
        const stat = internalSum.get(find(key)) ?? { sum: 0, count: 0 };
        concepts.push({
          conceptId: concepts.length,
          dimension: dim.name,
          memberNeuronIds: members.map((k) => offset + k).sort((a, b) => a - b),
          centerValue: members.reduce((s, k) => s + centers[k]!, 0) / members.length,
          meanInternalWeight: stat.count === 0 ? 0 : stat.sum / stat.count,
        });
      }
    }
    return concepts;
  }
}
