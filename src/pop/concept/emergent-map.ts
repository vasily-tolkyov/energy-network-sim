import type { EmergentConcept } from "./formation.js";
import type { SensoryEncoder } from "./sensory.js";

/**
 * EmergentMap（M3）：自形成概念的注册表与下游适配层。
 * 概念索引按维度内 centerValue 排序编号；population(dimension, index)
 * 返回成员感受野神经元（与 PopChannelMap 的 population(channel, bin)
 * 同签名——下游 R1/R2/R3/预测机器零改动接入）。
 * resolve(dimension, value)：连续值 → 最重叠概念的索引（无重叠 → null）。
 */

export class EmergentMap {
  readonly concepts: readonly EmergentConcept[];
  private readonly byDim = new Map<string, EmergentConcept[]>();

  constructor(
    concepts: readonly EmergentConcept[],
    readonly encoder: SensoryEncoder,
  ) {
    // 按维度分组、按中心值排序后重新编号
    const sorted = [...concepts].sort(
      (a, b) => a.dimension.localeCompare(b.dimension) || a.centerValue - b.centerValue,
    );
    this.concepts = sorted.map((c, i) => ({ ...c, conceptId: i }));
    for (const c of this.concepts) {
      const arr = this.byDim.get(c.dimension) ?? [];
      arr.push(c);
      this.byDim.set(c.dimension, arr);
    }
  }

  channelNames(): string[] {
    return [...this.byDim.keys()];
  }

  conceptCount(dimension: string): number {
    return this.byDim.get(dimension)?.length ?? 0;
  }

  /** PopChannelMap 兼容接口：维度第 index 个概念的成员神经元 */
  population(dimension: string, index: number): number[] {
    const arr = this.byDim.get(dimension);
    if (!arr) throw new Error(`unknown dimension: ${dimension}`);
    const c = arr[index];
    if (!c) throw new Error(`dimension ${dimension}: concept index ${index} out of range [0, ${arr.length})`);
    return [...c.memberNeuronIds];
  }

  concept(dimension: string, index: number): EmergentConcept {
    return this.byDim.get(dimension)![index]!;
  }

  /**
   * 连续值 → 概念索引：编码该值，与每个概念成员算重叠数，
   * 取重叠最大者（≥2 才认为该区域已形成；平手返回 null 如实）。
   * 合并的大概念（成员多）对其整个区域都有归属权——
   * 不要求激活覆盖概念的一半（大区域里每个值只覆盖一小片）。
   */
  resolve(dimension: string, value: number): number | null {
    const arr = this.byDim.get(dimension);
    if (!arr) return null;
    const activated = new Set(this.encoder.encodeDimension(dimension, value));
    let bestIdx = -1;
    let bestOverlap = 0;
    let tie = false;
    arr.forEach((c, index) => {
      const overlap = c.memberNeuronIds.filter((id) => activated.has(id)).length;
      if (overlap > bestOverlap) {
        bestIdx = index;
        bestOverlap = overlap;
        tie = false;
      } else if (overlap === bestOverlap && overlap > 0) {
        tie = true;
      }
    });
    if (bestOverlap < 2 || tie) return null;
    return bestIdx;
  }

  /** 连续帧 → 每维概念索引（无概念维度记 null） */
  resolveFrame(values: Readonly<Record<string, number>>): Record<string, number | null> {
    const out: Record<string, number | null> = {};
    for (const dim of this.byDim.keys()) {
      out[dim] = values[dim] === undefined ? null : this.resolve(dim, values[dim]!);
    }
    return out;
  }
}

/** 对齐报告条目：一个参照值与最重叠自形成概念的 IoU */
export interface AlignmentEntry {
  readonly dimension: string;
  readonly referenceValue: number;
  readonly conceptIndex: number | null;
  readonly conceptCenter: number | null;
  readonly iou: number;
}

/** 与参照值网格的对齐报告（M3 的对齐检验用） */
export function alignmentReport(
  map: EmergentMap,
  references: Readonly<Record<string, readonly number[]>>,
): AlignmentEntry[] {
  const entries: AlignmentEntry[] = [];
  for (const [dimension, values] of Object.entries(references)) {
    for (const value of values) {
      const activated = new Set(map.encoder.encodeDimension(dimension, value));
      const idx = map.resolve(dimension, value);
      let iou = 0;
      let center: number | null = null;
      if (idx !== null) {
        const c = map.concept(dimension, idx);
        center = c.centerValue;
        const members = new Set(c.memberNeuronIds);
        let inter = 0;
        for (const x of activated) if (members.has(x)) inter++;
        iou = inter / new Set([...activated, ...members]).size;
      }
      entries.push({ dimension, referenceValue: value, conceptIndex: idx, conceptCenter: center, iou });
    }
  }
  return entries;
}
