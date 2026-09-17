import type { EmergentMap } from "./emergent-map.js";

/**
 * 自形成概念 → R2Pop 通道图接口的适配器。
 * 把 EmergentMap 包装成 R2PopLayer 认识的通道图形状
 * （specs/neuronCount/population/encode/binOf/popSize/channelNames），
 * 使神经差分可以直接跑在自形成概念上——概念即"档"，
 * 档的边界来自共现形成，不是手工。
 */
export class EmergentChannelAdapter {
  readonly specs: readonly { readonly name: string; readonly bins: number }[];
  readonly neuronCount: number;
  readonly popSize: number;
  private readonly idBase: number;

  constructor(
    readonly map: EmergentMap,
    readonly dimensions: readonly string[],
    idBase?: number,
    readonly encodeMode: "concept" | "fields" = "concept",
  ) {
    this.specs = dimensions.map((name) => ({ name, bins: map.conceptCount(name) }));
    const sizes: number[] = [];
    for (const name of dimensions) {
      for (let i = 0; i < map.conceptCount(name); i++) sizes.push(map.population(name, i).length);
    }
    this.popSize = sizes.length === 0 ? 4 : sizes.reduce((a, b) => a + b, 0) / sizes.length;
    // 该适配器所覆盖维度在编码器空间中的占用（按 fieldsPerDim 连续排布）
    this.neuronCount = dimensions.length * map.encoder.fieldsPerDim;
    this.idBase = idBase ?? map.encoder.dimensionOffset(dimensions[0]!);
  }

  channelNames(): string[] {
    return [...this.dimensions];
  }

  /** 维度第 index 个概念的成员（编码器神经元 id，减去 idBase 变局部空间） */
  population(dimension: string, index: number): number[] {
    return this.map.population(dimension, index).map((id) => id - this.idBase);
  }

  /**
   * 概念索引帧 → 神经元集合。
   * mode="concept"（条件侧）：{dim: 概念索引} → 概念成员群体（粗粒度差分）；
   * mode="fields"（结果侧）：{dim: 连续值} → 该值的直接感受野编码——
   * 保留粗概念内部的细分差异（R2A 场级档差的前提）。
   */
  encode(values: Readonly<Record<string, number>>): number[] {
    if (this.encodeMode === "fields") {
      return Object.entries(values).flatMap(([dim, value]) =>
        this.map.encoder.encodeDimension(dim, value).map((id) => id - this.idBase),
      );
    }
    return Object.entries(values).flatMap(([dim, index]) => {
      if (index < 0) return [];
      return this.population(dim, index);
    });
  }

  /** 局部神经元 id → 所属概念（维度, 索引） */
  binOf(neuronId: number): { channel: string; bin: number } | null {
    const encoderId = neuronId + this.idBase;
    for (const dim of this.dimensions) {
      for (let i = 0; i < this.map.conceptCount(dim); i++) {
        if (this.map.concept(dim, i).memberNeuronIds.includes(encoderId)) {
          return { channel: dim, bin: i };
        }
      }
    }
    return null;
  }

  /** 局部神经元 id → 感受野中心值（场级档差用） */
  fieldCenterOf(neuronId: number): number | null {
    const f = this.map.encoder.fieldOf(neuronId + this.idBase);
    return f ? f.center : null;
  }
}
