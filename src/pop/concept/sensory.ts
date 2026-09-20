/**
 * S0 传感编码层（概念自形成的入口，非承诺编码）。
 *
 * 每个物理维度配一排高斯感受野神经元，中心均匀铺满量程、相邻重叠；
 * 连续值 x 激活 |x−c| < 2σ 的神经元（约 6–8 个/维/值）。
 * 关键性质是相似性保持：数值相近 → 激活群体重叠多；数值远 → 重叠少。
 * 这是"相似输入成阱"的引擎前提——没有任何"分档"被预先承诺，
 * 值概念将由共现统计在这些感受野上结晶。
 */

import { integer, finite } from "../../validate.js";

export interface DimensionSpec {
  readonly name: string;
  readonly min: number;
  readonly max: number;
  /** 可选：显式感受野宽度（默认 range/20）。离散世界按档间距标定分辨率用 */
  readonly sigma?: number;
}

export interface FieldEncoding {
  readonly dimension: string;
  readonly fieldIndex: number;
  readonly center: number;
}

export class SensoryEncoder {
  readonly dimensions: readonly DimensionSpec[];
  readonly fieldsPerDim: number;
  readonly neuronCount: number;
  private readonly offsets = new Map<string, number>();
  private readonly centers = new Map<string, Float64Array>();
  private readonly sigmas = new Map<string, number>();

  constructor(dimensions: readonly DimensionSpec[], fieldsPerDim = 40) {
    integer(fieldsPerDim, "fieldsPerDim", 2);
    this.dimensions = dimensions.map(d => ({ ...d }));
    this.fieldsPerDim = fieldsPerDim;
    let offset = 0;
    for (const dim of dimensions) {
      finite(dim.min, `${dim.name}.min`); finite(dim.max, `${dim.name}.max`);
      if (this.offsets.has(dim.name)) throw new Error(`duplicate dimension: ${dim.name}`);
      const range = dim.max - dim.min;
      if (!(range > 0)) throw new Error(`dimension ${dim.name}: max must be > min`);
      this.offsets.set(dim.name, offset);
      const centers = new Float64Array(fieldsPerDim);
      for (let k = 0; k < fieldsPerDim; k++) {
        centers[k] = dim.min + ((k + 0.5) / fieldsPerDim) * range;
      }
      this.centers.set(dim.name, centers);
      if (dim.sigma !== undefined) {
        finite(dim.sigma, `${dim.name}.sigma`);
        if (dim.sigma <= 0) throw new Error(`${dim.name}.sigma must be positive`);
      }
      this.sigmas.set(dim.name, dim.sigma ?? range / 20);
      offset += fieldsPerDim;
    }
    this.neuronCount = offset;
  }

  sigma(dimension: string): number {
    const s = this.sigmas.get(dimension);
    if (s === undefined) throw new Error(`unknown dimension: ${dimension}`);
    return s;
  }

  /** 连续值 → 激活的神经元 id（|x−c| < 2σ；量程外夹到边界） */
  encodeDimension(dimension: string, value: number): number[] {
    finite(value, dimension);
    const offset = this.offsets.get(dimension);
    const centers = this.centers.get(dimension);
    const sigma = this.sigmas.get(dimension);
    if (offset === undefined || !centers || !sigma) throw new Error(`unknown dimension: ${dimension}`);
    const dim = this.dimensions.find((d) => d.name === dimension)!;
    const x = Math.min(dim.max, Math.max(dim.min, value));
    const threshold = 2 * sigma;
    const out: number[] = [];
    for (let k = 0; k < centers.length; k++) {
      if (Math.abs(centers[k]! - x) < threshold) out.push(offset + k);
    }
    return out;
  }

  /** 编码整帧：维度名 → 实数值 */
  encode(values: Readonly<Record<string, number>>): number[] {
    return Object.entries(values).flatMap(([dim, x]) => this.encodeDimension(dim, x));
  }

  /** 神经元 id → 所属维度与感受野信息 */
  fieldOf(neuronId: number): FieldEncoding | null {
    for (const dim of this.dimensions) {
      const offset = this.offsets.get(dim.name)!;
      if (neuronId >= offset && neuronId < offset + this.fieldsPerDim) {
        const fieldIndex = neuronId - offset;
        return { dimension: dim.name, fieldIndex, center: this.centers.get(dim.name)![fieldIndex]! };
      }
    }
    return null;
  }

  fieldCenters(dimension: string): Float64Array {
    const c = this.centers.get(dimension);
    if (!c) throw new Error(`unknown dimension: ${dimension}`);
    return c;
  }

  /** 维度在神经元空间中的起始偏移 */
  dimensionOffset(dimension: string): number {
    const o = this.offsets.get(dimension);
    if (o === undefined) throw new Error(`unknown dimension: ${dimension}`);
    return o;
  }
}

/** 两个激活集合的 IoU（相似性保持的度量） */
export function iou(a: readonly number[], b: readonly number[]): number {
  const sa = new Set(a);
  const sb = new Set(b);
  let inter = 0;
  for (const x of sa) if (sb.has(x)) inter++;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : inter / union;
}
