/**
 * 群体编码：每个通道的每个档位对应一小群神经元（popSize 个），
 * 概念在反复共现中由这小群神经元聚成势阱，而不是指定单点。
 * 输入/查询钳制整个群体（或其子集作为部分线索），捕获机制补齐其余成员。
 */

import { integer } from "../validate.js";

export interface PopChannelSpec {
  readonly name: string;
  readonly bins: number;
}

export class PopChannelMap {
  readonly specs: readonly (PopChannelSpec & { readonly offset: number })[];
  readonly popSize: number;
  readonly neuronCount: number;
  private readonly byName = new Map<string, PopChannelSpec & { offset: number }>();

  constructor(specs: readonly PopChannelSpec[], popSize = 4) {
    integer(popSize, "popSize", 1);
    this.popSize = popSize;
    let offset = 0;
    const resolved: (PopChannelSpec & { offset: number })[] = [];
    for (const spec of specs) {
      integer(spec.bins, `channel ${spec.name} bins`, 1);
      if (this.byName.has(spec.name)) throw new Error(`duplicate channel: ${spec.name}`);
      resolved.push({ ...spec, offset });
      this.byName.set(spec.name, resolved[resolved.length - 1]!);
      offset += spec.bins * popSize;
    }
    this.specs = resolved;
    this.neuronCount = offset;
  }

  /** 一个档位的群体成员 id 列表 */
  population(channel: string, bin: number): number[] {
    const spec = this.byName.get(channel);
    if (!spec) throw new Error(`unknown channel: ${channel}`);
    if (!Number.isInteger(bin) || bin < 0 || bin >= spec.bins) {
      throw new Error(`channel ${channel}: bin ${bin} out of range [0, ${spec.bins})`);
    }
    const base = spec.offset + bin * this.popSize;
    return Array.from({ length: this.popSize }, (_, k) => base + k);
  }

  /** 编码一组取值：每个值取整个群体 */
  encode(values: Readonly<Record<string, number>>): number[] {
    return Object.entries(values).flatMap(([ch, bin]) => this.population(ch, bin));
  }

  /** 部分线索编码：每个值只取群体前 cueSize 个成员 */
  encodeCue(values: Readonly<Record<string, number>>, cueSize: number): number[] {
    return Object.entries(values).flatMap(([ch, bin]) =>
      this.population(ch, bin).slice(0, Math.max(1, cueSize)),
    );
  }

  /** 神经元 id → 所属通道与档位（读出用） */
  binOf(neuronId: number): { channel: string; bin: number } | null {
    for (const spec of this.specs) {
      if (neuronId >= spec.offset && neuronId < spec.offset + spec.bins * this.popSize) {
        return { channel: spec.name, bin: Math.floor((neuronId - spec.offset) / this.popSize) };
      }
    }
    return null;
  }

  channelNames(): string[] {
    return this.specs.map((s) => s.name);
  }
}

export type PopDecoded = number | null | "ambiguous";

/** 按群体激活率解码：取激活率最高的档；最高 < threshold → null；并列 → ambiguous */
export function decodePopulation(
  activeNeurons: Iterable<number>,
  map: PopChannelMap,
  channelNames: readonly string[],
  threshold = 0.5,
): Record<string, PopDecoded> {
  const active = new Set(activeNeurons);
  const out: Record<string, PopDecoded> = {};
  for (const name of channelNames) {
    const spec = map.specs.find((s) => s.name === name)!;
    let bestBin = -1;
    let bestRatio = 0;
    let tie = false;
    for (let b = 0; b < spec.bins; b++) {
      const base = spec.offset + b * map.popSize;
      let hit = 0;
      for (let k = 0; k < map.popSize; k++) if (active.has(base + k)) hit++;
      const ratio = hit / map.popSize;
      if (ratio > bestRatio + 1e-9) {
        bestRatio = ratio;
        bestBin = b;
        tie = false;
      } else if (Math.abs(ratio - bestRatio) <= 1e-9 && ratio > 0) {
        tie = true;
      }
    }
    out[name] = bestRatio < threshold ? null : tie ? "ambiguous" : bestBin;
  }
  return out;
}
