/**
 * 通道编码：连续/离散传感量 → 二值神经元群体（每档一个神经元）。
 * R1 的输入接口：一次经验 = 条件通道各取一档 + 结果通道各取一档。
 */

export interface ChannelSpec {
  readonly name: string;
  readonly bins: number;
}

export class ChannelMap {
  readonly specs: readonly (ChannelSpec & { readonly offset: number })[];
  readonly neuronCount: number;
  private readonly byName = new Map<string, ChannelSpec & { readonly offset: number }>();

  constructor(specs: readonly ChannelSpec[]) {
    let offset = 0;
    const resolved: (ChannelSpec & { offset: number })[] = [];
    for (const spec of specs) {
      if (spec.bins < 1) throw new Error(`channel ${spec.name}: bins must be >= 1`);
      resolved.push({ ...spec, offset });
      this.byName.set(spec.name, resolved[resolved.length - 1]!);
      offset += spec.bins;
    }
    this.specs = resolved;
    this.neuronCount = offset;
  }

  neuron(channel: string, bin: number): number {
    const spec = this.byName.get(channel);
    if (!spec) throw new Error(`unknown channel: ${channel}`);
    if (bin < 0 || bin >= spec.bins) {
      throw new Error(`channel ${channel}: bin ${bin} out of range [0, ${spec.bins})`);
    }
    return spec.offset + bin;
  }

  /** 编码一组取值（通道名→档位）为神经元 id 列表 */
  encode(values: Readonly<Record<string, number>>): number[] {
    return Object.entries(values).map(([ch, bin]) => this.neuron(ch, bin));
  }

  /** 通道定义（不含结果通道）与全部神经元 */
  channelNames(): string[] {
    return this.specs.map((s) => s.name);
  }
}

/** 预测读出解码结果：每通道恰一个激活档 → 该档；0 个 → null（无预测）；多个 → "ambiguous" */
export type DecodedOutcome = number | null | "ambiguous";

/**
 * 解码激活神经元中的通道取值。idOffset：通道神经元在网络中的起始偏移
 * （结果通道排在条件通道之后时传 conditionNeuronCount）。
 */
export function decodeChannels(
  activeNeurons: Iterable<number>,
  map: ChannelMap,
  channelNames: readonly string[],
  idOffset = 0,
): Record<string, DecodedOutcome> {
  const active = new Set(activeNeurons);
  const out: Record<string, DecodedOutcome> = {};
  for (const name of channelNames) {
    const spec = map.specs.find((s) => s.name === name)!;
    const hits: number[] = [];
    for (let b = 0; b < spec.bins; b++) {
      if (active.has(idOffset + spec.offset + b)) hits.push(b);
    }
    out[name] = hits.length === 1 ? hits[0]! : hits.length === 0 ? null : "ambiguous";
  }
  return out;
}
