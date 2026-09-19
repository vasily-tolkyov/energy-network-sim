import { EnergyNetwork, hebbianLearn } from "../index.js";
import type { Pair } from "../prototype/world.js";
import { neuralDifferential } from "../layers/differential.js";

/** R2 需要的通道图最小结构面（PopChannelMap 与自形成适配器均满足） */
export interface R2ChannelSurface {
  readonly specs: readonly { readonly name: string; readonly bins: number }[];
  readonly neuronCount: number;
  readonly popSize: number;
  population(dimension: string, index: number): number[];
  encode(values: Readonly<Record<string, number>>): number[];
  binOf(neuronId: number): { channel: string; bin: number } | null;
  /** 可选：神经元（局部空间）的感受野中心值——自形成概念下用于场级档差 */
  fieldCenterOf?(neuronId: number): number | null;
}

/**
 * R2 比较层（神经化，群体编码版）：
 * 对完整经验模式（条件群体 ∪ 结果群体，8 群体 = 32 神经元）做一次
 * 重合场分层差分，然后双侧读出：
 * - R2B 侧：条件成员的共同项与差分候选（影响因素提取用）；
 * - R2A 侧：结果成员的共同结果、结果差异与变化标志（含方向与数值）。
 * 两个分支共享同一次动力学——互相配合，不是先后两级。
 *
 * θHigh 窗口（簇内 0.8、交集边双倍 ≈1.6）：
 * 最紧的是碰墙对（交集 20）：单侧满场 24.8 < θHigh < 交集塌缩后场 30.4。
 * 退化对（两侧结果完全相同）结果侧差分恒为空 → 如实判为无变化。
 */

export interface R2PopAnalysis {
  /** R2A：结果是否变化（结果侧有差分成员且至少一个通道 |档差| > ε） */
  readonly outcomeChanged: boolean;
  /** R2A：逐结果通道的带符号档差（含方向与数值；未变化通道不出现） */
  readonly outcomeDelta: Readonly<Record<string, number>>;
  /** R2B：条件共同项 */
  readonly conditionCommonNeurons: readonly number[];
  /** R2B：条件差分候选的通道归属（变化标志与数值分离） */
  readonly diffChannels: readonly string[];
  /** 双源 AND 判为影响因素的通道（差分 ∩ 结果变化） */
  readonly influentialChannels: readonly string[];
  readonly thetaHigh: number;
  /**
   * 差分构型不可判定（分层窗口不存在或交集为空）——调用方不得据此归因。
   * 评审 F04/C07/C12 修复：修复前回退常数 θHigh=27，小模式上共同项被熄灭、
   * 再被当作双侧差分，恒定结果的世界也报"发现影响因素"。
   */
  readonly undecidable: boolean;
}

export class R2PopLayer {
  private readonly wellRepeats = 8;

  constructor(
    readonly conditionMap: R2ChannelSurface,
    readonly outcomeMap: R2ChannelSurface,
  ) {}

  private undecidableResult(): R2PopAnalysis {
    return {
      undecidable: true,
      outcomeChanged: false,
      outcomeDelta: {},
      conditionCommonNeurons: [],
      diffChannels: [],
      influentialChannels: [],
      thetaHigh: NaN,
    };
  }

  analyzePair(pair: Pair): R2PopAnalysis {
    const condN = this.conditionMap.neuronCount;
    const outN = this.outcomeMap.neuronCount;
    const outcomeChangeNeuron = condN + outN;
    const influenceBase = condN + outN + 1;
    const neuronCount = influenceBase + this.conditionMap.specs.length;
    const net = new EnergyNetwork({
      neuronCount,
      activationEnergy: 1.0,
      maintenanceEnergy: 0.5,
      learningRate: 0.1,
      maxWeight: 3.0,
    });

    const patternOf = (e: Pair["e0"]): number[] => [
      ...this.conditionMap.encode(e.conditions),
      ...this.outcomeMap.encode(e.outcomes).map((id) => id + condN),
    ];
    const patternA = patternOf(pair.e0);
    const patternB = patternOf(pair.e1);
    hebbianLearn(net, patternA, this.wellRepeats);
    hebbianLearn(net, patternB, this.wellRepeats);
    // θHigh 自适应 + 分层自校验（评审 F04/C07/C12 修复，取代回退常数 27）：
    // 初值取 (hSingle, hCommon) 中点（交集为空时取 hSingle+ε）；
    // 分层结果必须让**全部交集成员驻留**（common == ∩），否则说明阈值过高
    // 把共同项也熄灭了（它们会被误当双侧差分——恒定世界误报因素的根因），
    // 降档重试；至多 3 次仍不成立才如实报"不可判定"。
    const setA = new Set(patternA);
    const setB = new Set(patternB);
    let inter = 0;
    for (const x of setA) if (setB.has(x)) inter++;
    const w = 0.1 * this.wellRepeats;
    const hSingle = w * (Math.max(patternA.length, patternB.length) - 1);
    const hCommon = 2 * w * Math.max(0, inter - 1);
    let thetaHigh = inter === 0 ? hSingle + 0.1 : (hSingle + hCommon) / 2;
    let diff = neuralDifferential(net, patternA, patternB, thetaHigh);
    for (let attempt = 0; attempt < 3 && inter > 0 && diff.common.length < inter && thetaHigh * 0.7 > 0.05; attempt++) {
      thetaHigh *= 0.7;
      diff = neuralDifferential(net, patternA, patternB, thetaHigh);
    }
    if (inter > 0 && diff.common.length < inter) {
      return this.undecidableResult();
    }

    // —— R2A 侧读出：结果成员里的差分 ——
    const outDiffA = diff.onlyA.filter((id) => id >= condN);
    const outDiffB = diff.onlyB.filter((id) => id >= condN);
    // 逐通道带符号档差：按 (通道, 档) 去重（群体成员只计一次），flag 与数值分离
    const binsOf = (ids: readonly number[]): Map<string, Set<number>> => {
      const m = new Map<string, Set<number>>();
      for (const id of ids) {
        const b = this.outcomeMap.binOf(id - condN);
        if (b) {
          const s = m.get(b.channel) ?? new Set<number>();
          s.add(b.bin);
          m.set(b.channel, s);
        }
      }
      return m;
    };
    const binsA = binsOf(outDiffA);
    const binsB = binsOf(outDiffB);
    const outcomeDelta: Record<string, number> = {};
    if (this.outcomeMap.fieldCenterOf) {
      // 场级档差（自形成概念下）：粗概念内部的细分变化也能识别——
      // 用差分神经元的感受野中心均值做差，不依赖概念索引
      const f = (id: number) => this.outcomeMap.fieldCenterOf!(id);
      const meanBy = (ids: readonly number[]): Map<string, number> => {
        const sum = new Map<string, { s: number; n: number }>();
        for (const id of ids) {
          const b = this.outcomeMap.binOf(id - condN);
          const c = f(id - condN);
          if (b && c !== null) {
            const e = sum.get(b.channel) ?? { s: 0, n: 0 };
            e.s += c;
            e.n++;
            sum.set(b.channel, e);
          }
        }
        const out = new Map<string, number>();
        for (const [ch, e] of sum) out.set(ch, e.n === 0 ? 0 : e.s / e.n);
        return out;
      };
      const meanA = meanBy(outDiffA);
      const meanB = meanBy(outDiffB);
      for (const [ch, mb] of meanB) {
        outcomeDelta[ch] = mb - (meanA.get(ch) ?? 0);
      }
    } else {
      for (const [ch, bins] of binsA) {
        outcomeDelta[ch] = (outcomeDelta[ch] ?? 0) - [...bins].reduce((a, b) => a + b, 0);
      }
      for (const [ch, bins] of binsB) {
        outcomeDelta[ch] = (outcomeDelta[ch] ?? 0) + [...bins].reduce((a, b) => a + b, 0);
      }
    }

    // —— R2B 侧读出：条件成员里的差分 + 双源 AND ——
    const condDiff = [...diff.onlyA, ...diff.onlyB].filter((id) => id < condN);
    // 变化标志与数值一致：至少一个结果通道 |档差| > ε 才算结果变化
    // （评审 C07：修复前恒定结果的对因差分侧非空而误报变化，delta 却全为 0）
    const outcomeChanged = Object.values(outcomeDelta).some((d) => Math.abs(d) > 1e-9);
    this.conditionMap.specs.forEach((spec, k) => {
      const x = influenceBase + k;
      net.strengthen(outcomeChangeNeuron, x, 1.0);
      for (let bin = 0; bin < spec.bins; bin++) {
        // 边权按该概念实际成员数标定：每个变化概念贡献恒为 0.5，
        // 避免概念成员数差异破坏双源阈值语义
        const members = this.conditionMap.population(spec.name, bin);
        const w = members.length === 0 ? 0 : 0.5 / members.length;
        for (const from of members) {
          net.strengthen(from, x, w);
        }
      }
    });
    const clamped = [...condDiff];
    if (outcomeChanged) clamped.push(outcomeChangeNeuron);
    const influenceNeurons = this.conditionMap.specs.map((_, k) => influenceBase + k);
    const andResult = net.settleAnnealed(clamped, [], {
      seed: 1,
      extraCandidates: influenceNeurons,
      quenchCandidatesOnly: true,
      levels: 8,
      sweepsPerLevel: 10,
    });
    const active = new Set(andResult.activeNeurons);
    const influentialChannels = this.conditionMap.specs
      .map((spec, k) => (active.has(influenceBase + k) ? spec.name : null))
      .filter((n): n is string => n !== null)
      .sort();
    const diffChannels = [
      ...new Set(
        condDiff
          .map((id) => this.conditionMap.binOf(id)?.channel)
          .filter((n): n is string => n !== undefined),
      ),
    ].sort();

    return {
      undecidable: false,
      outcomeChanged,
      outcomeDelta,
      conditionCommonNeurons: diff.common.filter((id) => id < condN),
      diffChannels,
      influentialChannels,
      thetaHigh,
    };
  }
}
