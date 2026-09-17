import { EnergyNetwork, hebbianLearn } from "../index.js";
import { ChannelMap, decodeChannels } from "./channels.js";
import type { DecodedOutcome } from "./channels.js";
import type { Conditions, Experiment } from "./world.js";

/**
 * R3：规则的绑定、侧重与唤醒读出。
 * - 绑定：每次经验的条件∪结果共激活赫布绑定（基础耦合 η×r1Repeats）；
 * - 侧重（§7.1 类比）：R2B 判为影响因素的通道，其条件神经元→结果神经元
 *   的连接获得额外增强 η×G×r3Repeats；非影响背景保留基础耦合；
 * - 互斥：结果通道的档位之间建抑制边 Γ（一个通道不能同时取两档）；
 * - 唤醒：查询钳制条件神经元，候选集 = 查询 ∪ 结果通道神经元，
 *   局部退火求该候选集内的能耗极小模式 → 读出激活的结果档位。
 */

export interface R3Config {
  /** R1 经验绑定重复数，默认 4（基础边强 η×4） */
  readonly r1Repeats?: number;
  /** 影响侧重增益 G，默认 3；G=0 为断侧重消融 */
  readonly influenceGain?: number;
  /** R3 侧重绑定重复数，默认 4 */
  readonly r3Repeats?: number;
  /** 结果通道档位间抑制强度，默认 0.5 */
  readonly gamma?: number;
}

export class RuleMemory {
  readonly outcomeNeurons: readonly number[];

  constructor(
    readonly net: EnergyNetwork,
    readonly conditionMap: ChannelMap,
    readonly outcomeMap: ChannelMap,
    config: R3Config = {},
  ) {
    // 结果通道档位互斥抑制（结构先验：一个通道不能同时取两档）
    // 注意：网络中结果神经元排在条件神经元之后，必须加条件偏移
    const gamma = config.gamma ?? 0.5;
    const base = conditionMap.neuronCount;
    for (const spec of outcomeMap.specs) {
      for (let a = 0; a < spec.bins; a++) {
        for (let b = a + 1; b < spec.bins; b++) {
          net.strengthenInhibitory(base + spec.offset + a, base + spec.offset + b, gamma);
        }
      }
    }
    this.outcomeNeurons = Array.from(
      { length: outcomeMap.neuronCount },
      (_, k) => conditionMap.neuronCount + k,
    );
  }

  private outcomeIds(exp: Experiment): number[] {
    return this.outcomeMap
      .encode(exp.outcomes)
      .map((id) => id + this.conditionMap.neuronCount);
  }

  /** 教学模式：教师驱动一次经验的共激活绑定（此模式关闭预测输出）。基础耦合封顶 baseCap：背景连接不随频次增长（§7.1 的均匀 k_h） */
  teachExperience(exp: Experiment, repeats: number, baseCap = 0.4): void {
    const condIds = this.conditionMap.encode(exp.conditions);
    hebbianLearn(this.net, [...condIds, ...this.outcomeIds(exp)], repeats, undefined, baseCap);
  }

  /** R3 侧重：对本组判为影响因素的通道，增强其条件神经元→结果神经元连接（逐通道步长 = η·G·m²） */
  bindInfluence(exp: Experiment, channelBoost: Readonly<Record<string, number>>, repeats: number): void {
    if (repeats <= 0) return;
    const outs = this.outcomeIds(exp);
    for (const [ch, delta] of Object.entries(channelBoost)) {
      if (!(delta > 0)) continue;
      const from = this.conditionMap.neuron(ch, exp.conditions[ch]!);
      for (let r = 0; r < repeats; r++) {
        for (const to of outs) this.net.strengthen(from, to, delta);
      }
    }
  }

  /**
   * 唤醒与预测读出：在当前输入之上产生结果（不是只报告规则名）。
   * 返回解码后的结果通道与各内部读数（供评估）。
   */
  predict(
    query: Conditions,
    seed: number,
  ): { decoded: Record<string, DecodedOutcome>; activeNeurons: readonly number[]; energy: number } {
    const input = this.conditionMap.encode(query);
    const result = this.net.settleAnnealed(input, [], {
      seed,
      extraCandidates: this.outcomeNeurons,
      quenchCandidatesOnly: true, // 读出场景：候选集外冻结，防反传雪崩
    });
    return {
      decoded: decodeChannels(result.activeNeurons, this.outcomeMap, this.outcomeMap.channelNames(), this.conditionMap.neuronCount),
      activeNeurons: result.activeNeurons,
      energy: result.energy,
    };
  }

  /** 消融用：纯贪心读出（无退火竞争），结果取 settle 后的激活结果神经元 */
  predictGreedy(query: Conditions): { decoded: Record<string, DecodedOutcome> } {
    const input = this.conditionMap.encode(query);
    const result = this.net.settle(input);
    return {
      decoded: decodeChannels(result.activeNeurons, this.outcomeMap, this.outcomeMap.channelNames(), this.conditionMap.neuronCount),
    };
  }

  /** 响应模式学习：查询轮的新输入仍参与学习（更小步长 + 与基础耦合同一封顶，防漂移爆炸） */
  learnFromQuery(query: Conditions, predicted: Record<string, DecodedOutcome>, repeats: number): void {
    const condIds = this.conditionMap.encode(query);
    const outcomeIds: number[] = [];
    for (const [ch, bin] of Object.entries(predicted)) {
      if (typeof bin === "number") {
        outcomeIds.push(this.conditionMap.neuronCount + this.outcomeMap.neuron(ch, bin));
      }
    }
    if (outcomeIds.length > 0) hebbianLearn(this.net, [...condIds, ...outcomeIds], repeats, 0.05, 0.4);
  }
}
