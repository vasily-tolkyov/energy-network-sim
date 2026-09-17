import { EnergyNetwork, hebbianLearn, learnSequence } from "../index.js";
import type { ChannelMap } from "../prototype/channels.js";
import type { Conditions, Experiment, Outcomes } from "../prototype/world.js";

/**
 * R1 表层经验层：独立的 EnergyNetwork。
 * 每次实验在 R1 中形成一个"输入条件势阱 —有向通道→ 变化结果势阱"的势阱对：
 * - 条件∪结果共激活赫布绑定（对称 W，捕获/补全）；
 * - 条件→结果的有向通道（时序 D，链式生成的基础）。
 * 轮末恢复静息，连接保留（二态要求）。
 * 向后层传递的是从 R1 状态读出的表征（本轮激活的神经元集合），不转发外部探针。
 */

export interface R1Episode {
  readonly conditions: Conditions;
  readonly outcomes: Outcomes;
  readonly conditionNeurons: readonly number[];
  readonly outcomeNeurons: readonly number[];
}

export class R1Layer {
  readonly net: EnergyNetwork;
  private readonly episodes: R1Episode[] = [];

  constructor(
    readonly conditionMap: ChannelMap,
    readonly outcomeMap: ChannelMap,
    netConfig: { neuronCount?: number; activationEnergy?: number; maintenanceEnergy?: number; learningRate?: number; maxWeight?: number } = {},
  ) {
    this.net = new EnergyNetwork({
      neuronCount: netConfig.neuronCount ?? conditionMap.neuronCount + outcomeMap.neuronCount,
      activationEnergy: netConfig.activationEnergy ?? 1.0,
      maintenanceEnergy: netConfig.maintenanceEnergy ?? 0.5,
      learningRate: netConfig.learningRate ?? 0.1,
      maxWeight: netConfig.maxWeight ?? 3.0,
    });
  }

  private outcomeIds(outcomes: Outcomes): number[] {
    return this.outcomeMap.encode(outcomes).map((id) => id + this.conditionMap.neuronCount);
  }

  /** 教学模式：教师驱动一次实验的共激活绑定 + 条件→结果有向通道 */
  teachEpisode(exp: Experiment, repeats = 8, directedRepeats = 8): R1Episode {
    const conditionNeurons = this.conditionMap.encode(exp.conditions);
    const outcomeNeurons = this.outcomeIds(exp.outcomes);
    hebbianLearn(this.net, [...conditionNeurons, ...outcomeNeurons], repeats);
    learnSequence(this.net, [conditionNeurons, outcomeNeurons], directedRepeats);
    const episode: R1Episode = {
      conditions: exp.conditions,
      outcomes: exp.outcomes,
      conditionNeurons,
      outcomeNeurons,
    };
    this.episodes.push(episode);
    this.net.reset(); // 轮末恢复静息
    return episode;
  }

  /** R1 状态读出：全部已存经验（供 R2A/R2B 比较） */
  readoutEpisodes(): readonly R1Episode[] {
    return this.episodes;
  }
}
