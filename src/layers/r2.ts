import { EnergyNetwork, hebbianLearn } from "../index.js";
import type { ChannelMap } from "../prototype/channels.js";
import type { Pair } from "../prototype/world.js";
import { compareOutcomes, type OutcomeComparison } from "../prototype/r2.js";
import { neuralDifferential } from "./differential.js";

/**
 * R2 比较层：R2A 结果比较 + R2B 条件差分（神经化）+ 双源 AND 影响判定。
 *
 * R2B：每次比较在独立的 R2B 工作网中进行——两次实验的条件模式在其中
 * 成阱（hebbianLearn ×8），再用高阈值重合场分层筛出共同项与差分候选。
 * 每对独立工作网，避免跨对共享背景造成的阱合并污染。
 * R2A：结果集仅 2 神经元，重合场分层不适用——结果比较保留读出层操作
 * （复用 prototype/r2 的 compareOutcomes），如实记录此边界。
 * 影响判定：影响因素神经元以双源 AND 点火（差分信号 × R2A 结果变化信号，
 * θ=1.5，单源 1.0 不越阈，双源 2.0 越阈）。
 */

export interface R2PairAnalysis {
  readonly outcome: OutcomeComparison;
  /** R2B 神经差分：共同项 */
  readonly commonConditions: readonly number[];
  /** R2B 神经差分：差分候选神经元 */
  readonly conditionDiffNeurons: readonly number[];
  /** 差分候选的通道归属（变化标志，与变化数值分离） */
  readonly diffChannels: readonly string[];
  /** 判为影响因素的通道（差分 ∩ 结果变化，双源 AND 点火） */
  readonly influentialChannels: readonly string[];
  readonly thetaHigh: number;
}

export class R2Layer {
  /** θHigh 窗口：单阱自维持 (6−1)×0.8=4.0 < θHigh < 交集自维持 (5−1)×1.6=6.4 */
  readonly thetaHigh = 5.0;
  private readonly wellRepeats = 8;

  constructor(readonly conditionMap: ChannelMap) {}

  analyzePair(pair: Pair): R2PairAnalysis {
    const neuronCount = this.conditionMap.neuronCount + 1 + this.conditionMap.specs.length;
    const outcomeChangeNeuron = this.conditionMap.neuronCount;
    const influenceBase = this.conditionMap.neuronCount + 1;
    const net = new EnergyNetwork({
      neuronCount,
      activationEnergy: 1.0,
      maintenanceEnergy: 0.5,
      learningRate: 0.1,
      maxWeight: 3.0,
    });

    // —— R2B：条件模式成阱 → 重合场分层差分 ——
    const condA = this.conditionMap.encode(pair.e0.conditions);
    const condB = this.conditionMap.encode(pair.e1.conditions);
    hebbianLearn(net, condA, this.wellRepeats);
    hebbianLearn(net, condB, this.wellRepeats);
    const diff = neuralDifferential(net, condA, condB, this.thetaHigh);

    // —— R2A：结果比较（读出层） ——
    const outcome = compareOutcomes(pair.e0.outcomes, pair.e1.outcomes);

    // —— AND：差分神经元 ∩ 结果变化信号 → 影响因素神经元 ——
    // 每条件神经元边权 0.5：单通道变化（恒 2 个差分神经元）贡献 1.0，
    // 与结果变化信号 1.0 合成 2.0 > θ=1.5；缺结果变化信号则 1.0 不越阈。
    // 读出用候选集受限退火：防止共同项势阱在 settle 中自点燃造成假影响。
    this.conditionMap.specs.forEach((spec, k) => {
      const x = influenceBase + k;
      net.strengthen(outcomeChangeNeuron, x, 1.0);
      for (let b = 0; b < spec.bins; b++) net.strengthen(spec.offset + b, x, 0.5);
    });
    const clamped: number[] = [...diff.onlyA, ...diff.onlyB];
    if (outcome.outcomeChanged) clamped.push(outcomeChangeNeuron);
    const influenceNeurons = this.conditionMap.specs.map((_, k) => influenceBase + k);
    const andResult = net.settleAnnealed(clamped, [], {
      seed: 1,
      extraCandidates: influenceNeurons,
      quenchCandidatesOnly: true,
    });
    const active = new Set(andResult.activeNeurons);
    const influentialChannels = this.conditionMap.specs
      .map((spec, k) => (active.has(influenceBase + k) ? spec.name : null))
      .filter((n): n is string => n !== null)
      .sort();

    const channelOf = new Map<number, string>();
    for (const spec of this.conditionMap.specs) {
      for (let b = 0; b < spec.bins; b++) channelOf.set(spec.offset + b, spec.name);
    }
    const diffChannels = [
      ...new Set(
        [...diff.onlyA, ...diff.onlyB].map((id) => channelOf.get(id)).filter((n): n is string => n !== undefined),
      ),
    ].sort();

    return {
      outcome,
      commonConditions: diff.common,
      conditionDiffNeurons: [...diff.onlyA, ...diff.onlyB].sort((a, b) => a - b),
      diffChannels,
      influentialChannels,
      thetaHigh: this.thetaHigh,
    };
  }
}
