import { EnergyNetwork, hebbianLearn } from "../index.js";
import { decodePopulation, PopChannelMap } from "./popmap.js";
import type { PopDecoded } from "./popmap.js";
import type { Conditions, Experiment, Outcomes } from "../prototype/world.js";

/**
 * 群体编码 + 规则核的 R3 规则记忆。
 *
 * 结构（三层连接，概念全部由共现学习形成）：
 *   条件群体（每档 popSize 个神经元，各经验共享）
 *     → 规则核（每条经验私有的 coreSize 个神经元）
 *     → 结果群体（共享）
 *
 * - 教学：条件群体 ∪ 核 ∪ 结果群体共激活赫布绑定（基础封顶 baseCap）；
 *   核间建抑制边 Γ（候选规则互斥，赢家通吃）；
 * - 侧重（§7.1 类比）：影响因素通道群体 → 该经验核的连接 ×(η·G·m²) 增强；
 * - 预测：钳制查询条件群体，候选集 = 全部核 ∪ 结果群体，局部退火选出
 *   能耗极小的获胜核（匹配条件越多场越强），获胜核点亮其结果群体；
 * - 概念不再对应单点：档位 = 群体势阱，规则 = 私有核势阱。
 */

export interface PopMemoryConfig {
  readonly coreSize?: number;
  readonly maxRules?: number;
  readonly gammaCore?: number;
  readonly gammaOutcome?: number;
  readonly r1Repeats?: number;
  readonly influenceGain?: number;
  readonly r3Repeats?: number;
}

export class PopRuleMemory {
  readonly net: EnergyNetwork;
  readonly coreSize: number;
  private readonly maxRules: number;
  private coreCursor = 0;
  private readonly signatureToCore = new Map<string, readonly number[]>();
  private readonly cores: number[][] = [];

  constructor(
    readonly conditionMap: PopChannelMap,
    readonly outcomeMap: PopChannelMap,
    config: PopMemoryConfig & { activationEnergy?: number; maintenanceEnergy?: number; learningRate?: number; maxWeight?: number } = {},
  ) {
    this.coreSize = config.coreSize ?? 4;
    this.maxRules = config.maxRules ?? 128;
    const totalNeurons =
      conditionMap.neuronCount + outcomeMap.neuronCount + this.maxRules * this.coreSize;
    this.net = new EnergyNetwork({
      neuronCount: totalNeurons,
      activationEnergy: config.activationEnergy ?? 1.0,
      maintenanceEnergy: config.maintenanceEnergy ?? 0.5,
      learningRate: config.learningRate ?? 0.1,
      maxWeight: config.maxWeight ?? 3.0,
    });
    // 注意：结果通道的档位互斥不再预置——由 teachExclusion 从换对学习。
  }

  /**
   * 互斥关系从经验学习：教师对换过的两个值 = 同一通道的替代值，
   * 在两档群体之间写抑制边（条件通道与结果通道同样处理）。
   * 未被对换观察过的档之间没有抑制——不学不存在的关系。
   */
  teachExclusion(
    map: "condition" | "outcome",
    channel: string,
    binA: number,
    binB: number,
    strength: number,
  ): void {
    if (binA === binB || strength <= 0) return;
    const m = map === "condition" ? this.conditionMap : this.outcomeMap;
    const off = map === "condition" ? 0 : this.conditionMap.neuronCount;
    for (const x of m.population(channel, binA)) {
      for (const y of m.population(channel, binB)) {
        this.net.strengthenInhibitory(off + x, off + y, strength);
      }
    }
    // 登记该通道已被观察到的档位（否决布线用）
    const seen = this.observedBins.get(channel) ?? new Set<number>();
    seen.add(binA);
    seen.add(binB);
    this.observedBins.set(channel, seen);
  }

  private readonly observedBins = new Map<string, Set<number>>();

  private coreBase(): number {
    return this.conditionMap.neuronCount + this.outcomeMap.neuronCount;
  }

  private signature(conditions: Conditions): string {
    return Object.keys(conditions)
      .sort()
      .map((k) => `${k}=${conditions[k]}`)
      .join(";");
  }

  /** 分配（或复用）一条经验的私有核 */
  private coreFor(conditions: Conditions): readonly number[] {
    const sig = this.signature(conditions);
    const existing = this.signatureToCore.get(sig);
    if (existing) return existing;
    if (this.coreCursor >= this.maxRules) {
      throw new Error(`rule capacity exhausted (maxRules=${this.maxRules})`);
    }
    const base = this.coreBase() + this.coreCursor * this.coreSize;
    const core = Array.from({ length: this.coreSize }, (_, k) => base + k);
    this.coreCursor++;
    // 与既有核互斥（Γ_core）
    for (const other of this.cores) {
      for (const x of core) {
        for (const y of other) this.net.strengthenInhibitory(x, y, 1.5);
      }
    }
    this.cores.push(core);
    this.signatureToCore.set(sig, core);
    return core;
  }

  private outcomeIds(exp: Experiment): number[] {
    return this.outcomeMap.encode(exp.outcomes).map((id) => id + this.conditionMap.neuronCount);
  }

  get ruleCount(): number {
    return this.coreCursor;
  }

  /** 教学模式：两段绑定——条件群体↔核、核↔结果群体。
   * 不直接绑条件↔结果：规则核必须是条件与结果之间的唯一桥梁。
   * 试过再加封顶 0.1 的弱直连做插值偏置，实测在规则多时重新引入
   * 混答（直连把每个历史结果档都喂一点分），已回滚并记录在案。 */
  teachExperience(exp: Experiment, repeats: number, baseCap = 0.4): void {
    const core = this.coreFor(exp.conditions);
    const condPops = this.conditionMap.encode(exp.conditions);
    hebbianLearn(this.net, [...condPops, ...core], repeats, undefined, baseCap);
    // 结果按通道分别绑定到核：不允许出现 reb↔rs 等跨结果通道边——
    // 否则被驱动的结果档会经"结果→结果"边接力驱动别的档（实测发现的泄漏）。
    for (const spec of this.outcomeMap.specs) {
      const pops = this.outcomeMap.population(spec.name, exp.outcomes[spec.name]!);
      hebbianLearn(
        this.net,
        [...core, ...pops.map((id) => id + this.conditionMap.neuronCount)],
        repeats,
        undefined,
        baseCap,
      );
    }
    for (const [ch, bin] of Object.entries(exp.conditions)) {
      const seen = this.observedBins.get(ch) ?? new Set<number>();
      seen.add(bin);
      this.observedBins.set(ch, seen);
    }
  }

  /**
   * R3 侧重（含否决）：影响因素通道群体 → 核，按 η·G·m² 增强（加分）；
   * 同时给该通道已被观察到的其余档 → 核写否决抑制边（罚分）。
   * 否决是分级强度：一次影响因素不匹配的惩罚（gammaVeto×群体）须超过
   * 任何单通道最大支持，从而 精确匹配核 恒胜 含一处不匹配的核；
   * 全无精确匹配时，同级不匹配数的核按支持竞争——泛化兜底不死。
   */
  bindInfluence(
    exp: Experiment,
    channelBoost: Readonly<Record<string, number>>,
    repeats: number,
    gammaVeto = 1.5,
  ): void {
    // 记录侧重/否决参数的并集（逐通道取最大），供响应学习写入的新核"持证上岗"
    for (const [ch, delta] of Object.entries(channelBoost)) {
      this.lastBoost[ch] = Math.max(this.lastBoost[ch] ?? 0, delta);
    }
    this.lastGammaVeto = gammaVeto;
    const core = this.coreFor(exp.conditions);
    for (const [ch, delta] of Object.entries(channelBoost)) {
      if (delta > 0 && repeats > 0) {
        for (const from of this.conditionMap.population(ch, exp.conditions[ch]!)) {
          for (let r = 0; r < repeats; r++) {
            for (const to of core) this.net.strengthen(from, to, delta);
          }
        }
      }
      // 否决：同通道的其余已观察档 → 核 抑制
      const own = exp.conditions[ch]!;
      for (const alt of this.observedBins.get(ch) ?? []) {
        if (alt === own) continue;
        for (const from of this.conditionMap.population(ch, alt)) {
          for (const to of core) this.net.strengthenInhibitory(from, to, gammaVeto);
        }
      }
    }
  }

  /**
   * 预测：钳制查询条件群体，候选集 = 全部核 ∪ 结果群体，
   * 局部退火选出获胜核，读出其结果群体。
   */
  predict(
    query: Conditions,
    seed: number,
  ): { decoded: Record<string, PopDecoded>; activeNeurons: readonly number[]; energy: number } {
    const input = this.conditionMap.encode(query);
    const outcomeNeurons = Array.from(
      { length: this.outcomeMap.neuronCount },
      (_, k) => this.conditionMap.neuronCount + k,
    );
    const coreNeurons = this.cores.flat();
    const result = this.net.settleAnnealed(input, [], {
      seed,
      extraCandidates: [...outcomeNeurons, ...coreNeurons],
      quenchCandidatesOnly: true,
      levels: 12,
      sweepsPerLevel: 20,
    });
    return {
      decoded: decodePopulation(
        result.activeNeurons
          .filter((id) => id >= this.conditionMap.neuronCount)
          .map((id) => id - this.conditionMap.neuronCount),
        this.outcomeMap,
        this.outcomeMap.channelNames(),
      ),
      activeNeurons: result.activeNeurons,
      energy: result.energy,
    };
  }

  /** 响应模式学习：新查询分配/复用核并绑定预测结果（小步长同封顶） */
  learnFromQuery(query: Conditions, predicted: Record<string, PopDecoded>, repeats: number): void {
    const core = this.coreFor(query);
    const outcomeIds: number[] = [];
    for (const [ch, bin] of Object.entries(predicted)) {
      if (typeof bin === "number") {
        for (const id of this.outcomeMap.population(ch, bin)) {
          outcomeIds.push(id + this.conditionMap.neuronCount);
        }
      }
    }
    if (outcomeIds.length > 0) {
      hebbianLearn(this.net, [...this.conditionMap.encode(query), ...core, ...outcomeIds], repeats, 0.05, 0.4);
    }
  }

  /**
   * 门控版响应学习（注意力监测器用）：经核绑定**观察到的真实结果**，
   * 不是预测。证据分级：观察写入封顶 0.6，高于自写猜测的 0.4——
   * 纠错时正确绑定能压住旧的错误绑定（自强化错误的解药）。
   * 新核同时"持证上岗"：按最近一次侧重给它补正/否决边——
   * 否则未持证的新核会在竞争中充当干净外表的噪声竞争者。
   */
  learnFromObservation(query: Conditions, observed: Outcomes, repeats: number): void {
    const core = this.coreFor(query);
    const condIds = this.conditionMap.encode(query);
    if (Object.values(observed).some((v) => typeof v === "number")) {
      hebbianLearn(this.net, [...condIds, ...core], repeats, 0.1, 0.6);
      // 结果按通道分别绑定到核（同 teachExperience，防跨通道接力）
      for (const spec of this.outcomeMap.specs) {
        const bin = observed[spec.name];
        if (typeof bin === "number") {
          const pops = this.outcomeMap.population(spec.name, bin);
          hebbianLearn(this.net, [...core, ...pops.map((id) => id + this.conditionMap.neuronCount)], repeats, 0.1, 0.6);
        }
      }
    }
    // 持证：影响因素加分边 + 替代档否决边（与 bindInfluence 同一规则）
    for (const [ch, delta] of Object.entries(this.lastBoost)) {
      if (delta > 0) {
        for (const from of this.conditionMap.population(ch, query[ch]!)) {
          for (const to of core) this.net.strengthen(from, to, delta);
        }
      }
      for (const alt of this.observedBins.get(ch) ?? []) {
        if (alt === query[ch]) continue;
        for (const from of this.conditionMap.population(ch, alt)) {
          for (const to of core) this.net.strengthenInhibitory(from, to, this.lastGammaVeto);
        }
      }
    }
  }

  private lastBoost: Record<string, number> = {};
  private lastGammaVeto = 1.5;
}
