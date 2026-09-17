import { EnergyNetwork, hebbianLearn } from "../../index.js";
import type { SensoryEncoder } from "./sensory.js";
import type { EmergentMap } from "./emergent-map.js";

/**
 * 场级规则记忆（M4）：R3 直接工作在感受野场级，不在概念索引上。
 *
 * 设计依据（形成实验的如实结论）：共现成阱产出的是"粗粒度的共性区域"
 * （课程密集覆盖的低值区合并为一阱，稀疏处自动分离）——这是文档
 * "大范围共性及其细分结构应来自经验"的合理形态。细分发生在场级：
 * 每条规则核绑定的是该经验的**确切感受野集合**（场级签名），
 * 概念只用于解释（标签/对齐）与替代域否决的边界。
 * 读出：结果感受野的激活 → 中心值质心；双峰分布如实报歧义。
 */

export interface FieldRule {
  readonly core: readonly number[];
  readonly conditionFields: readonly number[];
  readonly outcomeFields: readonly number[];
}

export interface OutcomeCluster {
  /** 簇中心（成员感受野中心均值） */
  center: number;
  /** 簇内激活感受野数 */
  count: number;
}

export interface FieldPrediction {
  /** 各结果维度的读出中心值（单峰）；双峰或无激活 → null */
  readonly values: Record<string, number | null>;
  /**
   * 如实歧义上报：单峰维度给 1 簇；双峰维度给出全部簇（网络实际形成的
   * 分布——联盟答案分裂时不逼单选，把不确定性交出来）。
   */
  readonly distribution: Record<string, readonly OutcomeCluster[]>;
  readonly ambiguous: readonly string[];
  readonly activeNeurons: readonly number[];
  readonly energy: number;
}

export class FieldRuleMemory {
  readonly net: EnergyNetwork;
  private readonly rules: FieldRule[] = [];
  private readonly signatureToCore = new Map<string, readonly number[]>();
  private coreCursor = 0;
  private readonly coreSize: number;
  private readonly coreBase: number;
  private readonly poolBase: number;
  private readonly poolSize = 2;
  /** 池→核抑制强度：两核联盟时弱者净场 < θ 而胜者存活（按支持 ~20-27 标定） */
  private readonly poolGamma = 7.0;
  private lastBoost: Record<string, number> = {};
  private lastGammaVeto = 1.2;

  constructor(
    readonly encoder: SensoryEncoder,
    readonly emergent: EmergentMap,
    config: { coreSize?: number; maxRules?: number; activationEnergy?: number; maintenanceEnergy?: number; learningRate?: number; maxWeight?: number } = {},
  ) {
    this.coreSize = config.coreSize ?? 4;
    const maxRules = config.maxRules ?? 128;
    this.coreBase = encoder.neuronCount;
    this.poolBase = encoder.neuronCount + maxRules * this.coreSize;
    this.net = new EnergyNetwork({
      neuronCount: encoder.neuronCount + maxRules * this.coreSize + this.poolSize,
      activationEnergy: config.activationEnergy ?? 1.0,
      maintenanceEnergy: config.maintenanceEnergy ?? 0.5,
      learningRate: config.learningRate ?? 0.1,
      maxWeight: config.maxWeight ?? 3.0,
    });
  }

  get ruleCount(): number {
    return this.rules.length;
  }

  private signatureOf(fields: readonly number[]): string {
    return [...fields].sort((a, b) => a - b).join(",");
  }

  private coreFor(condFields: readonly number[]): readonly number[] {
    const sig = this.signatureOf(condFields);
    const existing = this.signatureToCore.get(sig);
    if (existing) return existing;
    const base = this.coreBase + this.coreCursor * this.coreSize;
    const core = Array.from({ length: this.coreSize }, (_, k) => base + k);
    this.coreCursor++;
    for (const rule of this.rules) {
      for (const x of core) {
        for (const y of rule.core) this.net.strengthenInhibitory(x, y, 3.0);
      }
    }
    // 全局抑制池接线（WTA 电路，前馈抑制版）：
    // 核神经元 → 池神经元 k：权重 1/(k+1)——池按核活动总量分级招募；
    // 池神经元 → 核神经元：γ_pool 前馈抑制（DI，非平衡驱动）——
    // 池压制核而不被反向压制，联盟越大压制越强，逐个淘汰至单核胜出。
    for (const x of core) {
      for (let k = 0; k < this.poolSize; k++) {
        // 单核（总量 4）时池沉睡（w_0=0.3 需 >5 才点燃），≥2 核时点燃压制
        this.net.strengthen(x, this.poolBase + k, 0.3 / (k + 1));
        this.net.strengthenDirectedInhibitory(this.poolBase + k, x, 20, 20);
      }
    }
    this.signatureToCore.set(sig, core);
    return core;
  }

  /**
   * 教学：场级三段绑定——条件场↔核、核↔结果场（按结果维度分别绑定，防跨维接力）。
   * 条件场集合即该经验的场级签名（概念不参与，保持细分精度）。
   */
  teachExperiment(
    conditions: Record<string, number>,
    outcomes: Record<string, number>,
    repeats = 4,
    baseCap = 0.4,
  ): void {
    const condFields = this.encoder.encode(conditions);
    const sig = this.signatureOf(condFields);
    const isNew = !this.signatureToCore.has(sig);
    const core = this.coreFor(condFields);
    hebbianLearn(this.net, [...condFields, ...core], repeats, undefined, baseCap);
    // 结果场星型绑定：只写 核↔结果场，不写 结果场↔结果场——
    // 否则相邻值的结果场跨经验串成自维持幻影块（实测发现：幻影块借
    // 换对抑制边反压真值区域）。读出必须由核驱动，不许自维持。
    for (const dim of this.encoder.dimensions) {
      if (outcomes[dim.name] === undefined) continue;
      const outFields = this.encoder.encodeDimension(dim.name, outcomes[dim.name]!);
      // 星型驱动边直接写到 0.6（高于条件段的 0.4）：核缺一个成员时结果场
      // 仍能被点燃——实测 3/4 核驱动 4 结果野 = 1.2 < θ 致结果读空
      for (const to of outFields) {
        for (const from of core) this.net.strengthen(from, to, 0.6, 0.6);
      }
    }
    if (isNew) {
      this.rules.push({ core, conditionFields: condFields, outcomeFields: this.encoder.encode(outcomes) });
    }
  }

  /**
   * 侧重+否决（场级）：影响因素维度的感受野 → 核 加分（η·G·m²）；
   * 否决 = 该维度**不在本核条件场中的全部感受野** → 核 抑制——
   * 场级否决按非重叠野数比例惩罚，同一粗概念内的相邻核也能区分
   * （概念级否决对同概念内的核无效，实测证实）。
   */
  bindInfluence(
    conditions: Record<string, number>,
    channelBoost: Readonly<Record<string, number>>,
    repeats = 4,
    gammaVeto = 1.2,
  ): void {
    for (const [dim, delta] of Object.entries(channelBoost)) {
      this.lastBoost[dim] = Math.max(this.lastBoost[dim] ?? 0, delta);
    }
    this.lastGammaVeto = gammaVeto;
    const condFields = this.encoder.encode(conditions);
    const core = this.coreFor(condFields);
    const own = new Set(condFields);
    for (const [dim, delta] of Object.entries(channelBoost)) {
      const ownFields = this.encoder.encodeDimension(dim, conditions[dim]!);
      if (delta > 0 && repeats > 0) {
        for (const from of ownFields) {
          for (let r = 0; r < repeats; r++) {
            for (const to of core) this.net.strengthen(from, to, delta);
          }
        }
      }
      // 场级否决：该维全部感受野中不属于本核条件场者 → 核 抑制。
      // （试过按影响幅度缩放否决强度：门控略升但整体下降，已回退——记录）
      const vetoGamma = gammaVeto;
      const offset = this.encoder.dimensionOffset(dim);
      for (let k = 0; k < this.encoder.fieldsPerDim; k++) {
        const from = offset + k;
        if (own.has(from)) continue;
        for (const to of core) this.net.strengthenInhibitory(from, to, vetoGamma);
      }
    }
  }

  /**
   * 预测：钳制查询的感受野，候选集 = 全部核 ∪ 全部结果感受野，
   * 局部退火选出获胜核，结果场中心值质心读出；双峰如实报歧义。
   */
  predict(query: Record<string, number>, seed: number): FieldPrediction {
    const input = this.encoder.encode(query);
    const coreNeurons = this.rules.flatMap((r) => [...r.core]);
    const result = this.net.settleAnnealed(input, [], {
      seed,
      extraCandidates: [
        ...coreNeurons,
        ...this.outcomeDimFields,
        ...Array.from({ length: this.poolSize }, (_, k) => this.poolBase + k),
      ],
      quenchCandidatesOnly: true,
      levels: 12,
      sweepsPerLevel: 20,
    });
    const active = new Set(result.activeNeurons);
    const values: Record<string, number | null> = {};
    const distribution: Record<string, readonly OutcomeCluster[]> = {};
    const ambiguous: string[] = [];
    for (const dim of this.outcomeDims) {
      const offset = this.encoder.dimensionOffset(dim);
      const centers = this.encoder.fieldCenters(dim);
      const actCenters: number[] = [];
      for (let k = 0; k < centers.length; k++) {
        if (active.has(offset + k)) actCenters.push(centers[k]!);
      }
      // 按 >2σ 间隙把激活感受野中心分簇，如实给出分布
      actCenters.sort((a, b) => a - b);
      const sigma = this.encoder.sigma(dim);
      const clusters: OutcomeCluster[] = [];
      let lastPoint: number | null = null;
      for (const c of actCenters) {
        const last = clusters.length > 0 ? clusters[clusters.length - 1]! : null;
        if (last !== null && lastPoint !== null && c - lastPoint > 2 * sigma) {
          clusters.push({ center: c, count: 1 });
        } else if (last !== null) {
          last.center = (last.center * last.count + c) / (last.count + 1);
          last.count++;
        } else {
          clusters.push({ center: c, count: 1 });
        }
        lastPoint = c;
      }
      distribution[dim] = clusters;
      if (clusters.length === 0) {
        values[dim] = null;
      } else if (clusters.length === 1) {
        values[dim] = clusters[0]!.center;
      } else {
        values[dim] = null;
        ambiguous.push(dim);
      }
    }
    return { values, distribution, ambiguous, activeNeurons: result.activeNeurons, energy: result.energy };
  }

  private outcomeDims: string[] = [];
  private outcomeDimFields: number[] = [];

  /** 指定哪些维度是结果维度（读出侧） */
  setOutcomeDimensions(dims: readonly string[]): void {
    this.outcomeDims = [...dims];
    this.outcomeDimFields = dims.flatMap((d) =>
      Array.from({ length: this.encoder.fieldsPerDim }, (_, k) => this.encoder.dimensionOffset(d) + k),
    );
  }

  /** 结果侧替代值互斥（写入本网络）：换对观察到的两个值区域之间写抑制边 */
  learnExclusion(dimension: string, valueA: number, valueB: number, strength = 3.0): void {
    const a = this.encoder.encodeDimension(dimension, valueA);
    const b = this.encoder.encodeDimension(dimension, valueB);
    for (const x of a) {
      for (const y of b) this.net.strengthenInhibitory(x, y, strength);
    }
  }

  learnFromObservation(
    conditions: Record<string, number>,
    observed: Record<string, number>,
    repeats = 2,
  ): void {
    this.teachExperimentWithCap(conditions, observed, repeats, 0.1, 0.6);
  }

  private teachExperimentWithCap(
    conditions: Record<string, number>,
    outcomes: Record<string, number>,
    repeats: number,
    eta: number,
    cap: number,
  ): void {
    const condFields = this.encoder.encode(conditions);
    const core = this.coreFor(condFields);
    hebbianLearn(this.net, [...condFields, ...core], repeats, eta, cap);
    // 结果场星型绑定（同 teachExperiment，防幻影块）
    for (const dim of this.encoder.dimensions) {
      if (outcomes[dim.name] === undefined) continue;
      const outFields = this.encoder.encodeDimension(dim.name, outcomes[dim.name]!);
      for (const to of outFields) {
        for (const from of core) this.net.strengthen(from, to, cap, cap);
      }
    }
    // 持证：按侧重并集补正/否决（场级，与 bindInfluence 同一规则）
    const own = new Set(condFields);
    for (const [dim, delta] of Object.entries(this.lastBoost)) {
      if (delta > 0) {
        for (const from of this.encoder.encodeDimension(dim, conditions[dim]!)) {
          for (const to of core) this.net.strengthen(from, to, delta);
        }
      }
      const offset = this.encoder.dimensionOffset(dim);
      for (let k = 0; k < this.encoder.fieldsPerDim; k++) {
        const from = offset + k;
        if (own.has(from)) continue;
        for (const to of core) this.net.strengthenInhibitory(from, to, this.lastGammaVeto);
      }
    }
  }
}
