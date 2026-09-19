import { OutcomeEvidence, sameSupport, compatibleSupports } from "../evidence.js";
import type { SettleTermination } from "../../types.js";
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

/** 内部可变形态（再观察时结果场元数据并集更新；对外只读视图） */
interface FieldRuleInternal {
  core: number[];
  conditionFields: number[];
  outcomeFields: number[];
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
  /** 获胜规则核索引（≥3/4 成员激活的核） */
  readonly winningCores: readonly number[];
  /** 退火是否以固定点终止（非平衡驱动下如实上报非收敛） */
  readonly converged: boolean;
  readonly terminationReason: SettleTermination;
}

export class FieldRuleMemory {
  readonly net: EnergyNetwork;
  private readonly rules: FieldRuleInternal[] = [];
  private readonly signatureToCore = new Map<string, readonly number[]>();
  /** 规则注册表：签名 → rules 下标。生命周期判定以它为准（与分配解耦） */
  private readonly sigRegistered = new Map<string, number>();
  private coreCursor = 0;
  private readonly coreSize: number;
  private readonly maxRules: number;
  private readonly coreBase: number;
  private readonly poolBase: number;
  private readonly poolSize = 2;
  /** 池→核抑制强度：两核联盟时弱者净场 < θ 而胜者存活（按本模块核支持 ~20-27 标定） */
  private readonly poolGamma = 20;
  private lastBoost: Record<string, number> = {};
  private lastGammaVeto = 1.2;
  /** 签名 → 最近一次写入的结果值（纠错否决用，与 PopRuleMemory 同一规则） */
  private readonly evidence = new OutcomeEvidence();
  get evidenceGeneration(): number { return this.evidence.generation; }
  get evidenceConflicts() { return this.evidence.conflictLog; }
  ruleEvidence(index: number) { return this.evidence.snapshot(Math.floor((this.rules[index]!.core[0]! - this.coreBase) / this.coreSize)); }
  /** 每维已观察到的条件值集合（否决源登记——评审 F05：只否决已观察替代值的域） */
  private readonly observedValues = new Map<string, Set<number>>();
  private readonly contrastVetoes = new Map<string, { from: number; to: number; delta: number }[]>();

  /** Contradictory observed episodes teach local pattern separation: only the
   * condition members absent from a core may inhibit it. Shared members never
   * do. This is associative exclusion, not a causal-factor assertion (R2).
   * Reconcile contributions when evidence changes, including world reversals. */
  private reconcileContrasts(index: number): void {
    const a = this.rules[index]!;
    const ae = this.ruleEvidence(index);
    this.rules.forEach((b, j) => {
      if (j === index) return;
      const key = `${Math.min(index, j)}:${Math.max(index, j)}`;
      for (const edge of this.contrastVetoes.get(key) ?? []) this.net.setInhibitionContribution(edge.from, edge.to, `contrast:${key}`, 0);
      const be = this.ruleEvidence(j);
      const conflict = Object.entries(ae).some(([dim, e]) => be[dim] && !compatibleSupports(e.support, be[dim]!.support));
      const edges: { from: number; to: number; delta: number }[] = [];
      if (conflict) for (const [source, target] of [[a, b], [b, a]] as const) {
        for (const from of source.conditionFields.filter(id => !target.conditionFields.includes(id))) {
          for (const to of target.core) {
            const before = this.net.getInhibitoryWeight(from, to);
            this.net.setInhibitionContribution(from, to, `contrast:${key}`, 3);
            edges.push({ from, to, delta: this.net.getInhibitoryWeight(from, to) - before });
          }
        }
      }
      this.contrastVetoes.set(key, edges);
    });
  }

  constructor(
    readonly encoder: SensoryEncoder,
    readonly emergent: EmergentMap,
    config: { coreSize?: number; maxRules?: number; activationEnergy?: number; maintenanceEnergy?: number; learningRate?: number; maxWeight?: number } = {},
  ) {
    this.coreSize = config.coreSize ?? 4;
    this.maxRules = config.maxRules ?? 128;
    this.coreBase = encoder.neuronCount;
    this.poolBase = encoder.neuronCount + this.maxRules * this.coreSize;
    this.net = new EnergyNetwork({
      neuronCount: encoder.neuronCount + this.maxRules * this.coreSize + this.poolSize,
      activationEnergy: config.activationEnergy ?? 1.0,
      maintenanceEnergy: config.maintenanceEnergy ?? 0.5,
      learningRate: config.learningRate ?? 0.1,
      maxWeight: config.maxWeight ?? 3.0,
    });
  }

  get ruleCount(): number {
    return this.rules.length;
  }

  /** 活跃核索引：规则核成员 ≥3/4 激活视为该核被捕获/接受 */
  activeCores(activeNeurons: Iterable<number>): number[] {
    const active = new Set(activeNeurons);
    const out: number[] = [];
    this.rules.forEach((rule, idx) => {
      let on = 0;
      for (const id of rule.core) if (active.has(id)) on++;
      if (on >= Math.max(1, Math.ceil((rule.core.length * 3) / 4))) out.push(idx);
    });
    return out;
  }

  /**
   * 未覆盖维度读出（与 PopRuleMemory.uncoveredDims 同一规则，场级版）：
   * 某维查询值的感受野对任何已注册规则核都没有 W 边 = 该值从未被学习过。
   */
  uncoveredDims(query: Record<string, number>): string[] {
    const out: string[] = [];
    if (this.rules.length === 0) return this.encoder.dimensions.map((d) => d.name).filter((n) => query[n] !== undefined);
    for (const dim of this.encoder.dimensions) {
      const v = query[dim.name];
      if (v === undefined) continue;
      const fields = this.encoder.encodeDimension(dim.name, v);
      let covered = false;
      for (const from of fields) {
        for (const rule of this.rules) {
          for (const to of rule.core) {
            if (this.net.getWeight(from, to) > 0) {
              covered = true;
              break;
            }
          }
          if (covered) break;
        }
        if (covered) break;
      }
      if (!covered) out.push(dim.name);
    }
    return out;
  }

  /** 规则核绑定结果场（捕获检验用：读出某核的答案内容） */
  ruleOutcomeFields(index: number): readonly number[] {
    return this.rules[index]!.outcomeFields;
  }

  /**
   * 覆盖读出（纯边权、无动力学——与失配场同一家族的网络量）：
   * 查询的感受野对每个已分配核的净支持场 = Σ(W−Γ)（查询场 → 核），返回最大值。
   * 低于 θ 表示没有核声称该查询——"我不知道"的网络证据（自主探索的无知场）。
   */
  coreFieldCoverage(query: Record<string, number>): number {
    const fields = this.encoder.encode(query);
    let best = 0;
    for (const rule of this.rules) {
      let s = 0;
      for (const from of fields) {
        for (const to of rule.core) {
          s += this.net.getWeight(from, to) - this.net.getInhibitoryWeight(from, to);
        }
      }
      if (s > best) best = s;
    }
    return best;
  }

  /** 某条规则核的成员神经元 */
  ruleCore(index: number): readonly number[] {
    return this.rules[index]!.core;
  }

  /** 全部规则核神经元（捕获检验的候选集） */
  allCoreNeurons(): number[] {
    return this.rules.flatMap((r) => [...r.core]);
  }

  private signatureOf(fields: readonly number[]): string {
    return [...fields].sort((a, b) => a - b).join(",");
  }

  private coreFor(condFields: readonly number[]): readonly number[] {
    const sig = this.signatureOf(condFields);
    const existing = this.signatureToCore.get(sig);
    if (existing) return existing;
    // 容量边界（评审 D01/F03 修复）：分配前检查，越界显式抛错——
    // 修复前第二核会静默覆盖抑制池神经元并产生超出 N 的索引
    // （TypedArray 越界写不报错，池区被别名污染）。
    if (this.coreCursor >= this.maxRules) {
      throw new Error(`rule capacity exhausted (maxRules=${this.maxRules})`);
    }
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
        this.net.strengthenDirectedInhibitory(this.poolBase + k, x, this.poolGamma, this.poolGamma);
      }
    }
    this.signatureToCore.set(sig, core);
    return core;
  }

  /**
   * 规则注册事务（生命周期唯一入口；评审 D02/F02、D03/F13 修复）：
   * - 首个**结果证据**写入时把核登记进 rules——判定以注册表为准，
   *   与签名分配（signatureToCore）解耦：先 bindInfluence 分配的核
   *   不阻止后续真实观察注册（修复前 isNew 看签名表，永不注册）；
   * - 已注册规则的再次观察：结果场元数据取**并集**——解释接口
   *   （ruleOutcomeFields/mismatchField）读到的是当前真实绑定。
   */
  private registerOrUpdateRule(
    sig: string,
    core: readonly number[],
    condFields: readonly number[],
    outcomes: Record<string, number>,
  ): void {
    const outFields = Object.values(this.evidence.snapshot(Math.floor((core[0]! - this.coreBase) / this.coreSize))).flatMap(e => e.support);
    const existing = this.sigRegistered.get(sig);
    if (existing === undefined) {
      this.sigRegistered.set(sig, this.rules.length);
      this.rules.push({ core: [...core], conditionFields: [...condFields], outcomeFields: outFields });
    } else {
      const rule = this.rules[existing]!;
      rule.outcomeFields = [...outFields];
    }
    this.reconcileContrasts(this.sigRegistered.get(sig)!);
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
    this.acceptOutcomes(core, outcomes);
    this.registerOrUpdateRule(sig, core, condFields, outcomes);
    this.registerObservedValues(conditions);
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
    veto = true,
  ): void {
    for (const [dim, delta] of Object.entries(channelBoost)) {
      this.lastBoost[dim] = Math.max(this.lastBoost[dim] ?? 0, delta);
    }
    this.lastGammaVeto = veto ? gammaVeto : 0;
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
      // 评审 F06 同一规则：零增益通道不加分也不写否决；veto=false 显式断否决
      if (!veto || delta <= 0) continue;
      // 场级否决（评审 F05 根因修复）：只从"该维已观察到的替代值的感受野"
      // 写否决边（去掉本核自己的场）——否决强度随观察覆盖增长而增长；
      // 修复前从该维全部非己感受野写入：落在未观察区域的输入会把所有核
      // 压灭（同量程网格外 128 探针 109 个读空，12.5% vs 最近邻 81.3%）。
      const vetoGamma = gammaVeto;
      const ownDimFields = new Set(ownFields);
      for (const from of this.vetoSourcesFor(dim, ownDimFields)) {
        for (const to of core) this.net.strengthenInhibitory(from, to, vetoGamma);
      }
    }
  }

  /** 否决源集合：该维已观察替代值的感受野（去掉本核自己的场） */
  private vetoSourcesFor(dim: string, ownDimFields: ReadonlySet<number>): number[] {
    const out = new Set<number>();
    for (const alt of this.observedValues.get(dim) ?? []) {
      for (const f of this.encoder.encodeDimension(dim, alt)) {
        if (!ownDimFields.has(f)) out.add(f);
      }
    }
    return [...out];
  }

  /** 登记观察到的条件值（否决源登记的素材——观察即经验） */
  private registerObservedValues(conditions: Record<string, number>): void {
    for (const [d, v] of Object.entries(conditions)) {
      if (this.outcomeDims.includes(d)) continue;
      const s = this.observedValues.get(d) ?? new Set<number>();
      s.add(v);
      this.observedValues.set(d, s);
    }
  }

  /**
   * 预测：钳制查询的感受野，候选集 = 全部核 ∪ 全部结果感受野，
   * 局部退火选出获胜核，结果场中心值质心读出；双峰如实报歧义。
   */
  predict(query: Record<string, number>, seed: number): FieldPrediction {
    const input = this.encoder.encode(query);
    // 核候选过滤（与 PopRuleMemory 同一修复，自主探索大核数实测同病：
    // 全核候选 + 淬火长尾爬降把单次读出拖到十秒级）：对钳制输入的
    // W 支持 ≤ θ/2 的核从静息无点火路径，结构性冻结；退火上坡点火理论上
    // 存在，实测对结果无影响。纯边权读出，不涉及任何语义判定。
    const minSupport = this.net.threshold / 2;
    const supportedCores: number[] = [];
    for (const rule of this.rules) {
      let s = 0;
      for (const from of input) {
        for (const to of rule.core) s += this.net.getWeight(from, to);
      }
      if (s > minSupport) supportedCores.push(...rule.core);
    }
    const result = this.net.settleAnnealed(input, [], {
      seed,
      extraCandidates: [
        ...supportedCores,
        ...this.outcomeDimFields,
        ...Array.from({ length: this.poolSize }, (_, k) => this.poolBase + k),
      ],
      quenchCandidatesOnly: true,
      // 长尾爬降防护：8×N 翻转上限，超限回退途中最低能态（语义不变）
      quenchMaxFlips: 8 * this.net.neuronCount,
      // 池电路专属：最优回退只在驱动静息态中选（多核共存态真实能量更低但
      // 破坏 WTA 单核语义；评审 F01 修复引入的显式开关）
      fallbackQuietOnly: true,
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
    // Same-code observations carry equal evidence weight. Refine only a neural
    // readout whose active support exactly matches one winning core's code;
    // never fill an absent/ambiguous neural answer from metadata.
    const winners = this.activeCores(result.activeNeurons);
    if (winners.length === 1) for (const [dim, e] of Object.entries(this.ruleEvidence(winners[0]!))) {
      const support = this.outcomeDimFields.filter(id => active.has(id) && this.encoder.fieldOf(id)?.dimension === dim);
      if (values[dim] !== null && e.samples > 1 && sameSupport(support, e.support)) values[dim] = e.value;
    }
    return {
      values,
      distribution,
      ambiguous,
      activeNeurons: result.activeNeurons,
      energy: result.energy,
      winningCores: this.activeCores(result.activeNeurons),
      converged: result.converged,
      terminationReason: result.terminationReason,
    };
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
    if (compatibleSupports(a, b)) return;
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

  private acceptOutcomes(core: readonly number[], outcomes: Record<string, number>): void {
    this.evidence.accept(this.net, Math.floor((core[0]! - this.coreBase) / this.coreSize), core,
      outcomes, "observation", (dim, value) => this.encoder.encodeDimension(dim, value));
  }

  private teachExperimentWithCap(
    conditions: Record<string, number>,
    outcomes: Record<string, number>,
    repeats: number,
    eta: number,
    cap: number,
  ): void {
    const condFields = this.encoder.encode(conditions);
    const sig = this.signatureOf(condFields);
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
    // 生命周期事务（评审 D02 修复 + 早前空白起步修复的完成版）：
    // 观察写入 = 结果证据 → 注册/更新规则。判定以注册表为准——
    // 先侧重（bindInfluence 分配核）后观察的顺序不再漏登记。
    this.acceptOutcomes(core, outcomes);
    this.registerOrUpdateRule(sig, core, condFields, outcomes);
    this.registerObservedValues(conditions);
    // 持证：按侧重并集补正/否决（场级，与 bindInfluence 同一规则）
    for (const [dim, delta] of Object.entries(this.lastBoost)) {
      // 评审 F06 同一规则：零增益通道不加分也不写否决
      if (delta <= 0) continue;
      for (const from of this.encoder.encodeDimension(dim, conditions[dim]!)) {
        for (const to of core) this.net.strengthen(from, to, delta);
      }
      // 否决源 = 已观察替代值的感受野（评审 F05 同一修复，不再全维写入）
      const ownDimFields = new Set(this.encoder.encodeDimension(dim, conditions[dim]!));
      for (const from of this.vetoSourcesFor(dim, ownDimFields)) {
        for (const to of core) this.net.strengthenInhibitory(from, to, this.lastGammaVeto);
      }
    }
  }
}
