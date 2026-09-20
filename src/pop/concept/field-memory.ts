import { predictionQuality } from "../prediction-quality.js";
import { integer, learning, nonnegative, nonempty } from "../../validate.js";
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
  private readonly lastBoost: Record<string, number> = {};
  private lastGammaVeto = 1.2;
  /** 签名 → 最近一次写入的结果值（纠错否决用，与 PopRuleMemory 同一规则） */
  private readonly evidence = new OutcomeEvidence();
  get evidenceGeneration(): number { return this.evidence.generation; }
  get evidenceConflicts() { return this.evidence.conflictLog; }
  /** 改判定额（探索器复验队列上限用） */
  get switchQuorum(): number { return this.evidence.switchQuorum; }
  private slotOf(core: readonly number[]): number { return Math.floor((core[0]! - this.coreBase) / this.coreSize); }
  ruleEvidence(index: number) { return this.evidence.snapshot(this.slotOf(this.rules[index]!.core)); }
  /** 每维已观察到的条件值集合（否决源登记——评审 F05：只否决已观察替代值的域） */
  private readonly observedValues = new Map<string, Set<number>>();
  private readonly contrastVetoes = new Map<string, { from: number; to: number; delta: number }[]>();
  /** 容量策略：throw（默认，评审契约）= 耗尽显式抛错；lru = 回收最久未用核槽 */
  private readonly eviction: "throw" | "lru";
  /** 使用跟踪：rules 下标 → 单调 tick（写入与预测获胜都算使用） */
  private tick = 0;
  private readonly lastTouch = new Map<number, number>();
  /** 已清空待复用的核槽位（槽序号，非 rules 下标） */
  private readonly freeSlots: number[] = [];
  /** Contradictory observed episodes teach local pattern separation: only the
   * condition members absent from a core may inhibit it. Shared members never
   * do. This is associative exclusion, not a causal-factor assertion (R2).
   * Reconcile contributions when evidence changes, including world reversals. */
  /** Contradictory observed episodes teach local pattern separation: only the
   * condition members absent from a core may inhibit it. Shared members never
   * do. This is associative exclusion, not a causal-factor assertion (R2).
   * Reconcile contributions when evidence changes, including world reversals.
   * 键用核槽位（神经元不移动），不用 rules 下标（淘汰 swap-remove 会改键）。 */
  private reconcileContrasts(index: number): void {
    const a = this.rules[index]!;
    const slotA = this.slotOf(a.core);
    const ae = this.ruleEvidence(index);
    this.rules.forEach((b, j) => {
      if (j === index) return;
      const slotB = this.slotOf(b.core);
      const key = `${Math.min(slotA, slotB)}:${Math.max(slotA, slotB)}`;
      for (const edge of this.contrastVetoes.get(key) ?? []) this.net.setInhibitionContribution(edge.from, edge.to, `contrast:${key}`, 0);
      const be = this.ruleEvidence(j);
      const overlappingConditions = this.encoder.dimensions.filter(d => !this.outcomeDims.includes(d.name)).every(d => {
        const af = a.conditionFields.filter(id => this.encoder.fieldOf(id)?.dimension === d.name);
        const bf = b.conditionFields.filter(id => this.encoder.fieldOf(id)?.dimension === d.name);
        return af.length === 0 || bf.length === 0 || compatibleSupports(af, bf);
      });
      const conflict = overlappingConditions && Object.entries(ae).some(([dim, e]) => be[dim] && !compatibleSupports(e.support, be[dim]!.support));
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
    config: { coreSize?: number; maxRules?: number; activationEnergy?: number; maintenanceEnergy?: number; learningRate?: number; maxWeight?: number; eviction?: "throw" | "lru" } = {},
  ) {
    this.coreSize = config.coreSize ?? 4;
    this.maxRules = config.maxRules ?? 128;
    integer(this.coreSize, "coreSize", 1); integer(this.maxRules, "maxRules", 1);
    this.eviction = config.eviction ?? "throw";
    if (this.eviction !== "throw" && this.eviction !== "lru") throw new Error(`unknown eviction: ${this.eviction}`);
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
    return [...this.rules[index]!.outcomeFields];
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

  /** 证据竞争（纯边权+元数据读出，不确定性地图用）：该查询最佳支持核的
   * 结果维中，存在 ≥2 个观察候选且前两名票差 < switchQuorum 的维度。
   * 无核覆盖返回空表——那是"未知"，不是"竞争"。 */
  contestedDims(query: Record<string, number>): string[] {
    const fields = this.encoder.encode(query);
    let best = 0;
    let bestIdx = -1;
    this.rules.forEach((rule, idx) => {
      let s = 0;
      for (const from of fields) for (const to of rule.core) s += this.net.getWeight(from, to) - this.net.getInhibitoryWeight(from, to);
      if (s > best) { best = s; bestIdx = idx; }
    });
    if (bestIdx < 0 || best < this.net.threshold) return [];
    const counts = this.evidence.candidateCounts(this.slotOf(this.rules[bestIdx]!.core));
    return Object.entries(counts).filter(([, c]) => c.length >= 2 && c[0]! - c[1]! < this.evidence.switchQuorum).map(([dim]) => dim);
  }

  /** 某条规则核的成员神经元 */
  ruleCore(index: number): readonly number[] {
    return [...this.rules[index]!.core];
  }

  /** 全部规则核神经元（捕获检验的候选集） */
  allCoreNeurons(): number[] {
    return this.rules.flatMap((r) => [...r.core]);
  }

  /** Lifecycle: allocation is separate from result evidence, registration and
   * prediction eligibility. Empty outcomes cannot advance the lifecycle. */
  private planWrite(conditions: Record<string, number>, outcomes: Record<string, number> | undefined,
    repeats: number, eta: number, cap: number): void {
    learning(repeats, eta, cap); nonempty(conditions, "conditions");
    for (const dim of Object.keys(conditions)) if (this.outcomeDims.includes(dim)) throw new Error(`outcome used as condition: ${dim}`);
    const condFields = this.encoder.encode(conditions);
    if (outcomes !== undefined) {
      nonempty(outcomes, "outcomes");
      for (const dim of Object.keys(outcomes)) if (!this.outcomeDims.includes(dim)) throw new Error(`unknown outcome dimension: ${dim}`);
      this.encoder.encode(outcomes);
    }
    if (!this.signatureToCore.has(this.signatureOf(condFields)) && this.eviction === "throw" && this.coreCursor >= this.maxRules)
      throw new Error(`rule capacity exhausted (maxRules=${this.maxRules})`);
  }

  private signatureOf(fields: readonly number[]): string {
    return [...fields].sort((a, b) => a - b).join(",");
  }

  private coreFor(condFields: readonly number[]): readonly number[] {
    const sig = this.signatureOf(condFields);
    const existing = this.signatureToCore.get(sig);
    if (existing) return existing;
    // 容量边界（评审 D01/F03 修复）：分配前检查；默认显式抛错——
    // 修复前第二核会静默覆盖抑制池神经元并产生超出 N 的索引。
    // eviction="lru" 时改为回收最久未用核槽（遗忘，见 evictForAllocation）。
    if (this.coreCursor - this.freeSlots.length >= this.maxRules) {
      if (this.eviction === "throw") {
        throw new Error(`rule capacity exhausted (maxRules=${this.maxRules})`);
      }
      this.evictForAllocation();
    }
    const slot = this.freeSlots.pop() ?? this.coreCursor++;
    const core = Array.from({ length: this.coreSize }, (_, k) => this.coreBase + slot * this.coreSize + k);
    this.wireNewCore(core);
    this.signatureToCore.set(sig, core);
    return core;
  }

  /** 新核结构接线：核间互斥 + 全局抑制池（WTA 电路，前馈抑制版）。
   * 新槽与回收槽共用——回收槽的全部突触已被 clearSynapses 清零。 */
  private wireNewCore(core: readonly number[]): void {
    for (const other of this.signatureToCore.values()) {
      for (const x of core) {
        for (const y of other) this.net.strengthenInhibitory(x, y, 3.0);
      }
    }
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
  }

  /** 遗忘 = 突触修剪：回收一个核槽给新经验。受害者顺序：
   * 1) 已分配但未登记的空槽（无经验，零代价）；
   * 2) 已登记规则中 lastTouch 最久远的（最久未被写入或预测获胜）。
   * 清除受害者自己的 supersede/contrast 否决贡献、证据登记、签名映射与
   * 全部突触，其他核的证据与边不受影响。swap-remove 保持 rules 紧致。 */
  private evictForAllocation(): void {
    for (const [sig, core] of this.signatureToCore) {
      if (!this.sigRegistered.has(sig)) {
        this.signatureToCore.delete(sig);
        this.net.clearSynapses(core);
        this.freeSlots.push(this.slotOf(core));
        return;
      }
    }
    let victimIdx = -1;
    let oldest = Infinity;
    for (let i = 0; i < this.rules.length; i++) {
      const t = this.lastTouch.get(i) ?? 0;
      if (t < oldest) { oldest = t; victimIdx = i; }
    }
    if (victimIdx < 0) throw new Error(`rule capacity exhausted (maxRules=${this.maxRules})`);
    const victim = this.rules[victimIdx]!;
    const victimSlot = this.slotOf(victim.core);
    this.evidence.remove(this.net, victimSlot);
    for (const key of [...this.contrastVetoes.keys()]) {
      const [a, b] = key.split(":").map(Number);
      if (a === victimSlot || b === victimSlot) {
        for (const e of this.contrastVetoes.get(key)!) this.net.setInhibitionContribution(e.from, e.to, `contrast:${key}`, 0);
        this.contrastVetoes.delete(key);
      }
    }
    for (const [sig, idx] of this.sigRegistered) {
      if (idx === victimIdx) { this.sigRegistered.delete(sig); this.signatureToCore.delete(sig); }
    }
    this.net.clearSynapses(victim.core);
    this.freeSlots.push(victimSlot);
    const last = this.rules.length - 1;
    if (victimIdx !== last) {
      this.rules[victimIdx] = this.rules[last]!;
      for (const [sig, idx] of this.sigRegistered) if (idx === last) this.sigRegistered.set(sig, victimIdx);
      const t = this.lastTouch.get(last);
      if (t === undefined) this.lastTouch.delete(victimIdx); else this.lastTouch.set(victimIdx, t);
    }
    this.rules.pop();
    this.lastTouch.delete(last);
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
    this.planWrite(conditions, outcomes, repeats, this.net.config.learningRate, baseCap);
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
    this.lastTouch.set(this.sigRegistered.get(sig)!, ++this.tick);
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
    this.planWrite(conditions, undefined, repeats, 0, 0);
    nonnegative(gammaVeto, "gammaVeto");
    for (const [dim, delta] of Object.entries(channelBoost)) {
      nonnegative(delta, `boost ${dim}`);
      if (this.outcomeDims.includes(dim)) throw new Error(`outcome used as influence: ${dim}`);
      this.encoder.encodeDimension(dim, conditions[dim]!);
    }
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
      // Key presence is R2 eligibility; gain controls W, veto controls Γ independently.
      if (!veto) continue;
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
  predict(query: Record<string, number>, seed: number,
    anneal?: { levels?: number; sweepsPerLevel?: number }): FieldPrediction {
    const input = this.encoder.encode(query);
    // 核候选过滤：裸 W 支持 ≤ θ/2 的核从静息无点火路径，结构性冻结
    // （门槛用裸 W——点火可行性只看兴奋场；F05 回归：被否决压制的核
    // 必须留在候选池里，由退火内的 Γ 竞争定胜负）。
    // #4 时间墙优化：候选超过 32 个时按**净支持**（W−Γ）排序只留前 32 名——
    // 否决已建立的核净支持自然靠后，不被误留（量程扩展场景实测回归）；
    // 共享条件维会让几十上百个核越过门槛，净支持悬殊时落后核无胜出路径。
    // 小世界（≤32 个入围核）行为与之前逐位一致。
    const minSupport = this.net.threshold / 2;
    const ranked: { s: number; core: readonly number[] }[] = [];
    for (const rule of this.rules) {
      let w = 0;
      let net = 0;
      for (const from of input) {
        for (const to of rule.core) {
          w += this.net.getWeight(from, to);
          net += this.net.getWeight(from, to) - this.net.getInhibitoryWeight(from, to);
        }
      }
      if (w > minSupport) ranked.push({ s: net, core: rule.core });
    }
    ranked.sort((a, b) => b.s - a.s);
    const supportedCores = ranked.slice(0, 32).flatMap(r => [...r.core]);
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
      levels: anneal?.levels ?? 12,
      // 退火扫描数随候选域规模走（大网络选错核的实测修复）：小世界保持 20 不变；
      // 候选域每大 8 个神经元加 1 次扫描——33 节点链的错读案例实测 20 次不够。
      sweepsPerLevel: anneal?.sweepsPerLevel ?? Math.max(20,
        Math.ceil((supportedCores.length + this.outcomeDimFields.length + this.poolSize) / 8)),
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
    for (const w of winners) this.lastTouch.set(w, ++this.tick);
    if (winners.length === 1) for (const [dim, e] of Object.entries(this.ruleEvidence(winners[0]!))) {
      const support = this.outcomeDimFields.filter(id => active.has(id) && this.encoder.fieldOf(id)?.dimension === dim);
      if (values[dim] !== null && e.samples > 1 && sameSupport(support, e.support)) values[dim] = e.value;
    }
    predictionQuality.record(values, result.converged, result.terminationReason);
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
    if (this.coreCursor > 0 && dims.join() !== this.outcomeDims.join()) throw new Error("cannot change outcome schema after allocation");
    const fields = dims.flatMap((d) =>
      Array.from({ length: this.encoder.fieldsPerDim }, (_, k) => this.encoder.dimensionOffset(d) + k),
    );
    this.outcomeDims = [...dims]; this.outcomeDimFields = fields;
  }

  /** 结果侧替代值互斥（写入本网络）：换对观察到的两个值区域之间写抑制边 */
  learnExclusion(dimension: string, valueA: number, valueB: number, strength = 3.0): void {
    nonnegative(strength, "exclusion strength");
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
    this.planWrite(conditions, outcomes, repeats, eta, cap);
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
      // Gain and veto are independent for eligible dimensions.
      if (conditions[dim] === undefined) continue;
      for (const from of this.encoder.encodeDimension(dim, conditions[dim]!)) {
        for (const to of core) this.net.strengthen(from, to, delta);
      }
      // 否决源 = 已观察替代值的感受野（评审 F05 同一修复，不再全维写入）
      const ownDimFields = new Set(this.encoder.encodeDimension(dim, conditions[dim]!));
      for (const from of this.vetoSourcesFor(dim, ownDimFields)) {
        for (const to of core) this.net.strengthenInhibitory(from, to, this.lastGammaVeto);
      }
    }
    this.lastTouch.set(this.sigRegistered.get(sig)!, ++this.tick);
  }
}
