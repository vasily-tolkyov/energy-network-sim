import { predictionQuality } from "./prediction-quality.js";
import { integer, learning, nonnegative, nonempty } from "../validate.js";
import { OutcomeEvidence } from "./evidence.js";
import type { SettleTermination } from "../types.js";
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
 * - 教学：两段绑定——条件群体↔核一段、核↔结果群体按通道各一段
 *   （基础封顶 baseCap；绝不允许三者同集合共激活，防条件→结果直连与
 *   结果通道间接力两种泄漏）；
 *   核间建抑制边 Γ（候选规则互斥），另有前馈抑制池 WTA 电路
 *   （分级招募、按联盟规模加压，逼出单核胜出——两两 Γ 的共激活代价
 *   固定、压不住多核共存，池是自 field-memory 移植的已知解）；
 * - 侧重（§7.1 类比）：影响因素通道群体 → 该经验核的连接 ×(η·G·m²) 增强；
 * - 预测：钳制查询条件群体，候选集 = 全部核 ∪ 结果群体，局部退火选出
 *   能耗极小的获胜核（匹配条件越多场越强），获胜核点亮其结果群体；
 * - 概念不再对应单点：档位 = 群体势阱，规则 = 私有核势阱。
 */

export interface PopMemoryConfig {
  readonly coreSize?: number;
  readonly maxRules?: number;
  /** 核间互斥强度 Γ_core（默认 3.0，评审 B07 修复：配置必须真实生效） */
  readonly gammaCore?: number;
}

export class PopRuleMemory {
  readonly net: EnergyNetwork;
  readonly coreSize: number;
  private readonly maxRules: number;
  private coreCursor = 0;
  private readonly signatureToCore = new Map<string, readonly number[]>();
  private readonly cores: number[][] = [];
  /** 核是否已有结果证据（评审 F02 的 Pop 侧同款修复：bindInfluence 可分配
   *  无结果绑定的核——它只能竞争却永远读空；预测候选只收有证据的核） */
  private readonly coreEvidence: boolean[] = [];
  /** 核最近一次写入的结果（纠错否决用：观察覆盖了谁就否决谁） */
  private readonly evidence = new OutcomeEvidence();
  get evidenceGeneration(): number { return this.evidence.generation; }
  get evidenceConflicts() { return this.evidence.conflictLog; }
  /** 前馈抑制池规模（WTA 电路，自 field-memory 移植） */
  private readonly poolSize = 2;
  /**
   * 池→核前馈抑制强度，按本模块核支持尺度重新标定（field-memory 用 20，
   * 其核支持 ~20-27；本模块持证精确核净场 ≈8-12，1-失配核净场 ≈+2）：
   * 须压灭 1-失配核（γ>2），同时让持证精确核在池点燃的暂态中存活（γ<8−θ）。
   */
  private readonly poolGamma = 4.5;
  /** 核间互斥强度（默认 3.0，可由 config.gammaCore 覆盖） */
  private readonly gammaCore: number;

  constructor(
    readonly conditionMap: PopChannelMap,
    readonly outcomeMap: PopChannelMap,
    config: PopMemoryConfig & { activationEnergy?: number; maintenanceEnergy?: number; learningRate?: number; maxWeight?: number } = {},
  ) {
    this.coreSize = config.coreSize ?? 4;
    this.maxRules = config.maxRules ?? 128;
    this.gammaCore = config.gammaCore ?? 3.0;
    integer(this.coreSize, "coreSize", 1); integer(this.maxRules, "maxRules", 1);
    nonnegative(this.gammaCore, "gammaCore");
    const totalNeurons =
      conditionMap.neuronCount + outcomeMap.neuronCount + this.maxRules * this.coreSize + this.poolSize;
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
    nonnegative(strength, "exclusion strength");
    if (map !== "condition" && map !== "outcome") throw new Error(`unknown map: ${map}`);
    const target = map === "condition" ? this.conditionMap : this.outcomeMap;
    target.population(channel, binA); target.population(channel, binB);
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

  /** Lifecycle: allocated -> result evidence -> registered -> prediction eligible.
   * validate/plan runs before coreFor or evidence/boost counters commit. */
  private planWrite(conditions: Conditions, outcomes?: Outcomes, repeats = 0, eta = 0, cap = 0): void {
    learning(repeats, eta, cap);
    nonempty(conditions, "conditions");
    this.conditionMap.encode(conditions);
    if (outcomes !== undefined) { nonempty(outcomes, "outcomes"); this.outcomeMap.encode(outcomes); }
    if (!this.signatureToCore.has(this.signature(conditions)) && this.coreCursor >= this.maxRules)
      throw new Error(`rule capacity exhausted (maxRules=${this.maxRules})`);
  }

  private coreBase(): number {
    return this.conditionMap.neuronCount + this.outcomeMap.neuronCount;
  }

  private poolBase(): number {
    return this.coreBase() + this.maxRules * this.coreSize;
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
    // 与既有核互斥（Γ_core，强度可配置——评审 B07：写死参数曾让消融配置表面变化）。
    // 默认 3.0 的由来：历史值 1.5 压不住强支持核（1-失配核净场可达 +2~+8 > θ−6），
    // 会在胜者两两抑制下复燃，与池构成"压制-熄火-复燃"弛豫振荡，淬火被迫跑满
    // maxFlips（实测单次预测均时 1.2s、最慢 4s）。3.0（联盟代价 4×3=12）
    // 使任何 1-失配核在胜者存活时净场 < 0，单核态成为稳定不动点。
    for (const other of this.cores) {
      for (const x of core) {
        for (const y of other) this.net.strengthenInhibitory(x, y, this.gammaCore);
      }
    }
    // 全局抑制池接线（WTA 电路，前馈抑制版，自 field-memory 移植）：
    // 核 → 池分级招募（0.3/0.15）：单核活跃（4×0.3=1.2<θ=1.5）池沉睡；
    // ≥2 核共存（8×0.3=2.4>θ）池点燃，DI 前馈抑制压回所有核——
    // 弱者（1-失配核，净场≈+2）被压灭，持证胜者（净场≈8-12）存活，
    // 池失去驱动后熄灭，逐个淘汰至单核胜出。
    // 两两互斥 Γ 的共激活代价是固定的、不随联盟规模增长（化学主题实测
    // 双核满激活共存），池让代价随联盟规模上涨，联盟自我拆台。
    for (const x of core) {
      for (let k = 0; k < this.poolSize; k++) {
        this.net.strengthen(x, this.poolBase() + k, 0.3 / (k + 1));
        this.net.strengthenDirectedInhibitory(this.poolBase() + k, x, this.poolGamma, this.poolGamma);
      }
    }
    this.cores.push(core);
    this.coreEvidence.push(false);
    this.signatureToCore.set(sig, core);
    return core;
  }

  /** 标记核已有结果证据（可参与预测候选） */
  private markCoreEvidence(core: readonly number[]): void {
    const idx = this.cores.indexOf(core as number[]);
    if (idx >= 0) this.coreEvidence[idx] = true;
  }

  private acceptOutcomes(core: readonly number[], outcomes: Outcomes, source: "observation" | "hypothesis"): Record<string, number> {
    return this.evidence.accept(this.net, this.cores.indexOf(core as number[]), core, outcomes, source,
      (dim, value) => this.outcomeMap.population(dim, value).map(id => id + this.conditionMap.neuronCount));
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
    this.planWrite(exp.conditions, exp.outcomes, repeats, this.net.config.learningRate, baseCap);
    const core = this.coreFor(exp.conditions);
    const condPops = this.conditionMap.encode(exp.conditions);
    const accepted = this.acceptOutcomes(core, exp.outcomes, "observation");
    hebbianLearn(this.net, [...condPops, ...core], repeats, undefined, baseCap);
    // 结果按通道分别绑定到核：不允许出现 reb↔rs 等跨结果通道边——
    // 否则被驱动的结果档会经"结果→结果"边接力驱动别的档（实测发现的泄漏）。
    for (const spec of this.outcomeMap.specs) {
      if (accepted[spec.name] === undefined) continue;
      const pops = this.outcomeMap.population(spec.name, accepted[spec.name]!);
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
    this.markCoreEvidence(core);

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
    veto = true,
  ): void {
    this.planWrite(exp.conditions, undefined, repeats, 0, 0);
    nonnegative(gammaVeto, "gammaVeto");
    for (const [ch, delta] of Object.entries(channelBoost)) {
      nonnegative(delta, `boost ${ch}`); this.conditionMap.population(ch, exp.conditions[ch]!);
    }
    // 记录侧重/否决参数的并集（逐通道取最大），供响应学习写入的新核"持证上岗"
    for (const [ch, delta] of Object.entries(channelBoost)) {
      this.lastBoost[ch] = Math.max(this.lastBoost[ch] ?? 0, delta);
    }
    this.lastGammaVeto = veto ? gammaVeto : 0;
    const core = this.coreFor(exp.conditions);
    for (const [ch, delta] of Object.entries(channelBoost)) {
      if (delta > 0 && repeats > 0) {
        for (const from of this.conditionMap.population(ch, exp.conditions[ch]!)) {
          for (let r = 0; r < repeats; r++) {
            for (const to of core) this.net.strengthen(from, to, delta);
          }
        }
      }
      // 评审 F06/A17 修复：否决边只对真正有加分（delta>0）的通道写——
      // 修复前零增益通道也写否决，"G=0 断侧重"消融实际没断否决，
      // 消融结论无法分离两个机制的因果贡献。veto=false 显式断否决（2×2 消融用）。
      if (!veto || delta <= 0) continue;
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
    anneal?: { levels?: number; sweepsPerLevel?: number },
  ): { decoded: Record<string, PopDecoded>; activeNeurons: readonly number[]; energy: number; converged: boolean; terminationReason: SettleTermination } {
    const input = this.conditionMap.encode(query);
    const outcomeNeurons = Array.from(
      { length: this.outcomeMap.neuronCount },
      (_, k) => this.conditionMap.neuronCount + k,
    );
    // 核候选过滤（自主探索大核数实测修复：75 核时全核候选的退火达 23s/次）：
    // 对钳制输入的 W 支持 ≤ θ/2 的核从静息无点火路径，结构性冻结——与候选集
    // 只含被触及势阱同一精神。如实边界：退火上坡点火理论上存在，但远失配核
    // 借此获胜既罕见又不合理；实测对结果无影响（种子 2 终止步数 95→81，
    // 因素发现与准确率不变）。纯边权读出，不涉及任何语义判定。
    const minSupport = this.net.threshold / 2;
    // 只收有结果证据的核（评审 F02 的 Pop 侧同款：bindInfluence 可分配
    // 无结果绑定的核——它只能竞争却永远读空，曾致 C05 全关角读空）
    const supportedCores = this.cores.filter((core, k) => {
      if (!this.coreEvidence[k]) return false;
      let s = 0;
      for (const from of input) {
        for (const to of core) s += this.net.getWeight(from, to);
      }
      return s > minSupport;
    });
    const coreCandidates = supportedCores.flat();
    const poolNeurons = Array.from({ length: this.poolSize }, (_, k) => this.poolBase() + k);
    const result = this.net.settleAnnealed(input, [], {
      seed,
      // 池神经元必须在候选集内，否则结构性冻结在静息、WTA 电路失效
      extraCandidates: [...outcomeNeurons, ...coreCandidates, ...poolNeurons],
      quenchCandidatesOnly: true,
      // 长尾爬降防护：大核数下淬火改为 8×N 翻转上限（超限回退途中最低能态，
      // 语义不变；默认 100×N 在 75 核规模实测达 23s/次）
      quenchMaxFlips: 8 * this.net.neuronCount,
      // 池电路专属：最优回退只在驱动静息态中选（多核共存态真实能量更低但
      // 破坏 WTA 单核语义；评审 F01 修复引入的显式开关）
      fallbackQuietOnly: true,
      // 驱动估计等中间读数可传轻量档（6×8）；最终评分用默认全档（12×20）
      levels: anneal?.levels ?? 12,
      sweepsPerLevel: anneal?.sweepsPerLevel ?? 20,
    });
    const decoded = decodePopulation(
        result.activeNeurons
          .filter((id) => id >= this.conditionMap.neuronCount)
          .map((id) => id - this.conditionMap.neuronCount),
        this.outcomeMap,
        this.outcomeMap.channelNames(),
      );
    predictionQuality.record(decoded, result.converged, result.terminationReason);
    return {
      decoded,
      activeNeurons: result.activeNeurons,
      energy: result.energy,
      // 非平衡驱动下不承诺无条件收敛：非收敛如实向上传播（评审 F01 修复附带）
      converged: result.converged,
      terminationReason: result.terminationReason,
    };
  }

  /**
   * 覆盖读出（纯边权、无动力学——与失配场同一家族的网络量）：
   * 给定条件群体对每个已分配核的净支持场 = Σ(W−Γ)（条件群体 → 核），
   * 返回最大值。低于 θ 表示没有核声称该条件组合——"我不知道"的网络证据。
   * 自主探索的无知场用它而不用退火：驱动只负责排序探索目标，
   * 最终判定仍走完整动力学（predict）。
   */
  coreFieldCoverage(conditions: Conditions): number {
    const pops = this.conditionMap.encode(conditions);
    let best = 0;
    for (const core of this.cores) {
      let s = 0;
      for (const from of pops) {
        for (const to of core) {
          s += this.net.getWeight(from, to) - this.net.getInhibitoryWeight(from, to);
        }
      }
      if (s > best) best = s;
    }
    return best;
  }

  /**
   * 未覆盖维度读出（评审修复：量程扩展场景中 4/5 维匹配的候选看似"置信"，
   * 但它在从未观察过的取值上纯属泛化猜测）：某条件维的钳制群体对任何
   * 有证据核都没有 W 边 = 该取值从未被观察/学习过。探索驱动必须把
   * "维从未被操纵过的值"当作未知，否则扩展区/门控角永远不被主动探针触及。
   */
  uncoveredDims(conditions: Conditions): string[] {
    const out: string[] = [];
    const evidenceCores: number[][] = [];
    this.cores.forEach((c, k) => {
      if (this.coreEvidence[k]) evidenceCores.push(c);
    });
    if (evidenceCores.length === 0) return Object.keys(conditions);
    for (const [ch, bin] of Object.entries(conditions)) {
      const pops = this.conditionMap.population(ch, bin);
      let covered = false;
      for (const from of pops) {
        for (const core of evidenceCores) {
          for (const to of core) {
            if (this.net.getWeight(from, to) > 0) {
              covered = true;
              break;
            }
          }
          if (covered) break;
        }
        if (covered) break;
      }
      if (!covered) out.push(ch);
    }
    return out;
  }

  /**
   * 点火不等式（评审 C01–C04 根因修复）：单核-单结果群体结构中，
   * 结果群体神经元的最大支持 = (coreSize + popSize − 1) × η × repeats，
   * 必须超过 θ 才能在 T=0 下被读出。不足时自动放大 repeats——
   * 如实推导写入强度，而不是给每个新世界单独加参数补丁。
   * 修复前 repeats=2、η=0.1 → 支持 1.4 < θ=1.5，小世界观察全部读空。
   */
  private effectiveRepeats(repeats: number, eta: number): number {
    const popSize = this.outcomeMap.popSize;
    const supportPerRepeat = eta * (this.coreSize + popSize - 1);
    const need = this.net.threshold * 1.2 + 1e-9;
    return Math.max(repeats, Math.ceil(need / supportPerRepeat));
  }

  /** 响应模式学习：新查询分配/复用核并绑定预测结果（小步长同封顶）。
   * 绑定必须分两段（化学主题实测修复）：条件↔核一段、核↔结果按通道各一段，
   * 绝不允许把三者放进同一集合——否则赫布学习会写下条件→结果直连边与
   * 结果通道间接力边（teachExperience 已记录在案的两种泄漏），
   * 几十个查询累积后输入群体绕开核直接驱动结果群体，规则核桥梁被短路。 */
  learnFromQuery(query: Conditions, predicted: Record<string, PopDecoded>, repeats: number): void {
    this.conditionMap.encode(query);
    learning(repeats, .05, .4);
    for (const [ch, v] of Object.entries(predicted)) {
      if (!this.outcomeMap.channelNames().includes(ch)) throw new Error(`unknown outcome: ${ch}`);
      if (typeof v === "number") this.outcomeMap.population(ch, v);
      else if (v !== null && v !== "ambiguous") throw new Error(`invalid prediction: ${ch}`);
    }
    if (Object.values(predicted).some(v => typeof v === "number")) this.planWrite(query);
    // 评审 B03 修复：无可写结果证据（全 null/歧义）时不分配核——
    // 修复前全空预测也白占一个私有核（容量被无证据查询消耗）。
    if (!Object.values(predicted).some((v) => typeof v === "number")) return;
    const effRepeats = this.effectiveRepeats(repeats, 0.05);
    const core = this.coreFor(query);
    const condIds = this.conditionMap.encode(query);
    const numeric = Object.fromEntries(Object.entries(predicted).filter((entry): entry is [string, number] => typeof entry[1] === "number"));
    const accepted = this.acceptOutcomes(core, numeric, "hypothesis");
    if (Object.keys(accepted).length === 0) return;
    hebbianLearn(this.net, [...condIds, ...core], effRepeats, 0.05, 0.4);
    for (const spec of this.outcomeMap.specs) {
      const bin = accepted[spec.name];
      if (typeof bin === "number") {
        const pops = this.outcomeMap.population(spec.name, bin);
        hebbianLearn(this.net, [...core, ...pops.map((id) => id + this.conditionMap.neuronCount)], effRepeats, 0.05, 0.4);
      }
    }
    this.markCoreEvidence(core);
    // 持证上岗（与 learnFromObservation 同一规则，化学主题实测补入）：
    // 新核必须带侧重加分边 + 替代档否决边，否则它绕开否决体系——
    // 无否决惩罚的核在任何近似查询上白拿匹配支持，压垮持证核，
    // 错误预测再被学习，形成自我强化级联（v2 课程完整模型曾因此崩盘）。
    for (const [ch, delta] of Object.entries(this.lastBoost)) {
      // 评审 F06 同一规则：零增益通道不加分也不写否决；lastGammaVeto=0 断否决
      if (delta <= 0 || query[ch] === undefined) continue;
      for (const from of this.conditionMap.population(ch, query[ch]!)) {
        for (const to of core) this.net.strengthen(from, to, delta);
      }
      for (const alt of this.observedBins.get(ch) ?? []) {
        if (alt === query[ch] || this.lastGammaVeto <= 0) continue;
        for (const from of this.conditionMap.population(ch, alt)) {
          for (const to of core) this.net.strengthenInhibitory(from, to, this.lastGammaVeto);
        }
      }
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
    this.planWrite(query, observed, repeats, .1, .6);
    const core = this.coreFor(query);
    const condIds = this.conditionMap.encode(query);
    const accepted = this.acceptOutcomes(core, observed, "observation");
    if (Object.values(accepted).some((v) => typeof v === "number")) {
      const effRepeats = this.effectiveRepeats(repeats, 0.1);
      hebbianLearn(this.net, [...condIds, ...core], effRepeats, 0.1, 0.6);
      // 结果按通道分别绑定到核（同 teachExperience，防跨通道接力）
      for (const spec of this.outcomeMap.specs) {
        const bin = accepted[spec.name];
        if (typeof bin === "number") {
          const pops = this.outcomeMap.population(spec.name, bin);
          hebbianLearn(this.net, [...core, ...pops.map((id) => id + this.conditionMap.neuronCount)], effRepeats, 0.1, 0.6);
        }
      }
      this.markCoreEvidence(core);

    }
    // 登记观察到的条件档（自主探索路径修复：探索从零起步、不经 teachExperience，
    // 若不在此登记，observedBins 永远为空、否决边永远不会被写入——
    // 观察即经验，与 teachExperience 同一登记规则）
    for (const [ch, bin] of Object.entries(query)) {
      const seen = this.observedBins.get(ch) ?? new Set<number>();
      seen.add(bin);
      this.observedBins.set(ch, seen);
    }
    // 持证：影响因素加分边 + 替代档否决边（与 bindInfluence 同一规则）
    for (const [ch, delta] of Object.entries(this.lastBoost)) {
      // 评审 F06 同一规则：零增益通道不加分也不写否决；lastGammaVeto=0 断否决
      if (delta <= 0 || query[ch] === undefined) continue;
      for (const from of this.conditionMap.population(ch, query[ch]!)) {
        for (const to of core) this.net.strengthen(from, to, delta);
      }
      for (const alt of this.observedBins.get(ch) ?? []) {
        if (alt === query[ch] || this.lastGammaVeto <= 0) continue;
        for (const from of this.conditionMap.population(ch, alt)) {
          for (const to of core) this.net.strengthenInhibitory(from, to, this.lastGammaVeto);
        }
      }
    }
  }

  private lastBoost: Record<string, number> = {};
  private lastGammaVeto = 1.5;
}
