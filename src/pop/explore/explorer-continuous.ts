import { SensoryEncoder } from "../concept/sensory.js";
import { ConceptFormation } from "../concept/formation.js";
import { EmergentMap } from "../concept/emergent-map.js";
import { EmergentChannelAdapter } from "../concept/emergent-channel-adapter.js";
import { FieldRuleMemory } from "../concept/field-memory.js";
import { R2PopLayer } from "../r2pop.js";
import { NeuralFocusNet } from "../attention/neural-focus.js";
import type { Conditions, Outcomes, Pair } from "../../prototype/world.js";
import type { ExperimentPlanner, Episode } from "./planner.js";

/**
 * 阶段 B 自主探索（连续世界）：与阶段 A 同一闭环，但值概念也要自己结晶。
 *
 * 两段式：
 * - B0 语料收集：场级规则记忆直接从感受野工作（无 R2、无侧重否决——
 *   概念还没形成），同一 WTA 探索循环收集实验语料；
 * - 形成间歇期：用**自己的实验史**跑概念形成（共现成阱 + 对换抑制），
 *   建 EmergentMap → R2 通道面 → 对全部自动对做差分 → 全局侧重 →
 *   全体规则核补持证（与阶段 A 的新档触发同一规则）；
 * - B2 全机器：恢复探索，此后每个新对即时差分+持证，直到终止判据。
 *
 * 如实边界：候选实验值仍来自分位网格（方法先验的躯体化），形成出的概念
 * 只用于 R2 通道面与解释报告，不反向改写候选集；终止判据与阶段 A 相同。
 */

export interface ContExploreStep {
  readonly index: number;
  readonly phase: "corpus" | "full";
  readonly conditions: Conditions;
  readonly predictedLit: number | null;
  readonly observed: Outcomes;
  readonly converged: boolean | null;
  readonly terminationReason: string;
  readonly anyOutputAbstained: boolean;
  readonly classification: Episode["classification"];
  readonly drive: number;
  readonly candidateCount: number;
  readonly rulesFormed: number;
}

export interface ContExploreConfig {
  readonly budget?: number;
  readonly verifyQuorum?: number;
  readonly learnRepeats?: number;
  readonly ignoranceDrive?: number;
  /** B0 最少真实观察数（默认 4）；形成还须每维至少两个值，与置信停止分开 */
  readonly corpusQuorum?: number;
  readonly fieldsPerDim?: number;
  /** 概念更新开关（默认 true；false = 冻结对照） */
  readonly conceptUpdate?: boolean;
  /** 概念更新的周期触发（每 N 次实验刷新一次，含细分检查），默认 25 */
  readonly reformPeriod?: number;
  /** 采样策略：balanced（默认，既有基准口径）| uncertainty（不确定性驱动：
   * 失配/低置信组合进入复验队列优先重做；quorum-met 停用，停止只看前沿清空） */
  readonly policy?: "balanced" | "uncertainty";
}

export class ContinuousExplorer {
  readonly mem: FieldRuleMemory;
  readonly enc: SensoryEncoder;
  /** 终止原因（评审"多种结束原因均为 false"修复）；运行中为 null */
  /** quorum-met is heuristic stopping, never a certificate of complete factors. */
  terminationReason: "budget-exhausted" | "frontier-exhausted" | "quorum-met" | null = null;
  private formation: ConceptFormation;
  private em: EmergentMap | null = null;
  private r2: R2PopLayer | null = null;
  private readonly planner: ExperimentPlanner;
  private readonly bench: { conduct(c: Conditions): Outcomes };
  private readonly condDimNames: readonly string[];
  private readonly outcomeSpan: Readonly<Record<string, number>>;
  private readonly cfg: Required<ContExploreConfig>;
  private readonly magSum = new Map<string, number>();
  private readonly magCount = new Map<string, number>();
  private boost: Record<string, number> = {};
  private withinStreak = 0;
  private stepSeed: number;
  private phase: "corpus" | "full" = "corpus";
  private sinceFormation = 0;
  readonly log: ContExploreStep[] = [];
  /** 概念形成历史（每次形成/更新一条，验收报告用） */
  readonly formationHistory: { conceptsPerDim: Record<string, number>; centers: Record<string, number[]> }[] = [];
  /** 最近一次概念形成报告（兼容既有测试与报告） */
  get formationReport(): { conceptsPerDim: Record<string, number>; centers: Record<string, number[]> } | null {
    return this.formationHistory.length > 0 ? this.formationHistory[this.formationHistory.length - 1]! : null;
  }

  constructor(
    condDims: readonly { name: string; min: number; max: number }[],
    outcomeDims: readonly { name: string; min: number; max: number }[],
    planner: ExperimentPlanner,
    private readonly specs: readonly { name: string; bins: number; values?: readonly number[] }[],
    bench: { conduct(c: Conditions): Outcomes },
    outcomeSpan: Readonly<Record<string, number>>,
    config: ContExploreConfig = {},
    seed = 1,
  ) {
    this.enc = new SensoryEncoder([...condDims, ...outcomeDims], config.fieldsPerDim ?? 40);
    this.formation = new ConceptFormation(this.enc);
    // FieldRuleMemory 内部不使用 emergent（仅持有引用）——B0 传空图，间歇期后 R2 有自己的图
    // eviction="lru"：容量满时回收最久未用核槽（遗忘=突触修剪），而不是撞墙抛错——
    // 终身探索的经验量随预算增长，固定容量墙会把长程学习卡死。
    this.mem = new FieldRuleMemory(this.enc, new EmergentMap([], this.enc), { maxRules: config.budget ?? 300, eviction: "lru" });
    this.mem.setOutcomeDimensions(outcomeDims.map((d) => d.name));
    this.planner = planner;
    this.bench = bench;
    this.condDimNames = condDims.map((d) => d.name);
    this.outcomeSpan = outcomeSpan;
    this.cfg = {
      budget: config.budget ?? 300,
      verifyQuorum: config.verifyQuorum ?? 8,
      learnRepeats: config.learnRepeats ?? 2,
      ignoranceDrive: config.ignoranceDrive ?? 2.0,
      corpusQuorum: config.corpusQuorum ?? 4,
      fieldsPerDim: config.fieldsPerDim ?? 40,
      conceptUpdate: config.conceptUpdate ?? true,
      reformPeriod: config.reformPeriod ?? 25,
      policy: config.policy ?? "balanced",
    };
    this.stepSeed = seed;
  }

  /** 复验队列（uncertainty 策略）：失配/低置信组合的签名 → 剩余复验次数。
   * 单次异常只吸引注意（进队列复验）；规则改判走记忆层的观察计票。 */
  private readonly reverify = new Map<string, number>();
  /** 规律性复检队列（uncertainty 策略）：最久未复检组合的签名 → 剩余次数。
   * 与异常复验分开管理，但清空它也是前沿清空停止的必要条件。 */
  private readonly healthcheck = new Map<string, number>();
  /** 各组合最近一次复检时的实验序号（轮扫进度） */
  private readonly healthChecked = new Map<string, number>();
  private healthSweepFrom = 0;
  /** 最后一次前沿推进（新组合首访）时的实验数——轮扫要赶上它才算扫完 */
  private lastFrontierAdvance = 0;

  get currentPhase(): "corpus" | "full" {
    return this.phase;
  }

  get influentialDims(): string[] {
    // 过滤零幅度键（评审 F04/C07：不可判定对曾留下 sum=0 的键被误报为因素）
    return [...this.magSum.entries()].filter(([, sum]) => sum > 1e-9).map(([k]) => k).sort();
  }

  /** 各维度累计的归因证据数（自动对中该维被判为影响因素的次数，概念更新验收用） */
  magEvidence(): Record<string, number> {
    return Object.fromEntries(this.magCount);
  }

  /** 无知场：无核覆盖（< θ）或有维度的取值从未被观察（uncoveredDims）→ 未知 */
  private ignoranceOf(c: Conditions): number {
    const unknown =
      this.mem.coreFieldCoverage(c) < this.mem.net.threshold || this.mem.uncoveredDims(c).length > 0;
    return unknown ? this.cfg.ignoranceDrive : 0.4;
  }

  /** 当前轮扫是否已赶上探索进度（全部已访组合在末次前沿推进后复检过） */
  private sweepComplete(): boolean {
    return this.planner.allEpisodes.every(
      ep => (this.healthChecked.get(this.planner.candidateId(ep.conditions)) ?? -1) >= this.lastFrontierAdvance);
  }

  /** 注入 2 个最久未复检的组合（轮扫制注入点；一轮扫完才开新一轮） */
  private injectHealthSweep(): void {
    const lastSeen = new Map<string, number>();
    for (const ep of this.planner.allEpisodes) lastSeen.set(this.planner.candidateId(ep.conditions), ep.index);
    const stale = [...lastSeen.entries()].filter(([id]) => !this.reverify.has(id) && !this.healthcheck.has(id));
    if (stale.every(([id]) => (this.healthChecked.get(id) ?? -1) >= this.healthSweepFrom)) {
      this.healthSweepFrom = this.planner.experimentCount; // 本轮扫完，开新一轮
    }
    for (const [id] of stale.filter(([id]) => (this.healthChecked.get(id) ?? -1) < this.healthSweepFrom)
      .sort((x, y) => (this.healthChecked.get(x[0]) ?? -1) - (this.healthChecked.get(y[0]) ?? -1)).slice(0, 2)) {
      this.healthcheck.set(id, 1);
    }
  }

  /** 推进一步。返回 false 表示该阶段结束（B0 达标或整体终止）。 */
  step(): boolean {
    if (this.planner.experimentCount >= this.cfg.budget) {
      this.terminationReason = "budget-exhausted";
      return false;
    }

    // 首实验：每个条件维取候选值中位数附近的点（任意但固定的起点）
    if (this.planner.experimentCount === 0) {
      const c: Conditions = {};
      for (const s of this.specs) {
        const vals = s.values!;
        c[s.name] = vals[Math.floor(vals.length / 2)]!;
      }
      const observed = this.bench.conduct(c);
      this.mem.learnFromObservation(c, observed, this.cfg.learnRepeats);
      this.planner.register({ conditions: c, outcomes: observed, classification: "unknown-change" });
      this.log.push({
        index: 0, phase: this.phase, conditions: c, predictedLit: null, observed,
        converged: null, terminationReason: "not-predicted", anyOutputAbstained: true,
        classification: "unknown-change", drive: this.cfg.ignoranceDrive, candidateCount: 0, rulesFormed: 1,
      });
      return true;
    }

    const frontier = this.planner.candidates();
    if (frontier.length === 0 && this.reverify.size === 0 && this.healthcheck.size === 0) {
      // 前沿清空还要求"当前轮扫赶上探索进度"：所有已访组合都在最后一次
      // 前沿推进之后复检存活过。否则继续注入复检，而不是在扫尾途中停止。
      if (this.cfg.policy !== "uncertainty" || this.sweepComplete()) {
        this.terminationReason = "frontier-exhausted";
        return false;
      }
      this.injectHealthSweep();
      if (this.healthcheck.size === 0) {
        this.terminationReason = "frontier-exhausted";
        return false;
      }
    }

    const drives = new Map<string, number>();
    for (const c of frontier) {
      const id = this.planner.candidateId(c);
      drives.set(id, this.ignoranceOf(c) + this.planner.boostOf(id));
    }
    // 队列优先级：异常复验 > 规律复检 > 前沿扩展（不确定性驱动的核心通道）
    const pool: Conditions[] = [...frontier];
    if (this.cfg.policy === "uncertainty") {
      for (const [queue, drive] of [[this.healthcheck, this.cfg.ignoranceDrive + 0.5],
        [this.reverify, this.cfg.ignoranceDrive * 2 + 1]] as const) {
        for (const [id, remaining] of queue) {
          const original = this.planner.allEpisodes.findLast(e => this.planner.candidateId(e.conditions) === id);
          if (!original || remaining <= 0) { queue.delete(id); continue; }
          pool.unshift(original.conditions);
          drives.set(id, drive);
        }
      }
    } else {
      this.reverify.clear();
      this.healthcheck.clear();
    }

    // Concept formation is a representation update from observations, not a
    // certificate that predictions have converged. Waiting for confident
    // predictions here creates a circular dependency on the unformed concepts.
    if (this.phase === "corpus" && this.planner.experimentCount >= this.cfg.corpusQuorum && this.corpusDiverseEnough()) {
      this.formConcepts();
      return true;
    }
    const anyUnknown = frontier.some(
      (c) => (drives.get(this.planner.candidateId(c)) ?? 0) >= this.cfg.ignoranceDrive - 1e-9,
    );
    // quorum-met 只在 balanced 策略下启用（启发式辅助）；uncertainty 策略
    // 的停止只看前沿清空 + 复验清空——停止判据必须包含未覆盖风险。
    if (this.cfg.policy === "balanced" && this.phase === "full" && !anyUnknown && this.withinStreak >= this.cfg.verifyQuorum) {
      this.terminationReason = "quorum-met";
      return false;
    }

    const focus = new NeuralFocusNet(pool.map((c) => this.planner.candidateId(c)), { inertia: 0 });
    focus.beginFrame();
    for (const c of pool) {
      const id = this.planner.candidateId(c);
      focus.setMismatch(id, (drives.get(id) ?? 0) * 4);
    }
    const chosenId = focus.select(this.stepSeed);
    const chosen = pool.find((c) => this.planner.candidateId(c) === chosenId)!;
    const chosenFromFrontier = frontier.some((c) => this.planner.candidateId(c) === chosenId);

    const predicted = this.mem.predict(chosen, this.stepSeed);
    const observed = this.bench.conduct(chosen);
    const predLit = predicted.values.lit ?? null;
    const anyOutputAbstained = Object.values(predicted.values).some((v) => v === null) || predicted.ambiguous.length > 0;
    const unconfident = !predicted.converged || anyOutputAbstained;
    const mismatch =
      !unconfident &&
      Object.entries(observed).some(([ch, v]) => {
        const p: number | null | undefined = predicted.values[ch];
        if (p === null || p === undefined) return true;
        // 结果维容差：lit 阈值 0.5 化，连续维 |Δ| ≤ 0.5（与评分同一口径）
        return ch === "lit" ? (p >= 0.5 ? 1 : 0) !== v : Math.abs(p - v) > 0.5;
      });
    const classification: Episode["classification"] = unconfident
      ? "unknown-change"
      : mismatch
        ? "prediction-violation"
        : "within-envelope";

    // 每次实验都是证据（评审 C05 根因修复，与离散版同一规则）：
    // 符合/偏差/未知都写观察——写的是世界真值而非自猜。
    // 修复前"符合不写"留下有侧重无结果的空核，门控角读数随之漂移。
    this.mem.learnFromObservation(chosen, observed, this.cfg.learnRepeats);
    this.withinStreak = classification === "within-envelope" ? this.withinStreak + 1 : 0;

    const { pairs, newBins } = this.planner.register({ conditions: chosen, outcomes: observed, classification });
    if (chosenFromFrontier) this.lastFrontierAdvance = this.planner.experimentCount;
    if (this.phase === "full") {
      this.sinceFormation++;
      // 概念更新：覆盖缺口（新值无任何概念）或周期到达 → 先重形成再差分，
      // 保证这对实验在新通道面上被归因
      if (this.shouldReform(chosen)) {
        // 重形成的全量重放已包含当前步的对——跳过 absorbPairs 的二次处理
        // （评审 §7 修复：重形成步的证据与加强曾被计算两次）
        this.formConcepts();
      } else {
        this.absorbPairs(pairs, newBins);
      }
    }
    if (classification === "prediction-violation") this.planner.boostNeighbors(chosen, 1.2);
    this.planner.decayBoosts();

    // 队列维护（uncertainty 策略）：失配/低置信 → 异常复验（上限 2×定额）；
    // 恢复置信 → 出队；例行复检发现失配 → 升级为完整复验。
    if (this.cfg.policy === "uncertainty") {
      const id = this.planner.candidateId(chosen);
      const anomalous = this.reverify.get(id);
      const routine = this.healthcheck.get(id);
      if (anomalous !== undefined) {
        if (classification === "within-envelope") this.reverify.delete(id);
        else if (anomalous <= 1) this.reverify.delete(id); // 耗尽：改判与否交给记忆层计票
        else this.reverify.set(id, anomalous - 1);
      } else if (routine !== undefined) {
        this.healthcheck.delete(id);
        this.healthChecked.set(id, this.planner.experimentCount - 1);
        // 例行复检只有"确信但被证伪"才升级为完整复验；非收敛/歧义不是异常证据
        if (classification === "prediction-violation") this.reverify.set(id, 2 * this.mem.switchQuorum);
      } else if (classification === "prediction-violation"
        || (classification === "unknown-change" && this.mem.contestedDims(chosen).length > 0)) {
        // 只有"确信的预测被证伪"或"确有候选在竞争"才复验——
        // 首次到访的低置信是正常学习起点，重复同一观察不增加信息。
        this.reverify.set(id, 2 * this.mem.switchQuorum);
      }
      // 规律性复检（轮扫制）：每 10 次实验注入一次；一轮扫完才开新一轮。
      if (this.planner.experimentCount % 10 === 0) this.injectHealthSweep();
    }

    this.log.push({
      index: this.planner.experimentCount - 1,
      phase: this.phase,
      conditions: chosen,
      predictedLit: predLit,
      converged: predicted.converged, terminationReason: predicted.terminationReason, anyOutputAbstained,
      observed,
      classification,
      drive: drives.get(chosenId) ?? 0,
      candidateCount: pool.length,
      rulesFormed: this.mem.ruleCount,
    });
    return true;
  }

  /** 覆盖风险报告（不确定性地图快照）：停止时必须随日志输出——
   * "没发现"要说清"哪里没覆盖"，不许用总体准确率掩盖。 */
  coverageReport(): { experiments: number; unvisited: number; reverifyPending: number; contested: number } {
    let contested = 0;
    for (const ep of this.planner.allEpisodes) {
      if (ep.conditions && this.mem.contestedDims(ep.conditions).length > 0) contested++;
    }
    return {
      experiments: this.planner.experimentCount,
      unvisited: this.planner.candidates().length,
      reverifyPending: this.reverify.size,
      contested,
    };
  }

  /** 语料充分性：每个条件维至少观察过 2 个不同取值（概念形成需要变异） */
  private corpusDiverseEnough(): boolean {
    const seen = new Map<string, Set<number>>();
    for (const ep of this.planner.allEpisodes) {
      for (const [d, v] of Object.entries(ep.conditions)) {
        const s = seen.get(d) ?? new Set<number>();
        s.add(v);
        seen.set(d, s);
      }
    }
    for (const name of this.condDimNames) {
      if ((seen.get(name)?.size ?? 0) < 2) return false;
    }
    return true;
  }

  /**
   * 概念更新触发（B2 相，规则核绑定的是感受野不是概念——概念层演化
   * 不会损坏任何已学规则，只管 R2 通道面与归因分辨率）：
   * - 覆盖缺口：最新实验的某条件维取值解析不到任何概念（无重叠 → null）；
   * - 周期刷新：每 reformPeriod 次实验一次（粗概念内部多峰化时的细分机会——
   *   "共性先行、细分后至"的后半句）。
   */
  private shouldReform(latest: Conditions): boolean {
    if (!this.cfg.conceptUpdate || this.em === null) return false;
    for (const name of this.condDimNames) {
      if (this.em.resolve(name, latest[name]!) === null) return true;
    }
    return this.sinceFormation >= this.cfg.reformPeriod;
  }

  /**
   * 概念形成/更新（同一入口）：从情景缓冲**全量重放**到全新的 ConceptFormation
   * （旧实例不可增量复用——重复呈现同一语料会把差分边全部顶到 cap、聚类糊掉），
   * 幅度累计器清零后在**新通道面上重分析全部自动对**——旧归因不混入新表面。
   */
  // Deliberate training epoch: replay recomputes R2 evidence counts from zero,
  // then bindInfluence adds another bounded plasticity pass to existing R3.
  // This is NOT new independent evidence and is NOT an idempotent refresh.
  private formConcepts(): void {
    this.formation = new ConceptFormation(this.enc);
    for (const ep of this.planner.allEpisodes) {
      this.formation.presentExperiment({ ...ep.conditions, ...ep.outcomes }, 2);
    }
    const allPairs = this.planner.allAutoPairs();
    // 注：换对抑制不再进形成网（死参数删除，评审 A12）；替代值区域由非共现
    // 统计自然分离；结果侧互斥在下方的 learnExclusion 循环（写入记忆层）。
    this.em = new EmergentMap(this.formation.extractConcepts(0.5), this.enc);
    const condAdapter = new EmergentChannelAdapter(this.em, this.condDimNames, 0);
    const outAdapter = new EmergentChannelAdapter(
      this.em,
      Object.keys(this.outcomeSpan),
      this.enc.dimensionOffset(Object.keys(this.outcomeSpan)[0]!),
      "fields",
    );
    this.r2 = new R2PopLayer(condAdapter, outAdapter);

    const conceptsPerDim: Record<string, number> = {};
    const centers: Record<string, number[]> = {};
    for (const name of [...this.condDimNames, ...Object.keys(this.outcomeSpan)]) {
      const cs = this.em.concepts.filter((c) => c.dimension === name);
      conceptsPerDim[name] = cs.length;
      centers[name] = cs.map((c) => +c.centerValue.toFixed(2));
    }
    this.formationHistory.push({ conceptsPerDim, centers });
    this.sinceFormation = 0;

    // 全量回溯（重分析）：幅度累计器清零——旧通道面的归因不混入新表面
    this.magSum.clear();
    this.magCount.clear();
    const bin = (e: { conditions: Conditions; outcomes: Outcomes }): { conditions: Conditions; outcomes: Outcomes } => ({
      conditions: Object.fromEntries(
        Object.entries(e.conditions).map(([d, v]) => [d, this.em!.resolve(d, v) ?? -1]),
      ),
      outcomes: e.outcomes,
    });
    for (const { pair } of allPairs) {
      const analysis = this.r2.analyzePair({ e0: bin(pair.e0), e1: bin(pair.e1) });
      for (const och of Object.keys(this.outcomeSpan)) {
        const a = pair.e0.outcomes[och]!;
        const b = pair.e1.outcomes[och]!;
        if (Math.abs(a - b) > 1e-9) this.mem.learnExclusion(och, a, b, 3.0);
      }
      if (analysis.undecidable) continue; // 不可判定对不进幅度累计（评审 F04）
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
          m += Math.abs(delta) / (this.outcomeSpan[och] ?? 1);
        }
        this.magSum.set(ch, (this.magSum.get(ch) ?? 0) + m / Object.keys(this.outcomeSpan).length);
        this.magCount.set(ch, (this.magCount.get(ch) ?? 0) + 1);
      }
    }
    this.refreshBoost();
    if (Object.keys(this.boost).length > 0) {
      for (const ep of this.planner.allEpisodes) {
        this.mem.bindInfluence(ep.conditions, this.boost, 2);
      }
    }
    this.phase = "full";
  }

  /** B2 增量：新对即时差分+持证（新档触发全体重持证，同阶段 A） */
  private absorbPairs(pairs: readonly Pair[], newBins: readonly string[]): void {
    const bin = (e: { conditions: Conditions; outcomes: Outcomes }): { conditions: Conditions; outcomes: Outcomes } => ({
      conditions: Object.fromEntries(
        Object.entries(e.conditions).map(([d, v]) => [d, this.em!.resolve(d, v) ?? -1]),
      ),
      outcomes: e.outcomes,
    });
    for (const pair of pairs) {
      const analysis = this.r2!.analyzePair({ e0: bin(pair.e0), e1: bin(pair.e1) });
      for (const och of Object.keys(this.outcomeSpan)) {
        const a = pair.e0.outcomes[och]!;
        const b = pair.e1.outcomes[och]!;
        if (Math.abs(a - b) > 1e-9) this.mem.learnExclusion(och, a, b, 3.0);
      }
      if (analysis.undecidable) continue; // 不可判定对不进幅度累计（评审 F04）
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
          m += Math.abs(delta) / (this.outcomeSpan[och] ?? 1);
        }
        this.magSum.set(ch, (this.magSum.get(ch) ?? 0) + m / Object.keys(this.outcomeSpan).length);
        this.magCount.set(ch, (this.magCount.get(ch) ?? 0) + 1);
      }
      this.refreshBoost();
      if (Object.keys(this.boost).length > 0) {
        this.mem.bindInfluence(pair.e0.conditions, this.boost, 2);
        this.mem.bindInfluence(pair.e1.conditions, this.boost, 2);
      }
    }
    if (newBins.length > 0 && Object.keys(this.boost).length > 0) {
      for (const ep of this.planner.allEpisodes) {
        this.mem.bindInfluence(ep.conditions, this.boost, 2);
      }
    }
  }

  private refreshBoost(): void {
    const boost: Record<string, number> = {};
    for (const [ch, sum] of this.magSum) {
      boost[ch] = 0.1 * 3 * (sum / Math.max(1, this.magCount.get(ch) ?? 1)) ** 2;
    }
    this.boost = boost;
  }
}
