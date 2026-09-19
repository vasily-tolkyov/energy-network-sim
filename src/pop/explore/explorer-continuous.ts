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
  /** B0 语料阶段的自然终止验证数（默认 4，比 B2 松——语料够用就形成概念） */
  readonly corpusQuorum?: number;
  readonly fieldsPerDim?: number;
  /** 概念更新开关（默认 true；false = 冻结对照） */
  readonly conceptUpdate?: boolean;
  /** 概念更新的周期触发（每 N 次实验刷新一次，含细分检查），默认 25 */
  readonly reformPeriod?: number;
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
    this.mem = new FieldRuleMemory(this.enc, new EmergentMap([], this.enc), { maxRules: config.budget ?? 300 });
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
    };
    this.stepSeed = seed;
  }

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

    const candidates = this.planner.candidates();
    if (candidates.length === 0) {
      this.terminationReason = "frontier-exhausted";
      return false;
    }

    const drives = new Map<string, number>();
    for (const c of candidates) {
      const id = this.planner.candidateId(c);
      drives.set(id, this.ignoranceOf(c) + this.planner.boostOf(id));
    }

    const quorum = this.phase === "corpus" ? this.cfg.corpusQuorum : this.cfg.verifyQuorum;
    const anyUnknown = candidates.some(
      (c) => (drives.get(this.planner.candidateId(c)) ?? 0) >= this.cfg.ignoranceDrive - 1e-9,
    );
    if (!anyUnknown && this.withinStreak >= quorum) {
      if (this.phase === "corpus") {
        // 语料充分性门（种子 2 实测病理修复）：每个条件维至少观察过 2 个不同
        // 取值才允许形成概念——零变异的维只会形成 1 个概念，R2 的差分通道面
        // 永远看不到这个维 → 影响因素漏判 → 下游级联崩溃。
        // 不充分时不终止、继续语料相（验证连胜保留，候选驱动会自然带来变异）。
        if (this.corpusDiverseEnough()) {
          this.formConcepts();
          return true; // 间歇期完成，进入 B2
        }
      } else {
        this.terminationReason = "quorum-met";
        return false; // B2 达标，整体终止
      }
    }

    const focus = new NeuralFocusNet(candidates.map((c) => this.planner.candidateId(c)), { inertia: 0 });
    focus.beginFrame();
    for (const c of candidates) {
      const id = this.planner.candidateId(c);
      focus.setMismatch(id, (drives.get(id) ?? 0) * 4);
    }
    const chosenId = focus.select(this.stepSeed);
    const chosen = candidates.find((c) => this.planner.candidateId(c) === chosenId)!;

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

    this.log.push({
      index: this.planner.experimentCount - 1,
      phase: this.phase,
      conditions: chosen,
      predictedLit: predLit,
      converged: predicted.converged, terminationReason: predicted.terminationReason, anyOutputAbstained,
      observed,
      classification,
      drive: drives.get(chosenId) ?? 0,
      candidateCount: candidates.length,
      rulesFormed: this.mem.ruleCount,
    });
    return true;
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
