import { PopRuleMemory } from "../popmemory.js";
import type { PopChannelMap } from "../popmap.js";
import { R2PopLayer } from "../r2pop.js";
import { NeuralFocusNet } from "../attention/neural-focus.js";
import type { Conditions, Outcomes, Pair } from "../../prototype/world.js";
import type { ExperimentPlanner, Episode } from "./planner.js";

/**
 * 自主探索闭环（"大脑"）：无教师时系统的学习主循环。
 *
 * 统一回路（教师带领 = 只需观察的探索特例）：进入系统的永远是
 * "观察片段"（条件+真值结果）；教师模式下片段由外部供给，探索模式下
 * 片段由自己的干预挣得。学习机器（写入/成对/R2/侧重否决/规则核）不变。
 *
 * 决策全神经化（驱动信号 → WTA 择选），方法先验躯体化（动作空间结构
 * 在 planner 里——如同人只有大脑是神经的，躯体是实体的）：
 * - 无知场：候选干预无任何核覆盖（纯边权读出 < θ，与失配场同家族）→ 驱动 2.0；
 * - 验证驱动：预测已置信 → 驱动 0.4（"顺便确认"——置信不等于正确）；
 * - 意外场：偏差事件的 Hamming-1 邻域获增压、随步衰减（planner 管理）；
 * - 择选：候选干预各为一对 driver/marker 神经元，驱动场绝对重写边强，
 *   互斥退火择出胜者（NeuralFocusNet 第三次复用；驱动 ×4 跨 marker θ）。
 *
 * 终止：无候选仍处未知（无知场全消）且连续 verifyQuorum 次验证无偏差。
 */

export interface ExploreStep {
  readonly index: number;
  readonly conditions: Conditions;
  readonly predicted: Record<string, number | null | "ambiguous">;
  readonly observed: Outcomes;
  readonly classification: Episode["classification"];
  readonly drive: number;
  readonly candidateCount: number;
  readonly rulesFormed: number;
  readonly influentialSoFar: readonly string[];
}

export interface ExploreConfig {
  /** 实验预算上限（安全绳，防不收敛），默认 300 */
  readonly budget?: number;
  /** 验证相连续无偏差的终止次数，默认 8 */
  readonly verifyQuorum?: number;
  /** 观察写入重复数，默认 2 */
  readonly learnRepeats?: number;
  /** 无知场强度，默认 2.0 */
  readonly ignoranceDrive?: number;
}

export class Explorer {
  readonly mem: PopRuleMemory;
  private readonly r2: R2PopLayer;
  private readonly planner: ExperimentPlanner;
  private readonly specs: readonly { readonly name: string; readonly bins: number }[];
  private readonly cfg: Required<ExploreConfig>;
  private readonly magSum = new Map<string, number>();
  private readonly magCount = new Map<string, number>();
  private boost: Record<string, number> = {};
  private withinStreak = 0;
  private stepSeed: number;
  readonly log: ExploreStep[] = [];

  constructor(
    cm: PopChannelMap,
    om: PopChannelMap,
    planner: ExperimentPlanner,
    specs: readonly { readonly name: string; readonly bins: number }[],
    private readonly bench: { conduct(c: Conditions): Outcomes },
    private readonly outcomeSpan: Readonly<Record<string, number>>,
    config: ExploreConfig = {},
    seed = 1,
  ) {
    this.mem = new PopRuleMemory(cm, om, { maxRules: 384 });
    this.r2 = new R2PopLayer(cm, om);
    this.planner = planner;
    this.specs = specs;
    this.cfg = {
      budget: config.budget ?? 300,
      // 默认 16（评审"提前终止"病理修复后重新标定：8 连胜在覆盖变厚后过早达成，
      // 导致因素发现不全；16 连胜下三种子因素 4/4 全对、粗粒度 98-100%）
      verifyQuorum: config.verifyQuorum ?? 16,
      learnRepeats: config.learnRepeats ?? 2,
      ignoranceDrive: config.ignoranceDrive ?? 2.0,
    };
    this.stepSeed = seed;
  }

  /** 终止原因（评审"多种结束原因均为 false"修复）：结构化区分
   *  预算耗尽 / 前沿枯竭 / 验证达标；运行中为 null */
  terminationReason: "budget-exhausted" | "frontier-exhausted" | "quorum-met" | null = null;

  /** 多样性门（终止判据用）：每个条件维至少观察过 min(2, bins) 个不同取值，
   * 且两端极值（档 0 与档 bins−1）都被操纵过——门控的关闭侧常在极端值上，
   * 只在中间档扫描会在门控区从未被触及时就收工（评审修复：13 步早停、
   * 门控关闭侧 4/16 的根因）。方法先验的躯体化：探维先探其两端。 */
  private diverseEnough(): boolean {
    const seen = new Map<string, Set<number>>();
    for (const ep of this.planner.allEpisodes) {
      for (const [d, v] of Object.entries(ep.conditions)) {
        const s = seen.get(d) ?? new Set<number>();
        s.add(v);
        seen.set(d, s);
      }
    }
    return this.specs.every((s) => {
      const got = seen.get(s.name) ?? new Set<number>();
      return got.size >= Math.min(2, s.bins) && got.has(0) && got.has(s.bins - 1);
    });
  }

  /** 当前 R2 判为影响因素的维度（累计幅度 > 0 者） */
  get influentialDims(): string[] {
    // 过滤零幅度键（评审 F04/C07：不可判定对曾留下 sum=0 的键被误报为因素）
    return [...this.magSum.entries()].filter(([, sum]) => sum > 1e-9).map(([k]) => k).sort();
  }

  /** 无知场：无核覆盖（纯边权读出 < θ）或有维度的取值从未被观察 → 未知 */
  private ignoranceOf(c: Conditions): number {
    const unknown =
      this.mem.coreFieldCoverage(c) < this.mem.net.threshold || this.mem.uncoveredDims(c).length > 0;
    return unknown ? this.cfg.ignoranceDrive : 0.4;
  }

  /** 推进一步：选一个实验、执行、学习。返回 false 表示终止。 */
  step(): boolean {
    if (this.planner.experimentCount >= this.cfg.budget) {
      this.terminationReason = "budget-exhausted";
      return false;
    }

    // 首实验：无任何经验时，从量程中点探一针（任意但固定的起点）
    if (this.planner.experimentCount === 0) {
      const c: Conditions = {};
      for (const s of this.specs) c[s.name] = Math.floor(s.bins / 2);
      const observed = this.bench.conduct(c);
      this.mem.learnFromObservation(c, observed, this.cfg.learnRepeats);
      const { pairs, newBins } = this.planner.register({
        conditions: c,
        outcomes: observed,
        classification: "unknown-change",
      });
      this.absorbPairs(pairs, newBins);
      this.log.push({
        index: 0,
        conditions: c,
        predicted: {},
        observed,
        classification: "unknown-change",
        drive: this.cfg.ignoranceDrive,
        candidateCount: 0,
        rulesFormed: this.mem.ruleCount,
        influentialSoFar: [],
      });
      return true;
    }

    const candidates = this.planner.candidates();
    if (candidates.length === 0) {
      this.terminationReason = "frontier-exhausted";
      return false;
    }

    // 驱动场：无知场 + 意外残余
    const drives = new Map<string, number>();
    for (const c of candidates) {
      const id = this.planner.candidateId(c);
      drives.set(id, this.ignoranceOf(c) + this.planner.boostOf(id));
    }

    // 终止判据：无知场全消（候选全部置信）且验证相连续无偏差达标，
    // 且每个条件维都被操纵过（多样性门——与连续版语料充分性同一规则：
    // 评审发现的提前终止病理：覆盖置信过早使探索在门控维从未被操纵时就收工）。
    // 修复前 40 步预算下 13 步即停、门控关闭侧只剩 4/16。
    const anyUnknown = candidates.some(
      (c) => (drives.get(this.planner.candidateId(c)) ?? 0) >= this.cfg.ignoranceDrive - 1e-9,
    );
    if (!anyUnknown && this.withinStreak >= this.cfg.verifyQuorum && this.diverseEnough()) {
      this.terminationReason = "quorum-met";
      return false;
    }

    // WTA 择选（NeuralFocusNet：驱动场即失配场，×4 跨过 marker 点火阈值 θ=1.5）
    const focus = new NeuralFocusNet(
      candidates.map((c) => this.planner.candidateId(c)),
      { inertia: 0 },
    );
    focus.beginFrame();
    for (const c of candidates) {
      const id = this.planner.candidateId(c);
      focus.setMismatch(id, (drives.get(id) ?? 0) * 4);
    }
    const chosenId = focus.select(this.stepSeed);
    const chosen = candidates.find((c) => this.planner.candidateId(c) === chosenId)!;

    // 执行：先全档预测（三类判定用），再上台观察
    const predicted = this.mem.predict(chosen, this.stepSeed).decoded;
    const observed = this.bench.conduct(chosen);
    const unconfident = Object.values(predicted).some((v) => v === null || v === "ambiguous");
    const mismatch = !unconfident && Object.entries(observed).some(([ch, v]) => predicted[ch] !== v);
    const classification: Episode["classification"] = unconfident
      ? "unknown-change"
      : mismatch
        ? "prediction-violation"
        : "within-envelope";

    // 每次实验都是证据（评审 C05 根因修复）：符合/偏差/未知都写观察——
    // 写的是世界真值而非自猜（自强化只发生在写预测时，learnFromQuery 那条路）。
    // 修复前"符合不写"留下有侧重无结果的空核，它被证据过滤排除后
    // 该条件只能由近似核代答，门控角读数随之漂移。
    // （注意力流式监测器的"符合不写"是另一场景：重复状态的连续流；
    // 探索里每次实验都是昂贵的新探针，确认本身值得记录。）
    this.mem.learnFromObservation(chosen, observed, this.cfg.learnRepeats);
    this.withinStreak = classification === "within-envelope" ? this.withinStreak + 1 : 0;

    const { pairs, newBins } = this.planner.register({ conditions: chosen, outcomes: observed, classification });
    this.absorbPairs(pairs, newBins);
    if (classification === "prediction-violation") {
      // 意外场：邻域增压（下一步更可能复查附近）
      this.planner.boostNeighbors(chosen, 1.2);
    }
    this.planner.decayBoosts();

    this.log.push({
      index: this.planner.experimentCount - 1,
      conditions: chosen,
      predicted,
      observed,
      classification,
      drive: drives.get(chosenId) ?? 0,
      candidateCount: candidates.length,
      rulesFormed: this.mem.ruleCount,
      influentialSoFar: this.influentialDims,
    });
    return true;
  }

  /** 自造对喂 R2：差分 → 幅度累计 → 侧重/否决（含新档触发的全体重持证） */
  private absorbPairs(pairs: readonly Pair[], newBins: readonly string[]): void {
    for (const pair of pairs) {
      const analysis = this.r2.analyzePair(pair);
      // 不可判定的差分构型不进幅度累计（评审 F04：小模式上窗口不存在时
      // R2 曾误报因素；现在由 R2 显式标记，此处跳过）
      if (analysis.undecidable) continue;
      // 对换结果学习互斥（替代教师对换演示）
      for (const spec of Object.keys(this.outcomeSpan)) {
        const a = pair.e0.outcomes[spec]!;
        const b = pair.e1.outcomes[spec]!;
        if (a !== b) this.mem.teachExclusion("outcome", spec, a, b, 3.0);
      }
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
          m += Math.abs(delta) / (this.outcomeSpan[och] ?? 1);
        }
        this.magSum.set(ch, (this.magSum.get(ch) ?? 0) + m / this.outcomeSpanCount);
        this.magCount.set(ch, (this.magCount.get(ch) ?? 0) + 1);
      }
      this.refreshBoost();
      // 新对的两个端点按当前侧重持证（增量补正/否决）
      if (Object.keys(this.boost).length > 0) {
        this.mem.bindInfluence(pair.e0, this.boost, 2);
        this.mem.bindInfluence(pair.e1, this.boost, 2);
      }
    }
    // 新档首次出现：全体重持证（旧核的否决集合要补上新档）
    if (newBins.length > 0 && Object.keys(this.boost).length > 0) {
      for (const ep of this.planner.allEpisodes) {
        this.mem.bindInfluence(ep, this.boost, 2);
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

  private get outcomeSpanCount(): number {
    return Object.keys(this.outcomeSpan).length;
  }
}
