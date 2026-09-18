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
}

export class ContinuousExplorer {
  readonly mem: FieldRuleMemory;
  readonly enc: SensoryEncoder;
  private readonly formation: ConceptFormation;
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
  readonly log: ContExploreStep[] = [];
  /** 形成间歇期的记录（验收报告用） */
  formationReport: { conceptsPerDim: Record<string, number>; centers: Record<string, number[]> } | null = null;

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
    this.mem = new FieldRuleMemory(this.enc, new EmergentMap([], this.enc), { maxRules: 192 });
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
    };
    this.stepSeed = seed;
  }

  get currentPhase(): "corpus" | "full" {
    return this.phase;
  }

  get influentialDims(): string[] {
    return [...this.magSum.keys()].sort();
  }

  private ignoranceOf(c: Conditions): number {
    return this.mem.coreFieldCoverage(c) < this.mem.net.threshold ? this.cfg.ignoranceDrive : 0.4;
  }

  /** 推进一步。返回 false 表示该阶段结束（B0 达标或整体终止）。 */
  step(): boolean {
    if (this.planner.experimentCount >= this.cfg.budget) return false;

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
        classification: "unknown-change", drive: this.cfg.ignoranceDrive, candidateCount: 0, rulesFormed: 1,
      });
      return true;
    }

    const candidates = this.planner.candidates();
    if (candidates.length === 0) return false;

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
    const unconfident = Object.values(predicted.values).some((v) => v === null) || predicted.ambiguous.length > 0;
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

    if (classification !== "within-envelope") {
      this.mem.learnFromObservation(chosen, observed, this.cfg.learnRepeats);
    }
    this.withinStreak = classification === "within-envelope" ? this.withinStreak + 1 : 0;

    const { pairs, newBins } = this.planner.register({ conditions: chosen, outcomes: observed, classification });
    if (this.phase === "full") this.absorbPairs(pairs, newBins);
    if (classification === "prediction-violation") this.planner.boostNeighbors(chosen, 1.2);
    this.planner.decayBoosts();

    this.log.push({
      index: this.planner.experimentCount - 1,
      phase: this.phase,
      conditions: chosen,
      predictedLit: predLit,
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

  /** 形成间歇期：自己的实验史 → 共现成阱 → 概念图 → R2 → 全量侧重/持证 */
  private formConcepts(): void {
    for (const ep of this.planner.allEpisodes) {
      this.formation.presentExperiment({ ...ep.conditions, ...ep.outcomes }, 2);
    }
    const allPairs = this.planner.allAutoPairs();
    for (const { pair, dim } of allPairs) {
      this.formation.presentSwap(dim, pair.e0.conditions[dim]!, pair.e1.conditions[dim]!, 3.0);
    }
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
    this.formationReport = { conceptsPerDim, centers };

    // 全量回溯：全部自动对差分 → 全局侧重 → 全体补持证 + 结果互斥
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
