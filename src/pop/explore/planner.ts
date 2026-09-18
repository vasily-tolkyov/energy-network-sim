import type { Conditions, Experiment, Pair } from "../../prototype/world.js";

/**
 * 实验规划器（方法先验的躯体化——"躯体"而非"大脑"）：
 *
 * 控制变量法不写成评分公式，而是写进动作空间结构：
 * - 候选干预 = 与某个已做实验**恰好相差一个维度**的条件组合
 *   （前沿扩展，Hamming-1）——动作空间里根本不存在"同时变两维"；
 * - 每个新实验自动与它的 Hamming-1 邻居成对（配对规则也是结构），
 *   对换结果顺带提供互斥学习素材；
 * - 规划器不决定"做哪个"——那是大脑（explorer 的 WTA）的事；
 *   这里只维护"能做什么"和"做过什么"。
 */

/** 已做实验的情景记录（条件+真值观察+三类判定结果） */
export interface Episode extends Experiment {
  readonly index: number;
  readonly classification: "within-envelope" | "prediction-violation" | "unknown-change";
}

export class ExperimentPlanner {
  private readonly episodes: Episode[] = [];
  private readonly conductedKeys = new Set<string>();
  /** 每个维度的已观察取值（新档检测用——触发全体重持证） */
  private readonly observedBins = new Map<string, Set<number>>();
  /** 偏差增压：候选干预的意外场残余（每步衰减） */
  private readonly violationBoost = new Map<string, number>();

  constructor(
    private readonly specs: readonly { readonly name: string; readonly bins: number }[],
  ) {}

  private keyOf(c: Conditions): string {
    return this.specs.map((s) => c[s.name]).join(",");
  }

  /** 两条件恰好相差一个维度时返回该维度名，否则 null（配对规则） */
  private differInOneDim(a: Conditions, b: Conditions): string | null {
    let diff: string | null = null;
    for (const s of this.specs) {
      if (a[s.name] !== b[s.name]) {
        if (diff !== null) return null;
        diff = s.name;
      }
    }
    return diff;
  }

  /** 登记一次实验并给出它自动成对的对（与全部历史实验比对，Hamming-1 即成对） */
  register(episode: Omit<Episode, "index">): { pairs: Pair[]; newBins: string[] } {
    const key = this.keyOf(episode.conditions);
    this.conductedKeys.add(key);
    const full: Episode = { ...episode, index: this.episodes.length };
    const pairs: Pair[] = [];
    for (const past of this.episodes) {
      if (this.differInOneDim(full.conditions, past.conditions) !== null) {
        pairs.push({ e0: past, e1: full });
      }
    }
    this.episodes.push(full);
    const newBins: string[] = [];
    for (const s of this.specs) {
      const v = full.conditions[s.name]!;
      const seen = this.observedBins.get(s.name) ?? new Set<number>();
      if (!seen.has(v)) {
        seen.add(v);
        this.observedBins.set(s.name, seen);
        newBins.push(s.name);
      }
    }
    return { pairs, newBins };
  }

  /** 候选干预：全部未做过的 Hamming-1 前沿（相对任何已做实验差恰好一维） */
  candidates(): Conditions[] {
    const out: Conditions[] = [];
    const seen = new Set<string>();
    for (const ep of this.episodes) {
      for (const s of this.specs) {
        for (let v = 0; v < s.bins; v++) {
          if (v === ep.conditions[s.name]) continue;
          const c: Conditions = { ...ep.conditions, [s.name]: v };
          const k = this.keyOf(c);
          if (this.conductedKeys.has(k) || seen.has(k)) continue;
          seen.add(k);
          out.push(c);
        }
      }
    }
    return out;
  }

  /** 候选干预 id（WTA 择选用——就是它的条件签名） */
  candidateId(c: Conditions): string {
    return this.keyOf(c);
  }

  /** 偏差事件后：给该实验的 Hamming-1 邻域候选加意外增压 */
  boostNeighbors(of: Conditions, amount: number): void {
    for (const c of this.candidates()) {
      if (this.differInOneDim(c, of) !== null) {
        const id = this.candidateId(c);
        this.violationBoost.set(id, (this.violationBoost.get(id) ?? 0) + amount);
      }
    }
  }

  /** 意外场是事件信号不是状态：每步衰减一半（与失配场逐帧清零同一哲学） */
  decayBoosts(): void {
    for (const [id, b] of this.violationBoost) {
      const next = b * 0.5;
      if (next < 0.05) this.violationBoost.delete(id);
      else this.violationBoost.set(id, next);
    }
  }

  boostOf(id: string): number {
    return this.violationBoost.get(id) ?? 0;
  }

  get experimentCount(): number {
    return this.episodes.length;
  }

  get allEpisodes(): readonly Episode[] {
    return this.episodes;
  }
}
