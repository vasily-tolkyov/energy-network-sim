import type { Conditions, Outcomes, Pair } from "./world.js";

/**
 * R2A/R2B：对 R1 读出（每次经验的条件/结果表征）的比较操作。
 * 如实声明：这里实现为对神经元群体读出集合的比较函数，
 * 不宣称是神经机制（对应材料实现中"暂存+差分+差异驱动开关"的功能位）。
 *
 * 约定（文档 §4）：变化标志与变化数值分开保存——从 1 变成 0 仍是变化。
 */

export interface ChannelDiff {
  readonly changed: boolean;
  readonly from: number;
  readonly to: number;
  /** 带符号档位差 to−from；未变化时为 0 */
  readonly delta: number;
}

export function diffValues(a: Readonly<Record<string, number>>, b: Readonly<Record<string, number>>): Record<string, ChannelDiff> {
  const out: Record<string, ChannelDiff> = {};
  for (const ch of Object.keys(a)) {
    const from = a[ch]!;
    const to = b[ch]!;
    out[ch] = { changed: from !== to, from, to, delta: to - from };
  }
  return out;
}

export function anyChanged(diff: Readonly<Record<string, ChannelDiff>>): boolean {
  return Object.values(diff).some((d) => d.changed);
}

/** R2A：两次实验的结果比较（共同结果 + 差异，flag 与数值分离） */
export interface OutcomeComparison {
  readonly common: Outcomes; // 未变化的结果通道取值
  readonly diffs: Record<string, ChannelDiff>;
  readonly outcomeChanged: boolean;
}

export function compareOutcomes(y0: Outcomes, y1: Outcomes): OutcomeComparison {
  const diffs = diffValues(y0, y1);
  const common: Outcomes = {};
  for (const [ch, d] of Object.entries(diffs)) {
    if (!d.changed) common[ch] = d.from;
  }
  return { common, diffs, outcomeChanged: anyChanged(diffs) };
}

/** R2B：条件比较 + 结合结果变化判断影响因素 */
export interface GroupInfluence {
  /** 影响因素通道（本组背景下，改变后伴随结果改变的因素） */
  readonly influential: readonly string[];
  /** 变化候选（实验中确实改变过的通道） */
  readonly changeCandidates: readonly string[];
  /**
   * 影响幅度 m_j：该因素改变时结果变化的平均归一化幅度（0..1）。
   * R2A 保留变化的方向与数值，这里把 |Δ| 归一化到通道档位跨度。
   */
  readonly magnitude: Readonly<Record<string, number>>;
  /** 逐对的条件差分与结果是否变化（证据保留） */
  readonly perPair: readonly {
    readonly conditionDiffs: Record<string, ChannelDiff>;
    readonly outcomeDiffs: Record<string, ChannelDiff>;
    readonly outcomeChanged: boolean;
  }[];
}

export function extractInfluences(
  conditionChannels: readonly string[],
  pairs: readonly Pair[],
  outcomeSpan: Readonly<Record<string, number>>,
): GroupInfluence {
  const perPair = pairs.map((p) => ({
    conditionDiffs: diffValues(p.e0.conditions, p.e1.conditions),
    outcomeDiffs: diffValues(p.e0.outcomes, p.e1.outcomes),
    outcomeChanged: compareOutcomes(p.e0.outcomes, p.e1.outcomes).outcomeChanged,
  }));
  const candidates = new Set<string>();
  const influential = new Set<string>();
  const magSum = new Map<string, number>();
  const magCount = new Map<string, number>();
  for (const { conditionDiffs, outcomeDiffs, outcomeChanged } of perPair) {
    for (const ch of conditionChannels) {
      const d = conditionDiffs[ch];
      if (d?.changed) {
        candidates.add(ch);
        // 变化候选 ∩ 结果变化 → 影响因素（本组背景内）
        if (outcomeChanged) {
          influential.add(ch);
          // 影响幅度：逐结果通道归一化 |Δ|，再对通道数取平均（m ∈ [0,1]）
          let m = 0;
          let counted = 0;
          for (const [och, od] of Object.entries(outcomeDiffs)) {
            const span = outcomeSpan[och] ?? 1;
            m += Math.abs(od.delta) / span;
            counted++;
          }
          m = counted === 0 ? 0 : m / counted;
          magSum.set(ch, (magSum.get(ch) ?? 0) + m);
          magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
        }
      }
    }
  }
  const magnitude: Record<string, number> = {};
  for (const ch of influential) {
    magnitude[ch] = (magSum.get(ch) ?? 0) / Math.max(1, magCount.get(ch) ?? 1);
  }
  return {
    influential: [...influential].sort(),
    changeCandidates: [...candidates].sort(),
    magnitude,
    perPair,
  };
}
