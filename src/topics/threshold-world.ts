import type { Conditions, Outcomes } from "../prototype/world.js";

/**
 * PLAN-010 陌生场景：多阈值耦合世界。
 * 与电路实验室刻意不同族：变量名/量程/真值结构无任何重叠。
 * - active = (alpha > 1) XOR (gamma > 1)（异或门，非电路世界的合取门）
 * - flow  = active ? 0.4·alpha + 1.2·beta : 0.15（beta 只在 active 时耦合进来）
 * - delta 是干扰维。
 * 真值只存在于实验台应答与评分侧，不进学习核。
 */

export const THRESHOLD_COND_DIMS = [
  { name: "alpha", min: 0, max: 2 },
  { name: "beta", min: 0, max: 2 },
  { name: "gamma", min: 0, max: 2 },
  { name: "delta", min: 0, max: 2 },
] as const;
export const THRESHOLD_OUTCOME_DIMS = [
  { name: "active", min: 0, max: 1 },
  { name: "flow", min: 0, max: 3 },
] as const;

/** 候选网格：不对称跨过阈值 1.0（0.9/1.1 跨阈，0.3/1.7 远离） */
export const THRESHOLD_GRID: Readonly<Record<string, readonly number[]>> = {
  alpha: [0.3, 0.9, 1.1, 1.7],
  beta: [0.3, 0.9, 1.1, 1.7],
  gamma: [0.3, 0.9, 1.1, 1.7],
  delta: [0.3, 0.9, 1.1, 1.7],
};

export function thresholdTruth(c: Conditions): Outcomes {
  const active = (c.alpha! > 1) !== (c.gamma! > 1) ? 1 : 0;
  const flow = active === 1 ? 0.4 * c.alpha! + 1.2 * c.beta! : 0.15;
  return { active, flow };
}

export class ThresholdBench {
  private cost = 0;
  conduct(c: Conditions): Outcomes {
    for (const d of THRESHOLD_COND_DIMS) {
      const v = c[d.name];
      if (v === undefined || !Number.isFinite(v) || v < d.min || v > d.max) {
        throw new Error(`非法条件：${d.name}=${v}`);
      }
    }
    this.cost++;
    return thresholdTruth(c);
  }
  get experimentsUsed(): number {
    return this.cost;
  }
}

export function thresholdKey(c: Conditions): string {
  return THRESHOLD_COND_DIMS.map((d) => c[d.name]).join(",");
}

export interface ThresholdProbe {
  conditions: Conditions;
  truth: Outcomes;
  seen: boolean;
}

/** 全部网格组合的评分探针（seen 标记已观察组合，评分不追加学习） */
export function thresholdProbes(seenKeys: ReadonlySet<string>): ThresholdProbe[] {
  const out: ThresholdProbe[] = [];
  const [a, b, g, d] = [THRESHOLD_GRID.alpha!, THRESHOLD_GRID.beta!, THRESHOLD_GRID.gamma!, THRESHOLD_GRID.delta!];
  for (const alpha of a) for (const beta of b) for (const gamma of g) for (const delta of d) {
    const conditions = { alpha, beta, gamma, delta };
    out.push({ conditions, truth: thresholdTruth(conditions), seen: seenKeys.has(thresholdKey(conditions)) });
  }
  return out;
}

/** 与 lcScore 同口径：亮灭粗粒度 + 连续量容差 0.5 的细粒度 */
export function thresholdScore(
  values: Record<string, number | null>,
  truth: Outcomes,
): { coarse: boolean; fine: boolean } {
  const a = values.active ?? null;
  const predActive = a === null ? null : a >= 0.5 ? 1 : 0;
  const coarse = predActive === truth.active;
  if (!coarse) return { coarse: false, fine: false };
  const f = values.flow ?? null;
  return { coarse: true, fine: f !== null && Math.abs(f - truth.flow!) <= 0.5 };
}
