import { mulberry32 } from "../prng.js";

/**
 * 容量测试用合成规则域：两因素 + 一干扰 → 单结果通道。
 * 地面真理 out = (3·f1 + 5·f2) mod 8：
 *   3、5 均为奇数 ⇒ f1/f2 任意非零档差都改变结果（控制变量下永远有影响），
 *   干扰通道从不出现在真理中。
 * 规则数（概念数）= 教学的 (f1,f2) 组合数 E，E ∈ [1, 64]。
 */

export const CAP_CONDITION_SPECS = [
  { name: "f1", bins: 8 },
  { name: "f2", bins: 8 },
  { name: "distraction", bins: 8 },
] as const;

export const CAP_OUTCOME_SPECS = [{ name: "out", bins: 8 }] as const;

export type CConditions = Record<string, number>;
export type COutcomes = Record<string, number>;

export function capTruth(c: CConditions): COutcomes {
  return { out: (3 * c.f1! + 5 * c.f2!) % 8 };
}

/** 平滑对照真理：out = min(7, f1+f2)——相邻输入映射相邻输出，可加性结构 */
export function capTruthSmooth(c: CConditions): COutcomes {
  return { out: Math.min(7, c.f1! + c.f2!) };
}

export interface CExperiment {
  readonly conditions: CConditions;
  readonly outcomes: COutcomes;
}
export interface CPair {
  readonly e0: CExperiment;
  readonly e1: CExperiment;
}
export interface CGroup {
  readonly name: string;
  readonly manipulated: string;
  readonly pairs: readonly CPair[];
}

function exp(f1: number, f2: number, distraction: number, truth: (c: CConditions) => COutcomes): CExperiment {
  const conditions = { f1, f2, distraction };
  return { conditions, outcomes: truth(conditions) };
}

/** 采样 E 个互不相同的 (f1,f2) 组合（可复现） */
export function sampleCombos(e: number, seed: number): [number, number][] {
  const rand = mulberry32(seed);
  const all: [number, number][] = [];
  for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) all.push([a, b]);
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [all[i], all[j]] = [all[j]!, all[i]!];
  }
  return all.slice(0, e);
}

/**
 * 为 E 个教学组合生成控制变量课程：
 * - distraction 组：每组合一对仅干扰档不同的实验（结果不变）；
 * - f1 组：每组合一对 (a,b) vs ((a+1)%8, b)（仅 f1 变，结果变）；
 * - f2 组：同理仅 f2 变。
 */
export function capCurriculum(
  combos: readonly [number, number][],
  truth: (c: CConditions) => COutcomes = capTruth,
): CGroup[] {
  const distPairs: CPair[] = [];
  const f1Pairs: CPair[] = [];
  const f2Pairs: CPair[] = [];
  for (const [a, b] of combos) {
    distPairs.push({ e0: exp(a, b, 0, truth), e1: exp(a, b, 1, truth) });
    f1Pairs.push({ e0: exp(a, b, 0, truth), e1: exp((a + 1) % 8, b, 0, truth) });
    f2Pairs.push({ e0: exp(a, b, 0, truth), e1: exp(a, (b + 1) % 8, 0, truth) });
  }
  return [
    { name: "distraction-group", manipulated: "distraction", pairs: distPairs },
    { name: "f1-group", manipulated: "f1", pairs: f1Pairs },
    { name: "f2-group", manipulated: "f2", pairs: f2Pairs },
  ];
}

export interface CQuery {
  readonly conditions: CConditions;
  readonly truth: COutcomes;
  readonly kind: "taught" | "heldout";
}

/** 查询：全部 64 组合（教学/留出由 kind 区分），干扰档取 0（与课程基线一致） */
export function capQueries(
  taught: ReadonlySet<string>,
  truth: (c: CConditions) => COutcomes = capTruth,
): CQuery[] {
  const out: CQuery[] = [];
  for (let a = 0; a < 8; a++) {
    for (let b = 0; b < 8; b++) {
      const conditions = { f1: a, f2: b, distraction: 0 };
      out.push({
        conditions,
        truth: truth(conditions),
        kind: taught.has(`${a},${b}`) ? "taught" : "heldout",
      });
    }
  }
  return out;
}
