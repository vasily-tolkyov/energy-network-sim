/**
 * 第四主题（自主探索验收场景）：电路实验台世界。
 *
 * 地面真理结构与前三个主题都不同：
 * - 门控是**合取**（亮 = 开关闭合 AND 电压≥1）——小球是单门控、化学是析取，
 *   合取是第三种形态；
 * - 亮度含**拮抗因素**（电阻拉低亮度）——化学全是正向可加，这里首次出现负向；
 * - 温度小幅正向加成；材料是干扰因素（可操纵但不影响结果）。
 *
 * 与前三主题的关键区别：没有教师课程。系统通过 LabBench.conduct 自主做实验
 * （提交完整条件 → 返回真值观察 → 计 1 成本），规则必须从零自己发现。
 * 地面真理只用于实验台应答与评分，不进学习核。
 */

import type { Conditions, Outcomes } from "../prototype/world.js";

export const LAB_CONDITION_SPECS = [
  { name: "switch", bins: 2 },
  { name: "voltage", bins: 4 },
  { name: "resistance", bins: 4 },
  { name: "temperature", bins: 3 },
  { name: "material", bins: 3 },
] as const;

export const LAB_OUTCOME_SPECS = [
  { name: "lit", bins: 2 },
  { name: "brightness", bins: 4 },
] as const;

/** 地面真理（实验台私有）：亮 = 闭合 AND 有压；亮度 = 1 + 1.25·压 − 0.75·阻 + 0.5·温，截断到 0..3 */
export function labTruth(c: Conditions): Outcomes {
  const lit = c.switch === 1 && c.voltage! >= 1 ? 1 : 0;
  if (lit === 0) return { lit: 0, brightness: 0 };
  const raw = 1 + 1.25 * c.voltage! - 0.75 * c.resistance! + 0.5 * c.temperature!;
  return { lit: 1, brightness: Math.min(3, Math.max(0, Math.floor(raw))) };
}

/**
 * 实验台（干预接口）：系统的"躯体"。
 * 动作空间约束（方法先验的躯体化）：一次实验必须提交**完整**条件——
 * 不存在"只提交一半条件"的动作；每次提交计 1 成本，逼出效率。
 */
export class LabBench {
  private cost = 0;

  conduct(conditions: Conditions): Outcomes {
    for (const spec of LAB_CONDITION_SPECS) {
      const v = conditions[spec.name];
      // 评审 A07 修复：NaN/非有限值绕过比较（NaN 比较恒 false 曾返回"灭灯"假象），
      // 显式拒绝且不计成本
      if (v === undefined || !Number.isFinite(v) || v < 0 || v >= spec.bins) {
        throw new Error(`非法实验条件：${spec.name}=${v}（应在 [0, ${spec.bins})）`);
      }
    }
    this.cost++;
    return labTruth(conditions);
  }

  get experimentsUsed(): number {
    return this.cost;
  }
}

export interface LabProbe {
  readonly conditions: Conditions;
  readonly truth: Outcomes;
  /**
   * 分桶（评审 F10/E01 修复：原 heldout 桶按构造全是"亮"，常数基线 100%，
   * 不能作为门控泛化证据）：self-taught = 已做实验；unseen-gate-off =
  未做且门控关闭；unseen-gate-on = 未做且门控开启。两个 unseen 桶的并集 =
   * 全部未观察组合，报告时必须同时给常数基线对照。
   */
  readonly kind: "self-taught" | "unseen-gate-off" | "unseen-gate-on";
}

/** 评分探针：条件全网格（288 个），按来源分档（self-taught 集合由探索轨迹给出） */
export function labProbes(conducted: ReadonlySet<string>): LabProbe[] {
  const key = (c: Conditions) =>
    LAB_CONDITION_SPECS.map((s) => c[s.name]).join(",");
  const out: LabProbe[] = [];
  for (let sw = 0; sw < 2; sw++) {
    for (let voltage = 0; voltage < 4; voltage++) {
      for (let resistance = 0; resistance < 4; resistance++) {
        for (let temperature = 0; temperature < 3; temperature++) {
          for (let material = 0; material < 3; material++) {
            const conditions: Conditions = {
              switch: sw,
              voltage,
              resistance,
              temperature,
              material,
            };
            const kind: LabProbe["kind"] = conducted.has(key(conditions))
              ? "self-taught"
              : sw === 0 || voltage === 0
                ? "unseen-gate-off"
                : "unseen-gate-on";
            out.push({ conditions, truth: labTruth(conditions), kind });
          }
        }
      }
    }
  }
  return out;
}

/** 探针条件 → 签名（探索轨迹记录用） */
export function labKey(c: Conditions): string {
  return LAB_CONDITION_SPECS.map((s) => c[s.name]).join(",");
}
