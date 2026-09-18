/**
 * 阶段 B 验收场景：连续版电路实验台世界。
 *
 * 与阶段 A（lab-world）**完全相同的因果结构**——合取门控（亮 = 闭合 AND 有压）、
 * 拮抗电阻、温度加成、材料干扰——但所有条件与结果都是实数：
 * 分档不再给定，值概念必须由系统自己的实验史结晶（概念自形成接入实验台）。
 * 这是受控对照：同一结构，唯一变量是"表征是给定的还是自形成的"。
 *
 * 地面真理只用于实验台应答与评分，不进学习核。
 */

import type { ContinuousDim } from "../pop/concept/world-continuous.js";

export const LAB_CONT_CONDITION_DIMS: ContinuousDim[] = [
  { name: "switchPos", min: 0, max: 1 },
  { name: "voltage", min: 0, max: 3 },
  { name: "resistance", min: 0, max: 3 },
  { name: "temperature", min: 0, max: 2 },
  { name: "material", min: 0, max: 2 },
];

export const LAB_CONT_OUTCOME_DIMS: ContinuousDim[] = [
  { name: "lit", min: 0, max: 1 },
  { name: "brightness", min: 0, max: 3 },
];

export type LCValues = Record<string, number>;

/** 地面真理（实验台私有）：亮 = 开关度≥0.5 AND 电压≥1；亮度 = 1 + 1.25v − 0.75r + 0.5t（连续，截断 0..3） */
export function labContTruth(c: LCValues): LCValues {
  const lit = c.switchPos! >= 0.5 && c.voltage! >= 1.0 ? 1 : 0;
  if (lit === 0) return { lit: 0, brightness: 0 };
  const raw = 1 + 1.25 * c.voltage! - 0.75 * c.resistance! + 0.5 * c.temperature!;
  return { lit: 1, brightness: Math.min(3, Math.max(0, raw)) };
}

/** 连续实验台（干预接口）：完整实数条件 → 真值观察 → 计 1 成本 */
export class LabContBench {
  private cost = 0;

  conduct(conditions: LCValues): LCValues {
    for (const d of LAB_CONT_CONDITION_DIMS) {
      const v = conditions[d.name];
      if (v === undefined || v < d.min - 1e-9 || v > d.max + 1e-9) {
        throw new Error(`非法实验条件：${d.name}=${v}（应在 [${d.min}, ${d.max}]）`);
      }
    }
    this.cost++;
    return labContTruth(conditions);
  }

  get experimentsUsed(): number {
    return this.cost;
  }
}

/** 每个条件维的候选实验值（方法先验：分位网格；动作空间里只有这些点） */
export const LAB_CONT_GRID: Readonly<Record<string, readonly number[]>> = {
  switchPos: [0.1, 0.5, 0.9],
  voltage: [0.3, 1.2, 2.1, 3.0],
  resistance: [0.3, 1.2, 2.1, 3.0],
  temperature: [0.3, 1.0, 1.7],
  material: [0.3, 1.0, 1.7],
};

export interface LCProbe {
  readonly conditions: LCValues;
  readonly truth: LCValues;
  readonly kind: "self-taught" | "gate-off" | "heldout";
}

/** 评分探针：网格全组合（3×4×4×3×3=432），按来源分档 */
export function labContProbes(conducted: ReadonlySet<string>): LCProbe[] {
  const key = (c: LCValues) => LAB_CONT_CONDITION_DIMS.map((d) => c[d.name]).join(",");
  const out: LCProbe[] = [];
  for (const switchPos of LAB_CONT_GRID.switchPos!) {
    for (const voltage of LAB_CONT_GRID.voltage!) {
      for (const resistance of LAB_CONT_GRID.resistance!) {
        for (const temperature of LAB_CONT_GRID.temperature!) {
          for (const material of LAB_CONT_GRID.material!) {
            const conditions: LCValues = { switchPos, voltage, resistance, temperature, material };
            const kind: LCProbe["kind"] =
              conducted.has(key(conditions))
                ? "self-taught"
                : switchPos < 0.5 || voltage < 1.0
                  ? "gate-off"
                  : "heldout";
            out.push({ conditions, truth: labContTruth(conditions), kind });
          }
        }
      }
    }
  }
  return out;
}

export function labContKey(c: LCValues): string {
  return LAB_CONT_CONDITION_DIMS.map((d) => c[d.name]).join(",");
}

/**
 * 容差评分：lit 精确（预测值阈值 0.5 化 0/1），brightness |预测−真值| ≤ 0.5。
 * 粗粒度 = 亮灭判定正确；细粒度 = 粗粒度 且 亮度在容差内。
 */
export function lcScore(
  values: Record<string, number | null>,
  truth: LCValues,
): { coarse: boolean; fine: boolean } {
  const litV = values.lit ?? null;
  const predLit = litV === null ? null : litV >= 0.5 ? 1 : 0;
  const coarse = predLit === truth.lit;
  if (!coarse) return { coarse: false, fine: false };
  if (truth.lit === 0) return { coarse: true, fine: true };
  const b = values.brightness ?? null;
  return { coarse: true, fine: b !== null && Math.abs(b - truth.brightness!) <= 0.5 };
}
