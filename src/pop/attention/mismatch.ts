import type { FieldRuleMemory } from "../concept/field-memory.js";
import type { CValues } from "../concept/world-continuous.js";

/**
 * 失配场（M3 的候选强度来源，全部网络量、无评分公式）：
 * 对象 X 的"失配" = 其最佳规则核 K 对**自己答案场**的覆盖，减去 K 对
 * **观察结果场**的覆盖。覆盖 = K 的核神经元到目标场的连接总和（W 支持
 * 减 Γ 否决），全部从网络边权读出，无任何外部公式：
 * - 观察符合规则：覆盖(观察) ≈ 覆盖(自身) → 失配 ≈ 0；
 * - 观察违反规则：观察场与 K 无连接/被否决 → 失配大；
 * - 无规则核（未知）：失配取最大（无任何覆盖）。
 */
export function mismatchField(
  mem: FieldRuleMemory,
  coreIdx: number | null,
  observedOutcomes: CValues,
): number {
  if (coreIdx === null) return 40; // 无核可覆盖 → 失配取上限（无任何覆盖可言）
  const core = mem.ruleCore(coreIdx);
  const ownFields = mem.ruleOutcomeFields(coreIdx);
  const obsFields = mem.encoder.encode(observedOutcomes);
  const cover = (fields: readonly number[]): number => {
    let s = 0;
    for (const k of core) {
      for (const f of fields) s += mem.net.getWeight(k, f) - mem.net.getInhibitoryWeight(k, f);
    }
    return s;
  };
  return Math.max(0, cover(ownFields) - cover(obsFields));
}
