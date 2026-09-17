import type { FieldRuleMemory } from "../concept/field-memory.js";
import type { CValues } from "../concept/world-continuous.js";

/**
 * 捕获检验（M2）：三类判定从手写分类公式改为纯动力学势阱归属。
 *
 * 检验方式：把对象的条件场与**观察到的结果场**一起钳制，让网络自己
 * 补全——能接收这组"条件+结果"的规则核会被捕获激活：
 * - 预测对应的核被捕获 → 符合（within-envelope）；
 * - 其他已学核被捕获 → 偏差（prediction-violation）；
 * - 无核被捕获 → 未知（unknown-change）。
 * 没有任何"比较逻辑"：判定 = 势阱归属。
 */

export type CaptureClass = "within-envelope" | "prediction-violation" | "unknown-change";

/** 预测对应核的索引来源：predict 的获胜核（由调用方给出） */
export function captureClassify(
  mem: FieldRuleMemory,
  conditions: CValues,
  observedOutcomes: CValues,
  forecastCoreIdx: number | null,
  seed = 1,
): { class: CaptureClass; capturedCores: number[] } {
  // 钳制：条件场 ∪ 观察结果场（结果维度也是编码器维度，直接编码）
  const input = mem.encoder.encode({ ...conditions, ...observedOutcomes });
  const result = mem.net.settleAnnealed(input, [], {
    seed,
    extraCandidates: mem.allCoreNeurons(),
    quenchCandidatesOnly: true,
    levels: 6,
    sweepsPerLevel: 8,
  });
  const captured = mem.activeCores(result.activeNeurons);
  // 无时间对应的预测时不能报"偏差"（文档三态原则）——一律报未知
  if (forecastCoreIdx === null) {
    return { class: "unknown-change", capturedCores: captured };
  }
  if (captured.includes(forecastCoreIdx)) {
    return { class: "within-envelope", capturedCores: captured };
  }
  if (captured.length > 0) {
    return { class: "prediction-violation", capturedCores: captured };
  }
  return { class: "unknown-change", capturedCores: captured };
}
