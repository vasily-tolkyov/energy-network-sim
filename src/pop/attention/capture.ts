import { compatibleSupports } from "../evidence.js";
import type { FieldRuleMemory } from "../concept/field-memory.js";
import type { CValues } from "../concept/world-continuous.js";

/** Joint capture requires a captured core AND compatible observed output on
 * every observed dimension. A core fired by conditions alone is insufficient. */
export interface ForecastSnapshot {
  coreIdx: number | null;
  values: Readonly<Record<string, number | null>>;
  generation: number;
  converged: boolean;
  terminationReason: string;
}

export type CaptureClass = "within-envelope" | "prediction-violation" | "unknown-change";

/** 预测对应核的索引来源：predict 的获胜核（由调用方给出） */
export function captureClassify(
  mem: FieldRuleMemory,
  conditions: CValues,
  observedOutcomes: CValues,
  forecast: number | null | ForecastSnapshot,
  seed = 1,
): { class: CaptureClass; capturedCores: number[]; converged: boolean; terminationReason: string } {
  const forecastCoreIdx = typeof forecast === "object" && forecast !== null ? forecast.coreIdx : forecast;
  const snapshot = typeof forecast === "object" && forecast !== null ? forecast : null;
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
  const status = { converged: result.converged, terminationReason: result.terminationReason };
  // 无时间对应的预测时不能报"偏差"（文档三态原则）——一律报未知
  if (forecastCoreIdx === null) {
    return { class: "unknown-change", capturedCores: captured, ...status };
  }
  const evidence = mem.ruleEvidence(forecastCoreIdx);
  const compatible = Object.entries(observedOutcomes).every(([dim, value]) => {
    const old = snapshot ? snapshot.values[dim] : evidence[dim]?.value;
    return typeof old === "number" && compatibleSupports(mem.encoder.encodeDimension(dim, old), mem.encoder.encodeDimension(dim, value));
  });
  if (captured.includes(forecastCoreIdx) && compatible && result.converged && (snapshot?.converged ?? true)) {
    return { class: "within-envelope", capturedCores: captured, ...status };
  }
  if (captured.length > 0) {
    return { class: "prediction-violation", capturedCores: captured, ...status };
  }
  return { class: "unknown-change", capturedCores: captured, ...status };
}
