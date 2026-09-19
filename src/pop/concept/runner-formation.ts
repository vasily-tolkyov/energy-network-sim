import { predictionQuality } from "../prediction-quality.js";
import { mkdirSync, writeFileSync } from "node:fs";
import { SensoryEncoder } from "./sensory.js";
import { ConceptFormation } from "./formation.js";
import { EmergentMap, alignmentReport } from "./emergent-map.js";
import { EmergentChannelAdapter } from "./emergent-channel-adapter.js";
import { FieldRuleMemory } from "./field-memory.js";
import { R2PopLayer } from "../r2pop.js";
import {
  CONT_CONDITION_DIMS,
  CONT_OUTCOME_DIMS,
  contCurriculum,
  contQueries,
  contTruth,
  type CValues,
} from "./world-continuous.js";

/**
 * M4：概念自形成版端到端——连续课程 → 形成 → 场级 R3 → 连续查询评分。
 * 运行：npm run build && node dist/src/pop/concept/runner-formation.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};
const SEP = "═".repeat(64);

// —— 形成 ——
const enc = new SensoryEncoder([...CONT_CONDITION_DIMS, ...CONT_OUTCOME_DIMS], 40);
const formation = new ConceptFormation(enc);
for (const group of contCurriculum()) {
  for (const pair of group.pairs) {
    for (const e of [pair.e0, pair.e1]) {
      formation.presentExperiment({ ...e.conditions, ...e.outcomes }, 4);
    }
  }
}
const emergent = new EmergentMap(formation.extractConcepts(0.5), enc);

out(SEP);
out("概念自形成：各维度的自形成值概念（中心值 / 成员数 / 内强度）");
out(SEP);
for (const dim of enc.dimensions) {
  const cs = emergent["byDim"].get(dim.name) ?? [];
  out(
    `  ${dim.name.padEnd(14)} ${cs.length} 个：` +
      cs.map((c) => `${c.centerValue.toFixed(2)}(${c.memberNeuronIds.length})`).join(" "),
  );
}

out("");
out(SEP);
out("对齐报告（参照值 → 自形成概念，IoU）");
out(SEP);
const refs: Record<string, number[]> = {
  speed: [0.8, 1.5, 3.2, 6.5],
  wallStiffness: [0.4, 1.0, 2.6],
  hitWall: [0, 1],
};
for (const e of alignmentReport(emergent, refs)) {
  out(
    `  ${e.dimension}=${e.referenceValue} → 概念#${e.conceptIndex ?? "无"}` +
      `${e.conceptCenter === null ? "" : `（中心 ${e.conceptCenter.toFixed(2)}）`} IoU=${e.iou.toFixed(2)}`,
  );
}

// —— 教学（场级 R3）——
const mem = new FieldRuleMemory(enc, emergent, { maxRules: 192 });
mem.setOutcomeDimensions(CONT_OUTCOME_DIMS.map((d) => d.name));
const condAdapter = new EmergentChannelAdapter(emergent, CONT_CONDITION_DIMS.map((d) => d.name), 0);
const outAdapter = new EmergentChannelAdapter(
  emergent,
  CONT_OUTCOME_DIMS.map((d) => d.name),
  enc.dimensionOffset(CONT_OUTCOME_DIMS[0]!.name),
  "fields", // 结果侧场级编码：保留粗概念内部的细分差异
);
const r2 = new R2PopLayer(condAdapter, outAdapter);
const OUTCOME_SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 4 };
// 第一遍：教学 + 逐组 R2 差分，累计全局影响因素（并集）
const globalMag = new Map<string, { sum: number; count: number }>();
for (const group of contCurriculum()) {
  for (const pair of group.pairs) {
    for (const e of [pair.e0, pair.e1]) mem.teachExperiment(e.conditions, e.outcomes, 4);
    // 结果侧的替代值同样从换对学习互斥（写入规则网络，读出档位竞争由此而来）
    for (const dim of CONT_OUTCOME_DIMS) {
      const a = pair.e0.outcomes[dim.name]!;
      const b = pair.e1.outcomes[dim.name]!;
      if (Math.abs(a - b) > 1e-9) mem.learnExclusion(dim.name, a, b, 3.0);
    }
    // R2 神经差分：EmergentMap 上 resolve 出的概念帧对（构造 Pair 形状复用 R2Pop）
    const analysis = r2.analyzePair({
      e0: { conditions: binarize(emergent, pair.e0.conditions), outcomes: pair.e0.outcomes },
      e1: { conditions: binarize(emergent, pair.e1.conditions), outcomes: pair.e1.outcomes },
    });
    if (analysis.undecidable) continue; // 不可判定对不进幅度累计（评审 F04）
    for (const ch of analysis.influentialChannels) {
      let m = 0;
      for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
        m += Math.abs(delta) / (OUTCOME_SPAN[och] ?? 1);
      }
      const e0 = globalMag.get(ch) ?? { sum: 0, count: 0 };
      e0.sum += m / CONT_OUTCOME_DIMS.length;
      e0.count++;
      globalMag.set(ch, e0);
    }
  }
}
// 全局侧重并集：否决权属于影响因素，每个核都按同一并集布线
const globalBoost: Record<string, number> = {};
for (const [ch, { sum, count }] of globalMag) {
  globalBoost[ch] = 0.1 * 3 * (sum / Math.max(1, count)) ** 2;
}
out(`  全局影响因素=${Object.entries(globalBoost).map(([ch, d]) => `${ch}(${d.toFixed(3)})`).join(",") || "无"}`);
for (const group of contCurriculum()) {
  for (const pair of group.pairs) {
    for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e.conditions, globalBoost, 4);
  }
}

function binarize(map: EmergentMap, values: CValues): CValues {
  const out2: CValues = {};
  for (const [dim, v] of Object.entries(values)) {
    out2[dim] = map.resolve(dim, v) ?? -1;
  }
  return out2;
}

// —— 评分：连续值容差判定 ——
out("");
out(SEP);
out("预测评分（容差：反弹 exact；反弹速度 |Δ| ≤ 0.75）");
out(SEP);
const TOL = 0.75;
const perKind: Record<string, { ok: number; total: number; coarseOk: number; uncertain: number }> = {};
for (const q of contQueries()) {
  const p = mem.predict(q.conditions, 1);
  const rebOk =
    p.values.rebound !== null && Math.abs((p.values.rebound ?? -99) - q.truth.rebound!) <= 0.5;
  const fineOk =
    rebOk &&
    (q.truth.rebound === 0 ||
      (p.values.reboundSpeed !== null &&
        Math.abs((p.values.reboundSpeed ?? -99) - q.truth.reboundSpeed!) <= TOL));
  const bucket = (perKind[q.kind] ??= { ok: 0, total: 0, coarseOk: 0, uncertain: 0 });
  bucket.total++;
  if (rebOk) bucket.coarseOk++;
  if (fineOk) bucket.ok++;
  if (p.ambiguous.length > 0) bucket.uncertain++;
}
let okSum = 0;
let totalSum = 0;
let coarseSum = 0;
let uncertainSum = 0;
for (const [kind, b] of Object.entries(perKind).sort()) {
  okSum += b.ok;
  totalSum += b.total;
  coarseSum += b.coarseOk;
  uncertainSum += b.uncertain;
  out(
    `  ${kind.padEnd(13)} 细粒度 ${b.ok}/${b.total}（${((b.ok / b.total) * 100).toFixed(1)}%）` +
      `  粗粒度 ${((b.coarseOk / b.total) * 100).toFixed(1)}%  歧义 ${b.uncertain}`,
  );
}
out(`  总计：细粒度 ${okSum}/${totalSum}（${((okSum / totalSum) * 100).toFixed(1)}%），粗粒度 ${((coarseSum / totalSum) * 100).toFixed(1)}%，如实歧义 ${uncertainSum}/${totalSum}`);
out(`  规则核数 ${mem.ruleCount}，网络规模 ${mem.net.neuronCount}（感受野 ${enc.neuronCount} + 核区）`);

mkdirSync("runs", { recursive: true });
out(`动力学质量与任一输出拒答（本日志全部 predict 调用，含训练期；答案质量另列）：${JSON.stringify(predictionQuality.snapshot())}`);
writeFileSync("runs/concept-formation-v1.log", lines.join("\n") + "\n");
predictionQuality.reset();
out(`\n日志已写入 runs/concept-formation-v1.log`);
