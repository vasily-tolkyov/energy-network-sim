import { mkdirSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { PopChannelMap } from "../popmap.js";
import { R2PopLayer } from "../r2pop.js";
import { ExperimentPlanner } from "./planner.js";
import { Explorer } from "./explorer.js";
import {
  LAB_CONDITION_SPECS,
  LAB_OUTCOME_SPECS,
  LabBench,
  labProbes,
  labKey,
} from "../../topics/lab-world.js";
import type { Conditions, Outcomes } from "../../prototype/world.js";

/**
 * M3 阶段 A 验收：完全陌生场景（电路世界）+ 零规则起步 + 自主探索。
 * 无教师课程、无人工选择学什么——只给实验台和预算。
 * 运行：npm run build && node dist/src/pop/explore/runner-explore.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const cm = new PopChannelMap(LAB_CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(LAB_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const OUTCOME_SPAN: Record<string, number> = { lit: 1, brightness: 3 };
const TRUTH_INFLUENTIAL = ["switch", "voltage", "resistance", "temperature"];
const EXHAUSTIVE = 2 * 4 * 4 * 3 * 3; // 288：穷举基线

function score(
  decoded: Record<string, number | null | "ambiguous">,
  truth: Outcomes,
): "correct" | "wrong" | "ambiguous" {
  if (decoded.lit === "ambiguous" || decoded.brightness === "ambiguous") return "ambiguous";
  if (decoded.lit === null) return "wrong";
  if (truth.lit === 0) return decoded.lit === 0 ? "correct" : "wrong";
  if (decoded.lit !== 1) return "wrong";
  return decoded.brightness === truth.brightness ? "correct" : "wrong";
}

function runSeed(seed: number): void {
  out(`\n── 种子 ${seed} ──`);
  const bench = new LabBench();
  const planner = new ExperimentPlanner(LAB_CONDITION_SPECS.map((s) => ({ ...s })));
  const explorer = new Explorer(cm, om, planner, cm.specs, bench, OUTCOME_SPAN, {}, seed);

  const t0 = performance.now();
  while (explorer.step()) {
    /* 推进到终止 */
  }
  const minutes = ((performance.now() - t0) / 60000).toFixed(1);

  const log = explorer.log;
  const conducted = new Set(log.map((s) => labKey(s.conditions)));
  const counts = { "within-envelope": 0, "prediction-violation": 0, "unknown-change": 0 };
  for (const s of log) counts[s.classification]++;

  out(
    `终止：实验 ${log.length} 次（穷举基线 ${EXHAUSTIVE}），规则核 ${explorer.mem.ruleCount} 条，` +
      `耗时 ${minutes} 分钟；判定：未知 ${counts["unknown-change"]} / 偏差 ${counts["prediction-violation"]} / 符合 ${counts["within-envelope"]}`,
  );
  out(
    `R2 发现的影响因素：[${explorer.influentialDims}]（真值：[${TRUTH_INFLUENTIAL}]，干扰 material 应缺席）`,
  );

  // 轨迹摘要：前 20 步 + 全部偏差事件
  out("  轨迹（前 20 步 + 全部偏差）：");
  for (const s of log) {
    if (s.index < 20 || s.classification === "prediction-violation") {
      const cond = LAB_CONDITION_SPECS.map((sp) => s.conditions[sp.name]).join(",");
      out(
        `    #${String(s.index).padStart(3)} [${cond}] 预测 ${JSON.stringify(s.predicted)} → 观察 ${JSON.stringify(s.observed)}` +
          `  ${s.classification}（驱动 ${s.drive.toFixed(1)}，候选 ${s.candidateCount}，核 ${s.rulesFormed}）`,
      );
    }
  }

  // 外部评分：全网格探针（不进学习）；每 48 个打一行进度（慢阶段可见）。
  // 评审 F10/E01 口径修复：unseen 桶 = 全部未观察组合（再分子集 gate-off/gate-on），
  // 报告常数基线对照、拒答率；细粒度改严格联合正确（灭灯也校验亮度为 0）。
  const probes = labProbes(conducted);
  const perKind: Record<string, { fine: number; strict: number; coarse: number; refused: number; total: number; truthLit: number }> =
    {};
  for (const [pi, p] of probes.entries()) {
    const { decoded } = explorer.mem.predict(p.conditions, seed);
    const verdict = score(decoded, p.truth);
    const b = (perKind[p.kind] ??= { fine: 0, strict: 0, coarse: 0, refused: 0, total: 0, truthLit: 0 });
    b.total++;
    if (p.truth.lit === 1) b.truthLit++;
    if (decoded.lit === null || decoded.lit === "ambiguous") b.refused++;
    if (decoded.lit === p.truth.lit) b.coarse++;
    if (verdict === "correct") b.fine++;
    // 严格联合：亮灭正确 且 亮度档精确（灭灯时亮度必须为 0，不许为空/错档）
    const strictOk =
      decoded.lit === p.truth.lit &&
      (p.truth.lit === 0 ? decoded.brightness === 0 : decoded.brightness === p.truth.brightness);
    if (strictOk) b.strict++;
    if ((pi + 1) % 48 === 0) out(`    …评分进度 ${pi + 1}/${probes.length}`);
  }
  let fine = 0;
  let coarse = 0;
  let strict = 0;
  for (const kind of ["self-taught", "unseen-gate-off", "unseen-gate-on"] as const) {
    const b = perKind[kind] ?? { fine: 0, strict: 0, coarse: 0, refused: 0, total: 0, truthLit: 0 };
    fine += b.fine;
    coarse += b.coarse;
    strict += b.strict;
    const constBase = Math.max(b.truthLit, b.total - b.truthLit) / Math.max(1, b.total);
    out(
      `    ${kind.padEnd(15)} 门控任务分 ${((b.fine / b.total) * 100).toFixed(1)}%  严格联合 ${((b.strict / b.total) * 100).toFixed(1)}%` +
        `  粗粒度亮灭 ${((b.coarse / b.total) * 100).toFixed(1)}%  拒答 ${((b.refused / b.total) * 100).toFixed(1)}%` +
        `  常数基线 ${(constBase * 100).toFixed(1)}%`,
    );
  }
  out(
    `    总计             门控任务分 ${((fine / probes.length) * 100).toFixed(1)}%  严格联合 ${((strict / probes.length) * 100).toFixed(1)}%` +
      `  粗粒度 ${((coarse / probes.length) * 100).toFixed(1)}%`,
  );
}

out("═".repeat(72));
out("M3 阶段 A 验收：陌生电路场景，零规则起步，自主控制变量探索");
out("═".repeat(72));
out(
  `网络：条件 ${cm.neuronCount} + 结果 ${om.neuronCount} + 核区 384×4 + 池 2 = ${
    cm.neuronCount + om.neuronCount + 384 * 4 + 2
  } 神经元（核按需分配）；动作空间 = Hamming-1 前沿（方法先验躯体化）；择选 = NeuralFocusNet WTA`,
);
out(`真值影响因素：[${TRUTH_INFLUENTIAL}]；干扰因素：material；门控：亮 = 闭合 AND 有压（合取，新形态）`);

// 逐种子可独立运行（SEEDS 环境变量，默认 1,2,3）：长任务分段可恢复、可检视
const seeds = (process.env.SEEDS ?? "1,2,3").split(",").map((s) => parseInt(s.trim(), 10));
for (const seed of seeds) {
  const mark = lines.length;
  runSeed(seed);
  // 逐种子独立落盘（explore-lab-seedN.log）：长任务分段可恢复、可检视、不互相覆盖
  mkdirSync("runs", { recursive: true });
  writeFileSync(`runs/explore-lab-seed${seed}.log`, lines.slice(mark).join("\n") + "\n");
}

out(`\n日志已写入 runs/explore-lab-seed*.log`);

out(`\n日志已写入 runs/explore-lab-v1.log`);
