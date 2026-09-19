import { predictionQuality } from "../prediction-quality.js";
import { mkdirSync, writeFileSync } from "node:fs";
import { ExperimentPlanner } from "./planner.js";
import { ContinuousExplorer } from "./explorer-continuous.js";
import {
  LAB_CONT_CONDITION_DIMS_EXT,
  LAB_CONT_OUTCOME_DIMS,
  LAB_CONT_GRID_EXT_BASE,
  LAB_CONT_EXT_NEW_VALUES,
  LabContBench,
  labContProbes,
  labContKey,
  lcScore,
} from "../../topics/lab-continuous-world.js";

/**
 * 概念更新验收：量程扩展场景（世界在中途展示比假设更大的范围）。
 * 对照：conceptUpdate=true（覆盖缺口/周期触发重形成）vs false（冻结）。
 * 判据：更新系统的解释层保持完整（新区域结晶出概念、R2 在新对上持续归因），
 * 冻结系统的解释层停滞（新区域永远不是概念、R2 对它的通道失明）。
 * 运行：npm run build && node dist/src/pop/explore/runner-explore-ext.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const OUTCOME_SPAN: Record<string, number> = { lit: 1, brightness: 3 };
const BASE_GRID = LAB_CONT_GRID_EXT_BASE;
const FULL_GRID: Record<string, readonly number[]> = {
  ...LAB_CONT_GRID_EXT_BASE,
  voltage: [...LAB_CONT_GRID_EXT_BASE.voltage!, ...LAB_CONT_EXT_NEW_VALUES.voltage!],
};
const SPECS = LAB_CONT_CONDITION_DIMS_EXT.map((d) => ({
  name: d.name,
  bins: FULL_GRID[d.name]!.length,
  values: BASE_GRID[d.name]!,
}));

function runMode(mode: "updating" | "frozen", seed: number): void {
  out(`\n── 模式 ${mode === "updating" ? "概念更新" : "冻结（对照）"}，种子 ${seed} ──`);
  const bench = new LabContBench(LAB_CONT_CONDITION_DIMS_EXT);
  const planner = new ExperimentPlanner(SPECS);
  const explorer = new ContinuousExplorer(
    LAB_CONT_CONDITION_DIMS_EXT,
    LAB_CONT_OUTCOME_DIMS,
    planner,
    SPECS,
    bench,
    OUTCOME_SPAN,
    { budget: 400, conceptUpdate: mode === "updating" },
    seed,
  );

  // Equal budget per exposure stage, independent of score/convergence.
  const phaseBudget = 400 / 2;
  while (explorer.log.length < phaseBudget && explorer.step()) {}
  const phase1Experiments = explorer.log.length;
  const voltageEvidencePhase1 = explorer.magEvidence().voltage ?? 0;
  out(
    `一阶段终止：实验 ${phase1Experiments} 次，形成 ${explorer.formationHistory.length} 次，` +
      `voltage 归因证据 ${voltageEvidencePhase1} 对，因素=[${explorer.influentialDims}]`,
  );

  // 量程扩展：世界展示比假设更大的范围
  planner.extendValues("voltage", LAB_CONT_EXT_NEW_VALUES.voltage!);
  out(`  ◄── 量程扩展：voltage 新候选值 [${LAB_CONT_EXT_NEW_VALUES.voltage}]`);
  while (explorer.step()) {}
  const phase2Experiments = explorer.log.length - phase1Experiments;
  const voltageEvidencePhase2 = explorer.magEvidence().voltage ?? 0;
  out(
    `二阶段终止：实验 +${phase2Experiments} 次（共 ${explorer.log.length}），形成 ${explorer.formationHistory.length} 次，` +
      `voltage 归因证据 ${voltageEvidencePhase1} → ${voltageEvidencePhase2} 对`,
  );

  // 概念演化轨迹
  out("  概念形成历史（voltage 概念中心）：");
  explorer.formationHistory.forEach((f, i) => {
    out(`    第 ${i + 1} 次形成（实验后 #${i === 0 ? phase1Experiments : "更新"}）：[${(f.centers.voltage ?? []).join(", ")}]`);
  });
  out(`  最终影响因素：[${explorer.influentialDims}]`);

  // 扩展区探针（voltage ∈ {3.9, 4.5} 的全部组合）
  const probes = labContProbes(new Set(explorer.log.map((s) => labContKey(s.conditions))), FULL_GRID).filter(
    (p) => p.kind === "extended",
  );
  let fine = 0;
  let coarse = 0;
  for (const p of probes) {
    const pred = explorer.mem.predict(p.conditions, seed);
    const s = lcScore(pred.values, p.truth);
    if (s.coarse) coarse++;
    if (s.fine) fine++;
  }
  out(
    `  扩展区探针（${probes.length} 个）：细粒度 ${fine}（${((fine / probes.length) * 100).toFixed(1)}%）` +
      `  粗粒度 ${coarse}（${((coarse / probes.length) * 100).toFixed(1)}%）`,
  );

  // 全网格摘要（参考）
  const all = labContProbes(new Set(explorer.log.map((s) => labContKey(s.conditions))), FULL_GRID);
  let af = 0;
  let ac = 0;
  for (const p of all) {
    const pred = explorer.mem.predict(p.conditions, seed);
    const s = lcScore(pred.values, p.truth);
    if (s.coarse) ac++;
    if (s.fine) af++;
  }
  out(
    `  全网格（${all.length} 个）：细粒度 ${((af / all.length) * 100).toFixed(1)}%  粗粒度 ${((ac / all.length) * 100).toFixed(1)}%`,
  );
}

out("═".repeat(72));
out("概念更新验收：量程扩展场景（世界中途变大），冻结 vs 更新 对照");
out("═".repeat(72));
out(
  `设定：电压量程 [0,4.5] 从起始对编码器可见（感受野覆盖），但候选网格初始只到 3.0；` +
    `一阶段最多 200 次实验后扩展候选值 [${LAB_CONT_EXT_NEW_VALUES.voltage}]。`,
);
out("判据：更新系统应在扩展区结晶新概念并继续归因；冻结系统的概念层停滞、R2 对新对的电压通道失明。");

const seeds = (process.env.SEEDS ?? "1").split(",").map((s) => parseInt(s.trim(), 10));
for (const seed of seeds) {
  for (const mode of ["updating", "frozen"] as const) {
    const mark = lines.length;
    runMode(mode, seed);
    mkdirSync("runs", { recursive: true });
    out(`动力学质量与任一输出拒答（本日志全部 predict 调用，含训练期；答案质量另列）：${JSON.stringify(predictionQuality.snapshot())}`);
    writeFileSync(`runs/explore-ext-${mode}-seed${seed}.log`, lines.slice(mark).join("\n") + "\n");
    predictionQuality.reset();
  }
}

out(`\n日志已写入 runs/explore-ext-*.log`);
