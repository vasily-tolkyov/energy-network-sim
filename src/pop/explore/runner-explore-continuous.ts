import { mkdirSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { ExperimentPlanner } from "./planner.js";
import { ContinuousExplorer } from "./explorer-continuous.js";
import {
  LAB_CONT_CONDITION_DIMS,
  LAB_CONT_OUTCOME_DIMS,
  LAB_CONT_GRID,
  LabContBench,
  labContProbes,
  labContKey,
  lcScore,
} from "../../topics/lab-continuous-world.js";

/**
 * 阶段 B 验收：连续电路场景，零规则起步，概念由自己的实验史结晶。
 * 与阶段 A 受控对照：同一因果结构，唯一变量是表征（给定档 vs 自形成概念）。
 * 运行：npm run build && SEEDS=1,2,3 node dist/src/pop/explore/runner-explore-continuous.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const SPECS = LAB_CONT_CONDITION_DIMS.map((d, i) => ({
  name: d.name,
  bins: LAB_CONT_GRID[d.name]!.length,
  values: LAB_CONT_GRID[d.name]!,
  index: i,
}));
const OUTCOME_SPAN: Record<string, number> = { lit: 1, brightness: 3 };
const TRUTH_INFLUENTIAL = ["switchPos", "voltage", "resistance", "temperature"];
const EXHAUSTIVE = 3 * 4 * 4 * 3 * 3; // 432：穷举基线

function runSeed(seed: number): void {
  out(`\n── 种子 ${seed} ──`);
  const bench = new LabContBench();
  const planner = new ExperimentPlanner(SPECS);
  const explorer = new ContinuousExplorer(
    LAB_CONT_CONDITION_DIMS,
    LAB_CONT_OUTCOME_DIMS,
    planner,
    SPECS,
    bench,
    OUTCOME_SPAN,
    {},
    seed,
  );

  const t0 = performance.now();
  while (explorer.step()) {
    /* 推进到终止（含 B0→间歇期→B2 全程） */
  }
  const minutes = ((performance.now() - t0) / 60000).toFixed(1);

  const log = explorer.log;
  const conducted = new Set(log.map((s) => labContKey(s.conditions)));
  const counts = { "within-envelope": 0, "prediction-violation": 0, "unknown-change": 0 };
  for (const s of log) counts[s.classification]++;
  const corpusSteps = log.filter((s) => s.phase === "corpus").length;

  out(
    `终止：实验 ${log.length} 次（语料相 ${corpusSteps} + 全机器相 ${log.length - corpusSteps}；穷举基线 ${EXHAUSTIVE}），` +
      `规则核 ${explorer.mem.ruleCount} 条，耗时 ${minutes} 分钟；判定：未知 ${counts["unknown-change"]} / 偏差 ${counts["prediction-violation"]} / 符合 ${counts["within-envelope"]}`,
  );

  // 概念形成报告（阶段 B 的核心验收点之一）
  if (explorer.formationReport) {
    const r = explorer.formationReport;
    out("  概念形成（由自己的实验史结晶）：");
    for (const d of [...LAB_CONT_CONDITION_DIMS, ...LAB_CONT_OUTCOME_DIMS]) {
      out(`    ${d.name.padEnd(11)} ${r.conceptsPerDim[d.name]} 个概念，中心=[${(r.centers[d.name] ?? []).join(", ")}]`);
    }
  }
  out(`  R2 发现的影响因素：[${explorer.influentialDims}]（真值：[${TRUTH_INFLUENTIAL}]，干扰 material 应缺席）`);

  // 轨迹摘要：前 15 步 + 相切换 + 全部偏差
  out("  轨迹（前 15 步 + 相切换 + 全部偏差）：");
  for (const s of log) {
    const isSwitch = s.index === corpusSteps;
    if (s.index < 15 || isSwitch || s.classification === "prediction-violation") {
      const cond = LAB_CONT_CONDITION_DIMS.map((d) => s.conditions[d.name]).join(",");
      out(
        `    #${String(s.index).padStart(3)} [${s.phase === "corpus" ? "语" : "全"}] [${cond}] 预测lit=${s.predictedLit ?? "∅"} → 观察 ${JSON.stringify(s.observed)}` +
          `  ${s.classification}（驱动 ${s.drive.toFixed(1)}，候选 ${s.candidateCount}，核 ${s.rulesFormed}）${isSwitch ? " ◄── 概念形成完成" : ""}`,
      );
    }
  }

  // 外部评分：网格全组合探针（不进学习）；每 96 个打一行进度。
  // 评审 F10 口径：报告拒答率与常数基线；细粒度口径见 lcScore（灭灯也校验亮度）。
  const probes = labContProbes(conducted);
  const perKind: Record<string, { fine: number; coarse: number; refused: number; total: number; truthLit: number }> =
    {};
  for (const [pi, p] of probes.entries()) {
    const pred = explorer.mem.predict(p.conditions, seed);
    const s = lcScore(pred.values, p.truth);
    const b = (perKind[p.kind] ??= { fine: 0, coarse: 0, refused: 0, total: 0, truthLit: 0 });
    b.total++;
    if (p.truth.lit === 1) b.truthLit++;
    if (pred.values.lit === null) b.refused++;
    if (s.coarse) b.coarse++;
    if (s.fine) b.fine++;
    if ((pi + 1) % 96 === 0) out(`    …评分进度 ${pi + 1}/${probes.length}`);
  }
  let fine = 0;
  let coarse = 0;
  for (const kind of ["self-taught", "unseen-gate-off", "unseen-gate-on"] as const) {
    const b = perKind[kind] ?? { fine: 0, coarse: 0, refused: 0, total: 0, truthLit: 0 };
    fine += b.fine;
    coarse += b.coarse;
    const constBase = Math.max(b.truthLit, b.total - b.truthLit) / Math.max(1, b.total);
    out(
      `    ${kind.padEnd(15)} 细粒度 ${b.fine}/${b.total}（${((b.fine / b.total) * 100).toFixed(1)}%）` +
        `  粗粒度亮灭 ${((b.coarse / b.total) * 100).toFixed(1)}%  拒答 ${((b.refused / b.total) * 100).toFixed(1)}%` +
        `  常数基线 ${(constBase * 100).toFixed(1)}%`,
    );
  }
  out(
    `    总计             细粒度 ${fine}/${probes.length}（${((fine / probes.length) * 100).toFixed(1)}%）` +
      `  粗粒度 ${((coarse / probes.length) * 100).toFixed(1)}%`,
  );
}

out("═".repeat(72));
out("阶段 B 验收：连续电路场景，零规则起步，概念自形成 + 自主控制变量探索");
out("═".repeat(72));
out(
  `编码：6 维（5 条件 + 2 结果中 lit/brightness 2 结果维）× 40 感受野；候选值 = 分位网格；` +
    `B0 语料相 → 概念形成间歇期 → B2 全机器相`,
);
out(`真值影响因素：[${TRUTH_INFLUENTIAL}]；干扰因素：material；门控：亮 = 开关度≥0.5 AND 电压≥1（合取，连续版）`);

const seeds = (process.env.SEEDS ?? "1,2,3").split(",").map((s) => parseInt(s.trim(), 10));
for (const seed of seeds) {
  const mark = lines.length;
  runSeed(seed);
  mkdirSync("runs", { recursive: true });
  writeFileSync(`runs/explore-cont-seed${seed}.log`, lines.slice(mark).join("\n") + "\n");
}

out(`\n日志已写入 runs/explore-cont-seed*.log`);
