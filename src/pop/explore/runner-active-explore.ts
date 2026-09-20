import { mkdirSync, writeFileSync } from "node:fs";
import { mulberry32 } from "../../prng.js";
import { ExperimentPlanner } from "./planner.js";
import { ContinuousExplorer } from "./explorer-continuous.js";
import { predictionQuality } from "../prediction-quality.js";
import type { Conditions, Outcomes } from "../../prototype/world.js";
import {
  THRESHOLD_COND_DIMS, THRESHOLD_OUTCOME_DIMS, THRESHOLD_GRID,
  ThresholdBench, thresholdKey, thresholdProbes, thresholdScore,
} from "../../topics/threshold-world.js";
import {
  LAB_CONT_CONDITION_DIMS, LAB_CONT_OUTCOME_DIMS, LAB_CONT_GRID,
  LabContBench, labContKey, labContProbes, lcScore,
} from "../../topics/lab-continuous-world.js";

/**
 * PLAN-010 主动探索验收：
 * - U1 稀有合取世界（唯一阳性藏在 128 组合中）：uncertainty / balanced / 随机三策略同预算对照；
 * - U2 电路回归对照（同预算 uncertainty vs balanced，禁止劣化掩盖）；
 * - U3/U4 陌生场景（多阈值耦合世界）：三种子，因素识别 + 网格精度 + 预算效率；
 * 逐种子落盘 runs/active-explore-*.json/log。真值只进实验台与评分侧。
 */

const lines: string[] = [];
const out = (s = "") => { console.log(s); lines.push(s); };
const seeds = (process.env.SEEDS ?? "1,2,3").split(",").map(s => parseInt(s.trim(), 10));

/* ---------- U1 稀有合取世界 ---------- */
const RARE_COND = [
  { name: "rareA", min: 0, max: 3 }, { name: "rareB", min: 0, max: 3 },
  { name: "rareC", min: 0, max: 3 }, { name: "rareD", min: 0, max: 1 },
];
const RARE_SPECS = [
  { name: "rareA", bins: 4, values: [0, 1, 2, 3] }, { name: "rareB", bins: 4, values: [0, 1, 2, 3] },
  { name: "rareC", bins: 4, values: [0, 1, 2, 3] }, { name: "rareD", bins: 2, values: [0, 1] },
];
const rareTruth = (c: Conditions): Outcomes => ({ y: c.rareA === 0 && c.rareB === 0 && c.rareC === 0 ? 1 : 0 });
class RareBench {
  conduct(c: Conditions): Outcomes { return rareTruth(c); }
}
function rareRun(policy: "uncertainty" | "balanced", budget: number, seed: number) {
  const planner = new ExperimentPlanner(RARE_SPECS);
  const explorer = new ContinuousExplorer(RARE_COND, [{ name: "y", min: 0, max: 1 }], planner, RARE_SPECS,
    new RareBench(), { y: 1 }, { budget, policy }, seed);
  while (explorer.step()) {}
  const positiveObserved = explorer.log.filter(s => s.observed.y === 1).length;
  // 读出阳性点（不追加学习）
  const read = explorer.mem.predict({ rareA: 0, rareB: 0, rareC: 0, rareD: 0 }, seed).values.y ?? null;
  return { policy, seed, experiments: explorer.log.length, terminationReason: explorer.terminationReason,
    positiveObserved, positiveRead: read, factors: explorer.influentialDims, coverage: explorer.coverageReport() };
}
function rareRandom(budget: number, seed: number) {
  const rng = mulberry32(seed);
  const all: Conditions[] = [];
  for (const a of [0, 1, 2, 3]) for (const b of [0, 1, 2, 3]) for (const c of [0, 1, 2, 3]) for (const d of [0, 1])
    all.push({ rareA: a, rareB: b, rareC: c, rareD: d });
  for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [all[i], all[j]] = [all[j]!, all[i]!]; }
  const picked = all.slice(0, budget);
  return { seed, experiments: picked.length, positiveObserved: picked.filter(c => rareTruth(c).y === 1).length };
}

/* ---------- U3/U4 多阈值耦合世界（陌生场景） ---------- */
const TH_SPECS = THRESHOLD_COND_DIMS.map(d => ({ name: d.name, bins: THRESHOLD_GRID[d.name]!.length, values: THRESHOLD_GRID[d.name]! }));
function thresholdRun(policy: "uncertainty" | "balanced", budget: number, seed: number) {
  const planner = new ExperimentPlanner(TH_SPECS);
  const bench = new ThresholdBench();
  const explorer = new ContinuousExplorer(THRESHOLD_COND_DIMS, THRESHOLD_OUTCOME_DIMS, planner, TH_SPECS,
    bench, { active: 1, flow: 3 }, { budget, policy }, seed);
  while (explorer.step()) {}
  const seen = new Set(explorer.log.map(s => thresholdKey(s.conditions)));
  const probes = thresholdProbes(seen);
  let coarse = 0, fine = 0, uCoarse = 0, uFine = 0, uN = 0;
  for (const p of probes) {
    const pred = explorer.mem.predict(p.conditions, seed);
    const s = thresholdScore(pred.values, p.truth);
    if (s.coarse) coarse++;
    if (s.fine) fine++;
    if (!p.seen) { uN++; if (s.coarse) uCoarse++; if (s.fine) uFine++; }
  }
  return { policy, seed, experiments: explorer.log.length, terminationReason: explorer.terminationReason,
    factors: explorer.influentialDims, coverage: explorer.coverageReport(),
    grid: { n: probes.length, coarse, fine }, unseen: { n: uN, coarse: uCoarse, fine: uFine },
    quality: predictionQuality.snapshot() };
}

/* ---------- U2 电路回归对照（同预算两策略） ---------- */
const LC_SPECS = LAB_CONT_CONDITION_DIMS.map(d => ({ name: d.name, bins: LAB_CONT_GRID[d.name]!.length, values: LAB_CONT_GRID[d.name]! }));
function circuitRun(policy: "uncertainty" | "balanced", budget: number, seed: number) {
  const planner = new ExperimentPlanner(LC_SPECS);
  const explorer = new ContinuousExplorer(LAB_CONT_CONDITION_DIMS, LAB_CONT_OUTCOME_DIMS, planner, LC_SPECS,
    new LabContBench(LAB_CONT_CONDITION_DIMS), { lit: 1, brightness: 3 }, { budget, policy }, seed);
  while (explorer.step()) {}
  const seen = new Set(explorer.log.map(s => labContKey(s.conditions)));
  const probes = labContProbes(seen);
  let coarse = 0;
  for (const p of probes) { if (lcScore(explorer.mem.predict(p.conditions, seed).values, p.truth).coarse) coarse++; }
  return { policy, seed, experiments: explorer.log.length, terminationReason: explorer.terminationReason,
    factors: explorer.influentialDims, coverage: explorer.coverageReport(), gridCoarse: coarse, gridN: probes.length };
}

/* ---------- 主流程 ---------- */
mkdirSync("runs", { recursive: true });
const report: Record<string, unknown> = { schema: "PLAN-010/v1", seeds, parts: {} };

out("══ U1 稀有合取世界（y=1 当且仅当 a=b=c=0；预算 32；128 组合） ══");
const u1 = { uncertainty: [] as unknown[], balanced: [] as unknown[], random: [] as unknown[] };
for (const seed of seeds) {
  u1.uncertainty.push(rareRun("uncertainty", 32, seed));
  u1.balanced.push(rareRun("balanced", 32, seed));
  u1.random.push(rareRandom(32, seed));
  const u = u1.uncertainty.at(-1)! as ReturnType<typeof rareRun>;
  const b = u1.balanced.at(-1)! as ReturnType<typeof rareRun>;
  const r = u1.random.at(-1)! as ReturnType<typeof rareRandom>;
  out(`seed ${seed}: uncertainty 阳性观察 ${u.positiveObserved} 次/读出 ${u.positiveRead}（${u.terminationReason}，未覆盖 ${u.coverage.unvisited}）| `
    + `balanced ${b.positiveObserved} 次/读出 ${b.positiveRead}（${b.terminationReason}）| 随机 ${r.positiveObserved} 次`);
}
report.parts = { ...report.parts as object, U1: u1 };

out("══ U3/U4 陌生场景：多阈值耦合世界（预算 120；256 组合） ══");
const u3 = { uncertainty: [] as unknown[], balanced: [] as unknown[] };
for (const seed of seeds) {
  u3.uncertainty.push(thresholdRun("uncertainty", 120, seed));
  u3.balanced.push(thresholdRun("balanced", 120, seed));
  const u = u3.uncertainty.at(-1)! as ReturnType<typeof thresholdRun>;
  const b = u3.balanced.at(-1)! as ReturnType<typeof thresholdRun>;
  out(`seed ${seed}: uncertainty 实验 ${u.experiments}（${u.terminationReason}）因素 [${u.factors}] 网格粗 ${u.grid.coarse}/${u.grid.n} 未观察粗 ${u.unseen.coarse}/${u.unseen.n}`);
  out(`         balanced   实验 ${b.experiments}（${b.terminationReason}）因素 [${b.factors}] 网格粗 ${b.grid.coarse}/${b.grid.n} 未观察粗 ${b.unseen.coarse}/${b.unseen.n}`);
  predictionQuality.reset();
}
report.parts = { ...report.parts as object, U3: u3 };

out("══ U2 电路回归对照（同预算 150，seed 1） ══");
const u2 = { uncertainty: circuitRun("uncertainty", 150, 1), balanced: circuitRun("balanced", 150, 1) };
out(`uncertainty: 粗 ${u2.uncertainty.gridCoarse}/${u2.uncertainty.gridN}（${u2.uncertainty.terminationReason}）因素 [${u2.uncertainty.factors}]`);
out(`balanced:    粗 ${u2.balanced.gridCoarse}/${u2.balanced.gridN}（${u2.balanced.terminationReason}）因素 [${u2.balanced.factors}]`);
report.parts = { ...report.parts as object, U2: u2 };
predictionQuality.reset();

writeFileSync("runs/active-explore.json", JSON.stringify(report, null, 2) + "\n");
writeFileSync("runs/active-explore.log", lines.join("\n") + "\n");
out("\n日志已写入 runs/active-explore.log / .json");
