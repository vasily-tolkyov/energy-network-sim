// 烟测：连续探索闭环 12 步（回路/写入/候选）
import { ExperimentPlanner } from "./dist/src/pop/explore/planner.js";
import { ContinuousExplorer } from "./dist/src/pop/explore/explorer-continuous.js";
import {
  LAB_CONT_CONDITION_DIMS,
  LAB_CONT_OUTCOME_DIMS,
  LAB_CONT_GRID,
  LabContBench,
} from "./dist/src/topics/lab-continuous-world.js";

const SPECS = LAB_CONT_CONDITION_DIMS.map((d) => ({
  name: d.name,
  bins: LAB_CONT_GRID[d.name].length,
  values: LAB_CONT_GRID[d.name],
}));
const bench = new LabContBench();
const planner = new ExperimentPlanner(SPECS);
const ex = new ContinuousExplorer(
  LAB_CONT_CONDITION_DIMS,
  LAB_CONT_OUTCOME_DIMS,
  planner,
  SPECS,
  bench,
  { lit: 1, brightness: 3 },
  { budget: 12 },
  1,
);
while (ex.step()) {}
for (const s of ex.log) {
  const cond = LAB_CONT_CONDITION_DIMS.map((d) => s.conditions[d.name]).join(",");
  console.log(
    `#${String(s.index).padStart(2)} [${cond}] obs=${JSON.stringify(s.observed)} ${s.classification} 驱动${s.drive.toFixed(1)} 候选${s.candidateCount} 核${s.rulesFormed}`,
  );
}
console.log(`实验台计数=${bench.experimentsUsed} 相=${ex.currentPhase}`);
