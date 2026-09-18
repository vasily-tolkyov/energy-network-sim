// 仪表化连续探索（种子 1，预算 300）：逐步分解 predict 耗时分布
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
  { budget: 300 },
  1,
);

let stepNo = 0;
const origPredict = ex.mem.predict.bind(ex.mem);
let predictMs = 0;
let predictMax = 0;
ex.mem.predict = (q, s, a) => {
  const t0 = performance.now();
  const r = origPredict(q, s, a);
  const ms = performance.now() - t0;
  predictMs += ms;
  predictMax = Math.max(predictMax, ms);
  return r;
};

const t0 = performance.now();
while (ex.step()) {
  stepNo++;
  if (stepNo % 10 === 0) {
    const s = ex.log[ex.log.length - 1];
    console.log(
      `步 ${stepNo} [${s.phase}] 候选 ${s.candidateCount} 核 ${s.rulesFormed} | predict 均时 ${(predictMs / 10).toFixed(0)}ms 峰值 ${predictMax.toFixed(0)}ms | 累计 ${((performance.now() - t0) / 1000).toFixed(0)}s`,
    );
    predictMs = 0;
    predictMax = 0;
  }
}
console.log(
  `终止于 ${ex.log.length} 步（相 ${ex.currentPhase}），总耗时 ${((performance.now() - t0) / 1000).toFixed(0)}s，因素=[${ex.influentialDims}]`,
);
