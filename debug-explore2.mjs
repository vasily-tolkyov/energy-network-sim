// 仪表化种子 2：逐步分解各阶段耗时，找慢点
import { PopChannelMap } from "./dist/src/pop/popmap.js";
import { ExperimentPlanner } from "./dist/src/pop/explore/planner.js";
import { Explorer } from "./dist/src/pop/explore/explorer.js";
import { LAB_CONDITION_SPECS, LAB_OUTCOME_SPECS, LabBench } from "./dist/src/topics/lab-world.js";

const cm = new PopChannelMap(LAB_CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(LAB_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const bench = new LabBench();
const planner = new ExperimentPlanner(LAB_CONDITION_SPECS.map((s) => ({ ...s })));
const ex = new Explorer(cm, om, planner, cm.specs, bench, { lit: 1, brightness: 3 }, { budget: 300 }, 2);

let stepNo = 0;
const origPredict = ex.mem.predict.bind(ex.mem);
let predictMs = 0;
let predictCalls = 0;
ex.mem.predict = (q, s, a) => {
  const t0 = performance.now();
  const r = origPredict(q, s, a);
  predictMs += performance.now() - t0;
  predictCalls++;
  return r;
};

const t0 = performance.now();
while (ex.step()) {
  stepNo++;
  if (stepNo % 25 === 0) {
    const s = ex.log[ex.log.length - 1];
    console.log(
      `步 ${stepNo} 候选 ${s.candidateCount} 核 ${s.rulesFormed} | 本批 predict 均时 ${(predictMs / Math.max(1, predictCalls)).toFixed(0)}ms (${predictCalls} 次) | 累计 ${((performance.now() - t0) / 1000).toFixed(0)}s`,
    );
    predictMs = 0;
    predictCalls = 0;
  }
}
console.log(`终止于 ${ex.log.length} 步，总耗时 ${((performance.now() - t0) / 1000).toFixed(0)}s，因素=[${ex.influentialDims}]`);
