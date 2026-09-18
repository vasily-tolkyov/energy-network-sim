// 烟测：25 步预算的探索闭环，验证回路运转（候选/择选/写入/成对/R2/否决）
import { PopChannelMap } from "./dist/src/pop/popmap.js";
import { ExperimentPlanner } from "./dist/src/pop/explore/planner.js";
import { Explorer } from "./dist/src/pop/explore/explorer.js";
import { LAB_CONDITION_SPECS, LAB_OUTCOME_SPECS, LabBench } from "./dist/src/topics/lab-world.js";

const cm = new PopChannelMap(LAB_CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(LAB_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const bench = new LabBench();
const planner = new ExperimentPlanner(LAB_CONDITION_SPECS.map((s) => ({ ...s })));
const ex = new Explorer(cm, om, planner, cm.specs, bench, { lit: 1, brightness: 3 }, { budget: 25 }, 1);
while (ex.step()) {}
for (const s of ex.log) {
  const cond = LAB_CONDITION_SPECS.map((sp) => s.conditions[sp.name]).join(",");
  console.log(
    `#${String(s.index).padStart(2)} [${cond}] ${JSON.stringify(s.observed)} ${s.classification} 驱动${s.drive.toFixed(1)} 候选${s.candidateCount} 核${s.rulesFormed} 因素=[${s.influentialSoFar}]`,
  );
}
console.log(`实验台计数=${bench.experimentsUsed}`);
