import { test } from "node:test";
import assert from "node:assert/strict";
import { ExperimentPlanner } from "../src/pop/explore/planner.js";
import { ContinuousExplorer } from "../src/pop/explore/explorer-continuous.js";
import {
  LAB_CONT_CONDITION_DIMS,
  LAB_CONT_OUTCOME_DIMS,
  LAB_CONT_GRID,
  LabContBench,
  labContTruth,
  lcScore,
} from "../src/topics/lab-continuous-world.js";

// 阶段 B（连续自主探索）锁定测试

const SPECS = LAB_CONT_CONDITION_DIMS.map((d) => ({
  name: d.name,
  bins: LAB_CONT_GRID[d.name]!.length,
  values: LAB_CONT_GRID[d.name]!,
}));

test("连续实验台：应答即真值、非法条件拒绝、成本计数", () => {
  const bench = new LabContBench();
  const c = { switchPos: 0.9, voltage: 2.1, resistance: 1.2, temperature: 1.0, material: 0.3 };
  assert.deepEqual(bench.conduct(c), labContTruth(c));
  assert.equal(bench.experimentsUsed, 1);
  assert.throws(() => bench.conduct({ ...c, voltage: 3.5 }));
  assert.equal(bench.experimentsUsed, 1);
});

test("容差评分：lit 阈值化、亮度 0.5 容差", () => {
  assert.deepEqual(lcScore({ lit: 0.95, brightness: 2.4 }, { lit: 1, brightness: 2.55 }), {
    coarse: true,
    fine: true,
  });
  assert.deepEqual(lcScore({ lit: 0.95, brightness: 1.0 }, { lit: 1, brightness: 2.55 }), {
    coarse: true,
    fine: false,
  });
  assert.deepEqual(lcScore({ lit: 0.95, brightness: 0 }, { lit: 0, brightness: 0 }), {
    coarse: false,
    fine: false,
  });
  assert.deepEqual(lcScore({ lit: null, brightness: null }, { lit: 0, brightness: 0 }), {
    coarse: false,
    fine: false,
  });
});

test("连续探索端到端（种子 1）：语料充分性门 → 概念形成 → 因素发现", { timeout: 600_000 }, () => {
  const bench = new LabContBench();
  const planner = new ExperimentPlanner(SPECS);
  const ex = new ContinuousExplorer(
    LAB_CONT_CONDITION_DIMS,
    LAB_CONT_OUTCOME_DIMS,
    planner,
    SPECS,
    bench,
    { lit: 1, brightness: 3 },
    {},
    1,
  );
  while (ex.step()) {}
  assert.ok(ex.log.length < 300, `应自然终止而非耗尽预算，实做 ${ex.log.length} 次`);
  // 种子 2 病理回归：概念形成必须在语料充分（每维 ≥2 值）之后——
  // switchPos 必须形成多概念且被判为影响因素（门控维度不可缺席）
  assert.ok(ex.formationReport !== null, "应经过概念形成间歇期");
  assert.ok(
    (ex.formationReport!.conceptsPerDim.switchPos ?? 0) >= 2,
    `switchPos 概念数 ${ex.formationReport!.conceptsPerDim.switchPos} < 2（语料不充分即形成）`,
  );
  assert.ok(ex.influentialDims.includes("switchPos"), `switchPos 应被发现，实得 [${ex.influentialDims}]`);
  assert.ok(!ex.influentialDims.includes("material"), "干扰维 material 不应被判为影响因素");
  // 门控关闭侧探针：粗粒度亮灭应基本正确（实测种子间 97.6%-100%，留诚实余量）
  let gateOk = 0;
  let gateTotal = 0;
  for (const voltage of LAB_CONT_GRID.voltage!) {
    for (const resistance of LAB_CONT_GRID.resistance!) {
      const q = { switchPos: 0.1, voltage, resistance, temperature: 1.0, material: 1.0 };
      const pred = ex.mem.predict(q, 1);
      gateTotal++;
      if ((pred.values.lit ?? 1) < 0.5) gateOk++;
    }
  }
  assert.ok(gateOk / gateTotal >= 0.75, `门控关闭侧粗粒度 ${gateOk}/${gateTotal} 过低`);
});
