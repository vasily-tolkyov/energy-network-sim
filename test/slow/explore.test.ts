import { test } from "node:test";
import assert from "node:assert/strict";
import { PopChannelMap } from "../../src/pop/popmap.js";
import { ExperimentPlanner } from "../../src/pop/explore/planner.js";
import { Explorer } from "../../src/pop/explore/explorer.js";
import {
  LAB_CONDITION_SPECS,
  LAB_OUTCOME_SPECS,
  LabBench,
  labTruth,
} from "../../src/topics/lab-world.js";

// 自主探索（M1/M2/M3 阶段 A）锁定测试

test("实验台：应答即真值、非法条件拒绝、成本计数", () => {
  const bench = new LabBench();
  const c = { switch: 1, voltage: 2, resistance: 2, temperature: 1, material: 0 };
  assert.deepEqual(bench.conduct(c), labTruth(c));
  assert.equal(bench.experimentsUsed, 1);
  assert.throws(() => bench.conduct({ switch: 1, voltage: 9, resistance: 2, temperature: 1, material: 0 }));
  assert.equal(bench.experimentsUsed, 1, "非法实验不计成本");
});

test("规划器：候选干预全部 Hamming-1（方法先验结构），配对自动且恰好一维", () => {
  const specs = [
    { name: "a", bins: 2 },
    { name: "b", bins: 3 },
  ];
  const p = new ExperimentPlanner(specs);
  p.register({ conditions: { a: 0, b: 0 }, outcomes: {}, classification: "unknown-change" });
  const cands = p.candidates();
  // 基点的 Hamming-1 前沿：a=1 与 b=1,b=2，共 3 个，不含对角（a1,b1）
  assert.equal(cands.length, 3);
  assert.ok(cands.every((c) => (c.a !== 0 ? 1 : 0) + (c.b !== 0 ? 1 : 0) === 1));
  const { pairs } = p.register({ conditions: { a: 1, b: 0 }, outcomes: {}, classification: "unknown-change" });
  assert.equal(pairs.length, 1, "只与基点成对（恰好一维）");
  assert.deepEqual(pairs[0]!.e0.conditions, { a: 0, b: 0 });
  // 重复条件不再是候选
  assert.ok(!p.candidates().some((c) => c.a === 1 && c.b === 0));
});

test("探索闭环（预算 40 步）：自主成对喂 R2，干扰维不被判为影响因素", { timeout: 300_000 }, () => {
  const cm = new PopChannelMap(LAB_CONDITION_SPECS.map((s) => ({ ...s })), 4);
  const om = new PopChannelMap(LAB_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
  const bench = new LabBench();
  const planner = new ExperimentPlanner(LAB_CONDITION_SPECS.map((s) => ({ ...s })));
  const ex = new Explorer(cm, om, planner, cm.specs, bench, { lit: 1, brightness: 3 }, { budget: 40 }, 1);
  while (ex.step()) {}
  // 修复后的诚实契约：允许提前终止，但提前终止必须是"验证达标"而非卡死/耗尽
  assert.ok(ex.log.length <= 40, "不得超过预算");
  if (ex.log.length < 40) {
    assert.equal(ex.terminationReason, "quorum-met", `提前终止原因应为 quorum-met，实为 ${ex.terminationReason}`);
  }
  assert.ok(ex.mem.ruleCount >= 20, `核数 ${ex.mem.ruleCount} 应随实验增长`);
  const found = ex.influentialDims;
  assert.ok(!found.includes("material"), "干扰维 material 不应被判为影响因素");
  assert.ok(found.includes("voltage"), `voltage 应被发现，实得 [${found}]`);
});

test("探索终止与门控自发现（种子 1，允许前沿或预算终止）", { timeout: 600_000 }, () => {
  const cm = new PopChannelMap(LAB_CONDITION_SPECS.map((s) => ({ ...s })), 4);
  const om = new PopChannelMap(LAB_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
  const bench = new LabBench();
  const planner = new ExperimentPlanner(LAB_CONDITION_SPECS.map((s) => ({ ...s })));
  const ex = new Explorer(cm, om, planner, cm.specs, bench, { lit: 1, brightness: 3 }, {}, 1);
  while (ex.step()) {}
  assert.ok(ex.log.length <= 300, `不得超过预算，实做 ${ex.log.length} 次`);
  assert.ok(["quorum-met", "budget-exhausted", "frontier-exhausted"].includes(ex.terminationReason!), "終止原因必须如实报告");
  if (ex.terminationReason === "quorum-met") assert.ok(ex.log.slice(-8).every(s => s.converged && s.classification === "within-envelope"));
  // 门控关闭侧（switch=0）探针：粗粒度亮灭判定应基本正确（否决承载门控）
  let gateOk = 0;
  let gateTotal = 0;
  for (let voltage = 0; voltage < 4; voltage++) {
    for (let resistance = 0; resistance < 4; resistance++) {
      const q = { switch: 0, voltage, resistance, temperature: 1, material: 1 };
      const { decoded } = ex.mem.predict(q, 1);
      gateTotal++;
      if (decoded.lit === 0) gateOk++;
    }
  }
  assert.ok(gateOk / gateTotal >= 0.75, `门控关闭侧粗粒度 ${gateOk}/${gateTotal} 过低`);
});
