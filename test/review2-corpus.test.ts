import { test } from "node:test";
import assert from "node:assert/strict";
import { ContinuousExplorer } from "../src/pop/explore/explorer-continuous.js";
import { ExperimentPlanner } from "../src/pop/explore/planner.js";

test("真实语料可形成概念；非收敛预测不能触发置信停止", () => {
  const dims = ["x", "z"].map(name => ({ name, min: 0, max: 1 }));
  const specs = dims.map(d => ({ name: d.name, bins: 3, values: [0, 0.5, 1] }));
  const planner = new ExperimentPlanner(specs);
  const ex = new ContinuousExplorer(dims, [{ name: "y", min: 0, max: 1 }], planner, specs,
    { conduct: c => ({ y: c.x! }) }, { y: 1 }, { budget: 9, fieldsPerDim: 12 }, 3);
  const predict = ex.mem.predict.bind(ex.mem);
  ex.mem.predict = (...args) => ({ ...predict(...args), converged: false, terminationReason: "quiet-constraint" });
  while (ex.step()) {}
  assert.equal(ex.currentPhase, "full");
  assert.ok(ex.formationHistory.length >= 1);
  assert.ok(ex.log.slice(1).every(s => !s.converged && s.classification !== "within-envelope"));
  assert.notEqual(ex.terminationReason, "quorum-met");
  assert.equal(ex.log.length, 9);
});
