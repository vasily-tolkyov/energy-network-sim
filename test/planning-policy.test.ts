import { test } from "node:test";
import assert from "node:assert/strict";
import { planGoal, candidateTier } from "../src/planning/planner.js";
import { TransitionMemory, type StepPrediction, type TransitionReader } from "../src/planning/transition-memory.js";
import { signature } from "../src/planning/space.js";
import { PATH_SPACE } from "../src/topics/path-world.js";

function prediction(next: number | null, reason = "fixed-point"): StepPrediction {
  return { state: { pos: 0 }, action: { id: 0, values: { move: 0 } }, seed: 1, observed: true,
    snapshot: { coreIdx: 0, values: { nextPos: next }, generation: 1, converged: reason === "fixed-point", terminationReason: reason },
    ambiguous: [], next: next === null ? null : { pos: next }, kind: reason === "no-quiet-candidate" ? "infeasible" : next === null ? "unknown" : "usable",
    approximate: next !== null && reason !== "fixed-point", audit: { energy: 0, traceEnd: 0, recomputed: 0, error: 0, mismatch: false }, milliseconds: 0 };
}
test("quiet failure is infeasible; unknown is retained; approximate valued steps remain usable", () => {
  for (const [p, status] of [[prediction(null, "no-quiet-candidate"), "no-known-route"],
    [prediction(null), "unexplored"], [prediction(1, "quiet-constraint"), "found"]] as const) {
    const model: TransitionReader = { space: PATH_SPACE, actions: [p.action], predict: () => structuredClone(p) };
    const plan = planGoal(model, { pos: 0 }, { pos: 1 }, 1);
    assert.equal(plan.status, status);
    if (status === "found") assert.equal(plan.steps[0]!.snapshot.terminationReason, "quiet-constraint");
    if (status === "unexplored") assert.equal(plan.unknownFrontier.length, 1);
  }
});
test("candidate ranking and cycle detection do not use distance to goal", () => {
  const visited = new Set([signature({ pos: 0 })]);
  assert.equal(candidateTier(prediction(1, "quiet-constraint"), visited), 0);
  assert.equal(candidateTier({ ...prediction(1), observed: false }, visited), 1);
  assert.equal(candidateTier(prediction(null), visited), 2);
  assert.equal(candidateTier(prediction(0), visited), 3);
  const p = prediction(0);
  const model: TransitionReader = { space: PATH_SPACE, actions: [p.action], predict: () => p };
  assert.equal(planGoal(model, { pos: 0 }, { pos: 7 }, 1).predictions.length, 1);
  assert.equal(planGoal(model, { pos: 0 }, { pos: 0 }, 1).predictions.length, 0);
});
test("all output dimensions required, invalid observation cannot mutate evidence", () => {
  const model = new TransitionMemory({ states: [{ name: "x", outcome: "nx", bins: 2 }, { name: "y", outcome: "ny", bins: 2 }],
    actions: [{ name: "a", bins: 2 }], diameter: 2 });
  assert.equal(model.decode({ nx: 1, ny: null }), null);
  assert.throws(() => model.observe({ x: 0, y: 0 }, model.actions[0]!, { nx: 1 }));
  assert.equal(model.mem.ruleCount, 0);
  assert.equal(model.mem.evidenceGeneration, 0);
});
