import { test } from "node:test";
import assert from "node:assert/strict";
import { PATH_SPACE, PathBench } from "../src/topics/path-world.js";
import { TransitionMemory } from "../src/planning/transition-memory.js";
import { collectTransitions } from "../src/planning/collect.js";
import { planGoal } from "../src/planning/planner.js";

test("path bench validates before spending and retains endpoint self-loops", () => {
  const b = new PathBench();
  for (const pos of [-1, 8, .5, NaN]) assert.throws(() => b.conduct({ pos }, { move: 0 }));
  assert.throws(() => b.conduct({ pos: 0 }, { move: .5 }));
  assert.equal(b.experimentsUsed, 0);
  assert.deepEqual(b.conduct({ pos: 0 }, { move: 0 }), { nextPos: 0 });
  assert.deepEqual(b.conduct({ pos: 7 }, { move: 1 }), { nextPos: 7 });
});

test("PLAN-009 P1/P2: path reach >=90%, median ratio <=1.5; rollout preserves readback", () => {
  const model = new TransitionMemory(PATH_SPACE);
  const training = collectTransitions(model, new PathBench(), 64, 1);
  assert.equal(training.distinctQueries, 16);
  assert.deepEqual(training.factors, ["move", "pos"]);
  const readback = () => Array.from({ length: 8 }, (_, pos) => model.actions.map(action => model.predict({ pos }, action, 100)));
  const before = readback();
  const generation = model.mem.evidenceGeneration;
  let reached = 0;
  const ratios: number[] = [];
  for (let start = 0; start < 8; start++) for (let goal = 0; goal < 8; goal++) {
    const p = planGoal(model, { pos: start }, { pos: goal }, 11 + start * 8 + goal);
    assert.ok(p.predictions.length <= p.predictBudget);
    assert.ok(p.predictions.every(x => !x.audit.mismatch && !!x.snapshot.terminationReason));
    let pos = start;
    const bench = new PathBench();
    for (const step of p.steps) pos = bench.conduct({ pos }, step.action.values).nextPos!;
    if (p.status === "found" && pos === goal) reached++;
    if (start !== goal) ratios.push(p.steps.length / Math.abs(goal - start));
  }
  assert.ok(reached / 64 >= .9, `${reached}/64 reached`);
  ratios.sort((a, b) => a - b);
  assert.ok(ratios[Math.floor(ratios.length / 2)]! <= 1.5);
  const after = readback();
  const stable = (rows: typeof before) => rows.map(row => row.map(p => ({ values: p.snapshot.values, next: p.next,
    converged: p.snapshot.converged, terminationReason: p.snapshot.terminationReason })));
  assert.deepEqual(stable(after), stable(before));
  assert.equal(model.mem.evidenceGeneration, generation);
});
