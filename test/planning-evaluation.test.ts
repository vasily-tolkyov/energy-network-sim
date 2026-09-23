import { test } from "node:test";
import assert from "node:assert/strict";
import { distribution, shortestReference, predictionMetrics, learningFingerprint } from "../src/topics/plan-evaluation.js";
import { TransitionMemory } from "../src/planning/transition-memory.js";
import { PATH_SPACE, PathBench } from "../src/topics/path-world.js";
import { CHAIN_SPACE, OrderedChainBench } from "../src/topics/ordered-chain-world.js";
import { collectTransitions } from "../src/planning/collect.js";
import { executeGoal } from "../src/planning/execute.js";

test("evaluation uses BFS only on bench side and reports empty denominators as N/A", () => {
  assert.equal(shortestReference(PATH_SPACE, new PathBench(), { pos: 1 }, { pos: 7 }), 6);
  assert.equal(shortestReference(CHAIN_SPACE, new OrderedChainBench(), { node: 3 }, { node: 2 }), null);
  assert.equal(distribution([]).median, null);
  assert.equal(distribution([1, 2, 3, 4]).median, 2.5);
  assert.equal(predictionMetrics([]).nonconvergedRate, null);
});

test("rollout preserves all synapses; source observations change evidence; snapshot stays immutable", async () => {
  const model = new TransitionMemory(PATH_SPACE);
  await collectTransitions(model, new PathBench(), 64, 2);
  const before = learningFingerprint(model);
  const forecast = model.predict({ pos: 0 }, model.actions[1]!, 1);
  assert.equal(learningFingerprint(model), before);
  assert.throws(() => { (forecast.snapshot.values as Record<string, number>).nextPos = 7; });
  const old = JSON.stringify(forecast.snapshot);
  model.observe({ pos: 0 }, model.actions[1]!, { nextPos: 7 });
  assert.notEqual(learningFingerprint(model), before);
  assert.equal(JSON.stringify(forecast.snapshot), old);
});

test("no-replanning control executes old chain and can succeed after displacement without detection credit", async () => {
  const model = new TransitionMemory(CHAIN_SPACE);
  await collectTransitions(model, new OrderedChainBench(), 40, 1);
  const bench = new OrderedChainBench(); let injected = false;
  const execution = await executeGoal(model, { conduct(state, action) {
    const out = bench.conduct(state, action);
    if (!injected && out.nextNode === 1) { injected = true; return { nextNode: 3 }; }
    return out;
  } }, { node: 0 }, { node: 4 }, 1, { replan: false });
  assert.equal(execution.reached, true);
  assert.equal(execution.replans.length, 0);
  assert.equal(execution.steps.length, 4);
  assert.deepEqual(execution.steps.map(s => s.actual.node), [3, 4, 4, 4]);
});
