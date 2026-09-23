import { test } from "node:test";
import assert from "node:assert/strict";
import { OrderedChainBench, CHAIN_SPACE } from "../src/topics/ordered-chain-world.js";
import { collectTransitions } from "../src/planning/collect.js";
import { TransitionMemory } from "../src/planning/transition-memory.js";
import { planGoal } from "../src/planning/planner.js";
import { executeGoal } from "../src/planning/execute.js";
import type { Frame, TransitionBench } from "../src/planning/space.js";

async function trained(seed: number): Promise<TransitionMemory> {
  const model = new TransitionMemory(CHAIN_SPACE);
  await collectTransitions(model, new OrderedChainBench(), 40, seed);
  return model;
}
test("chain bench is forward-only, validates inputs and charges only valid experiments", () => {
  const bench = new OrderedChainBench();
  assert.throws(() => bench.conduct({ node: 1.5 }, { advance: 0 }));
  assert.equal(bench.experimentsUsed, 0);
  assert.equal(bench.conduct({ node: 3 }, { advance: 0 }).nextNode, 4);
  assert.equal(bench.conduct({ node: 4 }, { advance: 0 }).nextNode, 4);
  bench.setGateOpen(false);
  assert.equal(bench.conduct({ node: 2 }, { advance: 0 }).nextNode, 2);
});
for (const seed of [1, 2, 3]) {
  test(`chain seed ${seed}: ordered route and no reverse door`, async () => {
    const model = await trained(seed);
    const execution = await executeGoal(model, new OrderedChainBench(), { node: 0 }, { node: 4 }, seed);
    assert.equal(execution.reached, true);
    assert.deepEqual([0, ...execution.steps.map(s => s.actual.node)], [0, 1, 2, 3, 4]);
    assert.equal(planGoal(model, { node: 3 }, { node: 2 }, seed).status, "no-known-route");
    assert.ok(execution.plans.flatMap(p => p.predictions).every(p => !p.audit.mismatch));
  });
  test(`chain seed ${seed}: gate closure triggers replanning and an honest stop at C`, async () => {
    const model = await trained(seed);
    const bench = new OrderedChainBench();
    const injecting: TransitionBench = { conduct: (state, action) => {
      if (state.node === 2) bench.setGateOpen(false);
      return bench.conduct(state, action);
    } };
    const execution = await executeGoal(model, injecting, { node: 0 }, { node: 4 }, seed);
    assert.equal(execution.reached, false);
    assert.equal(execution.finalState.node, 2);
    assert.ok(execution.replans.some(r => r.from.node === 2));
    assert.ok(execution.steps.some(s => s.state.node === 2 && s.capture.class !== "within-envelope"));
    assert.ok(execution.plans.slice(1).every(p => !p.steps.some(s => s.state.node === 2 && s.next?.node === 3)));
    assert.equal(execution.terminationReason, "no-known-route");
  });
  test(`chain seed ${seed}: displaced B→D replans from D and preserves the old snapshot`, async () => {
    const model = await trained(seed);
    const bench = new OrderedChainBench();
    let injected = false;
    const injecting: TransitionBench = { conduct: (state, action): Frame => {
      const actual = bench.conduct(state, action);
      if (!injected && actual.nextNode === 1) { injected = true; return { nextNode: 3 }; }
      return actual;
    } };
    const execution = await executeGoal(model, injecting, { node: 0 }, { node: 4 }, seed);
    assert.equal(execution.reached, true);
    assert.ok(execution.replans.some(r => r.from.node === 3));
    assert.equal(execution.steps[0]!.forecast.snapshot.values.nextNode, 1);
    assert.equal(execution.steps[0]!.actual.node, 3);
    assert.ok(execution.steps[0]!.evidenceGenerationAfter > execution.steps[0]!.forecast.snapshot.generation);
    assert.notEqual(execution.steps[0]!.capture.class, "within-envelope");
  });
}
