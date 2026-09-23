import { test } from "node:test";
import assert from "node:assert/strict";
import { TransitionMemory } from "../../src/planning/transition-memory.js";
import { collectTransitions } from "../../src/planning/collect.js";
import { planGoal } from "../../src/planning/planner.js";
import { executeGoal } from "../../src/planning/execute.js";
import { frames, signature } from "../../src/planning/space.js";
import { MULTISTAGE_WORLDS, MultistageBench, type MultistageWorld } from "../../src/topics/multistage-worlds.js";

/** 多阶段陌生场景验收：探索学习（全转移网格读回 + 因素归因）→ 规划 →
 * 执行（含重规划）→ 诚实性负对照。真值只存在于实验台。 */

const collectBudget = (w: MultistageWorld) =>
  [...w.space.states, ...w.space.actions].reduce((n, d) => n * d.bins, 1) * 2;

for (const world of MULTISTAGE_WORLDS) {
  test(`${world.name}：学习读回全网格精确，因素归因正确，干扰维缺席`, async () => {
    for (const seed of [1, 2]) {
      const m = new TransitionMemory(world.space);
      const collection = await collectTransitions(m, new MultistageBench(world), collectBudget(world), seed);
      let wrong = 0;
      let audits = 0;
      for (const state of frames(world.space.states)) {
        for (const action of m.actions) {
          const p = m.predict(state, action, seed);
          audits++;
          assert.ok(!p.audit.mismatch, `账本失配 ${world.name} seed${seed} ${JSON.stringify(state)}`);
          const want = world.truth(state, action.values);
          const wantKey = signature(Object.fromEntries(world.space.states.map(d => [d.name, want[d.name]!])));
          if (p.next === null || signature(p.next) !== wantKey) wrong++;
        }
      }
      assert.equal(wrong, 0, `${world.name} seed${seed} 读回错误 ${wrong} 处`);
      for (const f of world.factors) assert.ok(collection.factors.includes(f), `缺因素 ${f}`);
      for (const d of world.distractors) assert.ok(!collection.factors.includes(d), `干扰维 ${d} 误入`);
      void audits;
    }
  });

  test(`${world.name}：全部任务执行到达；负对照如实报无路线`, async () => {
    for (const seed of [1, 2]) {
      const m = new TransitionMemory(world.space);
      const bench = new MultistageBench(world);
      await collectTransitions(m, bench, collectBudget(world), seed);
      for (const task of world.tasks) {
        const exec = await executeGoal(m, bench, task.start, task.goal, seed);
        assert.ok(exec.reached, `${world.name} seed${seed} ${JSON.stringify(task.start)}→${JSON.stringify(task.goal)}: ${exec.terminationReason}`);
        assert.equal(exec.finalState ? signature(exec.finalState) : null, signature(task.goal));
      }
      if (world.unreachable) {
        const plan = planGoal(m, world.unreachable.start, world.unreachable.goal, seed);
        assert.notEqual(plan.status, "found", `负对照 ${world.name} 不应给出假链`);
      }
    }
  });
}
