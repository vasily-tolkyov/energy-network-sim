import { test } from "node:test";
import assert from "node:assert/strict";
import { TransitionMemory } from "../../src/planning/transition-memory.js";
import { collectTransitions } from "../../src/planning/collect.js";
import { planGoal } from "../../src/planning/planner.js";
import { executeGoal } from "../../src/planning/execute.js";
import { frames, signature, validateFrame, type Frame, type TransitionBench } from "../../src/planning/space.js";
import { MultistageBench, type MultistageWorld } from "../../src/topics/multistage-worlds.js";
import { W6, W7, W8 } from "../../src/topics/complex-worlds.js";

/** 复杂陌生场景验收：学习读回、因素归因、多阶段任务执行、动态世界全循环。 */

const collectBudget = (w: MultistageWorld) =>
  [...w.space.states, ...w.space.actions].reduce((n, d) => n * d.bins, 1) * 2;

function flipBench(world: typeof W7, flipAfter: number): TransitionBench & { flipNow(): void } {
  let calls = 0;
  let flipped = false;
  return {
    flipNow() { flipped = true; },
    conduct(state: Frame, action: Frame) {
      validateFrame(state, world.space.states); validateFrame(action, world.space.actions);
      if (++calls > flipAfter) flipped = true;
      const t = flipped ? world.flippedTruth : world.truth;
      return Object.fromEntries(world.space.states.map(d => [d.outcome, (t(state, action) as Record<string, number>)[d.name]!]));
    },
  };
}

test("factory：双资源深层分阶段（矿-能-合成）学习、归因与任务", async () => {
  for (const seed of [1, 2]) {
    const m = new TransitionMemory(W6.space);
    const bench = new MultistageBench(W6);
    const collection = await collectTransitions(m, bench, collectBudget(W6), seed);
    let wrong = 0;
    for (const state of frames(W6.space.states)) {
      for (const action of m.actions) {
        const p = m.predict(state, action, seed);
        const want = W6.truth(state, action.values);
        if (p.next === null || signature(p.next) !== signature(Object.fromEntries(W6.space.states.map(d => [d.name, want[d.name]!])))) wrong++;
        assert.ok(!p.audit.mismatch, "账本失配");
      }
    }
    assert.equal(wrong, 0, `seed${seed} 读回错误 ${wrong} 处`);
    for (const f of W6.factors) assert.ok(collection.factors.includes(f), `缺因素 ${f}`);
    for (const task of W6.tasks) {
      const exec = await executeGoal(m, bench, task.start, task.goal, seed);
      assert.ok(exec.reached, `seed${seed} 任务未到达: ${exec.terminationReason}`);
    }
  }
});

test("world-flip：失配检测、绕行到达、计票翻转旧规则、无捷径对照诚实报无路", async () => {
  for (const seed of [1, 2]) {
    const m = new TransitionMemory(W7.space);
    await collectTransitions(m, new MultistageBench(W7), collectBudget(W7), seed);
    const goal = W7.tasks[0]!.goal;
    assert.equal(planGoal(m, W7.tasks[0]!.start, goal, seed).status, "found", "翻转前应可规划");
    // 主案例：翻转开启捷径——世界把 agent 弹到 7，失配观察即捷径证据，重规划绕行到达
    const bench = flipBench(W7, 2);
    const exec = await executeGoal(m, bench, W7.tasks[0]!.start, goal, seed);
    assert.ok(exec.replans.length > 0, "应有失配触发的重规划");
    assert.ok(exec.steps.some(s => s.capture.class !== "within-envelope"), "必须有失配检测记录");
    assert.ok(exec.reached, `seed${seed} 应经捷径到达: ${exec.terminationReason}`);
    // 再探索：同一记忆在翻转后的世界重新收集（预算 ×4：计票翻转需要定额观察）
    await collectTransitions(m, bench, collectBudget(W7) * 2, seed);
    assert.deepEqual(m.predict({ pos: 4 }, m.actions[1]!, seed).next, { pos: 4 }, "被堵规则应已按计票翻转");
    assert.deepEqual(m.predict({ pos: 2 }, m.actions[1]!, seed).next, { pos: 7 }, "捷径应已学到");
    // 无捷径对照：翻转只关门不开路——必须诚实报无路线，不得给假链
    const m2 = new TransitionMemory(W7.space);
    await collectTransitions(m2, new MultistageBench(W7), collectBudget(W7), seed);
    let calls = 0;
    const noShortcut: TransitionBench = {
      conduct(state: Frame, action: Frame) {
        validateFrame(state, W7.space.states); validateFrame(action, W7.space.actions);
        const t = ++calls > 2
          ? (s: Frame, a: Frame) => (a.act === 0 ? { pos: s.pos! } : s.pos === 4 ? { pos: 4 } : { pos: Math.min(9, s.pos! + 1) })
          : W7.truth;
        return Object.fromEntries(W7.space.states.map(d => [d.outcome, (t(state, action) as Record<string, number>)[d.name]!]));
      },
    };
    const execN = await executeGoal(m2, noShortcut, W7.tasks[0]!.start, goal, seed);
    assert.ok(!execN.reached, "无捷径对照不应到达");
    assert.notEqual(execN.plans.at(-1)!.status, "found", "无捷径对照必须诚实报无路线");
  }
});

test("two-gates：24 节点双门长程学习、归因与长链任务", { timeout: 3600_000 }, async () => {
  const m = new TransitionMemory(W8.space);
  const bench = new MultistageBench(W8);
  const collection = await collectTransitions(m, bench, collectBudget(W8), 1);
  // 抽样读回（全网格 192 组合的确定性 64 子集；成本考虑，全量覆盖已由 runner 的成本实测记录）
  const combos = frames(W8.space.states).flatMap(state => m.actions.map(action => ({ state, action })));
  const step = combos.length / 64;
  let wrong = 0;
  for (let k = 0; k < 64; k++) {
    const { state, action } = combos[Math.floor(k * step)]!;
    const p = m.predict(state, action, 1);
    const want = W8.truth(state, action.values);
    if (p.next === null || signature(p.next) !== signature(Object.fromEntries(W8.space.states.map(d => [d.name, want[d.name]!])))) wrong++;
    assert.ok(!p.audit.mismatch, "账本失配");
  }
  assert.equal(wrong, 0, `抽样读回错误 ${wrong} 处`);
  for (const f of W8.factors) assert.ok(collection.factors.includes(f), `缺因素 ${f}`);
  const task = W8.tasks[0]!;
  const exec = await executeGoal(m, bench, task.start, task.goal, 1);
  assert.ok(exec.reached, `双门长链未到达: ${exec.terminationReason}`);
});
