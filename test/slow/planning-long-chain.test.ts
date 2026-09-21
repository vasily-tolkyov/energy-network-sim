import { test } from "node:test";
import assert from "node:assert/strict";
import { TransitionMemory } from "../../src/planning/transition-memory.js";
import { collectTransitions } from "../../src/planning/collect.js";
import { planGoal } from "../../src/planning/planner.js";
import { validateFrame, type Frame, type TransitionSpace } from "../../src/planning/space.js";

/** 长链回归（感受野渗色修复）：相邻档只共享边缘感受野，转移经验不得在
 * 编码层互相渗色——修复前 16 节点起出现跳读与自闭环死点（no-known-route）。
 * 只测规划正确性（执行路径由 planning-chain 测试在 5 节点覆盖）：
 * 大网络的执行全程受 quiet-constraint 预算约束，耗时按分钟计，不进常规套件。 */
function chainWorld(n: number): { space: TransitionSpace; bench: { conduct(s: Frame, a: Frame): Frame } } {
  const space: TransitionSpace = {
    states: [{ name: "node", outcome: "nextNode", bins: n }],
    actions: [{ name: "advance", bins: 2 }],
    diameter: n - 1,
  };
  const bench = {
    conduct(s: Frame, a: Frame): Frame {
      validateFrame(s, space.states); validateFrame(a, space.actions);
      return { nextNode: a.advance === 1 ? Math.min(n - 1, s.node! + 1) : s.node! };
    },
  };
  return { space, bench };
}

test("长链 16 步：三种子规划成立且逐步预测精确", () => {
  for (const seed of [1, 2, 3]) {
    const { space, bench } = chainWorld(17);
    const m = new TransitionMemory(space);
    collectTransitions(m, bench, 17 * 2 * 4, seed);
    const plan = planGoal(m, { node: 0 }, { node: 16 }, seed);
    assert.equal(plan.status, "found", `seed ${seed}: ${plan.status}`);
    assert.equal(plan.steps.length, 16);
    plan.steps.forEach((s, i) => assert.equal(s.next?.node, i + 1, `seed ${seed} 第 ${i} 步`));
  }
});

test("长链 32 步（单种子）：规划成立且逐步预测精确", () => {
  const { space, bench } = chainWorld(33);
  const m = new TransitionMemory(space);
  collectTransitions(m, bench, 33 * 2 * 4, 1);
  const plan = planGoal(m, { node: 0 }, { node: 32 }, 1);
  assert.equal(plan.status, "found");
  assert.equal(plan.steps.length, 32);
  plan.steps.forEach((s, i) => assert.equal(s.next?.node, i + 1, `第 ${i} 步`));
});
