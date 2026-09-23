import { test } from "node:test";
import assert from "node:assert/strict";
import { SceneParser } from "../src/perception/scene-parser.js";
import { MockBackend } from "../src/perception/decision-backend.js";
import { ActionExecutor } from "../src/execution/action-executor.js";
import { MC_PERCEPTION, MC_EXECUTION } from "../src/topics/minecraft-spec.js";
import { TransitionMemory } from "../src/planning/transition-memory.js";
import { collectTransitions } from "../src/planning/collect.js";
import { planGoal } from "../src/planning/planner.js";
import { executeGoal } from "../src/planning/execute.js";
import { validateFrame, type Frame, type TransitionSpace } from "../src/planning/space.js";

/** Minecraft 形状的状态流 mock：位置 0..5、昼/夜、存活；规则：
 * 黑夜在野外（pos≥3）每步扣血；遮蔽区（pos≤2）安全；白天不扣血；死亡即终点。
 * 目标：从 pos 0 白天出发，活着到达 pos 5（需要在黑夜前冲过或等待白天）。 */
const MC_SPACE: TransitionSpace = {
  states: [
    { name: "pos", outcome: "nextPos", bins: 6 },
    { name: "time", outcome: "nextTime", bins: 2 }, // 0昼 1夜
    { name: "alive", outcome: "nextAlive", bins: 2 },
  ],
  actions: [{ name: "move", bins: 2 }], // 0=等待 1=前进
  diameter: 8,
};
const mcTruth = (s: Frame, a: Frame): Frame => {
  if (s.alive === 0) return { pos: s.pos!, time: s.time!, alive: 0 };
  const ntime = a.move === 0 ? (s.time === 0 ? 1 : 0) : s.time!; // 等待会度过时段
  const np = a.move === 1 ? Math.min(5, s.pos! + 1) : s.pos!;
  const die = ntime === 1 && np >= 3;
  return { pos: np, time: ntime, alive: die ? 0 : 1 };
};

test("minecraft-mock：文字状态流 → 感知帧 → 学习 → 规划 → 执行（全链路）", async () => {
  // 感知：文字状态 → 条件帧（Minecraft 感知规格实测）
  const parser = new SceneParser(new MockBackend());
  const stateText = "生命值满血，饥饿正常，白天，附近没有敌对生物，脚下是坚实地面。";
  const perceived = await parser.parse(stateText, MC_PERCEPTION);
  assert.deepEqual(perceived.frame, { health: 2, hunger: 1, time: 0, threat: 0, ground: 0 });

  // 学习：mock 世界全转移覆盖
  const m = new TransitionMemory(MC_SPACE);
  const bench = { conduct: (s: Frame, a: Frame) => {
    validateFrame(s, MC_SPACE.states); validateFrame(a, MC_SPACE.actions);
    const n = mcTruth(s, a);
    return Object.fromEntries(MC_SPACE.states.map(d => [d.outcome, n[d.name]!]));
  } };
  await collectTransitions(m, bench, 48, 1);

  // 规划：白天出发活到 pos5——黑夜在野外会死，必须在白天冲刺或等待
  const plan = planGoal(m, { pos: 0, time: 0, alive: 1 }, { pos: 5, time: 0, alive: 1 }, 1);
  assert.equal(plan.status, "found");
  // 执行：任何一步死亡都算失败
  const exec = await executeGoal(m, bench, { pos: 0, time: 0, alive: 1 }, { pos: 5, time: 0, alive: 1 }, 1);
  assert.ok(exec.steps.every(s => s.actual.alive === 1), "全程存活");
  assert.equal(exec.finalState.pos, 5);

  // 执行映射：规划链 → mineflayer 风格命令（可行性判定通过）；
  // 本 mock 世界的动作空间只有 move，执行规格按实际动作帧过滤子集
  const executor = new ActionExecutor(new MockBackend());
  const moveSpec = { ...MC_EXECUTION, actions: MC_EXECUTION.actions.filter(a => a.dimension === "move") };
  const rec = await executor.map({ move: 1 }, "白天，生命值健康，无威胁", moveSpec);
  assert.ok(rec.feasible);
});
