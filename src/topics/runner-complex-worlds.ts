import { mkdirSync, writeFileSync } from "node:fs";
import { mulberry32 } from "../prng.js";
import { TransitionMemory } from "../planning/transition-memory.js";
import { collectTransitions } from "../planning/collect.js";
import { planGoal } from "../planning/planner.js";
import { executeGoal } from "../planning/execute.js";
import { frames, signature, validateFrame, type Frame, type TransitionBench } from "../planning/space.js";
import { MultistageBench, type MultistageWorld } from "./multistage-worlds.js";
import { W6, W7, W8 } from "./complex-worlds.js";

/**
 * 复杂陌生场景验收 + 成本记录：
 * W6 双资源工厂（深层分阶段）、W7 世界规则中途翻转（动态世界全循环）、
 * W8 三门长走廊（32 节点大网格长程规划）。
 * 逐世界记录：神经元数、堆内存、学习/预测/规划/执行耗时、读回精度、
 * 因素归因、任务到达率。落盘 runs/complex-worlds-*.json。
 * 运行：node --expose-gc dist/src/topics/runner-complex-worlds.js（SEEDS 分段）
 */

const seeds = (process.env.SEEDS ?? "1,2").split(",").map(s => parseInt(s.trim(), 10));
const heapMB = () => {
  if (globalThis.gc) globalThis.gc();
  return process.memoryUsage().heapUsed / 1048576;
};
const collectBudget = (w: MultistageWorld) =>
  [...w.space.states, ...w.space.actions].reduce((n, d) => n * d.bins, 1) * 2;

function readback(model: TransitionMemory, world: MultistageWorld, seed: number, sampleCap = Infinity) {
  let wrong = 0, total = 0, ledgerMismatches = 0;
  const predictTimes: number[] = [];
  const combos = frames(world.space.states).flatMap(state => model.actions.map(action => ({ state, action })));
  // 大网格抽样读回（确定性选取；小网格全量）
  const rng = mulberry32(seed);
  const picked = combos.length <= sampleCap ? combos : combos.filter(() => rng() < sampleCap / combos.length).slice(0, sampleCap);
  for (const { state, action } of picked) {
    const t = performance.now();
    const p = model.predict(state, action, seed);
    predictTimes.push(performance.now() - t);
    total++;
    if (p.audit.mismatch) ledgerMismatches++;
    const want = world.truth(state, action.values);
    if (p.next === null || signature(p.next) !== signature(Object.fromEntries(world.space.states.map(d => [d.name, want[d.name]!])))) wrong++;
  }
  predictTimes.sort((a, b) => a - b);
  return { wrong, total, ledgerMismatches, predictMsMedian: predictTimes[Math.floor(total / 2)] };
}

function runWorld(world: MultistageWorld, seed: number) {
  const t0 = performance.now();
  const before = heapMB();
  const model = new TransitionMemory(world.space);
  const bench = new MultistageBench(world);
  let t = performance.now();
  const collection = collectTransitions(model, bench, collectBudget(world), seed);
  const collectMs = performance.now() - t;
  console.log(`  [${world.name} seed${seed}] 学习完成 ${collection.experiments} 次，${collectMs.toFixed(0)}ms`);
  const memMB = heapMB() - before;
  const rb = readback(model, world, seed, world.space.states[0]!.bins > 16 ? 128 : Infinity);
  console.log(`  [${world.name} seed${seed}] 读回完成 ${rb.total - rb.wrong}/${rb.total}，预测中位 ${(rb.predictMsMedian ?? 0).toFixed(1)}ms`);
  const tasks = world.tasks.map(task => {
    t = performance.now();
    const plan = planGoal(model, task.start, task.goal, seed);
    const planMs = performance.now() - t;
    t = performance.now();
    const exec = executeGoal(model, bench, task.start, task.goal, seed);
    const execMs = performance.now() - t;
    return { planStatus: plan.status, planSteps: plan.steps.length, planMs: +planMs.toFixed(0),
      reached: exec.reached, terminationReason: exec.terminationReason, replans: exec.replans.length, execMs: +execMs.toFixed(0) };
  });
  return {
    world: world.name, seed, neurons: model.mem.net.neuronCount, heapMB: +memMB.toFixed(1),
    collectExperiments: collection.experiments, collectMs: +collectMs.toFixed(0), factors: collection.factors,
    readback: { ...rb, predictMsMedian: +(rb.predictMsMedian ?? 0).toFixed(1) }, tasks,
    totalMs: +(performance.now() - t0).toFixed(0),
  };
}

/** W7 动态世界全循环：原模型规划执行 → 翻转 → 失配检测 → 诚实无路线 →
 * 再探索（同记忆）→ 计票翻转 + 学捷径 → 重规划绕行到达。 */
const W7_FLIP = 2; // 世界在第 2 次执行干预后永久翻转
function runWorldFlip(seed: number) {
  const model = new TransitionMemory(W7.space);
  const bench = new MultistageBench(W7);
  collectTransitions(model, bench, collectBudget(W7), seed);
  const goal = W7.tasks[0]!.goal;
  const plan = planGoal(model, W7.tasks[0]!.start, goal, seed);
  let flipped = false;
  const flippingBench: TransitionBench = {
    conduct(state: Frame, action: Frame) {
      validateFrame(state, W7.space.states); validateFrame(action, W7.space.actions);
      const t = flipped ? W7.flippedTruth : W7.truth;
      return Object.fromEntries(W7.space.states.map(d => [d.outcome, (t(state, action) as Record<string, number>)[d.name]!]));
    },
  };
  let calls = 0;
  const countingFlippingBench: TransitionBench = {
    conduct(state, action) {
      if (++calls > W7_FLIP) flipped = true;
      return flippingBench.conduct(state, action);
    },
  };
  const exec = executeGoal(model, countingFlippingBench, W7.tasks[0]!.start, goal, seed);
  // 翻转后：失配检测存在；捷径经失配观察学到，重规划绕行到达
  const detected = exec.steps.some(s => s.capture.class !== "within-envelope");
  // 再探索：同一记忆在翻转后的世界重新收集
  collectTransitions(model, { conduct: (s, a) => flippingBench.conduct(s, a) } , collectBudget(W7) * 2, seed);
  const blocked = model.predict({ pos: 4 }, model.actions[1]!, seed);
  const shortcut = model.predict({ pos: 2 }, model.actions[1]!, seed);
  // 无捷径对照：翻转只关门不开路——必须诚实报无路线
  const m2 = new TransitionMemory(W7.space);
  collectTransitions(m2, new MultistageBench(W7), collectBudget(W7), seed);
  let calls2 = 0;
  const noShortcut: TransitionBench = {
    conduct(state: Frame, action: Frame) {
      validateFrame(state, W7.space.states); validateFrame(action, W7.space.actions);
      const t = ++calls2 > 2
        ? (s: Frame, a: Frame) => (a.act === 0 ? { pos: s.pos! } : s.pos === 4 ? { pos: 4 } : { pos: Math.min(9, s.pos! + 1) })
        : W7.truth;
      return Object.fromEntries(W7.space.states.map(d => [d.outcome, (t(state, action) as Record<string, number>)[d.name]!]));
    },
  };
  const execN = executeGoal(m2, noShortcut, W7.tasks[0]!.start, goal, seed);
  return {
    seed, initialPlan: plan.status, detected, reachedViaDetour: exec.reached,
    replans: exec.replans.length,
    blockedReads4: blocked.next, shortcutReads2: shortcut.next,
    noShortcutReached: execN.reached, noShortcutLastPlan: execN.plans.at(-1)!.status,
  };
}

mkdirSync("runs", { recursive: true });
const report: Record<string, unknown> = { schema: "complex-worlds/v1", seeds };
const rows = [];
for (const seed of seeds) {
  for (const world of [W6, W8]) rows.push(runWorld(world, seed));
  (report as any).w6w8 = rows;
}
(report as any).w7 = seeds.map(runWorldFlip);
writeFileSync(`runs/complex-worlds.json`, JSON.stringify(report, null, 2) + "\n");
for (const r of rows) {
  console.log(`${r.world} seed${r.seed} N=${r.neurons} 堆+${r.heapMB}MB | 学习 ${r.collectExperiments}次 ${r.collectMs}ms | 读回 ${r.readback.total - r.readback.wrong}/${r.readback.total} 预测中位 ${r.readback.predictMsMedian}ms | 任务 ${r.tasks.map(t => `${t.planStatus}/${t.reached}`).join(",")} | 合计 ${r.totalMs}ms`);
}
for (const r of (report as any).w7 as ReturnType<typeof runWorldFlip>[]) {
  console.log(`world-flip seed${r.seed} | 初计划 ${r.initialPlan} | 检测 ${r.detected} 绕行到达 ${r.reachedViaDetour} 重规划${r.replans}次 | 堵点读 ${JSON.stringify(r.blockedReads4)} 捷径读 ${JSON.stringify(r.shortcutReads2)} | 无捷径对照到达 ${r.noShortcutReached} 终计划 ${r.noShortcutLastPlan}`);
}
