import { mkdirSync, writeFileSync } from "node:fs";
import { mulberry32 } from "../prng.js";
import { TransitionMemory } from "../planning/transition-memory.js";
import { collectTransitions } from "../planning/collect.js";
import { planGoal } from "../planning/planner.js";
import { executeGoal } from "../planning/execute.js";
import { frames, signature, type Frame, type TransitionSpace } from "../planning/space.js";
import { MultistageBench, W1, W2, W3, W4, W5, type MultistageWorld } from "./multistage-worlds.js";
import { BOX_SPACE, BOX_START, BOX_GOAL_STACK2, boxTruth } from "./box-world.js";

/**
 * 通用预测基准：一套原型（TransitionMemory + collectTransitions + planGoal +
 * executeGoal，同一 seed，零逐场景调参）在 5 个转移结构互异的陌生场景上跑
 * 完整链：自由探索学习 → 读回校验 → 规划 → 物理执行；另加箱子世界作规模锚点。
 * 重点记录退火耗时：每条 predict 自带毫秒数（含审计重算），逐场景汇总
 * n/mean/p50/p95/max/total。落盘 runs/general-bench.json。
 * 运行：node --expose-gc dist/src/topics/runner-general-bench.js
 */

const seed = 1;
const heapMB = () => {
  if (globalThis.gc) globalThis.gc();
  return process.memoryUsage().heapUsed / 1048576;
};
const collectBudget = (space: TransitionSpace) =>
  [...space.states, ...space.actions].reduce((n, d) => n * d.bins, 1) * 2;

const annealStats = (xs: readonly number[]) => {
  if (!xs.length) return { n: 0, totalMs: 0, mean: 0, p50: 0, p95: 0, max: 0 };
  const s = [...xs].sort((a, b) => a - b);
  const total = xs.reduce((a, b) => a + b, 0);
  return { n: xs.length, totalMs: Math.round(total), mean: +(total / xs.length).toFixed(1),
    p50: +s[Math.floor(xs.length * 0.5)]!.toFixed(1),
    p95: +s[Math.min(xs.length - 1, Math.floor(xs.length * 0.95))]!.toFixed(1),
    max: +s.at(-1)!.toFixed(1) };
};

function readback(model: TransitionMemory, world: MultistageWorld, sampleCap: number) {
  let wrong = 0, total = 0;
  const times: number[] = [];
  const combos = frames(world.space.states).flatMap(state => model.actions.map(action => ({ state, action })));
  const rng = mulberry32(seed);
  const picked = combos.length <= sampleCap ? combos : combos.filter(() => rng() < sampleCap / combos.length).slice(0, sampleCap);
  for (const { state, action } of picked) {
    const t = performance.now();
    const p = model.predict(state, action, seed);
    times.push(performance.now() - t);
    total++;
    const want = world.truth(state, action.values);
    if (p.next === null || signature(p.next) !== signature(Object.fromEntries(world.space.states.map(d => [d.name, want[d.name]!])))) wrong++;
  }
  return { wrong, total, readbackAnneal: annealStats(times) };
}

async function runWorld(world: MultistageWorld, readbackCap: number) {
  const t0 = performance.now();
  const before = heapMB();
  const model = new TransitionMemory(world.space);
  const bench = new MultistageBench(world);
  let t = performance.now();
  const collection = await collectTransitions(model, bench, collectBudget(world.space), seed);
  const collectMs = performance.now() - t;
  const rb = readback(model, world, readbackCap);
  const tasks = [];
  const allPredictMs: number[] = [];
  for (const task of world.tasks) {
    t = performance.now();
    const exec = await executeGoal(model, bench, task.start, task.goal, seed);
    const execMs = performance.now() - t;
    const plan0 = exec.plans[0]!;
    for (const plan of exec.plans) for (const p of plan.predictions) allPredictMs.push(p.milliseconds);
    tasks.push({ planStatus: plan0.status, planSteps: plan0.steps.length,
      planAnneal: annealStats(plan0.predictions.map(p => p.milliseconds)),
      reached: exec.reached, terminationReason: exec.terminationReason, replans: exec.replans.length,
      execMs: +execMs.toFixed(0) });
  }
  // 诚实性负对照：不可达目标必须如实报无路线
  let unreachable: unknown = null;
  if (world.unreachable) {
    t = performance.now();
    const plan = planGoal(model, world.unreachable.start, world.unreachable.goal, seed);
    unreachable = { status: plan.status, anneal: annealStats(plan.predictions.map(p => p.milliseconds)) };
    for (const p of plan.predictions) allPredictMs.push(p.milliseconds);
  }
  return {
    world: world.name, seed, neurons: model.mem.net.neuronCount, heapMB: +(heapMB() - before).toFixed(1),
    rules: model.mem.ruleCount, collectExperiments: collection.experiments, collectMs: +collectMs.toFixed(0),
    readback: { wrong: rb.wrong, total: rb.total }, readbackAnneal: rb.readbackAnneal,
    tasks, unreachable, anneal: annealStats(allPredictMs),
    totalMs: +(performance.now() - t0).toFixed(0),
  };
}

const boxAsWorld: MultistageWorld = {
  name: "box-world(规模锚点)", space: BOX_SPACE, truth: boxTruth,
  tasks: [{ start: BOX_START, goal: BOX_GOAL_STACK2 }],
  factors: ["row", "col", "carry", "stack", "move", "handle"], distractors: [],
};

mkdirSync("runs", { recursive: true });
const rows = [];
for (const w of [W1, W2, W3, W4, W5]) {
  const r = await runWorld(w, Infinity);
  rows.push(r);
  console.log(`${r.world} N=${r.neurons} 规则${r.rules} | 读回 ${r.readback.total - r.readback.wrong}/${r.readback.total} | 任务 ${r.tasks.map(t => `${t.planStatus}/${t.reached ? "到达" : "未到"}`).join(",")}${r.unreachable ? ` | 负对照 ${(r.unreachable as any).status}` : ""} | 退火 n=${r.anneal.n} 中位${r.anneal.p50}ms p95=${r.anneal.p95}ms 共${r.anneal.totalMs}ms | 合计 ${(r.totalMs / 1000).toFixed(1)}s`);
}
const box = await runWorld(boxAsWorld, 64); // 大网格抽样读回
rows.push(box);
console.log(`${box.world} N=${box.neurons} 规则${box.rules} | 读回 ${box.readback.total - box.readback.wrong}/${box.readback.total}(抽样) | 任务 ${box.tasks.map(t => `${t.planStatus}/${t.reached ? "到达" : "未到"}`).join(",")} | 退火 n=${box.anneal.n} 中位${box.anneal.p50}ms p95=${box.anneal.p95}ms 共${box.anneal.totalMs}ms | 合计 ${(box.totalMs / 1000).toFixed(1)}s`);
writeFileSync("runs/general-bench.json", JSON.stringify({ schema: "general-bench/v1", seed, note: "与 MC 现场演示并行，耗时含 CPU 争用（保守上界）", rows }, null, 2) + "\n");
console.log("落盘 runs/general-bench.json");
