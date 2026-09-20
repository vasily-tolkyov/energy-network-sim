/**
 * #4 稀疏化规模基准：链世界逐档扩大，测量构建/学习/预测时间与堆内存占用。
 * 对照：同一配置在旧稠密引擎（/tmp/before-network/network.js）上的可达成规模。
 * 运行：node --expose-gc scripts/scale-benchmark.mjs
 */
import { writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";
import { TransitionMemory } from "../dist/src/planning/transition-memory.js";
import { collectTransitions } from "../dist/src/planning/collect.js";
import { planGoal } from "../dist/src/planning/planner.js";
import { validateFrame } from "../dist/src/planning/space.js";

function chainWorld(n) {
  const space = {
    states: [{ name: "node", outcome: "nextNode", bins: n }],
    actions: [{ name: "advance", bins: 2 }],
    diameter: n - 1,
  };
  const bench = {
    conduct(s, a) {
      validateFrame(s, space.states); validateFrame(a, space.actions);
      return { nextNode: a.advance === 1 ? Math.min(n - 1, s.node + 1) : s.node };
    },
  };
  return { space, bench };
}
const heapMB = () => {
  globalThis.gc();
  return process.memoryUsage().heapUsed / 1048576;
};

const rows = [];
for (const bins of [17, 33, 65, 129]) {
  const { space, bench } = chainWorld(bins);
  const before = heapMB();
  let t = performance.now();
  const m = new TransitionMemory(space);
  const buildMs = performance.now() - t;
  const n = m.mem.net.neuronCount;
  t = performance.now();
  collectTransitions(m, bench, bins * 2 * 2, 1);
  const learnMs = performance.now() - t;
  const memMB = heapMB() - before;
  // 预测时间（5 次取中位；大档位的单次预测成本本身就是被测量）
  const times = [];
  for (let s = 1; s <= 5; s++) {
    t = performance.now();
    m.predict({ node: Math.floor(bins / 2) }, m.actions[1], s);
    times.push(performance.now() - t);
  }
  times.sort((a, b) => a - b);
  const predictMs = times[2];
  // 规划全程只测到 33 档（更大档位的规划成本 = 步数 × 单次预测，可由上行外推）
  let planStatus = "skipped", planSteps = 0, planMs = 0;
  if (bins <= 33) {
    t = performance.now();
    const plan = planGoal(m, { node: 0 }, { node: bins - 1 }, 1);
    planMs = performance.now() - t;
    planStatus = plan.status;
    planSteps = plan.steps.length;
  }
  rows.push({ bins, neurons: n, buildMs: +buildMs.toFixed(1), learnMs: +learnMs.toFixed(0),
    heapMB: +memMB.toFixed(1), predictMsMedian: +predictMs.toFixed(1), planStatus,
    planSteps, planMs: +planMs.toFixed(0) });
  console.log(`bins=${bins} N=${n} | 构建 ${buildMs.toFixed(1)}ms | 学习 ${learnMs.toFixed(0)}ms | 堆内存 +${memMB.toFixed(1)}MB | 单次预测中位 ${predictMs.toFixed(1)}ms | 规划 ${planStatus} ${planSteps}步 ${planMs.toFixed(0)}ms`);
}

// 稠密对照（旧引擎能撑到的规模）：N=1600 与 N=4000 的纯分配成本
try {
  const { EnergyNetwork: Dense } = await import(pathToFileURL(path.resolve("/tmp/before-network/network.js")).href);
  const { EnergyNetwork: Sparse } = await import("../dist/src/network.js");
  for (const n of [1600, 4000, 8000]) {
    for (const [label, Ctor] of [["稠密", Dense], ["稀疏", Sparse]]) {
      const before = heapMB();
      const t = performance.now();
      const net = new Ctor({ neuronCount: n, activationEnergy: 1, maintenanceEnergy: 0.5 });
      const ms = performance.now() - t;
      const mb = heapMB() - before;
      // 触及若干条边，模拟真实使用
      for (let k = 0; k < n * 20; k++) net.strengthen(k % n, (k * 7) % n, 0.3);
      const mb2 = heapMB() - before;
      console.log(`N=${n} ${label}: 分配 ${ms.toFixed(1)}ms，空网堆内存 +${mb.toFixed(1)}MB，2万边后 +${mb2.toFixed(1)}MB`);
      rows.push({ denseCompare: { n, label, allocMs: +ms.toFixed(1), emptyMB: +mb.toFixed(1), withEdgesMB: +mb2.toFixed(1) } });
      void net;
    }
  }
} catch (e) {
  console.log("稠密对照失败：", e.message);
}

writeFileSync("runs/scale-benchmark.json", JSON.stringify(rows, null, 2) + "\n");
console.log("已写入 runs/scale-benchmark.json");
