/** W8-32 压力复测（方案 A 后）：512 规则、核度曾 ~2000、单预测 ~40s。
 * 目标：互斥全图移除后回到秒级。 */
import { TransitionMemory } from "../dist/src/planning/transition-memory.js";
import { collectTransitions } from "../dist/src/planning/collect.js";
import { MultistageBench } from "../dist/src/topics/multistage-worlds.js";

const space = {
  states: [
    { name: "pos", outcome: "nextPos", bins: 32 },
    { name: "k1", outcome: "nextK1", bins: 2 },
    { name: "k2", outcome: "nextK2", bins: 2 },
    { name: "k3", outcome: "nextK3", bins: 2 },
  ],
  actions: [{ name: "move", bins: 2 }],
  diameter: 38,
};
const truth = (s, a) => {
  let np = s.pos;
  if (a.move === 1 && s.pos < 31) np = s.pos + 1;
  if (a.move === 0 && s.pos > 0) np = s.pos - 1;
  if (a.move === 1) {
    if (s.pos === 7 && s.k1 === 0) np = 7;
    if (s.pos === 15 && s.k2 === 0) np = 15;
    if (s.pos === 23 && s.k3 === 0) np = 23;
  }
  return { pos: np, k1: np === 4 ? 1 : s.k1, k2: np === 12 ? 1 : s.k2, k3: np === 20 ? 1 : s.k3 };
};
const world = { name: "w8-32", space, truth, tasks: [], factors: [], distractors: [] };

const t0 = performance.now();
const m = new TransitionMemory(space);
await collectTransitions(m, new MultistageBench(world), 1024, 1);
console.log(`学习 1024 次：${((performance.now() - t0) / 1000).toFixed(1)}s（方案 A 前：67s+ 且含 210 万条互斥边写入）`);
console.log(`N=${m.mem.net.neuronCount} 规则=${m.mem.ruleCount}`);
for (const s of [1, 2, 3]) {
  const t = performance.now();
  const p = m.predict({ pos: 16, k1: 1, k2: 1, k3: 0 }, m.actions[1], s);
  console.log(`seed${s} 预测 ${((performance.now() - t) / 1000).toFixed(1)}s → ${JSON.stringify(p.next)} ${p.snapshot.terminationReason}`);
}
