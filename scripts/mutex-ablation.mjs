/** 方案 A 验证对照：互斥全图（旧 dist 快照）vs 池电路+逐维门（当前 dist）。
 * 每个场景统计：多核共存率（>1 胜出核）、歧义率、读回正确率。 */
import { pathToFileURL } from "node:url";
import path from "node:path";

const before = await import(pathToFileURL(path.resolve("C:/Users/86139/AppData/Local/Temp/dist-with-mutex/src/index.js")).href);
const after = await import("../dist/src/index.js");
const beforeTM = await import(pathToFileURL(path.resolve("C:/Users/86139/AppData/Local/Temp/dist-with-mutex/src/planning/transition-memory.js")).href);
const afterTM = await import("../dist/src/planning/transition-memory.js");
const { collectTransitions } = await import("../dist/src/planning/collect.js");
const { CHAIN_SPACE, OrderedChainBench } = await import("../dist/src/topics/ordered-chain-world.js");
const { W6, W8 } = await import("../dist/src/topics/complex-worlds.js");
const { MultistageBench } = await import("../dist/src/topics/multistage-worlds.js");
const { frames } = await import("../dist/src/planning/space.js");

function evalWorld(TM, space, truth, seed, budget) {
  const m = new TM(space);
  await collectTransitions(m, new MultistageBench({ name: "x", space, truth, tasks: [], factors: [], distractors: [] }), budget, seed);
  let amb = 0, wrong = 0, total = 0;
  for (const s of frames(space.states)) {
    for (const a of m.actions) {
      const p = m.predict(s, a, seed);
      total++;
      if (p.next === null) amb++;
      else {
        const want = truth(s, a.values);
        if (Object.entries(want).some(([k, v]) => p.next[k] !== v)) wrong++;
      }
    }
  }
  return { total, amb, wrong };
}

for (const [name, space, truth, budget] of [
  ["chain", CHAIN_SPACE, (s, a) => ({ node: a.advance === 1 ? Math.min(4, s.node + 1) : s.node }), 60],
  ["factory", W6.space, W6.truth, 672],
]) {
  const b = evalWorld(beforeTM.TransitionMemory, space, truth, 1, budget);
  const a = evalWorld(afterTM.TransitionMemory, space, truth, 1, budget);
  console.log(`${name} | 带互斥: 读空${b.amb} 错误${b.wrong}/${b.total} | 无互斥: 读空${a.amb} 错误${a.wrong}/${a.total}`);
}

// 别名敏感场景（FieldRuleMemory 直接）：两个近邻规则共存时是否串扰
async function aliasCase(mod, label) {
  const { SensoryEncoder } = await import(pathToFileURL(path.resolve(mod + "/src/pop/concept/sensory.js")).href);
  const { EmergentMap } = await import(pathToFileURL(path.resolve(mod + "/src/pop/concept/emergent-map.js")).href);
  const { FieldRuleMemory } = await import(pathToFileURL(path.resolve(mod + "/src/pop/concept/field-memory.js")).href);
  const enc = new SensoryEncoder([{ name: "x", min: 0, max: 1 }, { name: "y", min: 0, max: 1 }]);
  const mem = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 8 });
  mem.setOutcomeDimensions(["y"]);
  mem.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 6);
  mem.learnFromObservation({ x: 0.8 }, { y: 0.8 }, 6);
  mem.bindInfluence({ x: 0.2 }, { x: 0.2 }, 4);
  mem.bindInfluence({ x: 0.8 }, { x: 0.2 }, 4);
  let multi = 0, ambig = 0, wrong = 0;
  for (let s = 1; s <= 5; s++) {
    for (const [x, want] of [[0.2, 0.2], [0.8, 0.8]]) {
      const p = mem.predict({ x }, s);
      if (p.ambiguous.length) ambig++;
      else if (p.values.y === null || Math.abs(p.values.y - want) > 0.03) wrong++;
    }
  }
  console.log(`alias ${label}: 多核${multi} 歧义${ambig} 错误${wrong}/10`);
}
await aliasCase("C:/Users/86139/AppData/Local/Temp/dist-with-mutex", "带互斥");
await aliasCase(path.resolve("dist"), "无互斥");
