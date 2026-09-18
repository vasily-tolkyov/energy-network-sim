// 计时：小球世界单预测耗时（含抑制池），并统计淬火翻转数
import { PopChannelMap } from "./dist/src/pop/popmap.js";
import { PopRuleMemory } from "./dist/src/pop/popmemory.js";
import { R2PopLayer } from "./dist/src/pop/r2pop.js";
import { CONDITION_SPECS, OUTCOME_SPECS, curriculum, queries } from "./dist/src/prototype/world.js";

const cm = new PopChannelMap(CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const SPAN = { rebound: 1, reboundSpeed: 3 };
const mem = new PopRuleMemory(cm, om, { maxRules: 128 });
const r2 = new R2PopLayer(cm, om);
for (const group of curriculum()) {
  const magSum = new Map(); const magCount = new Map();
  for (const pair of group.pairs) {
    for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
    for (const spec of OUTCOME_SPECS) {
      const a = pair.e0.outcomes[spec.name]; const b = pair.e1.outcomes[spec.name];
      if (a !== b) mem.teachExclusion("outcome", spec.name, a, b, 3.0);
    }
    const analysis = r2.analyzePair(pair);
    for (const ch of analysis.influentialChannels) {
      let m = 0;
      for (const [och, delta] of Object.entries(analysis.outcomeDelta)) m += Math.abs(delta) / (SPAN[och] ?? 1);
      magSum.set(ch, (magSum.get(ch) ?? 0) + m / om.specs.length);
      magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
    }
  }
  const boost = {};
  for (const [ch, sum] of magSum) boost[ch] = 0.1 * 3 * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
  for (const pair of group.pairs) for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
}
console.log(`核 ${mem.ruleCount} 条，网络 ${mem.net.neuronCount} 神经元`);
const qs = queries();
let tAll = 0;
let worst = { ms: 0, i: -1 };
for (let i = 0; i < Math.min(12, qs.length); i++) {
  const t0 = performance.now();
  mem.predict(qs[i].conditions, 1);
  const ms = performance.now() - t0;
  tAll += ms;
  if (ms > worst.ms) worst = { ms, i };
}
console.log(`前 12 查询均时 ${(tAll / 12).toFixed(1)}ms，最慢 #${worst.i} ${worst.ms.toFixed(0)}ms`);
