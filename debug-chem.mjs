// 调试：重现 v2 课程 + 查询学习（seed 1），逐查询 dump 核块与结果群体激活
import { PopChannelMap } from "./dist/src/pop/popmap.js";
import { PopRuleMemory } from "./dist/src/pop/popmemory.js";
import { R2PopLayer } from "./dist/src/pop/r2pop.js";
import { CHEM_CONDITION_SPECS, CHEM_OUTCOME_SPECS, chemCurriculum, chemQueries } from "./dist/src/topics/chem-world.js";

const cm = new PopChannelMap(CHEM_CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(CHEM_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const CORE_BASE = cm.neuronCount + om.neuronCount;
const OUTCOME_SPAN = { reacted: 1, rate: 3 };

const mem = new PopRuleMemory(cm, om, { maxRules: 128 });
const r2 = new R2PopLayer(cm, om);
const curriculum = chemCurriculum(true);
for (const group of curriculum) {
  const magSum = new Map();
  const magCount = new Map();
  for (const pair of group.pairs) {
    for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
    for (const spec of CHEM_OUTCOME_SPECS) {
      const a = pair.e0.outcomes[spec.name];
      const b = pair.e1.outcomes[spec.name];
      if (a !== b) mem.teachExclusion("outcome", spec.name, a, b, 3.0);
    }
    const analysis = r2.analyzePair(pair);
    for (const ch of analysis.influentialChannels) {
      let m = 0;
      for (const [och, delta] of Object.entries(analysis.outcomeDelta)) m += Math.abs(delta) / (OUTCOME_SPAN[och] ?? 1);
      magSum.set(ch, (magSum.get(ch) ?? 0) + m / om.specs.length);
      magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
    }
  }
  const boost = {};
  for (const [ch, sum] of magSum) boost[ch] = 0.1 * 3 * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
  for (const pair of group.pairs) for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
}

const queries = chemQueries(curriculum);
let i = 0;
for (const q of queries) {
  i++;
  const { decoded, activeNeurons } = mem.predict(q.conditions, 1);
  // 核块激活计数
  const blockCount = new Map();
  for (const id of activeNeurons) {
    if (id >= CORE_BASE) {
      const b = Math.floor((id - CORE_BASE) / 4);
      blockCount.set(b, (blockCount.get(b) ?? 0) + 1);
    }
  }
  // 结果群体激活计数（reacted: 0-7, rate: 8-23，每档 4 个）
  const outCount = [];
  for (let k = 0; k < om.neuronCount; k++) {
    if (activeNeurons.includes(cm.neuronCount + k)) outCount.push(k);
  }
  const blocks = [...blockCount.entries()].map(([b, n]) => `#${b}:${n}/4`).join(" ");
  const wrong = decoded.reacted !== q.truth.reacted || (q.truth.reacted === 1 && decoded.rate !== q.truth.rate);
  if (i <= 8 || q.kind === "seen-combo" || (wrong && i <= 20)) {
    console.log(
      `q${i} [${q.kind}] cat${q.conditions.catalyst} conc${q.conditions.concentration} temp${q.conditions.temperature} ves${q.conditions.vessel}` +
        ` → ${JSON.stringify(decoded)} 真值=${JSON.stringify(q.truth)}`,
    );
    console.log(`    核块: ${blocks || "(无)"}  结果神经元: [${outCount.join(",")}]`);
  }
  mem.learnFromQuery(q.conditions, decoded, 1);
}
console.log(`总核数 ${mem.ruleCount}`);
