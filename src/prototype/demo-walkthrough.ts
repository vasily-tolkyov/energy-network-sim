import { EnergyNetwork } from "../index.js";
import { ChannelMap } from "./channels.js";
import { extractInfluences } from "./r2.js";
import { RuleMemory } from "./r3.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  queries,
  groundTruth,
} from "./world.js";

/**
 * 走查演示：展示学习集实例、一次预测的完整过程、汇总结果。
 * 运行：npm run build && node dist/src/prototype/demo-walkthrough.js
 */

const CN: Record<string, string> = {
  color: "颜色", speed: "速度", direction: "方向", wallStiffness: "墙刚度",
  damping: "阻尼", hitWall: "碰墙", rebound: "反弹", reboundSpeed: "反弹速度档",
};
const fmt = (rec: Record<string, number>): string =>
  Object.entries(rec).map(([k, v]) => `${CN[k] ?? k}=${v}`).join(" ");

const cm = new ChannelMap(CONDITION_SPECS.map((s) => ({ ...s })));
const om = new ChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })));
const net = new EnergyNetwork({
  neuronCount: cm.neuronCount + om.neuronCount,
  activationEnergy: 1.0,
  maintenanceEnergy: 0.5,
  learningRate: 0.1,
  maxWeight: 3.0,
});
const mem = new RuleMemory(net, cm, om, { r1Repeats: 4, influenceGain: 3, r3Repeats: 4, gamma: 3.0 });
const OUTCOME_SPAN = { rebound: 1, reboundSpeed: 3 };

const SEP = "─".repeat(72);
const show = (t: string) => console.log(`\n${SEP}\n${t}\n${SEP}`);

// ── 1. 学习集 ────────────────────────────────────────────────────
show("① 学习集（控制变量课程，节选每组第一对实验）");
for (const g of curriculum()) {
  const p = g.pairs[0]!;
  console.log(`组「${g.name}」（操纵：${CN[g.manipulated]}）`);
  console.log(`  实验A: ${fmt(p.e0.conditions)} → ${fmt(p.e0.outcomes)}`);
  console.log(`  实验B: ${fmt(p.e1.conditions)} → ${fmt(p.e1.outcomes)}`);
  console.log(
    `  神经元编码：A 条件 [${cm.encode(p.e0.conditions)}] 结果 [${om.encode(p.e0.outcomes).map((i) => i + cm.neuronCount)}]`,
  );
}

// ── 2. 教学（R1 绑定 + R2A/R2B 比较 + R3 侧重）──────────────────
show("② 教学与影响因素提取");
for (const g of curriculum()) {
  for (const p of g.pairs) for (const e of [p.e0, p.e1]) mem.teachExperience(e, 4);
  const inf = extractInfluences(cm.channelNames(), g.pairs, OUTCOME_SPAN);
  const boost: Record<string, number> = {};
  for (const ch of inf.influential) boost[ch] = 0.1 * 3 * (inf.magnitude[ch] ?? 0) ** 2;
  for (const p of g.pairs) for (const e of [p.e0, p.e1]) mem.bindInfluence(e, boost, 4);
  console.log(
    `  ${g.name.padEnd(16)} 变化候选=[${inf.changeCandidates.map((c) => CN[c])}] → ` +
      `影响因素=[${inf.influential.map((c) => `${CN[c]}(m=${(inf.magnitude[c] ?? 0).toFixed(2)})`)}]`,
  );
}

// ── 3. 一次预测的完整过程 ────────────────────────────────────────
show("③ 预测过程实例：未见过颜色的小球（颜色=6 从未出现在课程中）");
const q = { color: 6, direction: 0, damping: 0, hitWall: 1, speed: 1, wallStiffness: 1 };
const truth = groundTruth(q);
console.log(`查询条件：${fmt(q)}`);
console.log(`地面真理（教师私有）：${fmt(truth)}`);
const input = cm.encode(q);
console.log(`钳制输入神经元：[${input}]`);
{
  const bits = new Uint8Array(27);
  for (const i of input) bits[i] = 1;
  console.log("各结果档位的吸引场（学习形成，含影响因素侧重）：");
  for (const spec of om.specs) {
    const row = [];
    for (let b = 0; b < spec.bins; b++) {
      row.push(`档${b}:${net.localField(cm.neuronCount + spec.offset + b, bits).toFixed(2)}`);
    }
    console.log(`  ${CN[spec.name]}：${row.join("  ")}`);
  }
}
const r = net.settleAnnealed(input, [], { seed: 1, extraCandidates: mem.outcomeNeurons, quenchCandidatesOnly: true });
console.log(`局部退火：初始 E=${r.initialEnergy.toFixed(2)} → 退火末 E=${r.annealEndEnergy.toFixed(2)} → 淬火后 E=${r.energy.toFixed(2)}`);
console.log(`翻转提议 ${r.proposals} 次（候选集 = 6 个结果神经元），上坡接受 ${r.acceptedUphill} 次`);
console.log(`最终激活神经元：[${r.activeNeurons}]`);
const { decoded } = mem.predict(q, 1);
console.log(`读出解码：${JSON.stringify(decoded)} → 预测「${CN.rebound}=${decoded.rebound}，${CN.reboundSpeed}=${decoded.reboundSpeed}」`);
console.log(`判定：${decoded.rebound === truth.rebound && decoded.reboundSpeed === truth.reboundSpeed ? "✓ 正确" : "✗ 错误"}`);

// ── 4. 诚实展示一个失败例 ────────────────────────────────────────
show("④ 失败例（高速角，细粒度档偏低 1-2 档的已知残余失败）");
const qBad = { color: 0, direction: 0, damping: 0, hitWall: 1, speed: 3, wallStiffness: 2 };
const bad = mem.predict(qBad, 1).decoded;
console.log(`查询：${fmt(qBad)}`);
console.log(`预测：${fmt({ rebound: bad.rebound as number, reboundSpeed: bad.reboundSpeed as number })}，真值：${fmt(groundTruth(qBad))}`);

// ── 5. 汇总 ──────────────────────────────────────────────────────
show("⑤ 汇总（本演示网络，48 查询）");
let correct = 0;
let coarse = 0;
const all = queries();
for (const qq of all) {
  const d = mem.predict(qq.conditions, 1).decoded;
  if (d.rebound === qq.truth.rebound) coarse++;
  if (d.rebound === qq.truth.rebound && (qq.truth.rebound === 0 || d.reboundSpeed === qq.truth.reboundSpeed)) correct++;
}
console.log(`细粒度（反弹+速度档）：${correct}/${all.length}（${((correct / all.length) * 100).toFixed(1)}%）`);
console.log(`粗粒度（反弹有无）：${coarse}/${all.length}（${((coarse / all.length) * 100).toFixed(1)}%）`);
console.log(`（多配置×多种子的完整结果见 npm run proto / runs/proto-v1.log）`);
