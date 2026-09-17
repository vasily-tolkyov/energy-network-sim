import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork } from "../src/index.js";
import { ChannelMap } from "../src/prototype/channels.js";
import { RuleMemory } from "../src/prototype/r3.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  queries,
} from "../src/prototype/world.js";
import { R1Layer } from "../src/layers/r1.js";
import { R2Layer } from "../src/layers/r2.js";

// 三层架构（R1/R2/R3 各自独立成网）端到端锁定测试。

const cm = new ChannelMap(CONDITION_SPECS.map((s) => ({ ...s })));
const om = new ChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })));

test("R1：每次实验形成条件阱→结果阱的有向通道（单向），部分条件线索补全结果", () => {
  const r1 = new R1Layer(cm, om);
  const groups = curriculum();
  const pair = groups[1]!.pairs[0]!; // speed-group 第一对
  const ep = r1.teachEpisode(pair.e0, 8, 8);
  const cond = ep.conditionNeurons;
  const outIds = ep.outcomeNeurons;
  // 有向通道：条件→结果 > 0，结果→条件 = 0
  assert.ok(r1.net.getDirectedWeight(cond[0]!, outIds[0]!) > 0);
  assert.equal(r1.net.getDirectedWeight(outIds[0]!, cond[0]!), 0);
  // 部分条件线索 → settle 捕获并补全结果神经元（对称 W 承载关联）
  const r = r1.net.settle(cond.slice(0, 3));
  const active = new Set(r.activeNeurons);
  for (const id of outIds) assert.ok(active.has(id), `outcome neuron ${id} should be recruited`);
});

test("R2：神经差分提取正确的影响因素（双源 AND：结果无变化则无影响）", () => {
  const r2 = new R2Layer(cm);
  const byName = Object.fromEntries(curriculum().map((g) => [g.name, g]));
  for (const [name, g] of Object.entries(byName)) {
    for (const pair of g.pairs) {
      const a = r2.analyzePair(pair);
      // 差分候选通道恒等于被操纵通道（每对只变一个条件）
      assert.deepEqual(a.diffChannels, [g.manipulated], `${name} diff`);
      // 影响因素 = 差分 ∩ 结果变化；该课程中 stiffness-group 第二对 (0↔1) 结果档恰好相同，
      // 属退化比较，按文档语义本组背景下不得判定影响
      const expected = a.outcome.outcomeChanged ? [g.manipulated] : [];
      assert.deepEqual(a.influentialChannels, expected, `${name} influential`);
    }
  }
});

function runPipeline(gain: number, learnFromQueries: boolean): { fine: number; coarse: number; missWall: number } {
  const r1 = new R1Layer(cm, om);
  const r2 = new R2Layer(cm);
  const r3Net = new EnergyNetwork({
    neuronCount: cm.neuronCount + om.neuronCount,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const r3 = new RuleMemory(r3Net, cm, om, { r1Repeats: 4, influenceGain: gain, r3Repeats: 4, gamma: 3.0 });
  const SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 3 };
  for (const group of curriculum()) {
    const magSum = new Map<string, number>();
    const magCount = new Map<string, number>();
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) {
        r1.teachEpisode(e, 4, 8);
        r3.teachExperience(e, 4);
      }
      const analysis = r2.analyzePair(pair);
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        let counted = 0;
        for (const [och, od] of Object.entries(analysis.outcome.diffs)) {
          m += Math.abs(od.delta) / (SPAN[och] ?? 1);
          counted++;
        }
        magSum.set(ch, (magSum.get(ch) ?? 0) + m / counted);
        magCount.set(ch, (magCount.get(ch) ?? 1) + 1);
      }
    }
    const boost: Record<string, number> = {};
    for (const [ch, sum] of magSum) {
      boost[ch] = 0.1 * gain * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
    }
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) r3.bindInfluence(e, boost, 4);
    }
  }
  let correct = 0;
  let coarse = 0;
  let missCorrect = 0;
  let missTotal = 0;
  const all = queries();
  for (const q of all) {
    const { decoded } = r3.predict(q.conditions, 1);
    if (learnFromQueries) r3.learnFromQuery(q.conditions, decoded, 1);
    if (decoded.rebound === q.truth.rebound) coarse++;
    if (decoded.rebound === q.truth.rebound && (q.truth.rebound === 0 || decoded.reboundSpeed === q.truth.reboundSpeed)) correct++;
    if (q.kind === "miss-wall") {
      missTotal++;
      if (decoded.rebound === q.truth.rebound) missCorrect++;
    }
  }
  return { fine: correct / all.length, coarse: coarse / all.length, missWall: missCorrect / missTotal };
}

test("三层端到端：细粒度 ≥ 80%，粗粒度与错过墙门控 100%（与单网原型同水平）", () => {
  const r = runPipeline(3, true);
  assert.ok(r.fine >= 0.8, `fine ${(r.fine * 100).toFixed(1)}% < 80%`);
  assert.equal(r.coarse, 1);
  assert.equal(r.missWall, 1);
});

test("三层消融：G=0 断侧重后显著退化", () => {
  const full = runPipeline(3, true);
  const ablated = runPipeline(0, true);
  assert.ok(ablated.fine < full.fine * 0.6, `ablation ${(ablated.fine * 100).toFixed(1)}% should be much worse`);
});
