import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork } from "../src/index.js";
import { ChannelMap } from "../src/prototype/channels.js";
import { extractInfluences } from "../src/prototype/r2.js";
import { RuleMemory } from "../src/prototype/r3.js";
import {
  CAP_CONDITION_SPECS,
  CAP_OUTCOME_SPECS,
  capCurriculum,
  capQueries,
  capTruth,
  sampleCombos,
} from "../src/topics/capacity-world.js";

// 容量结论锁定测试（mod-8 对抗性真理结构，seed=1，与 runner-capacity 同构）

const cm = new ChannelMap(CAP_CONDITION_SPECS.map((s) => ({ ...s })));
const om = new ChannelMap(CAP_OUTCOME_SPECS.map((s) => ({ ...s })));
const SPAN = { out: 7 };

function teachAndScore(E: number, seed: number): { taughtAcc: number } {
  const combos = sampleCombos(E, seed);
  const net = new EnergyNetwork({
    neuronCount: cm.neuronCount + om.neuronCount,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const mem = new RuleMemory(net, cm, om, { r1Repeats: 4, influenceGain: 3, r3Repeats: 4, gamma: 3.0 });
  for (const group of capCurriculum(combos, capTruth)) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
    }
    const influence = extractInfluences(cm.channelNames(), group.pairs, SPAN);
    const boost: Record<string, number> = {};
    for (const ch of influence.influential) boost[ch] = 0.1 * 3 * (influence.magnitude[ch] ?? 0) ** 2;
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  const taughtSet = new Set(combos.map(([a, b]) => `${a},${b}`));
  let correct = 0;
  let total = 0;
  for (const q of capQueries(taughtSet, capTruth)) {
    if (q.kind !== "taught") continue;
    const { decoded } = mem.predict(q.conditions, seed);
    total++;
    if (decoded.out === q.truth.out) correct++;
  }
  return { taughtAcc: correct / total };
}

test("容量：E=8 时教学规则高准确率（≥70%），E=64 时显著退化（≤30%），单调劣化", () => {
  const acc8 = teachAndScore(8, 1).taughtAcc;
  const acc64 = teachAndScore(64, 1).taughtAcc;
  assert.ok(acc8 >= 0.7, `E=8 taught accuracy ${(acc8 * 100).toFixed(1)}% < 70%`);
  assert.ok(acc64 <= 0.3, `E=64 taught accuracy ${(acc64 * 100).toFixed(1)}% > 30%`);
  assert.ok(acc8 > acc64, "capacity must degrade with rule count");
});

test("容量：mod-8 真理下留出查询不可插值（heldout ≈ 0）", () => {
  const combos = sampleCombos(16, 1);
  const net = new EnergyNetwork({
    neuronCount: cm.neuronCount + om.neuronCount,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const mem = new RuleMemory(net, cm, om, { r1Repeats: 4, influenceGain: 3, r3Repeats: 4, gamma: 3.0 });
  for (const group of capCurriculum(combos, capTruth)) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
    }
    const influence = extractInfluences(cm.channelNames(), group.pairs, SPAN);
    const boost: Record<string, number> = {};
    for (const ch of influence.influential) boost[ch] = 0.1 * 3 * (influence.magnitude[ch] ?? 0) ** 2;
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  const taughtSet = new Set(combos.map(([a, b]) => `${a},${b}`));
  let heldCorrect = 0;
  let heldTotal = 0;
  for (const q of capQueries(taughtSet, capTruth)) {
    if (q.kind !== "heldout") continue;
    const { decoded } = mem.predict(q.conditions, 1);
    heldTotal++;
    if (decoded.out === q.truth.out) heldCorrect++;
  }
  // 实测 8% 左右（随机基线 1/8 = 12.5% 附近），不允许超过 25%
  assert.ok(heldCorrect / heldTotal <= 0.25, `heldout ${((heldCorrect / heldTotal) * 100).toFixed(1)}% too high for non-interpolable truth`);
});
