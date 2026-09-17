import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork } from "../src/index.js";
import { ChannelMap } from "../src/prototype/channels.js";
import { extractInfluences } from "../src/prototype/r2.js";
import { RuleMemory } from "../src/prototype/r3.js";
import {
  PLANT_CONDITION_SPECS,
  PLANT_OUTCOME_SPECS,
  plantCurriculum,
  plantQueries,
  plantTruth,
} from "../src/topics/plant-world.js";

// 第二主题（植物生长）迁移测试：原型机制与参数完全冻结，
// 只换领域数据（双因素交互 + 肥料/温度加成 + 虫害门控 + 品种干扰）。

const cm = new ChannelMap(PLANT_CONDITION_SPECS.map((s) => ({ ...s })));
const om = new ChannelMap(PLANT_OUTCOME_SPECS.map((s) => ({ ...s })));
const SPAN: Record<string, number> = { alive: 1, growth: 3 };

function teachAll(gain: number): RuleMemory {
  const net = new EnergyNetwork({
    neuronCount: cm.neuronCount + om.neuronCount,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const mem = new RuleMemory(net, cm, om, { r1Repeats: 4, influenceGain: gain, r3Repeats: 4, gamma: 3.0 });
  for (const group of plantCurriculum()) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
    }
    const influence = extractInfluences(cm.channelNames(), group.pairs, SPAN);
    const boost: Record<string, number> = {};
    for (const ch of influence.influential) boost[ch] = 0.1 * gain * (influence.magnitude[ch] ?? 0) ** 2;
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  return mem;
}

test("新主题影响因素提取：品种/干扰不判影响，光照水量肥料温度判影响，虫害幅度最大", () => {
  const byName = Object.fromEntries(plantCurriculum().map((g) => [g.name, g]));
  const expectations: Record<string, readonly string[]> = {
    "species-group": [],
    "light-group": ["light"],
    "water-group": ["water"],
    "fertilizer-group": ["fertilizer"],
    "temperature-group": ["temperature"],
    "pest-group": ["pest"],
  };
  for (const [name, expected] of Object.entries(expectations)) {
    const inf = extractInfluences(cm.channelNames(), byName[name]!.pairs, SPAN);
    assert.deepEqual(inf.influential, [...expected], name);
  }
  const pestInf = extractInfluences(cm.channelNames(), byName["pest-group"]!.pairs, SPAN);
  const lightInf = extractInfluences(cm.channelNames(), byName["light-group"]!.pairs, SPAN);
  assert.ok((pestInf.magnitude.pest ?? 0) > (lightInf.magnitude.light ?? 0));
});

test("新主题端到端：门控（虫害对照）100%，粗粒度存活 100%，细粒度 ≥ 65%", () => {
  const mem = teachAll(3);
  let correct = 0;
  let coarse = 0;
  const all = plantQueries();
  for (const q of all) {
    const { decoded } = mem.predict(q.conditions, 1);
    if (decoded.alive === q.truth.alive) coarse++;
    if (decoded.alive === q.truth.alive && (q.truth.alive === 0 || decoded.growth === q.truth.growth)) correct++;
  }
  assert.equal(coarse, all.length, "coarse alive prediction must be 100%");
  assert.ok(correct / all.length >= 0.65, `fine accuracy ${((correct / all.length) * 100).toFixed(1)}% < 65%`);
});

test("新主题消融：G=0 时虫害门控显著退化（完整模型 100%，断侧重 ≤70%）", () => {
  const pestQueries = plantQueries().filter((q) => q.kind === "pest-control");
  const rate = (gain: number): number => {
    let ok = 0;
    let total = 0;
    for (let seed = 1; seed <= 3; seed++) {
      const mem = teachAll(gain);
      for (const q of pestQueries) {
        const { decoded } = mem.predict(q.conditions, seed);
        if (decoded.alive === q.truth.alive) ok++;
        total++;
      }
    }
    return ok / total;
  };
  const full = rate(3);
  const ablated = rate(0);
  assert.equal(full, 1, "完整模型虫害门控应 100%");
  // 实测 G=0 约 56%（部分靠原始匹配数猜中；runner 含查询学习时进一步恶化到 11.5%）
  assert.ok(ablated <= 0.7, `G=0 门控应显著退化，实得 ${(ablated * 100).toFixed(1)}%`);
});

test("未见品种 × 已学生长条件仍可预测存活（干扰因素不丢规则适用性）", () => {
  const mem = teachAll(3);
  const q = { species: 3, light: 1, water: 1, fertilizer: 1, temperature: 1, pest: 0 };
  const { decoded } = mem.predict(q, 1);
  assert.equal(decoded.alive, plantTruth(q).alive);
});
