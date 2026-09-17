import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork } from "../src/index.js";
import { ChannelMap } from "../src/prototype/channels.js";
import { compareOutcomes, extractInfluences, diffValues } from "../src/prototype/r2.js";
import { RuleMemory } from "../src/prototype/r3.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  queries,
  groundTruth,
  type Query,
} from "../src/prototype/world.js";

// R1/R2A/R2B/R3 预测原型的端到端锁定测试。
// 参数与 runner 一致；完整实验报告见 runs/proto-v1.log。

const cm = new ChannelMap(CONDITION_SPECS.map((s) => ({ ...s })));
const om = new ChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })));
const OUTCOME_SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 3 };

function teachAll(gain: number): RuleMemory {
  const net = new EnergyNetwork({
    neuronCount: cm.neuronCount + om.neuronCount,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const mem = new RuleMemory(net, cm, om, { r1Repeats: 4, influenceGain: gain, r3Repeats: 4, gamma: 3.0 });
  for (const group of curriculum()) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
    }
    const influence = extractInfluences(cm.channelNames(), group.pairs, OUTCOME_SPAN);
    const boost: Record<string, number> = {};
    for (const ch of influence.influential) {
      const m = influence.magnitude[ch] ?? 0;
      boost[ch] = 0.1 * gain * m * m;
    }
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  return mem;
}

test("R2A/R2B：变化标志与变化数值分离，控制变量提取影响因素", () => {
  // 从 1 变到 0 仍是变化（文档 §4 明确要求）
  const d = diffValues({ a: 1, b: 2 }, { a: 0, b: 2 });
  assert.equal(d.a!.changed, true);
  assert.equal(d.a!.delta, -1);
  assert.equal(d.b!.changed, false);
  // 结果共同部分保留
  const oc = compareOutcomes({ rebound: 1, reboundSpeed: 2 }, { rebound: 1, reboundSpeed: 3 });
  assert.equal(oc.outcomeChanged, true);
  assert.deepEqual(oc.common, { rebound: 1 });
  // 真实课程：颜色/阻尼/方向不判为影响因素，速度/刚度/碰墙判为影响因素
  const groups = curriculum();
  const byName = Object.fromEntries(groups.map((g) => [g.name, g]));
  for (const [name, expected] of [
    ["color-group", []],
    ["speed-group", ["speed"]],
    ["stiffness-group", ["wallStiffness"]],
    ["hitwall-group", ["hitWall"]],
    ["damping-group", []],
    ["direction-group", []],
  ] as const) {
    const inf = extractInfluences(cm.channelNames(), byName[name]!.pairs, OUTCOME_SPAN);
    assert.deepEqual(inf.influential, [...expected], name);
    // 影响幅度：门控因素应显著大于线性因素
    if (name === "hitwall-group") assert.ok((inf.magnitude.hitWall ?? 0) > 0.5);
  }
});

test("端到端：教学后粗粒度反弹预测 100%（含未见颜色），错过墙门控 100%", () => {
  const mem = teachAll(3);
  for (const q of queries()) {
    const { decoded } = mem.predict(q.conditions, 1);
    assert.equal(decoded.rebound, q.truth.rebound, `rebound mispredicted: ${JSON.stringify(q.conditions)}`);
  }
});

test("端到端：细粒度（含反弹速度档）准确率 ≥ 70%，且 G=0 消融显著更差", () => {
  const all: Query[] = queries();
  const run = (gain: number): number => {
    const mem = teachAll(gain);
    let correct = 0;
    for (const q of all) {
      const { decoded } = mem.predict(q.conditions, 1);
      const rsOk = q.truth.rebound === 0 || decoded.reboundSpeed === q.truth.reboundSpeed;
      if (decoded.rebound === q.truth.rebound && rsOk) correct++;
    }
    return correct / all.length;
  };
  const full = run(3);
  const ablated = run(0);
  assert.ok(full >= 0.7, `full model fine-grained accuracy ${(full * 100).toFixed(1)}% < 70%`);
  assert.ok(ablated < full, `G=0 ablation (${(ablated * 100).toFixed(1)}%) should be worse than full (${(full * 100).toFixed(1)}%)`);
});

test("读出消融：纯贪心读出产生歧义，退火竞争读出收敛单档", () => {
  const mem = teachAll(3);
  const q = { color: 0, direction: 0, damping: 0, hitWall: 1, speed: 1, wallStiffness: 1 };
  const greedy = mem.predictGreedy(q).decoded;
  assert.ok(
    greedy.reboundSpeed === "ambiguous" || greedy.rebound === "ambiguous",
    "greedy readout should leave channel competition unresolved",
  );
  const annealed = mem.predict(q, 1).decoded;
  assert.equal(typeof annealed.rebound, "number");
  assert.equal(typeof annealed.reboundSpeed, "number");
});

test("编码：未见颜色的神经元不参与任何已学连接（规则不因颜色丢失适用性）", () => {
  const mem = teachAll(3);
  const color6 = cm.neuron("color", 6);
  // color6 从未出现在课程中：与任何结果神经元的连接应为 0
  for (const outId of mem.outcomeNeurons) {
    assert.equal(mem.net.getWeight(color6, outId), 0);
  }
  // 但未见颜色 × 已学运动条件仍能预测反弹（§7.1 文档场景）
  const { decoded } = mem.predict({ color: 6, direction: 0, damping: 0, hitWall: 1, speed: 1, wallStiffness: 1 }, 1);
  assert.equal(decoded.rebound, groundTruth({ color: 6, direction: 0, damping: 0, hitWall: 1, speed: 1, wallStiffness: 1 }).rebound);
});
