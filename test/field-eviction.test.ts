import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, hebbianLearn } from "../src/index.js";
import { SensoryEncoder } from "../src/pop/concept/sensory.js";
import { EmergentMap } from "../src/pop/concept/emergent-map.js";
import { FieldRuleMemory } from "../src/pop/concept/field-memory.js";

function fieldMem(maxRules: number, eviction: "throw" | "lru" = "lru") {
  const enc = new SensoryEncoder([{ name: "x", min: 0, max: 1 }, { name: "y", min: 0, max: 1 }]);
  const mem = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules, eviction });
  mem.setOutcomeDimensions(["y"]);
  return mem;
}
const reads = (m: FieldRuleMemory, x: number) =>
  Array.from({ length: 5 }, (_, i) => m.predict({ x }, i + 1).values.y);

test("clearSynapses zeroes all four matrices and drops inhibition ownership", () => {
  const net = new EnergyNetwork({ neuronCount: 4, activationEnergy: 1, maintenanceEnergy: 0.5, maxWeight: 8 });
  net.strengthen(0, 1, 2);
  net.strengthenInhibitory(0, 2, 1.5);
  net.strengthenDirectedInhibitory(2, 1, 3, 4);
  net.setInhibitionContribution(1, 3, "test:edge", 2);
  net.clearSynapses([1]);
  assert.equal(net.getWeight(0, 1), 0);
  assert.equal(net.getInhibitoryWeight(0, 2), 1.5); // 未涉及 1 的边不动
  assert.equal(net.getDirectedInhibitoryWeight(2, 1), 0);
  // 归属账被丢弃后，撤回旧贡献不得复活任何抑制
  net.setInhibitionContribution(1, 3, "test:edge", 0);
  assert.equal(net.getInhibitoryWeight(1, 3), 0);
  assert.throws(() => net.clearSynapses([4]));
  assert.throws(() => net.clearSynapses([0.5]));
});

test("eviction=lru recycles the least recently used rule without debris", () => {
  const mem = fieldMem(2);
  mem.learnFromObservation({ x: 0.1 }, { y: 0.2 }, 6);
  mem.learnFromObservation({ x: 0.9 }, { y: 0.8 }, 6);
  assert.equal(mem.ruleCount, 2);
  // 使用 x=0.1 的规则（预测获胜即使用），x=0.9 的规则成为最久未用
  assert.ok(reads(mem, 0.1).every(v => typeof v === "number" && Math.abs(v - 0.2) < 0.03));
  // 第三条经验触发淘汰：受害者应为 x=0.9 的规则
  mem.learnFromObservation({ x: 0.5 }, { y: 0.5 }, 6);
  assert.equal(mem.ruleCount, 2);
  // 幸存规则读回不变（无残骸干扰）
  assert.ok(reads(mem, 0.1).every(v => typeof v === "number" && Math.abs(v - 0.2) < 0.03));
  assert.ok(reads(mem, 0.5).every(v => typeof v === "number" && Math.abs(v - 0.5) < 0.03));
  // 被淘汰的规则已遗忘：读空或读新主，不得读出旧值 0.8
  const stale = reads(mem, 0.9);
  assert.ok(stale.every(v => v == null || Math.abs(v - 0.8) > 0.03), `stale readback: ${stale}`);
});

test("eviction keeps mutual exclusion: recycled core still wins alone", () => {
  const mem = fieldMem(2);
  mem.learnFromObservation({ x: 0.1 }, { y: 0.2 }, 6);
  mem.learnFromObservation({ x: 0.9 }, { y: 0.8 }, 6);
  mem.learnFromObservation({ x: 0.5 }, { y: 0.5 }, 6); // 淘汰一条
  mem.learnFromObservation({ x: 0.4 }, { y: 0.4 }, 6); // 再淘汰一条
  assert.equal(mem.ruleCount, 2);
  for (const x of [0.4, 0.5]) {
    for (let s = 1; s <= 5; s++) {
      const p = mem.predict({ x }, s);
      assert.ok(p.winningCores.length <= 1, `x=${x} seed=${s} multi-winner: ${p.winningCores}`);
    }
  }
});

test("default eviction=throw keeps the review capacity contract", () => {
  const mem = fieldMem(1, "throw");
  mem.learnFromObservation({ x: 0.1 }, { y: 0.2 }, 2);
  assert.throws(() => mem.learnFromObservation({ x: 0.9 }, { y: 0.8 }, 2), /capacity/);
  assert.equal(mem.ruleCount, 1);
});

test("evidence of surviving rules is untouched by a neighbour's eviction", () => {
  const mem = fieldMem(2);
  mem.learnFromObservation({ x: 0.1 }, { y: 0.2 }, 6);
  mem.learnFromObservation({ x: 0.9 }, { y: 0.8 }, 6);
  const before = [0, 1].map(i => JSON.stringify(mem.ruleEvidence(i)));
  mem.learnFromObservation({ x: 0.5 }, { y: 0.5 }, 6); // 淘汰最久未用（x=0.1，无预测使用）
  // 幸存者（x=0.9）证据逐字节不变
  const survivors = [0, 1].map(i => mem.ruleEvidence(i));
  const kept = survivors.find(e => JSON.stringify(e) === before[1]);
  assert.ok(kept, "surviving rule evidence changed after eviction");
});
