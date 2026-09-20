import { test } from "node:test";
import assert from "node:assert/strict";
import { SensoryEncoder } from "../src/pop/concept/sensory.js";
import { EmergentMap } from "../src/pop/concept/emergent-map.js";
import { FieldRuleMemory } from "../src/pop/concept/field-memory.js";

/** 改判计票（与成核同一赫布原则：重复才构成规则）：
 * 单次异常不改判、不污染读出；重复确认才翻转；翻转可逆。 */
function mem() {
  const enc = new SensoryEncoder([{ name: "x", min: 0, max: 1 }, { name: "y", min: 0, max: 1 }]);
  const m = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 8 });
  m.setOutcomeDimensions(["y"]);
  return m;
}
const read = (m: FieldRuleMemory) => m.predict({ x: 0.2 }, 1).values.y;

test("单次异常不改判已确立的规则，也不造成读出歧义", () => {
  const m = mem();
  for (let i = 0; i < 4; i++) m.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 2);
  m.learnFromObservation({ x: 0.2 }, { y: 0.8 }, 2); // 单次异常（如外力扰动）
  for (let s = 1; s <= 5; s++) {
    const p = m.predict({ x: 0.2 }, s);
    assert.ok(Math.abs(p.values.y! - 0.2) < 0.03, `seed ${s}: ${p.values.y}`);
    assert.equal(p.ambiguous.length, 0);
  }
});

test("同一异常重复到定额才翻转（世界真实变化可学会）", () => {
  const m = mem();
  for (let i = 0; i < 4; i++) m.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 2);
  m.learnFromObservation({ x: 0.2 }, { y: 0.8 }, 2);
  assert.ok(Math.abs(read(m)! - 0.2) < 0.03, "1 次不应翻转");
  m.learnFromObservation({ x: 0.2 }, { y: 0.8 }, 2);
  assert.ok(Math.abs(read(m)! - 0.2) < 0.03, "2 次不应翻转（quorum=3）");
  m.learnFromObservation({ x: 0.2 }, { y: 0.8 }, 2);
  assert.ok(Math.abs(read(m)! - 0.8) < 0.03, "3 次达到定额，应翻转");
});

test("翻转是双向的：世界反转回来后旧值按票翻回", () => {
  const m = mem();
  for (let i = 0; i < 2; i++) m.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 2);
  for (let i = 0; i < 2; i++) m.learnFromObservation({ x: 0.2 }, { y: 0.8 }, 2);
  assert.ok(Math.abs(read(m)! - 0.8) < 0.03, "平票新者胜");
  m.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 2);
  assert.ok(Math.abs(read(m)! - 0.2) < 0.03, "旧值票数仍在，一次即翻回");
});

test("来源优先级不走计票：单次真观察立即纠正错误猜测", () => {
  const m = mem();
  m.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 2);
  // 假设无权否决观察（第二轮契约）
  for (let s = 1; s <= 3; s++) assert.ok(Math.abs(m.predict({ x: 0.2 }, s).values.y! - 0.2) < 0.03);
});

test("计数为 1 的规则是临时的：平票挑战者可翻转", () => {
  const m = mem();
  m.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 2);
  m.learnFromObservation({ x: 0.2 }, { y: 0.8 }, 2);
  assert.ok(Math.abs(read(m)! - 0.8) < 0.03);
});
