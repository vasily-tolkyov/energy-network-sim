import { test } from "node:test";
import assert from "node:assert/strict";
import { NeuralFocusNet } from "../src/pop/attention/neural-focus.js";
import { pretrainContinuous } from "../src/pop/concept/pretrain.js";
import { captureClassify } from "../src/pop/attention/capture.js";
import { mismatchField } from "../src/pop/attention/mismatch.js";
import { contTruth, type CValues } from "../src/pop/concept/world-continuous.js";

// 神经化注意力（捕获判定 + 失配场 WTA 择焦）锁定测试

test("WTA 择焦：失配大者胜、惯性持焦（失配随帧衰减）、失配更大者换焦", () => {
  const f = new NeuralFocusNet(["A", "B", "C"], { inertia: 2.0 });
  f.beginFrame();
  f.setMismatch("A", 8);
  f.setMismatch("B", 3);
  assert.equal(f.select(1), "A");
  // 失配随帧清零：A 新失配 5 + 惯性 2 = 7 > B 的 6 → 持焦
  f.beginFrame();
  f.setMismatch("A", 5);
  f.setMismatch("B", 6);
  assert.equal(f.select(2), "A");
  // B=9 > 5+2 → 换焦
  f.beginFrame();
  f.setMismatch("B", 9);
  assert.equal(f.select(3), "B");
});

test("WTA 择焦：失配场是事件信号，帧清零后旧失配不再占优", () => {
  const f = new NeuralFocusNet(["A", "B"], { inertia: 2.0 });
  f.beginFrame();
  f.setMismatch("A", 10);
  assert.equal(f.select(1), "A");
  f.beginFrame(); // 清零
  f.setMismatch("B", 3);
  // A 清零后只剩惯性 2 < B 的 3 → 换焦
  assert.equal(f.select(2), "B");
});

test("捕获判定三态：符合 / 偏差 / 未知（含无预测时报未知不报偏差）", () => {
  const { mem } = pretrainContinuous();
  const cond: CValues = { color: 0.5, direction: 0.2, damping: 0.2, hitWall: 1, speed: 1.5, wallStiffness: 1.0 };
  const fc = mem.predict(cond, 1);
  const fcCore = fc.winningCores.length > 0 ? fc.winningCores[0]! : null;
  assert.notEqual(fcCore, null);
  // 符合：观察 = 预测结果
  const good = captureClassify(mem, cond, fc.winningCores.length > 0 ? contTruth(cond) : contTruth(cond), fcCore);
  assert.equal(good.class, "within-envelope");
  // 偏差：观察与预测核答案冲突（碰墙翻转）
  const bad = captureClassify(mem, { ...cond, hitWall: 0 }, contTruth({ ...cond, hitWall: 0 }), fcCore);
  assert.equal(bad.class, "prediction-violation");
  // 未知：无预测核
  const none = captureClassify(mem, cond, contTruth(cond), null);
  assert.equal(none.class, "unknown-change");
  // 无预测核时，即使观察与某规则吻合，也只能报未知，不能报偏差
  const none2 = captureClassify(mem, { ...cond, hitWall: 0 }, contTruth({ ...cond, hitWall: 0 }), null);
  assert.equal(none2.class, "unknown-change");
});

test("失配场：符合≈0，违反大，未知最大", () => {
  const { mem } = pretrainContinuous();
  const cond: CValues = { color: 0.5, direction: 0.2, damping: 0.2, hitWall: 1, speed: 1.5, wallStiffness: 1.0 };
  const fc = mem.predict(cond, 1);
  const fcCore = fc.winningCores[0]!;
  const good = mismatchField(mem, fcCore, contTruth(cond));
  const bad = mismatchField(mem, fcCore, contTruth({ ...cond, hitWall: 0 }));
  const none = mismatchField(mem, null, contTruth(cond));
  assert.ok(good < bad, `good ${good.toFixed(2)} < bad ${bad.toFixed(2)}`);
  assert.ok(bad > 0);
  assert.ok(none >= bad, "unknown should be maximal mismatch");
});
