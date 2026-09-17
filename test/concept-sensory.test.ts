import { test } from "node:test";
import assert from "node:assert/strict";
import { SensoryEncoder, iou } from "../src/pop/concept/sensory.js";

// M1：S0 感受野编码层——相似性保持是"相似输入成阱"的引擎前提

test("编码：每个值激活约 6-8 个感受野神经元，量程外夹到边界", () => {
  const enc = new SensoryEncoder([{ name: "speed", min: 0, max: 8 }], 40);
  assert.equal(enc.neuronCount, 40);
  const mid = enc.encodeDimension("speed", 4);
  assert.ok(mid.length >= 5 && mid.length <= 10, `mid activation ${mid.length}`);
  const below = enc.encodeDimension("speed", -3);
  assert.ok(below.length > 0 && below.every((id) => id < 10), "clamped to low edge");
  const above = enc.encodeDimension("speed", 99);
  assert.ok(above.length > 0 && above.every((id) => id >= 30), "clamped to high edge");
});

test("相似性保持：近值 IoU 显著大于远值，且随距离单调下降", () => {
  const enc = new SensoryEncoder([{ name: "speed", min: 0, max: 8 }], 40);
  const at = (x: number) => enc.encodeDimension("speed", x);
  const near = iou(at(4), at(4.4));
  const mid = iou(at(4), at(5.2));
  const far = iou(at(4), at(7.6));
  assert.ok(near > 0.5, `near IoU ${near.toFixed(2)} should be high`);
  assert.ok(mid < near, `mid IoU ${mid.toFixed(2)} < near ${near.toFixed(2)}`);
  assert.equal(far, 0, "far values share no neurons");
});

test("整帧编码与反查：fieldOf 返回正确的维度与中心", () => {
  const enc = new SensoryEncoder(
    [
      { name: "speed", min: 0, max: 8 },
      { name: "stiffness", min: 0, max: 3 },
    ],
    40,
  );
  const frame = enc.encode({ speed: 4, stiffness: 1.5 });
  const dims = new Set(frame.map((id) => enc.fieldOf(id)!.dimension));
  assert.deepEqual([...dims].sort(), ["speed", "stiffness"]);
  const speedIds = frame.filter((id) => enc.fieldOf(id)!.dimension === "speed");
  for (const id of speedIds) {
    const f = enc.fieldOf(id)!;
    assert.ok(Math.abs(f.center - 4) < 2 * enc.sigma("speed"));
  }
});

test("参数扫描：感受野数 20/40/80 均可编码且保持相似性", () => {
  for (const n of [20, 40, 80]) {
    const enc = new SensoryEncoder([{ name: "x", min: 0, max: 1 }], n);
    const near = iou(enc.encodeDimension("x", 0.5), enc.encodeDimension("x", 0.53));
    const far = iou(enc.encodeDimension("x", 0.5), enc.encodeDimension("x", 0.95));
    assert.ok(near > far, `fieldsPerDim=${n}`);
  }
});
