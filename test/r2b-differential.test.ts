import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, hebbianLearn } from "../src/index.js";
import { neuralDifferential } from "../src/layers/differential.js";

// R2B 神经差分引擎验证：
// 实验 A 条件 {0..7}、实验 B 条件 {3..10}，共享 {3..7}（5 个）。
// 关键结构条件：交集自维持 (5−1)×2w=6.4 必须大于单阱自维持 (8−1)×w=5.6，
// 即 |交集| > (|阱|+1)/2——高阈值窗口 (5.6, 6.4) 才存在，取 θHigh=6.0。

test("重合场分层：交集驻留为共同项，单侧成员为差分候选", () => {
  const net = new EnergyNetwork({
    neuronCount: 24,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const A = [0, 1, 2, 3, 4, 5, 6, 7];
  const B = [3, 4, 5, 6, 7, 8, 9, 10];
  hebbianLearn(net, A, 8);
  hebbianLearn(net, B, 8); // 共享成员的互连边被双倍加强
  const r = neuralDifferential(net, A, B, 6.0);
  assert.deepEqual(r.common, [3, 4, 5, 6, 7]);
  assert.deepEqual(r.onlyA, [0, 1, 2]);
  assert.deepEqual(r.onlyB, [8, 9, 10]);
  assert.deepEqual([...r.persisted], [3, 4, 5, 6, 7]);
});

test("阈值窗口：θHigh 过低不分层（全部驻留），过高全熄灭", () => {
  const build = () => {
    const net = new EnergyNetwork({
      neuronCount: 24,
      activationEnergy: 1.0,
      maintenanceEnergy: 0.5,
      learningRate: 0.1,
      maxWeight: 3.0,
    });
    hebbianLearn(net, [0, 1, 2, 3, 4, 5, 6, 7], 8);
    hebbianLearn(net, [3, 4, 5, 6, 7, 8, 9, 10], 8);
    return net;
  };
  const A = [0, 1, 2, 3, 4, 5, 6, 7];
  const B = [3, 4, 5, 6, 7, 8, 9, 10];
  // θHigh=4.0 < 单阱自维持场(≈5.6)：11 个成员全部驻留，无法分出交集
  const low = neuralDifferential(build(), A, B, 4.0);
  assert.equal(low.persisted.length, 11);
  // θHigh=7.0 > 交集自维持场(≈6.4)：全部熄灭
  const high = neuralDifferential(build(), A, B, 7.0);
  assert.equal(high.persisted.length, 0);
});

test("双源 AND 门：影响因素神经元必须同时收到差分信号与结果变化信号才越阈", () => {
  const net = new EnergyNetwork({
    neuronCount: 8,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const d = 0; // 差分信号（该条件确实变化）
  const o = 1; // R2A 结果变化信号
  const x = 2; // 影响因素神经元
  net.strengthen(d, x, 1.0);
  net.strengthen(o, x, 1.0);
  // θ=1.5：单源场 1.0 不够，双源 2.0 才越阈
  assert.equal(net.settle([d]).activeNeurons.includes(x), false, "仅差分不足");
  assert.equal(net.settle([o]).activeNeurons.includes(x), false, "仅结果变化不足");
  assert.equal(net.settle([]).activeNeurons.includes(x), false, "无输入");
  assert.ok(net.settle([d, o]).activeNeurons.includes(x), "双源齐备 → 判为影响因素");
});
