import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, hebbianLearn } from "../src/index.js";

// 要求 5、7、10：输入约束下的极小能耗模式选择，且只靠局部规律

test("要求5：无加强连接时，输入只激活被钳制子集，其余保持静息", () => {
  const net = new EnergyNetwork({ neuronCount: 16, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  const result = net.settle([2, 7]);
  assert.deepEqual(result.activeNeurons, [2, 7]);
  assert.equal(result.energy, 2 * (1.0 + 0.5));
});

test("要求7+10：settle 收敛模式 = 满足输入约束的能耗最小模式（枚举仅用于测试侧验证）", () => {
  const n = 12;
  const net = new EnergyNetwork({ neuronCount: n, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  // 训练一个 6 神经元全连接簇 {0..5}，簇内连接 0.8；θ = 1.5
  hebbianLearn(net, [0, 1, 2, 3, 4, 5], 8);
  const clamped = [0, 1];
  const result = net.settle(clamped);

  // 测试侧暴力枚举全部 2^12 个满足钳制约束的模式，取最小能量。
  // 注意：枚举只出现在测试中用于验证；引擎本身只执行局部翻转规律（要求 10）。
  let minEnergy = Infinity;
  for (let mask = 0; mask < (1 << n); mask++) {
    if ((mask & 0b11) !== 0b11) continue; // 钳制 0、1 必须激活
    const pattern = new Uint8Array(n);
    for (let i = 0; i < n; i++) pattern[i] = (mask >> i) & 1;
    const e = net.energy(pattern);
    if (e < minEnergy) minEnergy = e;
  }
  assert.ok(
    Math.abs(result.energy - minEnergy) < 1e-9,
    `settled energy ${result.energy} should equal brute-force minimum ${minEnergy}`,
  );
  // 具体地，该场景下最小模式应为整个簇被招募
  assert.deepEqual(result.activeNeurons, [0, 1, 2, 3, 4, 5]);
});

test("要求10：收敛点没有任何单翻转改进（局部极小），从初始条件纯局部可达", () => {
  const net = new EnergyNetwork({ neuronCount: 12, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  hebbianLearn(net, [0, 1, 2, 3, 4, 5], 8);
  const clamped = new Set([0, 1]);
  const result = net.settle([0, 1]);
  const settled = new Uint8Array(12);
  for (const i of result.activeNeurons) settled[i] = 1;
  const e0 = net.energy(settled);
  for (let i = 0; i < 12; i++) {
    if (clamped.has(i)) continue;
    const flipped = Uint8Array.from(settled);
    flipped[i] = flipped[i] === 1 ? 0 : 1;
    assert.ok(
      net.energy(flipped) >= e0 - 1e-9,
      `flipping neuron ${i} must not decrease energy at the settled local minimum`,
    );
  }
});

test("要求10：能量轨迹严格单调下降，有限步收敛", () => {
  const net = new EnergyNetwork({ neuronCount: 12, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  hebbianLearn(net, [0, 1, 2, 3, 4, 5], 8);
  const result = net.settle([0, 1]);
  const { energies, flipCount } = result.trace;
  assert.equal(energies.length, flipCount + 1);
  for (let k = 1; k < energies.length; k++) {
    assert.ok(
      energies[k]! < energies[k - 1]!,
      `energy must strictly decrease at each accepted flip: step ${k}`,
    );
  }
  // 有限状态 + 严格下降 ⇒ 必在有限步内终止（此处 flip 数 ≤ 神经元数）
  assert.ok(flipCount <= 12);
});
