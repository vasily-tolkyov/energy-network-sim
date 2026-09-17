import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, hebbianLearn } from "../src/index.js";

// 要求 6：赫布加强连接，强度越强能耗越低

test("赫布共激活使固定模式的能量随连接强度单调下降", () => {
  const net = new EnergyNetwork({ neuronCount: 8, activationEnergy: 2.0, maintenanceEnergy: 0.5 });
  const pattern = [0, 1, 2, 3];
  const bits = new Uint8Array(8);
  for (const i of pattern) bits[i] = 1;

  const energies: number[] = [net.energy(bits)];
  for (let r = 0; r < 10; r++) {
    hebbianLearn(net, pattern, 1);
    energies.push(net.energy(bits));
  }
  for (let k = 1; k < energies.length; k++) {
    assert.ok(
      energies[k]! < energies[k - 1]!,
      `stronger connections must lower total energy (step ${k})`,
    );
  }
  // 10 次重复后连接打满 maxWeight=1.0：E = 4·2.5 − 6·1.0 = 4
  assert.ok(Math.abs(energies[10]! - 4.0) < 1e-9);
});

test("加强连接使被钳制输入招募整个簇，且训练越强收敛能耗越低", () => {
  const weak = new EnergyNetwork({ neuronCount: 8, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  hebbianLearn(weak, [0, 1, 2, 3, 4, 5], 5); // 簇内连接 0.5
  const strong = new EnergyNetwork({ neuronCount: 8, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  hebbianLearn(strong, [0, 1, 2, 3, 4, 5], 8); // 簇内连接 0.8

  const beforeWeak = weak.settle([0, 1]);
  // 弱连接（0.5）：神经元 2 的场 = 2×0.5 = 1.0 < θ=1.5，无法招募
  assert.deepEqual(beforeWeak.activeNeurons, [0, 1]);

  const afterStrong = strong.settle([0, 1]);
  // 强连接（0.8）：神经元 2 的场 = 2×0.8 = 1.6 > θ=1.5，级联招募全簇
  assert.deepEqual(afterStrong.activeNeurons, [0, 1, 2, 3, 4, 5]);

  // 更强的连接 → 更低的收敛总能耗（要求 6 后半句）
  assert.ok(afterStrong.energy < beforeWeak.energy);
});
