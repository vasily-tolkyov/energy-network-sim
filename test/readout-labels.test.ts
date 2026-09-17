import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, hebbianLearn, detectWells, ReadoutModule } from "../src/index.js";

// 要求 9：新输入到来时，读出被激活势阱的标签

function setup(): { net: EnergyNetwork; readout: ReadoutModule } {
  const net = new EnergyNetwork({ neuronCount: 16, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  hebbianLearn(net, [0, 1, 2, 3], 8);
  hebbianLearn(net, [0, 1], 2);
  hebbianLearn(net, [10, 11, 12, 13], 8);
  hebbianLearn(net, [10, 11], 2);
  const wells = detectWells(net);
  const readout = new ReadoutModule(wells, net.config.readoutThreshold);
  // 教师程序赋标签（要求 8 后半句）
  for (const well of wells) {
    readout.labelWell(well.wellId, well.memberNeuronIds.includes(0) ? "alpha" : "beta");
  }
  return { net, readout };
}

test("输入簇 A 的部分神经元 → 级联招募全簇 → 读出标签 alpha", () => {
  const { net, readout } = setup();
  const result = net.settle([0, 2]); // 部分输入：θ=1.5，神经元 1 场强 2×0.8=1.6 被招募
  assert.deepEqual(result.activeNeurons, [0, 1, 2, 3]);

  const readouts = readout.identify(result.activeNeurons);
  assert.equal(readouts.length, 1);
  assert.equal(readouts[0]!.label, "alpha");
  assert.equal(readouts[0]!.activationRatio, 1.0);
});

test("输入同时触及两簇 → 读出两个标签并按激活率排序", () => {
  const { net, readout } = setup();
  const result = net.settle([0, 2, 10, 12]);
  const readouts = readout.identify(result.activeNeurons);
  assert.deepEqual(readouts.map((r) => r.label), ["alpha", "beta"]);
});

test("未触及任何势阱的输入 → 无读出", () => {
  const { net, readout } = setup();
  const result = net.settle([5, 6]); // 无连接区域，只激活钳制神经元
  assert.deepEqual(readout.identify(result.activeNeurons), []);
});

test("模型自身赋标签（无教师）：autoLabel 生成占位标签", () => {
  const net = new EnergyNetwork({ neuronCount: 16, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  hebbianLearn(net, [0, 1, 2, 3], 8);
  hebbianLearn(net, [0, 1], 2);
  const wells = detectWells(net);
  const readout = new ReadoutModule(wells, net.config.readoutThreshold);
  readout.autoLabel();
  const result = net.settle([0, 2]);
  const readouts = readout.identify(result.activeNeurons);
  assert.equal(readouts.length, 1);
  assert.equal(readouts[0]!.label, `well-${readouts[0]!.wellId}`);
});
