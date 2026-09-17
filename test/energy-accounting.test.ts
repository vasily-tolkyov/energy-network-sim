import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork } from "../src/index.js";

// 要求 1–4：能耗记账

test("要求1：静息不耗能——全静息能量为 0，静息→静息不计费", () => {
  const net = new EnergyNetwork({ neuronCount: 8, activationEnergy: 2.0, maintenanceEnergy: 0.5 });
  assert.equal(net.energy(), 0);
  const ledger = net.runStep([]);
  assert.equal(ledger.activationCost, 0);
  assert.equal(ledger.maintenanceCost, 0);
});

test("要求2：从静息到激活耗能恰为 Ea", () => {
  const net = new EnergyNetwork({ neuronCount: 8, activationEnergy: 2.0, maintenanceEnergy: 0.5 });
  const ledger = net.runStep([5]);
  assert.equal(ledger.activationCount, 1);
  assert.equal(ledger.activationCost, 2.0);
  assert.equal(ledger.maintenanceCost, 0);
});

test("要求3：维持激活每步耗能 Em，k 步累计 k·Em", () => {
  const net = new EnergyNetwork({ neuronCount: 8, activationEnergy: 2.0, maintenanceEnergy: 0.5 });
  net.runStep([3]); // 启动
  for (let k = 0; k < 4; k++) net.runStep([3]); // 维持 4 步
  const ledger = net.ledger();
  assert.equal(ledger.activationCost, 2.0);
  assert.equal(ledger.maintenanceCost, 4 * 0.5);
});

test("要求3补充：激活→静息当步不计维持能耗", () => {
  const net = new EnergyNetwork({ neuronCount: 8, activationEnergy: 2.0, maintenanceEnergy: 0.5 });
  net.runStep([3]);
  const ledger = net.runStep([]); // 熄灭
  assert.equal(ledger.maintenanceCost, 0);
  assert.equal(net.activeNeurons().length, 0);
});

test("要求4：Ea > Em 强制成立，否则拒绝构造", () => {
  assert.throws(
    () => new EnergyNetwork({ neuronCount: 8, activationEnergy: 0.5, maintenanceEnergy: 2.0 }),
    /must be greater than/,
  );
  assert.throws(
    () => new EnergyNetwork({ neuronCount: 8, activationEnergy: 1.0, maintenanceEnergy: 1.0 }),
    /must be greater than/,
  );
});
