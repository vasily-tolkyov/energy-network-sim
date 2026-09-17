import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, hebbianLearn, detectWells } from "../src/index.js";

// 要求 8：不相邻极大值 + 0.8 邻域 = 势阱

function trainedTwoClusterNet(): EnergyNetwork {
  const net = new EnergyNetwork({ neuronCount: 16, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  // 簇 A {0..3}、簇 B {10..13}：簇内全连接 0.8，峰边 (0,1)/(10,11) 加强到 1.0
  hebbianLearn(net, [0, 1, 2, 3], 8);
  hebbianLearn(net, [0, 1], 2);
  hebbianLearn(net, [10, 11, 12, 13], 8);
  hebbianLearn(net, [10, 11], 2);
  return net;
}

test("学习后检测出两个不相邻极大值，各自构成一个势阱", () => {
  const wells = detectWells(trainedTwoClusterNet());
  assert.equal(wells.length, 2);

  const peaks = wells.map((w) => new Set([w.peak.from, w.peak.to]));
  // 极大值互不相邻：两个峰边没有公共端点
  const [p0, p1] = peaks;
  assert.ok(![...p0!].some((node) => p1!.has(node)), "peaks must be non-adjacent");
  for (const well of wells) {
    assert.ok(Math.abs(well.peak.weight - 1.0) < 1e-9);
  }

  const memberSets = wells.map((w) => [...w.memberNeuronIds].join(",")).sort();
  assert.deepEqual(memberSets, ["0,1,2,3", "10,11,12,13"]);
});

test("0.8 邻域规则：低于峰值 80% 的边不纳入势阱", () => {
  const net = trainedTwoClusterNet();
  hebbianLearn(net, [3, 7], 1); // 弱边 (3,7)=0.1 < 0.8×1.0
  const wells = detectWells(net);
  assert.equal(wells.length, 2);
  const wellA = wells.find((w) => w.memberNeuronIds.includes(0))!;
  assert.ok(!wellA.memberNeuronIds.includes(7), "neuron 7 must stay outside the 0.8 neighbourhood");
  assert.deepEqual([...wellA.memberNeuronIds], [0, 1, 2, 3]);
});

test("未经学习的网络没有势阱", () => {
  const net = new EnergyNetwork({ neuronCount: 8, activationEnergy: 1.0, maintenanceEnergy: 0.5 });
  assert.deepEqual(detectWells(net), []);
});
