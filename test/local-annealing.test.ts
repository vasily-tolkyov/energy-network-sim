import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, detectWells } from "../src/index.js";

// 局部退火：候选集内翻越协同势垒，候选集外结构性冻结
//
// 场景（N=10，Ea=2、Em=1，θ=3）：
// - 势阱 A = {0,1,2}：边 (0,1)=2.0、(0,2)=2.0、(1,2)=2.2。
//   从 {0} 单独激活 1 或 2 都要 +1.0 上坡（协同势垒），
//   但联合激活 {0,1,2} 的能量 2.8 低于 {0} 的 3.0。
// - 势阱 B = {5..9}：与输入完全不相干的负能耗势阱，E = −1.2。
//   无约束全局最小会点燃它（E=1.6），局部退火必须视而不见。

function buildNet(): EnergyNetwork {
  const net = new EnergyNetwork({
    neuronCount: 10,
    activationEnergy: 2.0,
    maintenanceEnergy: 1.0,
    maxWeight: 3.0,
  });
  net.strengthen(0, 1, 2.0);
  net.strengthen(0, 2, 2.0);
  net.strengthen(1, 2, 2.2);
  for (let i = 5; i < 10; i++) {
    for (let j = i + 1; j < 10; j++) net.strengthen(i, j, 1.6);
  }
  net.strengthen(5, 6, 0.2); // 峰边 1.8
  return net;
}

function bruteForceMin(net: EnergyNetwork, n: number, clampMask: number, domainMask: number): number {
  let min = Infinity;
  for (let mask = 0; mask < (1 << n); mask++) {
    if ((mask & clampMask) !== clampMask) continue;
    if (mask & ~domainMask) continue; // 域外必须为 0
    const bits = new Uint8Array(n);
    for (let i = 0; i < n; i++) bits[i] = (mask >> i) & 1;
    min = Math.min(min, net.energy(bits));
  }
  return min;
}

test("贪心被协同势垒挡住：输入 [0] 只激活 {0}", () => {
  const net = buildNet();
  const result = net.settle([0]);
  assert.deepEqual(result.activeNeurons, [0]);
  assert.ok(Math.abs(result.energy - 3.0) < 1e-9);
});

test("局部退火翻越协同势垒，收敛到候选集 C 内的全局最小", () => {
  const net = buildNet();
  const wells = detectWells(net);
  assert.equal(wells.length, 2);
  const result = net.settleAnnealed([0], wells, { seed: 7 });
  // C = {0} ∪ 势阱A = {0,1,2}；势阱B 不触及输入，不在 C 内
  assert.deepEqual(result.candidateSet, [0, 1, 2]);
  assert.deepEqual(result.activeNeurons, [0, 1, 2]);
  // 候选集内枚举最小 = 2.8；无约束全局最小 = 1.6（会点燃势阱B）
  const minWithinC = bruteForceMin(net, 10, 0b1, 0b0000000111);
  const globalMin = bruteForceMin(net, 10, 0b1, 0b1111111111);
  assert.ok(Math.abs(minWithinC - 2.8) < 1e-9);
  assert.ok(Math.abs(globalMin - 1.6) < 1e-9);
  assert.ok(Math.abs(result.energy - minWithinC) < 1e-9, "局部退火 = C 内最小");
  assert.ok(Math.abs(result.energy - globalMin) > 1e-9, "局部退火 ≠ 无约束全局最小（语义正确）");
  assert.ok(result.acceptedUphill > 0, "应发生过上坡翻转（翻越势垒的证据）");
});

test("不相干劲阱在任何种子下都不被点燃", () => {
  for (let seed = 1; seed <= 5; seed++) {
    const net = buildNet();
    const wells = detectWells(net);
    const result = net.settleAnnealed([0], wells, { seed });
    const active = new Set(result.activeNeurons);
    for (const id of [5, 6, 7, 8, 9]) {
      assert.ok(!active.has(id), `seed=${seed}: neuron ${id} must stay at rest`);
    }
    assert.ok(Math.abs(result.energy - 2.8) < 1e-9, `seed=${seed}: all seeds converge to C-minimum`);
  }
});

test("确定性：相同种子两次运行结果完全一致", () => {
  const wells = detectWells(buildNet());
  const a = buildNet().settleAnnealed([0], wells, { seed: 42 });
  const b = buildNet().settleAnnealed([0], wells, { seed: 42 });
  assert.deepEqual(a.activeNeurons, b.activeNeurons);
  assert.equal(a.energy, b.energy);
  assert.equal(a.acceptedUphill, b.acceptedUphill);
});

test("输入未触及任何势阱时退化为贪心结果", () => {
  const net = buildNet();
  const wells = detectWells(net);
  const result = net.settleAnnealed([4], wells, { seed: 1 });
  assert.deepEqual(result.candidateSet, [4]);
  assert.deepEqual(result.activeNeurons, [4]);
  assert.equal(result.acceptedUphill, 0);
});

test("淬火相能量轨迹严格单调下降", () => {
  const net = buildNet();
  const wells = detectWells(net);
  const result = net.settleAnnealed([0], wells, { seed: 7 });
  const { energies } = result.trace;
  for (let k = 1; k < energies.length; k++) {
    assert.ok(energies[k]! < energies[k - 1]!, `quench energy must strictly decrease at step ${k}`);
  }
});
