import { test } from "node:test";
import assert from "node:assert/strict";
import {
  EnergyNetwork,
  hebbianLearn,
  detectWells,
  learnSequence,
  runTransitions,
} from "../src/index.js";

// 中规模冒烟测试（N=512，8 势阱 × 16 神经元，2 条 4 节链），
// 锁定大规模实验使用的参数组合的行为。大规模完整实验见 src/experiment-scale.ts。

const N = 512;
const WELL_SIZE = 16;
const SEQ = { steps: 80, temperature: 0.12, driveBeta: 0.5, fatigueRate: 0.7, fatigueRecoveryRate: 0.15 } as const;

function clusterAt(k: number): number[] {
  return Array.from({ length: WELL_SIZE }, (_, i) => k * WELL_SIZE + i);
}

function buildTrained(): EnergyNetwork {
  const net = new EnergyNetwork({
    neuronCount: N,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 1.0,
    maxDirectedWeight: 1.0,
  });
  for (let k = 0; k < 8; k++) {
    hebbianLearn(net, clusterAt(k), 8);
    hebbianLearn(net, [k * WELL_SIZE, k * WELL_SIZE + 1], 2);
  }
  return net;
}

test("中规模：8 个 16 神经元势阱全部检出，25% 线索捕获完成率 ≥ 90%", () => {
  const net = buildTrained();
  const wells = detectWells(net);
  assert.equal(wells.length, 8);
  for (let k = 0; k < 8; k++) {
    const cue = clusterAt(k).slice(0, WELL_SIZE / 4);
    const r = net.settle(cue);
    const active = new Set(r.activeNeurons);
    const members = new Set(clusterAt(k));
    let hit = 0;
    for (const id of members) if (active.has(id)) hit++;
    assert.ok(hit / WELL_SIZE >= 0.9, `cluster ${k} completion ${hit}/${WELL_SIZE}`);
    for (const id of active) {
      assert.ok(members.has(id), `cluster ${k} capture must not recruit stray neuron ${id}`);
    }
  }
});

test("中规模：4 节链在噪声+驱动下生成有序转移", () => {
  const net = buildTrained();
  const wells = detectWells(net);
  const chain = [0, 1, 2, 3].map(clusterAt);
  learnSequence(net, chain, 1, 0.15);
  const ids = chain.map((cl) => wells.findIndex((w) => w.memberNeuronIds.includes(cl[0]!)));
  assert.ok(!ids.includes(-1));
  let anyFull = false;
  for (let seed = 1; seed <= 3; seed++) {
    net.settle(chain[0]!.slice(0, WELL_SIZE / 4));
    const r = runTransitions(net, wells, { ...SEQ, seed });
    let want = 1;
    for (const t of r.transitions) {
      if (want < ids.length && t.from === ids[want - 1] && t.to === ids[want]) want++;
    }
    if (want === ids.length) anyFull = true;
  }
  assert.ok(anyFull, "at least one seed should generate the full ordered chain");
});

test("中规模对照：T=0 与 β=0 均无转移", () => {
  const net = buildTrained();
  const wells = detectWells(net);
  const chain = [0, 1, 2, 3].map(clusterAt);
  learnSequence(net, chain, 1, 0.15);
  for (const opts of [
    { ...SEQ, temperature: 0 },
    { ...SEQ, driveBeta: 0 },
  ]) {
    net.settle(chain[0]!.slice(0, WELL_SIZE / 4));
    const r = runTransitions(net, wells, { ...opts, seed: 1 });
    assert.equal(r.transitions.length, 0);
  }
});
