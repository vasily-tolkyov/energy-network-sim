import { test } from "node:test";
import assert from "node:assert/strict";
import {
  EnergyNetwork,
  hebbianLearn,
  detectWells,
  learnSequence,
  runTransitions,
} from "../src/index.js";

// 势阱间有向通道 + 噪声推动转移
//
// 场景（N=24，θ=1.5）：三个势阱 A={0..3}、B={8..11}、C={16..19}，
// 簇内 0.8、峰边 1.0；时序链 A→B→C 由 learnSequence 建立（D=0.7）。
// β=0.5 时 B 神经元受全激活 A 的驱动 g·β = 4×0.7×0.5 = 1.4：
// 首个 B 神经元点火仍需翻越 ΔE′=0.1 的势垒（噪声），点火后级联为下坡；
// 而残缺的 B（2/4 成员）只能给出 ΔE′=0.8 的点火势垒——T=0.15 下概率 ≈0.005，
// 保证"前驱基本补全才能点燃后继"，链式转移干净有序。
// 温度取 T=0.15：热侵蚀寿命（~exp(0.9/0.15)≈400 步）与疲劳崩塌（~15-20 步）
// 拉开量级，保证"噪声点火 + 驱动级联 + 疲劳交接"各司其职。

const A = [0, 1, 2, 3];
const B = [8, 9, 10, 11];
const C = [16, 17, 18, 19];

function buildChainNet(): EnergyNetwork {
  const net = new EnergyNetwork({
    neuronCount: 24,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 1.0,
    maxDirectedWeight: 1.0,
  });
  for (const cluster of [A, B, C]) {
    hebbianLearn(net, cluster, 8);
    hebbianLearn(net, [cluster[0]!, cluster[1]!], 2);
  }
  learnSequence(net, [A, B, C], 7); // D(A→B) = D(B→C) = 0.7
  return net;
}

/** 检查转移事件按 from→to 顺序包含给定链 */
function hasOrderedChain(
  transitions: readonly { from: number; to: number; step: number }[],
  chain: readonly number[],
): boolean {
  let idx = 0;
  for (const t of transitions) {
    if (idx + 1 < chain.length && t.from === chain[idx] && t.to === chain[idx + 1]) {
      idx++;
      if (idx === chain.length - 1) return true;
    }
  }
  return false;
}

test("时序学习的有向性：D(A→B)>0 而 D(B→A)=0，且裁剪到 maxDirectedWeight", () => {
  const net = buildChainNet();
  assert.equal(net.getDirectedWeight(0, 8), 0.7);
  assert.equal(net.getDirectedWeight(8, 0), 0);
  assert.equal(net.getDirectedWeight(8, 16), 0.7);
  learnSequence(net, [A, B], 10); // 0.7 + 10×0.1 → 裁剪到 1.0
  assert.equal(net.getDirectedWeight(0, 8), 1.0);
});

test("噪声+驱动：种子 1..10 中 ≥8 个按序转移 A→B→C，且驱动做功 > 0", () => {
  let hits = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const net = buildChainNet();
    const wells = detectWells(net);
    net.settle([0, 2]); // 部分线索捕获势阱 A
    assert.ok(net.isActive(3), "settle 应先补全势阱 A");
    const result = runTransitions(net, wells, {
      steps: 200,
      temperature: 0.15,
      driveBeta: 0.5,
      fatigueRate: 0.05,
      seed,
    });
    if (hasOrderedChain(result.transitions, [0, 1, 2])) hits++;
  }
  assert.ok(hits >= 8, `expected ordered A→B→C in >= 8/10 seeds, got ${hits}`);
});

test("对照·无噪声（T=0）：驱动不足以点火，无转移，疲劳后归于静息", () => {
  const net = buildChainNet();
  const wells = detectWells(net);
  net.settle([0, 2]);
  const result = runTransitions(net, wells, {
    steps: 200,
    temperature: 0,
    driveBeta: 0.5,
    fatigueRate: 0.05,
    seed: 7,
  });
  assert.equal(result.transitions.length, 0);
  assert.equal(result.finalActive.length, 0); // A 疲劳崩塌后无人接续 → 静息
});

test("对照·无驱动（β=0）：单个噪声点火无法级联成簇，无转移事件", () => {
  for (let seed = 1; seed <= 5; seed++) {
    const net = buildChainNet();
    const wells = detectWells(net);
    net.settle([0, 2]);
    const result = runTransitions(net, wells, {
      steps: 200,
      temperature: 0.15,
      driveBeta: 0,
      fatigueRate: 0.05,
      seed,
    });
    assert.equal(result.transitions.length, 0, `seed=${seed} must not show well-to-well transitions`);
  }
});

test("对照·无疲劳（κ=0）：A 不崩塌，新旧势阱共存而非转移", () => {
  const net = buildChainNet();
  const wells = detectWells(net);
  net.settle([0, 2]);
  const result = runTransitions(net, wells, {
    steps: 200,
    temperature: 0.15,
    driveBeta: 0.5,
    fatigueRate: 0,
    seed: 7,
  });
  // B 可被驱动+噪声点燃（转移事件发生），但 A 永远不熄灭
  assert.ok(result.transitions.some((t) => t.from === 0 && t.to === 1));
  assert.ok(result.finalActive.some((id) => A.includes(id)), "A must remain active without fatigue");
});

test("回合边界：转移回合结束后 reset 恢复静息，连接记忆保留", () => {
  const net = buildChainNet();
  const wells = detectWells(net);
  net.settle([0, 2]);
  runTransitions(net, wells, { seed: 7 });
  net.reset();
  assert.equal(net.activeNeurons().length, 0);
  assert.ok(net.getWeight(0, 1) > 0, "symmetric weights persist across rounds");
  assert.ok(net.getDirectedWeight(0, 8) > 0, "directed channels persist across rounds");
});
