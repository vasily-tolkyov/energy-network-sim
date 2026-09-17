import { test } from "node:test";
import assert from "node:assert/strict";
import {
  EnergyNetwork,
  hebbianLearn,
  detectWells,
  learnBranches,
  runTransitions,
} from "../src/index.js";

// 分支竞争：同一前驱 A 的两条等强/不等强通道 A→B、A→D。
// 语义（用户约定）：两条通道都有效，但激活演化在噪声下只推进到一条；
// 进入哪条是概率事件，概率随通道强度倾斜。
// 机制：候选后继间的抑制边 Γ（先点燃者抬高对手势垒，赢家通吃）。
//
// 场景（N=48，θ=1.5）：A={0..3}、B={8..11}、D={16..19}，簇内 0.8 + 峰边 1.0。
// 等强 d=0.55：点火势垒 ΔE′=0.4（竞赛窗口约 3-4 步）；抑制 γ=0.6，赢家每增加 1 成员，输家点火势垒 +0.6 → 先发者迅速锁死对手。
// T=0.15、κ=0.05（4 成员簇死亡同步，无需慢恢复）。

const A = [0, 1, 2, 3];
const B = [8, 9, 10, 11];
const D = [16, 17, 18, 19];

function buildBranchNet(opts: { etaB?: number; etaD?: number; gamma?: number }): EnergyNetwork {
  const net = new EnergyNetwork({
    neuronCount: 48,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 1.0,
    maxDirectedWeight: 1.0,
  });
  for (const c of [A, B, D]) {
    hebbianLearn(net, c, 8);
    hebbianLearn(net, [c[0]!, c[1]!], 2);
  }
  learnBranches(net, A, [B, D], {
    repeats: 1,
    eta: [opts.etaB ?? 0.55, opts.etaD ?? 0.55],
    gamma: opts.gamma ?? 0.6,
  });
  return net;
}

/** 统计一次运行中 B（wellId 1）与 D（wellId 2）各自的主导步数 */
function raceOutcome(net: EnergyNetwork, seed: number): { bSteps: number; dSteps: number } {
  const wells = detectWells(net);
  net.settle([0, 2]); // 捕获 A
  const r = runTransitions(net, wells, {
    steps: 120,
    temperature: 0.15,
    driveBeta: 0.5,
    fatigueRate: 0.05,
    seed,
  });
  // 用激活集累计步数而非占优时间线——共存期前驱会掩盖后继
  return { bSteps: r.wellActiveSteps[1] ?? 0, dSteps: r.wellActiveSteps[2] ?? 0 };
}
test("等强双通道：每次运行互斥地只进入一个后继，且不偏向任何一方", () => {
  let bWins = 0;
  let dWins = 0;
  for (let seed = 1; seed <= 20; seed++) {
    const { bSteps, dSteps } = raceOutcome(buildBranchNet({}), seed);
    // 赢家主导其寿命全程（≥10 步），输家只允许竞赛期的零星闪烁
    const [winSteps, loseSteps] = bSteps >= dSteps ? [bSteps, dSteps] : [dSteps, bSteps];
    assert.ok(winSteps >= 10, `seed=${seed}: winner must persist (B=${bSteps}, D=${dSteps})`);
    assert.ok(loseSteps <= 3, `seed=${seed}: loser must be suppressed (B=${bSteps}, D=${dSteps})`);
    if (bSteps >= dSteps) bWins++;
    else dWins++;
  }
  // 等强通道 ~50/50；20 种子下极端分布（≤3 或 ≥17）概率 < 0.3%
  assert.ok(bWins >= 3 && dWins >= 3, `expected roughly even split, got B=${bWins} D=${dWins}`);
});

test("不等强双通道：强通道赢多数，弱通道仍有少数获胜（概率性）", () => {
  let bWins = 0;
  let dWins = 0;
  for (let seed = 1; seed <= 40; seed++) {
    // d_B=0.55（ΔE′=0.4）vs d_D=0.4（ΔE′=0.7）
    const { bSteps, dSteps } = raceOutcome(buildBranchNet({ etaB: 0.55, etaD: 0.4 }), seed);
    if (bSteps >= dSteps) bWins++;
    else dWins++;
  }
  assert.ok(bWins > dWins, `stronger channel should win more: B=${bWins} D=${dWins}`);
  assert.ok(dWins >= 1, `weaker channel should still win sometimes: D=${dWins}`);
});

test("抑制项进入能量函数：同时激活两个竞争势阱要付出 Γ 惩罚", () => {
  const net = buildBranchNet({});
  const onlyB = new Uint8Array(48);
  for (const i of [...A, ...B]) onlyB[i] = 1;
  const bothBD = new Uint8Array(48);
  for (const i of [...A, ...B, ...D]) bothBD[i] = 1;
  const onlyD = new Uint8Array(48);
  for (const i of [...A, ...D]) onlyD[i] = 1;
  // E(A∪B∪D) − E(A∪B) − E(A∪D) + E(A) = B↔D 的抑制总量 = 16 对 × 0.6
  const onlyA = new Uint8Array(48);
  for (const i of A) onlyA[i] = 1;
  const penalty =
    net.energy(bothBD) - net.energy(onlyB) - net.energy(onlyD) + net.energy(onlyA);
  assert.ok(Math.abs(penalty - 16 * 0.6) < 1e-9, `inhibitory penalty = ${penalty}`);
});
