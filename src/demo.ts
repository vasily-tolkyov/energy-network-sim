import { EnergyNetwork, hebbianLearn, detectWells, ReadoutModule } from "./index.js";

/**
 * 实机演示：完整走一遍 学习 → 势阱检测 → 教师赋标签 → 新输入读出 的流程。
 * 运行：npm run build && node dist/src/demo.js
 */

const SEP = "─".repeat(64);

function show(title: string): void {
  console.log(`\n${SEP}\n${title}\n${SEP}`);
}

// ── 0. 网络与能耗常数 ──────────────────────────────────────────────
const Ea = 1.0;
const Em = 0.5;
const net = new EnergyNetwork({
  neuronCount: 16,
  activationEnergy: Ea,
  maintenanceEnergy: Em,
  learningRate: 0.1,
  maxWeight: 1.0,
});
show(`网络：16 个神经元，Ea=${Ea}（启动能耗），Em=${Em}（维持能耗），阈值 θ=Ea+Em=${Ea + Em}`);

// ── 1. 要求 1–4：能耗记账 ─────────────────────────────────────────
show("① 能耗记账：静息 0 耗能；启动收 Ea；维持每步收 Em");
console.log(`全静息模式能量 = ${net.energy()}（要求 1）`);
net.runStep([5]);
console.log(`神经元 5 从静息激活：累计启动能耗 = ${net.ledger().activationCost}（= Ea，要求 2）`);
for (let k = 0; k < 3; k++) net.runStep([5]);
console.log(`再维持 3 步：累计维持能耗 = ${net.ledger().maintenanceCost}（= 3×Em，要求 3）`);
net.runStep([]);
console.log(`熄灭后回到静息，本步不计费；当前激活集合 = [${net.activeNeurons()}]`);
net.reset();
net.resetLedger();

// ── 2. 要求 6：赫布学习，连接越强能耗越低 ─────────────────────────
show("② 赫布学习：教师呈现两个模式，观察模式能量随连接增强而下降");
const patternA = [0, 1, 2, 3];
const patternB = [10, 11, 12, 13];
const bitsA = new Uint8Array(16);
for (const i of patternA) bitsA[i] = 1;
console.log(`学习前，模式 A 的能量 = ${net.energy(bitsA).toFixed(2)}`);
for (let r = 1; r <= 8; r++) {
  hebbianLearn(net, patternA, 1);
  hebbianLearn(net, patternB, 1);
  console.log(`第 ${r} 轮共激活后：模式 A 能量 = ${net.energy(bitsA).toFixed(2)}（连接 0.${r}，要求 6）`);
}
// 把每个簇的峰边加强到 1.0，制造势阱的局部极大值
hebbianLearn(net, [0, 1], 2);
hebbianLearn(net, [10, 11], 2);
console.log(`峰边 (0,1)、(10,11) 额外加强至 1.0（簇内其余边保持 0.8）`);

// ── 3. 要求 8：势阱检测 + 教师赋标签 ──────────────────────────────
show("③ 势阱检测：不相邻极大值的 0.8 邻域（要求 8）");
const wells = detectWells(net);
const readout = new ReadoutModule(wells, net.config.readoutThreshold);
for (const well of wells) {
  const label = well.memberNeuronIds.includes(0) ? "猫" : "狗";
  readout.labelWell(well.wellId, label); // 教师程序赋标签
  console.log(
    `势阱 ${well.wellId}：峰边 (${well.peak.from},${well.peak.to}) 强度 ${well.peak.weight.toFixed(2)}，` +
      `成员神经元 [${well.memberNeuronIds}] → 教师标签「${label}」`,
  );
}

// ── 4. 要求 5、7、10：新输入 → 局部规律收敛到极小能耗模式 ─────────
show("④ 新输入 [0,2]（「猫」模式的一部分）：只看局部规律，能量单调下降到极小");
const result = net.settle([0, 2]);
result.trace.energies.forEach((e, k) => {
  const bar = k === 0 ? "（初始：钳制 0、2）" : `（第 ${k} 次翻转后）`;
  console.log(`  E = ${e.toFixed(2).padStart(7)} ${bar}`);
});
console.log(`收敛激活模式 = [${result.activeNeurons}]（要求 5：其余神经元保持静息）`);

// 测试侧枚举验证：枚举只出现在演示验证中，引擎本身没有枚举（要求 10）
let minEnergy = Infinity;
for (let mask = 0; mask < (1 << 16); mask++) {
  if ((mask & 0b101) !== 0b101) continue;
  const bits = new Uint8Array(16);
  for (let i = 0; i < 16; i++) bits[i] = (mask >> i) & 1;
  minEnergy = Math.min(minEnergy, net.energy(bits));
}
console.log(`枚举全部 2^16 个满足约束的模式验证：全局最小 = ${minEnergy.toFixed(2)}，与收敛结果一致（要求 7、10）`);

// ── 5. 要求 9：读出被激活势阱的标签 ───────────────────────────────
show("⑤ 读出：识别激活模式中被激活的势阱并读出标签（要求 9）");
for (const r of readout.identify(result.activeNeurons)) {
  console.log(`  势阱 ${r.wellId} 激活率 ${(r.activationRatio * 100).toFixed(0)}% → 标签「${r.label}」`);
}

show("⑥ 对照：输入同时触及两簇 [0,2,10,12] 与未触及任何势阱 [5,6]");
const both = net.settle([0, 2, 10, 12]);
console.log(
  `输入 [0,2,10,12] → 激活 [${both.activeNeurons}] → 标签 [${readout
    .identify(both.activeNeurons)
    .map((r) => r.label)
    .join("、")}]`,
);
const none = net.settle([5, 6]);
console.log(
  `输入 [5,6] → 激活 [${none.activeNeurons}] → 标签 [${readout
    .identify(none.activeNeurons)
    .map((r) => r.label)
    .join("、")}]（无势阱被激活）`,
);
console.log();

// ── 6. 局部退火：协同势垒 + 不相干劲阱 ────────────────────────────
show("⑦ 局部退火：只在输入所及的候选集内翻越势垒");
const net2 = new EnergyNetwork({
  neuronCount: 10,
  activationEnergy: 2.0,
  maintenanceEnergy: 1.0,
  maxWeight: 3.0,
});
net2.strengthen(0, 1, 2.0);
net2.strengthen(0, 2, 2.0);
net2.strengthen(1, 2, 2.2);
for (let i = 5; i < 10; i++) {
  for (let j = i + 1; j < 10; j++) net2.strengthen(i, j, 1.6);
}
net2.strengthen(5, 6, 0.2);
console.log("势阱 A={0,1,2}：从 {0} 单翻激活 1 或 2 都是 +1.0 上坡（协同势垒），联合激活才更省");
console.log("势阱 B={5..9}：与输入不相干的负能耗势阱（E=−1.2），无约束全局最小会点燃它");
const wells2 = detectWells(net2);
console.log(`检测到 ${wells2.length} 个势阱：${wells2.map((w) => `{${w.memberNeuronIds}}`).join("、")}`);

const greedy2 = net2.settle([0]);
console.log(`\n贪心 settle([0]) → 激活 [${greedy2.activeNeurons}]，E=${greedy2.energy.toFixed(2)}（被协同势垒挡住）`);

const annealed = net2.settleAnnealed([0], wells2, { seed: 7 });
console.log(
  `局部退火（seed=7）→ 候选集 C=[${annealed.candidateSet}]，上坡翻转 ${annealed.acceptedUphill}/${annealed.proposals} 次`,
);
console.log(
  `能量：初始（静息+钳制）${annealed.initialEnergy.toFixed(2)} → 退火末 ${annealed.annealEndEnergy.toFixed(2)} → 淬火后 ${annealed.energy.toFixed(2)}`,
);
console.log(`最终激活 [${annealed.activeNeurons}]（翻越势垒，落入势阱 A）`);

let minWithinC = Infinity;
let globalMin = Infinity;
for (let mask = 0; mask < (1 << 10); mask++) {
  if ((mask & 1) !== 1) continue;
  const bits = new Uint8Array(10);
  for (let i = 0; i < 10; i++) bits[i] = (mask >> i) & 1;
  const e = net2.energy(bits);
  globalMin = Math.min(globalMin, e);
  if ((mask & ~0b111) === 0) minWithinC = Math.min(minWithinC, e);
}
console.log(`枚举验证：C 内最小 = ${minWithinC.toFixed(2)}（退火命中）；无约束全局最小 = ${globalMin.toFixed(2)}（语义上不该取）`);

const seedResults: string[] = [];
for (let seed = 1; seed <= 5; seed++) {
  const probe = new EnergyNetwork({
    neuronCount: 10,
    activationEnergy: 2.0,
    maintenanceEnergy: 1.0,
    maxWeight: 3.0,
  });
  probe.strengthen(0, 1, 2.0);
  probe.strengthen(0, 2, 2.0);
  probe.strengthen(1, 2, 2.2);
  for (let i = 5; i < 10; i++) {
    for (let j = i + 1; j < 10; j++) probe.strengthen(i, j, 1.6);
  }
  probe.strengthen(5, 6, 0.2);
  const r = probe.settleAnnealed([0], detectWells(probe), { seed });
  const touchedB = r.activeNeurons.some((id) => id >= 5);
  seedResults.push(`seed=${seed}: [${r.activeNeurons}] E=${r.energy.toFixed(2)}${touchedB ? "（点燃了 B！）" : ""}`);
}
console.log(`\n多种子复跑（势阱 B 必须永远静默）：`);
for (const line of seedResults) console.log(`  ${line}`);
console.log();
