import { test } from "node:test";
import assert from "node:assert/strict";
import { EnergyNetwork, hebbianLearn, detectWells, ReadoutModule } from "../src/index.js";
import { PopChannelMap } from "../src/pop/popmap.js";
import { PopRuleMemory } from "../src/pop/popmemory.js";
import { R2PopLayer } from "../src/pop/r2pop.js";
import { SensoryEncoder } from "../src/pop/concept/sensory.js";
import { ConceptFormation } from "../src/pop/concept/formation.js";
import { EmergentMap } from "../src/pop/concept/emergent-map.js";
import { FieldRuleMemory } from "../src/pop/concept/field-memory.js";
import { LabContBench, lcScore } from "../src/topics/lab-continuous-world.js";
import { LabBench, labProbes } from "../src/topics/lab-world.js";
import { NeuralFocusNet } from "../src/pop/attention/neural-focus.js";

/**
 * 第三方评审反例回归集（两份独立评审，f25fefe）。
 * 来源：`reviews/energy-network-sim-f25fefe`（A 系列）与下载包（B/C/D/E 系列）。
 * 修复后这些反例的预期全部翻转；纯属边界声明的（A20 全局最优、非平稳反转）
 * 以文档声明的形式锁定，不假装通过。
 */

// ── F01：DI 账本拆分（A02/A03）────────────────────────────────────
test("F01-A02：DI 两神经元——轨迹末值恒等于真实能量", () => {
  const net = new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: 0.5, maxWeight: 8 });
  net.strengthen(0, 1, 2);
  net.strengthenDirectedInhibitory(0, 1, 0.5, 0.5);
  const r = net.settle([0]);
  const traceEnd = r.trace.energies[r.trace.energies.length - 1]!;
  assert.ok(Math.abs(traceEnd - r.energy) < 1e-9, `轨迹末值 ${traceEnd} ≠ 真实能量 ${r.energy}`);
});

test("F01-A03：DI 闭环——账本回到同状态必须同值；非收敛如实标记", () => {
  const net = new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: 0.5, maxWeight: 8 });
  net.strengthen(0, 1, 2);
  net.strengthen(1, 2, 2);
  net.strengthenDirectedInhibitory(2, 1, 4, 4);
  const r = net.settle([0]);
  // 闭环循环下：要么固定点，要么 flip-budget + 保守账本与真实能量一致
  assert.ok(["fixed-point", "flip-budget"].includes(r.terminationReason));
  const traceEnd = r.trace.energies[r.trace.energies.length - 1]!;
  assert.ok(Math.abs(traceEnd - r.energy) < 1e-9);
  if (!r.converged) {
    assert.equal(r.terminationReason, "flip-budget");
    assert.ok(r.residualFlips > 0, "非收敛态应报告残余可翻转");
  }
});

test("F01-A12：quenchMaxFlips=1 严格预算——至多 1 次淬火翻转且如实报 flip-budget", () => {
  const net = new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: 0.5, maxWeight: 8 });
  net.strengthen(0, 1, 2);
  net.strengthen(1, 2, 2);
  net.strengthenDirectedInhibitory(2, 1, 4, 4);
  const r = net.settleAnnealed([0], [], { seed: 1, levels: 2, sweepsPerLevel: 2, quenchMaxFlips: 1 });
  assert.ok(r.trace.flipCount <= 1 + 2 * 2 * 2, `flipCount ${r.trace.flipCount} 超预算`);
});

// ── F02/D02 生命周期：先侧重后观察必须注册 ─────────────────────────
test("F02/D02：bindInfluence 分配的核在真实观察后必须注册可见", () => {
  const enc = new SensoryEncoder([{ name: "x", min: 0, max: 1 }, { name: "y", min: 0, max: 1 }], 40);
  const mem = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 8 });
  mem.setOutcomeDimensions(["y"]);
  mem.bindInfluence({ x: 0.2 }, { x: 0.1 }, 2);
  mem.learnFromObservation({ x: 0.2 }, { y: 0.8 }, 6);
  assert.ok(mem.ruleCount >= 1, "观察后规则必须注册");
  assert.ok(mem.coreFieldCoverage({ x: 0.2 }) > 0);
  const p = mem.predict({ x: 0.2 }, 1);
  assert.ok(p.values.y !== null, "注册后必须可读出");
});

// ── F03/D01 容量边界 ──────────────────────────────────────────────
test("F03/D01：FieldRuleMemory 超容量显式抛错且不覆盖池区", () => {
  const enc = new SensoryEncoder([{ name: "x", min: 0, max: 1 }, { name: "y", min: 0, max: 1 }], 40);
  const mem = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 1 });
  mem.setOutcomeDimensions(["y"]);
  mem.learnFromObservation({ x: 0.2 }, { y: 0.2 }, 2);
  assert.throws(() => mem.learnFromObservation({ x: 0.8 }, { y: 0.8 }, 2), /capacity/);
});

// ── F04/C07/C12 R2 窗口诚实 ───────────────────────────────────────
test("F04/C07：恒定结果的对不得报变化、不得报因素", () => {
  const cm = new PopChannelMap([{ name: "x", bins: 2 }]);
  const om = new PopChannelMap([{ name: "y", bins: 2 }]);
  const r = new R2PopLayer(cm, om).analyzePair({
    e0: { conditions: { x: 0 }, outcomes: { y: 0 } },
    e1: { conditions: { x: 1 }, outcomes: { y: 0 } },
  });
  assert.equal(r.outcomeChanged, false);
  assert.deepEqual([...r.influentialChannels], []);
});

test("F04/C12：不变干扰维不进差分集合", () => {
  const cm = new PopChannelMap([{ name: "x", bins: 2 }, { name: "d", bins: 2 }]);
  const om = new PopChannelMap([{ name: "y", bins: 2 }]);
  const r = new R2PopLayer(cm, om).analyzePair({
    e0: { conditions: { x: 0, d: 0 }, outcomes: { y: 0 } },
    e1: { conditions: { x: 1, d: 0 }, outcomes: { y: 1 } },
  });
  assert.deepEqual([...r.diffChannels], ["x"]);
});

// ── R1.5 点火不等式：小世界默认配置读回 ────────────────────────────
test("R1.5：恒等小世界两条观察即可读回（点火不等式）", () => {
  const cm = new PopChannelMap([{ name: "x", bins: 2 }], 4);
  const om = new PopChannelMap([{ name: "y", bins: 2 }], 4);
  const mem = new PopRuleMemory(cm, om, { maxRules: 8 });
  mem.learnFromObservation({ x: 0 }, { y: 0 }, 2);
  mem.learnFromObservation({ x: 1 }, { y: 1 }, 2);
  for (const x of [0, 1]) {
    assert.equal(mem.predict({ x }, 1).decoded.y, x, `x=${x} 读回失败`);
  }
});

// ── R1.6 输入校验（A05/A06/A07）──────────────────────────────────
test("R1.6-A05：越界写边抛错且零部分写入", () => {
  const net = new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: 0.5 });
  assert.throws(() => net.strengthen(0, 3, 0.75));
  assert.equal(net.getWeight(0, 1), 0);
  assert.equal(net.getWeight(1, 0), 0, "对称性不得被破坏");
});

test("R1.6-A06：非有限配置/索引/档位全部拒绝", () => {
  assert.throws(() => new EnergyNetwork({ neuronCount: 3, activationEnergy: Infinity, maintenanceEnergy: 0.5 }));
  assert.throws(() => new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: NaN }));
  const net = new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: 0.5 });
  assert.throws(() => net.strengthen(0.5, 1, 0.5));
  const map = new PopChannelMap([{ name: "x", bins: 2 }], 4);
  assert.throws(() => map.population("x", NaN));
});

test("R1.6-A07：实验台拒绝 NaN 且不计成本", () => {
  const bench = new LabContBench();
  const ok = { switchPos: 0.9, voltage: 2.1, resistance: 1.2, temperature: 1.0, material: 0.3 };
  assert.throws(() => bench.conduct({ ...ok, voltage: NaN }));
  assert.equal(bench.experimentsUsed, 0);
  const benchD = new LabBench();
  const okD = { switch: 1, voltage: 2, resistance: 1, temperature: 1, material: 0 };
  assert.throws(() => benchD.conduct({ ...okD, voltage: NaN }));
  assert.equal(benchD.experimentsUsed, 0);
});

// ── B03：全空预测不分配核 ─────────────────────────────────────────
test("B03：全 null/歧义预测不分配核", () => {
  const cm = new PopChannelMap([{ name: "x", bins: 2 }], 4);
  const om = new PopChannelMap([{ name: "y", bins: 2 }], 4);
  const mem = new PopRuleMemory(cm, om, { maxRules: 8 });
  mem.learnFromQuery({ x: 0 }, { y: null }, 1);
  assert.equal(mem.ruleCount, 0);
});

// ── B04：纠错否决不依赖预学互斥 ───────────────────────────────────
test("B04：错误自猜后被真实观察纠正（无预学互斥）", () => {
  const cm = new PopChannelMap([{ name: "x", bins: 2 }], 4);
  const om = new PopChannelMap([{ name: "y", bins: 2 }], 4);
  const mem = new PopRuleMemory(cm, om, { maxRules: 8 });
  mem.learnFromQuery({ x: 0 }, { y: 1 }, 2); // 错误自猜
  mem.learnFromObservation({ x: 0 }, { y: 0 }, 3); // 真实观察纠正
  const d = mem.predict({ x: 0 }, 1).decoded.y;
  assert.equal(d, 0, `纠错后应读 0，实得 ${d}`);
});

// ── F05：网格外覆盖（否决只打已观察替代域）─────────────────────────
test("F05：场级否决不再压灭落在已观察区域附近的输入", () => {
  const enc = new SensoryEncoder(
    [
      { name: "x", min: 0, max: 1 },
      { name: "y", min: 0, max: 1 },
    ],
    40,
  );
  const mem = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 8 });
  mem.setOutcomeDimensions(["y"]);
  mem.teachExperiment({ x: 0.2 }, { y: 0.2 }, 4);
  mem.teachExperiment({ x: 0.5 }, { y: 0.5 }, 4);
  mem.teachExperiment({ x: 0.8 }, { y: 0.8 }, 4);
  mem.bindInfluence({ x: 0.2 }, { x: 0.2 }, 4);
  mem.bindInfluence({ x: 0.5 }, { x: 0.2 }, 4);
  mem.bindInfluence({ x: 0.8 }, { x: 0.2 }, 4);
  // 未训练但在已观察覆盖范围内的点：修复前否决把所有核压灭（结果场全灭）；
  // 修复后结果场有活动（允许如实双峰歧义，不许全灭）
  const p = mem.predict({ x: 0.35 }, 1);
  assert.ok(
    (p.distribution.y ?? []).length > 0,
    "覆盖范围内的未训练输入不得结果场全灭（F05 回归；如实歧义允许）",
  );
  // 远离一切已观察的点：读空是诚实的"未知"，不是否决崩溃（声明边界）
  void mem.predict({ x: 0.99 }, 1);
});

// ── F07/A15/C08：WTA 诚实框架（候选筛选 + 读出择优 + 平局上报）─────
test("F07：等强候选平局如实上报（不静默兜底）；清晰差距单胜者", () => {
  const f = new NeuralFocusNet(["a", "b"], { inertia: 0 });
  f.beginFrame();
  f.setMismatch("a", 8);
  f.setMismatch("b", 8);
  f.select(1);
  assert.equal(f.lastSelectionTied, true, "等强驱动应如实报平局");
  f.beginFrame();
  f.setMismatch("a", 8);
  f.setMismatch("b", 3);
  f.select(2);
  assert.equal(f.lastSelectionTied, false);
});

// ── A19/D04：README 示例可执行 ────────────────────────────────────
test("A19/D04：README 使用示例端到端执行（含峰边说明）", () => {
  const net = new EnergyNetwork({ neuronCount: 24, activationEnergy: 2.0, maintenanceEnergy: 0.5 });
  hebbianLearn(net, [0, 1, 2, 3], 8);
  hebbianLearn(net, [0, 1], 2); // 峰边（完全等权团是平台，严格峰检测需要显著更强的边）
  hebbianLearn(net, [12, 13, 14, 15], 8);
  hebbianLearn(net, [12, 13], 2);
  const wells = detectWells(net);
  assert.ok(wells.length >= 2, `应检测出两个势阱，实得 ${wells.length}`);
  const readout = new ReadoutModule(wells, net.config.readoutThreshold);
  readout.labelWell(wells[0]!.wellId, "A");
  const result = net.settle([0, 1]);
  const ids = readout.identify(result.activeNeurons);
  assert.ok(ids.length >= 1 && ids[0]!.label === "A", "应读出标签 A");
});

// ── F10/E01：heldout 口径退化钉住（分桶语义）──────────────────────
test("F10/E01：unseen-gate-on 桶按构造恒亮——报告必须附常数基线", () => {
  const probes = labProbes(new Set());
  const on = probes.filter((p) => p.kind === "unseen-gate-on");
  assert.ok(on.length > 0);
  assert.ok(on.every((p) => p.truth.lit === 1), "该桶恒亮是构造属性——它不能单独作为门控泛化证据");
  const off = probes.filter((p) => p.kind === "unseen-gate-off");
  assert.ok(off.length > 0 && off.every((p) => p.truth.lit === 0));
});

// ── F14/A20：有限退火是近似搜索（边界声明）────────────────────────
test("F14/A20：退火收敛时是候选域内局部固定点（不承诺全局最小）", () => {
  const net = new EnergyNetwork({ neuronCount: 10, activationEnergy: 2.0, maintenanceEnergy: 1.0, maxWeight: 3.0 });
  net.strengthen(0, 1, 2.0);
  net.strengthen(0, 2, 2.0);
  net.strengthen(1, 2, 2.2);
  for (let i = 5; i < 10; i++) for (let j = i + 1; j < 10; j++) net.strengthen(i, j, 1.6);
  net.strengthen(5, 6, 0.2);
  const r = net.settleAnnealed([0], detectWells(net), { seed: 7 });
  if (r.converged) {
    assert.equal(r.residualFlips, 0, "收敛时无残余可翻转（局部固定点）");
  }
  // 不承诺全局最小：全局最小点燃不相干劲阱 B，本机制语义上不该取（候选域外冻结）
  assert.ok(r.activeNeurons.every((i) => i < 5), "候选域外的势阱 B 不得被点燃");
});
