import { PredictionQuality } from "../pop/prediction-quality.js";
import { mkdirSync, writeFileSync } from "node:fs";
import { EnergyNetwork, hebbianLearn, detectWells } from "../index.js";
import { ChannelMap } from "../prototype/channels.js";
import { extractInfluences } from "../prototype/r2.js";
import { RuleMemory } from "../prototype/r3.js";
import {
  CAP_CONDITION_SPECS,
  CAP_OUTCOME_SPECS,
  capCurriculum,
  capQueries,
  capTruth,
  capTruthSmooth,
  sampleCombos,
} from "./capacity-world.js";

/**
 * 预测核心容量评估（本机实机）：
 * - C1 规则容量：合成规则域（两因素+一干扰 → 单结果通道 8 档），两种真理结构
 *   （mod-8 对抗性 / 平滑可加）× 教学规则数 E ∈ {8,16,24,32,48,64}；
 * - C2 装配容量：N=8000 下势阱数 100/200/400 的检出率与捕获率（资源侧）。
 * 机制与参数与预测原型完全一致（θ=1.5、η=0.1、base 0.4、γ=3.0、G=3）。
 * 运行：npm run build && node dist/src/topics/runner-capacity.js [--skip-c2]
 */

const lines: string[] = [];
const quality = new PredictionQuality();
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

// ── C1 规则容量 ──────────────────────────────────────────────────
const cm = new ChannelMap(CAP_CONDITION_SPECS.map((s) => ({ ...s })));
const om = new ChannelMap(CAP_OUTCOME_SPECS.map((s) => ({ ...s })));
const SPAN = { out: 7 };

function teachCapacityNet(
  combos: readonly [number, number][],
  gain: number,
  truth: (c: Record<string, number>) => Record<string, number>,
): RuleMemory {
  const net = new EnergyNetwork({
    neuronCount: cm.neuronCount + om.neuronCount,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 3.0,
  });
  const mem = new RuleMemory(net, cm, om, { r1Repeats: 4, influenceGain: gain, r3Repeats: 4, gamma: 3.0 });
  for (const group of capCurriculum(combos, truth)) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
    }
    const influence = extractInfluences(cm.channelNames(), group.pairs, SPAN);
    const boost: Record<string, number> = {};
    for (const ch of influence.influential) boost[ch] = 0.1 * gain * (influence.magnitude[ch] ?? 0) ** 2;
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  return mem;
}

out("═".repeat(64));
out("C1 规则容量：预测准确率 vs 教学规则数 E（annealed 读出，无查询学习）");
out("═".repeat(64));

for (const [variant, truthFn] of [
  ["mod-8（对抗性，不可插值）", capTruth],
  ["平滑可加（out=min(7,f1+f2)）", capTruthSmooth],
] as const) {
  out(`\n真理结构：${variant}`);
  out("E".padStart(4) + "taught 准确率".padStart(16) + "heldout 准确率".padStart(16) + "单查询ms".padStart(10));

  for (const E of [8, 16, 24, 32, 48, 64]) {
    let taughtCorrect = 0;
    let taughtTotal = 0;
    let heldCorrect = 0;
    let heldTotal = 0;
    let queryMs = 0;
    for (const seed of [1, 2]) {
      const combos = sampleCombos(E, seed);
      const mem = teachCapacityNet(combos, 3, truthFn);
      const taughtSet = new Set(combos.map(([a, b]) => `${a},${b}`));
      for (const q of capQueries(taughtSet, truthFn)) {
        const t0 = performance.now();
        const prediction = mem.predict(q.conditions, seed);
        const { decoded } = prediction;
        quality.record(decoded, prediction.converged, prediction.terminationReason);
        queryMs += performance.now() - t0;
        const ok = decoded.out === q.truth.out;
        if (q.kind === "taught") {
          taughtTotal++;
          if (ok) taughtCorrect++;
        } else {
          heldTotal++;
          if (ok) heldCorrect++;
        }
      }
    }
    out(
      String(E).padStart(4) +
        `${((taughtCorrect / taughtTotal) * 100).toFixed(1)}%`.padStart(16) +
        (heldTotal === 0 ? "N/A (n=0)" : `${((heldCorrect / heldTotal) * 100).toFixed(1)}%`).padStart(16) +
        (queryMs / (taughtTotal + heldTotal)).toFixed(2).padStart(10),
    );
  }
}

// G=0 参照（E=32）
{
  let correct = 0;
  let total = 0;
  for (const seed of [1, 2]) {
    const combos = sampleCombos(32, seed);
    const mem = teachCapacityNet(combos, 0, capTruth);
    for (const q of capQueries(new Set(combos.map(([a, b]) => `${a},${b}`)), capTruth)) {
      const prediction = mem.predict(q.conditions, seed);
        const { decoded } = prediction;
        quality.record(decoded, prediction.converged, prediction.terminationReason);
      total++;
      if (decoded.out === q.truth.out) correct++;
    }
  }
  out(`参照 E=32 且 G=0（断侧重，mod-8）：总准确率 ${((correct / total) * 100).toFixed(1)}%`);
}

// ── C2 装配容量（资源侧）─────────────────────────────────────────
out(`C1 动力学与任一输出拒答（答案质量另列）：${JSON.stringify(quality.snapshot())}`);
const skipC2 = process.argv.includes("--skip-c2");
if (!skipC2) {
out("");
out("═".repeat(64));
out("C2 装配容量：N=8000、16 神经元/势阱（资源侧上限扫描）");
out("═".repeat(64));
out("势阱数".padStart(8) + "检出率".padStart(10) + "捕获完成率".padStart(12) + "误激活".padStart(9) + "捕获ms/次".padStart(12));

const N2 = 8000;
const WS = 16;
for (const count of [100, 200, 400]) {
  const net = new EnergyNetwork({
    neuronCount: N2,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 1.0,
  });
  const clusters = Array.from({ length: count }, (_, k) =>
    Array.from({ length: WS }, (_, i) => k * WS + i),
  );
  const t0 = performance.now();
  for (const c of clusters) {
    hebbianLearn(net, c, 8);
    hebbianLearn(net, [c[0]!, c[1]!], 2);
  }
  const learnMs = performance.now() - t0;
  const t1 = performance.now();
  const wells = detectWells(net);
  const detectMs = performance.now() - t1;
  // 检出率
  let found = 0;
  for (const c of clusters) {
    const cs = new Set(c);
    if (
      wells.some((w) => {
        const inter = w.memberNeuronIds.filter((id) => cs.has(id)).length;
        return inter / new Set([...c, ...w.memberNeuronIds]).size >= 0.8;
      })
    ) {
      found++;
    }
  }
  // 捕获
  let completion = 0;
  let stray = 0;
  let settleMs = 0;
  let nonconverged = 0;
  for (const c of clusters) {
    const cue = c.slice(0, WS / 4);
    const t2 = performance.now();
    const r = net.settle(cue);
    if (!r.converged) nonconverged++;
    settleMs += performance.now() - t2;
    const active = new Set(r.activeNeurons);
    let hit = 0;
    for (const id of c) if (active.has(id)) hit++;
    completion += hit / c.length;
    for (const id of active) if (!cs_has(c, id)) stray++;
  }
  out(
    String(count).padStart(8) +
      `${((found / count) * 100).toFixed(0)}%`.padStart(10) +
      `${((completion / count) * 100).toFixed(1)}%`.padStart(12) +
      (stray / count).toFixed(2).padStart(9) +
      (settleMs / count).toFixed(1).padStart(12) +
      `  [learn ${learnMs.toFixed(0)}ms, detect ${detectMs.toFixed(0)}ms] nonconverged=${nonconverged}/${count}; refusal=N/A (capture completion is reported)`,
  );
}
} // end if (!skipC2)

function cs_has(c: number[], id: number): boolean {
  return id >= c[0]! && id <= c[c.length - 1]!;
}

mkdirSync("runs", { recursive: true });
writeFileSync("runs/capacity-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/capacity-v1.log`);
