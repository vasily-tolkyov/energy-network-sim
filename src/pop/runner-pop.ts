import { mkdirSync, writeFileSync } from "node:fs";
import { PopChannelMap } from "./popmap.js";
import { PopRuleMemory } from "./popmemory.js";
import { R2PopLayer } from "./r2pop.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  queries,
  type Outcomes,
} from "../prototype/world.js";
import {
  CAP_CONDITION_SPECS,
  CAP_OUTCOME_SPECS,
  capCurriculum,
  capQueries,
  capTruth,
  capTruthSmooth,
  sampleCombos,
} from "../topics/capacity-world.js";

/**
 * 群体编码 + 规则核版本的预测实验：
 * - P1 小球端到端（与单网 proto-v1 / 三层 layers-v1 对照）；
 * - P2 容量重跑（mod-8 / 平滑两结构 × E 规则数）。
 * 运行：npm run build && node dist/src/pop/runner-pop.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

// ── P1 小球端到端 ────────────────────────────────────────────────
const cm = new PopChannelMap(CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const OUTCOME_SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 3 };

interface RunConfig {
  readonly label: string;
  readonly gain: number;
  readonly learnFromQueries: boolean;
}

function runBall(cfg: RunConfig, seed: number): { fine: number; coarse: number; missWall: number; missTotal: number } {
  const mem = new PopRuleMemory(cm, om, { maxRules: 128 });
  const r2 = new R2PopLayer(cm, om);
  let correct = 0;
  let coarse = 0;
  let missCorrect = 0;
  let missTotal = 0;
  for (const group of curriculum()) {
    const magSum = new Map<string, number>();
    const magCount = new Map<string, number>();
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
      // 互斥从经验学习：对换过的结果档之间写抑制边（替代原预置 Γ）
      for (const spec of OUTCOME_SPECS) {
        const a = pair.e0.outcomes[spec.name]!;
        const b = pair.e1.outcomes[spec.name]!;
        if (a !== b) mem.teachExclusion("outcome", spec.name, a, b, 3.0);
      }
      // R2 神经化：全模式差分，R2A/R2B 双侧读出
      const analysis = r2.analyzePair(pair);
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
          m += Math.abs(delta) / (OUTCOME_SPAN[och] ?? 1);
        }
        // 与读出层口径一致：按全部结果通道数取平均（2），不按变化通道数
        magSum.set(ch, (magSum.get(ch) ?? 0) + m / om.specs.length);
        magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
      }
    }
    const boost: Record<string, number> = {};
    for (const [ch, sum] of magSum) {
      boost[ch] = 0.1 * cfg.gain * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
    }
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  const all = queries();
  for (const q of all) {
    const { decoded } = mem.predict(q.conditions, seed);
    if (cfg.learnFromQueries) mem.learnFromQuery(q.conditions, decoded, 1);
    if (decoded.rebound === q.truth.rebound) coarse++;
    if (decoded.rebound === q.truth.rebound && (q.truth.rebound === 0 || decoded.reboundSpeed === q.truth.reboundSpeed)) correct++;
    if (q.kind === "miss-wall") {
      missTotal++;
      if (decoded.rebound === q.truth.rebound) missCorrect++;
    }
  }
  return {
    fine: correct / all.length,
    coarse: coarse / all.length,
    missWall: missTotal === 0 ? 1 : missCorrect / missTotal,
    missTotal,
  };
}

out("═".repeat(64));
out("P1 群体编码+规则核：小球端到端（pop=4/档，核=4/规则）");
out("═".repeat(64));
out(`网络规模：条件 ${cm.neuronCount} + 结果 ${om.neuronCount} + 核区 128×4（按需分配）= ${cm.neuronCount + om.neuronCount + 512} 神经元`);

for (const cfg of [
  { label: "完整（G=3，查询参与学习）", gain: 3, learnFromQueries: true },
  { label: "消融 G=0", gain: 0, learnFromQueries: true },
  { label: "消融 无查询学习", gain: 3, learnFromQueries: false },
] as const) {
  let fine = 0;
  let coarse = 0;
  let miss = 0;
  let rules = 0;
  for (const seed of [1, 2, 3]) {
    const r = runBall(cfg, seed);
    fine += r.fine;
    coarse += r.coarse;
    miss += r.missWall;
    rules = 0;
  }
  out(
    `  ${cfg.label}：细粒度 ${((fine / 3) * 100).toFixed(1)}%，粗粒度 ${((coarse / 3) * 100).toFixed(1)}%，` +
      `错过墙门控 ${((miss / 3) * 100).toFixed(1)}%`,
  );
  void rules;
}

// ── P2 容量重跑 ──────────────────────────────────────────────────
out("");
out("═".repeat(64));
out("P2 容量重跑（群体编码+规则核版）");
out("═".repeat(64));

const ccm = new PopChannelMap(CAP_CONDITION_SPECS.map((s) => ({ ...s })), 4);
const com = new PopChannelMap(CAP_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const CAP_SPAN = { out: 7 };

function teachCap(combos: readonly [number, number][], truth: typeof capTruth): PopRuleMemory {
  const mem = new PopRuleMemory(ccm, com, { maxRules: 192 });
  const r2 = new R2PopLayer(ccm, com);
  for (const group of capCurriculum(combos, truth)) {
    const magSum = new Map<string, number>();
    const magCount = new Map<string, number>();
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
      for (const spec of CAP_OUTCOME_SPECS) {
        const a = pair.e0.outcomes[spec.name]!;
        const b = pair.e1.outcomes[spec.name]!;
        if (a !== b) mem.teachExclusion("outcome", spec.name, a, b, 3.0);
      }
      const analysis = r2.analyzePair(pair);
      for (const ch of analysis.influentialChannels) {
        const m = Math.abs(analysis.outcomeDelta.out ?? 0) / CAP_SPAN.out;
        magSum.set(ch, (magSum.get(ch) ?? 0) + m);
        magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
      }
    }
    const boost: Record<string, number> = {};
    for (const [ch, sum] of magSum) {
      boost[ch] = 0.1 * 3 * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
    }
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  return mem;
}

for (const [variant, truthFn] of [
  ["mod-8（对抗性）", capTruth],
  ["平滑可加", capTruthSmooth],
] as const) {
  out(`\n真理结构：${variant}`);
  out("E".padStart(4) + "taught 准确率".padStart(16) + "heldout 准确率".padStart(16));
  for (const E of [8, 16, 24, 32, 48, 64]) {
    let taughtCorrect = 0;
    let taughtTotal = 0;
    let heldCorrect = 0;
    let heldTotal = 0;
    for (const seed of [1, 2]) {
      const combos = sampleCombos(E, seed);
      const mem = teachCap(combos, truthFn);
      const taughtSet = new Set(combos.map(([a, b]) => `${a},${b}`));
      for (const q of capQueries(taughtSet, truthFn)) {
        const { decoded } = mem.predict(q.conditions, seed);
        if (q.kind === "taught") {
          taughtTotal++;
          if (decoded.out === q.truth.out) taughtCorrect++;
        } else {
          heldTotal++;
          if (decoded.out === q.truth.out) heldCorrect++;
        }
      }
    }
    out(
      String(E).padStart(4) +
        `${((taughtCorrect / taughtTotal) * 100).toFixed(1)}%`.padStart(16) +
        `${(heldTotal === 0 ? 0 : (heldCorrect / heldTotal) * 100).toFixed(1)}%`.padStart(16),
    );
  }
}
out(`\n容量版网络规模：条件 ${ccm.neuronCount} + 结果 ${com.neuronCount} + 核区 192×4 = ${ccm.neuronCount + com.neuronCount + 768} 神经元`);

mkdirSync("runs", { recursive: true });
writeFileSync("runs/pop-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/pop-v1.log`);
