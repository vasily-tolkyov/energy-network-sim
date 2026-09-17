import { mkdirSync, writeFileSync } from "node:fs";
import { EnergyNetwork } from "../index.js";
import { ChannelMap } from "./channels.js";
import { extractInfluences } from "./r2.js";
import { RuleMemory } from "./r3.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  queries,
  type Outcomes,
  type Query,
} from "./world.js";

/**
 * R1/R2A/R2B/R3 预测原型实机实验。
 *
 * 管线：
 * - R1：每次经验（条件∪结果共激活）赫布绑定进网络，轮末恢复静息；
 * - R2A/R2B：逐组比较条件/结果读出，变化标志与数值分离，得出影响因素；
 * - R3：侧重绑定（影响通道 ×(1+G) 增益）+ 结果通道互斥抑制；
 * - 预测：未教学查询 → 局部退火读出结果档位 → 与地面真理比对。
 *
 * 消融：G=0（断侧重）、greedy（纯贪心读出、无退火竞争）、
 * 无查询学习（关响应模式学习）。
 * 运行：npm run build && node dist/src/prototype/runner.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const conditionMap = new ChannelMap(CONDITION_SPECS.map((s) => ({ ...s })));
const outcomeMap = new ChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })));
const TOTAL_NEURONS = conditionMap.neuronCount + outcomeMap.neuronCount;

const NET_PARAMS = {
  neuronCount: TOTAL_NEURONS,
  activationEnergy: 1.0,
  maintenanceEnergy: 0.5, // θ = 1.5：基础场（6 通道 × 0.4 ≈ 2.4）可越阈，竞争交给 Γ+退火
  learningRate: 0.1,
  maxWeight: 3.0,
} as const;
const R1_REPEATS = 4; // 基础耦合 0.4
const R3_REPEATS = 4; // 侧重增强总量 η·G·m²·4
const GAMMA = 3.0; // 结果通道内互斥抑制：保证退火读出单档
const QUERY_LEARN_REPEATS = 1; // 响应模式学习的较小步长
const OUTCOME_SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 3 };

interface RunConfig {
  readonly label: string;
  readonly gain: number;
  readonly readout: "annealed" | "greedy";
  readonly learnFromQueries: boolean;
}

interface Score {
  correct: number;
  wrong: number;
  ambiguous: number;
  total: number;
  coarseCorrect: number; // 只评 rebound 档（粗粒度"是否反弹"）
}

function scoreQuery(decoded: Record<string, number | null | "ambiguous">, truth: Outcomes): "correct" | "wrong" | "ambiguous" {
  const reb = decoded.rebound;
  if (reb === "ambiguous" || decoded.reboundSpeed === "ambiguous") return "ambiguous";
  if (reb === null) return "wrong"; // 无预测视为未答对
  if (truth.rebound === 0) return reb === 0 ? "correct" : "wrong";
  if (reb !== 1) return "wrong";
  return decoded.reboundSpeed === truth.reboundSpeed ? "correct" : "wrong";
}

function runConfig(cfg: RunConfig, seeds: number[]): void {
  out(`\n── ${cfg.label} ──`);
  const perKind: Record<string, Score> = {};
  const failures: string[] = [];
  for (const seed of seeds) {
    const net = new EnergyNetwork(NET_PARAMS);
    const mem = new RuleMemory(net, conditionMap, outcomeMap, {
      r1Repeats: R1_REPEATS,
      influenceGain: cfg.gain,
      r3Repeats: R3_REPEATS,
      gamma: GAMMA,
    });
    // —— 教学：R1 绑定 + R2A/R2B 比较 + R3 侧重 ——
    for (const group of curriculum()) {
      for (const pair of group.pairs) {
        for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, R1_REPEATS);
      }
      const influence = extractInfluences(
        conditionMap.channelNames(),
        group.pairs,
        OUTCOME_SPAN,
      );
      // §7.1 类比：k_j = k_h·(1 + G·m_j²)，m_j 为 R2A 保留的结果变化幅度
      const boost: Record<string, number> = {};
      for (const ch of influence.influential) {
        const m = influence.magnitude[ch] ?? 0;
        boost[ch] = NET_PARAMS.learningRate * cfg.gain * m * m;
      }
      for (const pair of group.pairs) {
        for (const e of [pair.e0, pair.e1]) {
          mem.bindInfluence(e, boost, R3_REPEATS);
        }
      }
      if (seed === seeds[0]) {
        out(
          `  组 ${group.name.padEnd(16)} 变化候选=[${influence.changeCandidates}] ` +
            `影响因素=[${influence.influential.map((ch) => `${ch}(m=${(influence.magnitude[ch] ?? 0).toFixed(2)})`)}] ` +
            `首对结果变化=${influence.perPair[0]?.outcomeChanged ?? "?"}`,
        );
      }
    }
    // —— 查询：预测与评分 ——
    for (const q of queries()) {
      const predicted =
        cfg.readout === "annealed" ? mem.predict(q.conditions, seed).decoded : mem.predictGreedy(q.conditions).decoded;
      if (cfg.learnFromQueries) mem.learnFromQuery(q.conditions, predicted, QUERY_LEARN_REPEATS);
      const verdict = scoreQuery(predicted, q.truth);
      const bucket = (perKind[q.kind] ??= { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarseCorrect: 0 });
      bucket[verdict]++;
      bucket.total++;
      if (predicted.rebound === q.truth.rebound) bucket.coarseCorrect++;
      if (verdict !== "correct" && seed === seeds[0] && failures.length < 12) {
        failures.push(
          `    [${q.kind}] 查询 ${JSON.stringify(q.conditions)} → 预测 ${JSON.stringify(predicted)}（真值 ${JSON.stringify(q.truth)}）`,
        );
      }
    }
  }
  let correct = 0;
  let total = 0;
  let coarse = 0;
  for (const kind of ["seen-combo", "unseen-color", "unseen-combo", "miss-wall"] as const) {
    const s = perKind[kind] ?? { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarseCorrect: 0 };
    correct += s.correct;
    total += s.total;
    coarse += s.coarseCorrect;
    out(
      `  ${kind.padEnd(13)} 细粒度 ${String(s.correct).padStart(3)}/${s.total}（${((s.correct / s.total) * 100).toFixed(1)}%）` +
        `  粗粒度反弹 ${((s.coarseCorrect / s.total) * 100).toFixed(1)}%  错 ${s.wrong}  歧义 ${s.ambiguous}`,
    );
  }
  out(`  总计：细粒度 ${correct}/${total}（${((correct / total) * 100).toFixed(1)}%），粗粒度反弹 ${((coarse / total) * 100).toFixed(1)}%`);
  if (failures.length > 0) {
    out("  失败样例（seed 首个，至多 12 条）：");
    for (const f of failures) out(f);
  }
}

out("R1/R2A/R2B/R3 预测原型实机实验");
out(`网络 ${TOTAL_NEURONS} 神经元（条件 ${conditionMap.neuronCount} + 结果 ${outcomeMap.neuronCount}），θ=1.5，η=0.1`);
out(`课程：${curriculum().length} 组控制变量实验；查询：${queries().length} 个（含未教学组合与错过墙对照）`);

const seeds = [1, 2, 3];
const configs: RunConfig[] = [
  { label: "完整模型（G=3，退火读出，查询参与学习）", gain: 3, readout: "annealed", learnFromQueries: true },
  { label: "消融 G=0（断 R2B 侧重）", gain: 0, readout: "annealed", learnFromQueries: true },
  { label: "消融 greedy（纯贪心读出，无退火竞争）", gain: 3, readout: "greedy", learnFromQueries: true },
  { label: "消融 无查询学习", gain: 3, readout: "annealed", learnFromQueries: false },
];
for (const cfg of configs) runConfig(cfg, seeds);

mkdirSync("runs", { recursive: true });
writeFileSync("runs/proto-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/proto-v1.log`);
