import { mkdirSync, writeFileSync } from "node:fs";
import { EnergyNetwork } from "../index.js";
import { ChannelMap } from "../prototype/channels.js";
import { RuleMemory } from "../prototype/r3.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  queries,
  type Outcomes,
} from "../prototype/world.js";
import { R1Layer } from "./r1.js";
import { R2Layer } from "./r2.js";

/**
 * 三层神经网络（R1/R2/R3 各自独立成网）端到端预测实验，
 * 与单网原型（runs/proto-v1.log：细粒度 83.3%、粗粒度 100%）对比。
 *
 * 管线：R1 存每次实验的"条件阱→结果阱"势阱对（有向通道）→
 * R2A 结果比较（读出）+ R2B 神经差分（条件阱重合场分层）+ 双源 AND 影响因素 →
 * R3 绑定背景+因素+结果为规则阱，新输入传播做预测。
 * 运行：npm run build && node dist/src/layers/runner-layers.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const conditionMap = new ChannelMap(CONDITION_SPECS.map((s) => ({ ...s })));
const outcomeMap = new ChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })));
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
  coarseCorrect: number;
}

function score(decoded: Record<string, number | null | "ambiguous">, truth: Outcomes): "correct" | "wrong" | "ambiguous" {
  if (decoded.rebound === "ambiguous" || decoded.reboundSpeed === "ambiguous") return "ambiguous";
  if (decoded.rebound === null) return "wrong";
  if (truth.rebound === 0) return decoded.rebound === 0 ? "correct" : "wrong";
  if (decoded.rebound !== 1) return "wrong";
  return decoded.reboundSpeed === truth.reboundSpeed ? "correct" : "wrong";
}

function runConfig(cfg: RunConfig, seeds: number[]): void {
  out(`\n── ${cfg.label} ──`);
  const perKind: Record<string, Score> = {};
  for (const seed of seeds) {
    // R1：经验层，存每次实验的条件阱→结果阱势阱对
    const r1 = new R1Layer(conditionMap, outcomeMap);
    // R2：比较层（R2B 神经差分）
    const r2 = new R2Layer(conditionMap);
    // R3：规则层（独立网络，与 R1/R2 各自的网络分离）
    const r3Net = new EnergyNetwork({
      neuronCount: conditionMap.neuronCount + outcomeMap.neuronCount,
      activationEnergy: 1.0,
      maintenanceEnergy: 0.5,
      learningRate: 0.1,
      maxWeight: 3.0,
    });
    const r3 = new RuleMemory(r3Net, conditionMap, outcomeMap, {
      r1Repeats: 4,
      influenceGain: cfg.gain,
      r3Repeats: 4,
      gamma: 3.0,
    });
    for (const group of curriculum()) {
      const magSum = new Map<string, number>();
      const magCount = new Map<string, number>();
      for (const pair of group.pairs) {
        for (const e of [pair.e0, pair.e1]) {
          const ep = r1.teachEpisode(e, 4, 8);
          r3.teachExperience({ conditions: e.conditions, outcomes: e.outcomes }, 4);
          void ep;
        }
        const analysis = r2.analyzePair(pair);
        for (const ch of analysis.influentialChannels) {
          let m = 0;
          let counted = 0;
          for (const [och, od] of Object.entries(analysis.outcome.diffs)) {
            m += Math.abs(od.delta) / (OUTCOME_SPAN[och] ?? 1);
            counted++;
          }
          magSum.set(ch, (magSum.get(ch) ?? 0) + (counted ? m / counted : 0));
          magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
        }
      }
      const boost: Record<string, number> = {};
      for (const [ch, sum] of magSum) {
        const m = sum / Math.max(1, magCount.get(ch) ?? 1);
        boost[ch] = 0.1 * cfg.gain * m * m;
      }
      for (const pair of group.pairs) {
        for (const e of [pair.e0, pair.e1]) {
          r3.bindInfluence({ conditions: e.conditions, outcomes: e.outcomes }, boost, 4);
        }
      }
      if (seed === seeds[0]) {
        const first = r2.analyzePair(group.pairs[0]!);
        out(
          `  组 ${group.name.padEnd(16)} R2B差分通道=[${first.diffChannels}] ` +
            `影响因素=[${first.influentialChannels}] θHigh=${first.thetaHigh}`,
        );
      }
    }
    for (const q of queries()) {
      const decoded =
        cfg.readout === "annealed" ? r3.predict(q.conditions, seed).decoded : r3.predictGreedy(q.conditions).decoded;
      if (cfg.learnFromQueries) r3.learnFromQuery(q.conditions, decoded, 1);
      const verdict = score(decoded, q.truth);
      const bucket = (perKind[q.kind] ??= { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarseCorrect: 0 });
      bucket[verdict]++;
      bucket.total++;
      if (decoded.rebound === q.truth.rebound) bucket.coarseCorrect++;
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
}

out("三层神经网络（R1/R2/R3 各自独立成网）端到端预测实验");
out("对照：单网原型 proto-v1 细粒度 83.3%、粗粒度 100%");

const seeds = [1, 2, 3];
const configs: RunConfig[] = [
  { label: "完整三层模型（G=3，退火读出，查询参与学习）", gain: 3, readout: "annealed", learnFromQueries: true },
  { label: "消融 G=0（断 R2B 侧重）", gain: 0, readout: "annealed", learnFromQueries: true },
  { label: "消融 无查询学习", gain: 3, readout: "annealed", learnFromQueries: false },
];
for (const cfg of configs) runConfig(cfg, seeds);

mkdirSync("runs", { recursive: true });
writeFileSync("runs/layers-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/layers-v1.log`);
