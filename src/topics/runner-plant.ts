import { mkdirSync, writeFileSync } from "node:fs";
import { EnergyNetwork } from "../index.js";
import { ChannelMap } from "../prototype/channels.js";
import { extractInfluences } from "../prototype/r2.js";
import { RuleMemory } from "../prototype/r3.js";
import {
  PLANT_CONDITION_SPECS,
  PLANT_OUTCOME_SPECS,
  plantCurriculum,
  plantQueries,
  type POutcomes,
} from "./plant-world.js";

/**
 * 第二主题（植物生长）的学习-预测实机实验。
 * 原型机制（channels/r2/r3、网络引擎）完全冻结，本文件只定义新领域数据并复用。
 * 运行：npm run build && node dist/src/topics/runner-plant.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const conditionMap = new ChannelMap(PLANT_CONDITION_SPECS.map((s) => ({ ...s })));
const outcomeMap = new ChannelMap(PLANT_OUTCOME_SPECS.map((s) => ({ ...s })));
const TOTAL = conditionMap.neuronCount + outcomeMap.neuronCount;

// 与冻结原型完全相同的参数（θ=1.5，η=0.1，base 0.4，γ=3.0，G=3，R1/R3 各 4 轮）
const NET_PARAMS = {
  neuronCount: TOTAL,
  activationEnergy: 1.0,
  maintenanceEnergy: 0.5,
  learningRate: 0.1,
  maxWeight: 3.0,
} as const;
const OUTCOME_SPAN: Record<string, number> = { alive: 1, growth: 3 };

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

function score(decoded: Record<string, number | null | "ambiguous">, truth: POutcomes): "correct" | "wrong" | "ambiguous" {
  if (decoded.alive === "ambiguous" || decoded.growth === "ambiguous") return "ambiguous";
  if (decoded.alive === null) return "wrong";
  if (truth.alive === 0) return decoded.alive === 0 ? "correct" : "wrong";
  if (decoded.alive !== 1) return "wrong";
  return decoded.growth === truth.growth ? "correct" : "wrong";
}

function runConfig(cfg: RunConfig, seeds: number[]): void {
  out(`\n── ${cfg.label} ──`);
  const perKind: Record<string, Score> = {};
  const failures: string[] = [];
  for (const seed of seeds) {
    const net = new EnergyNetwork(NET_PARAMS);
    const mem = new RuleMemory(net, conditionMap, outcomeMap, {
      r1Repeats: 4,
      influenceGain: cfg.gain,
      r3Repeats: 4,
      gamma: 3.0,
    });
    for (const group of plantCurriculum()) {
      for (const pair of group.pairs) {
        for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
      }
      const influence = extractInfluences(conditionMap.channelNames(), group.pairs, OUTCOME_SPAN);
      const boost: Record<string, number> = {};
      for (const ch of influence.influential) {
        const m = influence.magnitude[ch] ?? 0;
        boost[ch] = NET_PARAMS.learningRate * cfg.gain * m * m;
      }
      for (const pair of group.pairs) {
        for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
      }
      if (seed === seeds[0]) {
        out(
          `  组 ${group.name.padEnd(18)} 变化候选=[${influence.changeCandidates}] → ` +
            `影响因素=[${influence.influential.map((ch) => `${ch}(m=${(influence.magnitude[ch] ?? 0).toFixed(2)})`)}]`,
        );
      }
    }
    for (const q of plantQueries()) {
      const decoded =
        cfg.readout === "annealed" ? mem.predict(q.conditions, seed).decoded : mem.predictGreedy(q.conditions).decoded;
      if (cfg.learnFromQueries) mem.learnFromQuery(q.conditions, decoded, 1);
      const verdict = score(decoded, q.truth);
      const bucket = (perKind[q.kind] ??= { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarseCorrect: 0 });
      bucket[verdict]++;
      bucket.total++;
      if (decoded.alive === q.truth.alive) bucket.coarseCorrect++;
      if (verdict !== "correct" && seed === seeds[0] && failures.length < 10) {
        failures.push(
          `    [${q.kind}] ${JSON.stringify(q.conditions)} → ${JSON.stringify(decoded)}（真值 ${JSON.stringify(q.truth)}）`,
        );
      }
    }
  }
  let correct = 0;
  let total = 0;
  let coarse = 0;
  for (const kind of ["seen-combo", "unseen-species", "unseen-combo", "pest-control"] as const) {
    const s = perKind[kind] ?? { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarseCorrect: 0 };
    correct += s.correct;
    total += s.total;
    coarse += s.coarseCorrect;
    out(
      `  ${kind.padEnd(15)} 细粒度 ${String(s.correct).padStart(3)}/${s.total}（${((s.correct / s.total) * 100).toFixed(1)}%）` +
        `  粗粒度存活 ${((s.coarseCorrect / s.total) * 100).toFixed(1)}%  错 ${s.wrong}  歧义 ${s.ambiguous}`,
    );
  }
  out(`  总计：细粒度 ${correct}/${total}（${((correct / total) * 100).toFixed(1)}%），粗粒度存活 ${((coarse / total) * 100).toFixed(1)}%`);
  if (failures.length > 0) {
    out("  失败样例（seed 首个，至多 10 条）：");
    for (const f of failures) out(f);
  }
}

out("第二主题：植物生长实验（原型机制冻结，迁移学习-预测测试）");
out(`网络 ${TOTAL} 神经元（条件 ${conditionMap.neuronCount} + 结果 ${outcomeMap.neuronCount}），参数与小球原型完全相同`);
out(`课程：${plantCurriculum().length} 组控制变量实验；查询：${plantQueries().length} 个`);

const seeds = [1, 2, 3];
const configs: RunConfig[] = [
  { label: "完整模型（G=3，退火读出，查询参与学习）", gain: 3, readout: "annealed", learnFromQueries: true },
  { label: "消融 G=0（断 R2B 侧重）", gain: 0, readout: "annealed", learnFromQueries: true },
  { label: "消融 greedy（纯贪心读出）", gain: 3, readout: "greedy", learnFromQueries: true },
  { label: "消融 无查询学习", gain: 3, readout: "annealed", learnFromQueries: false },
];
for (const cfg of configs) runConfig(cfg, seeds);

mkdirSync("runs", { recursive: true });
writeFileSync("runs/proto-plant-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/proto-plant-v1.log`);
