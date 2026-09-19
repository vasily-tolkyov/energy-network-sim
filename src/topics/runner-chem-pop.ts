import { predictionQuality } from "../pop/prediction-quality.js";
import { mkdirSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { PopChannelMap } from "../pop/popmap.js";
import { PopRuleMemory } from "../pop/popmemory.js";
import { R2PopLayer } from "../pop/r2pop.js";
import type { Conditions, Group, Outcomes } from "../prototype/world.js";
import { CHEM_CONDITION_SPECS, CHEM_OUTCOME_SPECS, chemCurriculum, chemQueries } from "./chem-world.js";

/**
 * 第三主题（化学反应）的独立测试：预测核心机制（PopRuleMemory/R2PopLayer/
 * 网络引擎）完全冻结，本文件只定义新领域数据并复用，外加如实评分与展示。
 * 运行：npm run build && node dist/src/topics/runner-chem-pop.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const cm = new PopChannelMap(CHEM_CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(CHEM_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const OUTCOME_SPAN: Record<string, number> = { reacted: 1, rate: 3 };
const CORE_BASE = cm.neuronCount + om.neuronCount;
const CORE_SIZE = 4;

interface RunConfig {
  readonly label: string;
  readonly gain: number;
  readonly learnFromQueries: boolean;
  /** 断否决开关（2×2 消融，评审 F06）：false 时 bindInfluence 不写否决边 */
  readonly veto?: boolean;
}

function teach(mem: PopRuleMemory, r2: R2PopLayer, gain: number, verbose: boolean, curriculum: Group[], veto = true): void {
  for (const group of curriculum) {
    const magSum = new Map<string, number>();
    const magCount = new Map<string, number>();
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
      for (const spec of CHEM_OUTCOME_SPECS) {
        const a = pair.e0.outcomes[spec.name]!;
        const b = pair.e1.outcomes[spec.name]!;
        if (a !== b) mem.teachExclusion("outcome", spec.name, a, b, 3.0);
      }
      const analysis = r2.analyzePair(pair);
          if (analysis.undecidable) continue; // 不可判定对不进幅度累计（评审 F04）
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
          m += Math.abs(delta) / (OUTCOME_SPAN[och] ?? 1);
        }
        magSum.set(ch, (magSum.get(ch) ?? 0) + m / om.specs.length);
        magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
      }
      if (verbose) {
        out(
          `    对 ${group.manipulated}: 变化=${analysis.outcomeChanged}` +
            ` 条件差分=[${analysis.diffChannels}] → 影响因素=[${analysis.influentialChannels}]` +
            ` 结果档差=${JSON.stringify(analysis.outcomeDelta)}`,
        );
      }
    }
    const boost: Record<string, number> = {};
    for (const [ch, sum] of magSum) {
      boost[ch] = 0.1 * gain * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
    }
    if (verbose) {
      out(
        `  组 ${group.name.padEnd(20)} 影响因素侧重：${JSON.stringify(
          Object.fromEntries(Object.entries(boost).map(([k, v]) => [k, +v.toFixed(3)])),
        )}`,
      );
    }
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4, 1.5, veto);
    }
  }
}

function score(
  decoded: Record<string, number | null | "ambiguous">,
  truth: Outcomes,
): "correct" | "wrong" | "ambiguous" {
  if (decoded.reacted === "ambiguous" || decoded.rate === "ambiguous") return "ambiguous";
  if (decoded.reacted === null) return "wrong";
  if (truth.reacted === 0) return decoded.reacted === 0 ? "correct" : "wrong";
  if (decoded.reacted !== 1) return "wrong";
  return decoded.rate === truth.rate ? "correct" : "wrong";
}

function runChem(cfg: RunConfig, seed: number, verboseR2: boolean, curriculum: Group[]) {
  const mem = new PopRuleMemory(cm, om, { maxRules: 128 });
  const r2 = new R2PopLayer(cm, om);
  teach(mem, r2, cfg.gain, verboseR2, curriculum, cfg.veto ?? true);
  const perKind: Record<string, { correct: number; wrong: number; ambiguous: number; total: number; coarse: number }> =
    {};
  const failures: string[] = [];
  let msTotal = 0;
  let qCount = 0;
  let activeTotal = 0;
  for (const q of chemQueries(curriculum)) {
    const t0 = performance.now();
    const { decoded, activeNeurons } = mem.predict(q.conditions, seed);
    msTotal += performance.now() - t0;
    qCount++;
    activeTotal += activeNeurons.length;
    if (cfg.learnFromQueries) mem.learnFromQuery(q.conditions, decoded, 1);
    const verdict = score(decoded, q.truth);
    const b = (perKind[q.kind] ??= { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarse: 0 });
    b[verdict]++;
    b.total++;
    if (decoded.reacted === q.truth.reacted) b.coarse++;
    if (verdict !== "correct" && failures.length < 8) {
      failures.push(
        `    [${q.kind}] ${JSON.stringify(q.conditions)} → ${JSON.stringify(decoded)}（真值 ${JSON.stringify(q.truth)}）`,
      );
    }
  }
  return { perKind, failures, ruleCount: mem.ruleCount, msPerQuery: msTotal / qCount, activePerQuery: activeTotal / qCount, mem };
}

// ── 展示 1：学习集 ────────────────────────────────────────────────
out("═".repeat(72));
out("第三主题（独立测试）：化学反应实验 —— 预测核心（群体编码+规则核，机制冻结）");
out("═".repeat(72));
out(
  `网络规模：条件 ${cm.neuronCount}（5 通道×档×4）+ 结果 ${om.neuronCount} + 核区 128×${CORE_SIZE} = ${
    cm.neuronCount + om.neuronCount + 128 * CORE_SIZE
  } 神经元`,
);
out("");
out("【学习集】教师程序带领的控制变量实验（每组只操纵一个因素；下为 v1 稀疏门控覆盖版）：");
for (const g of chemCurriculum(false)) {
  out(`  组 ${g.name}（操纵 ${g.manipulated}）：`);
  for (const p of g.pairs) {
    const fmt = (c: Conditions, o: Outcomes) =>
      `cat${c.catalyst} conc${c.concentration} temp${c.temperature} stir${c.stirring} ves${c.vessel}` +
      ` → reacted=${o.reacted} rate=${o.rate}`;
    out(`    ${fmt(p.e0.conditions, p.e0.outcomes)}  ⇄  ${fmt(p.e1.conditions, p.e1.outcomes)}`);
  }
}
out(
  `  （v2 均衡门控覆盖版另加 5 对：催化剂对换在 conc=0/2/3、temp=0 背景各一次，` +
    `温度对换补 conc=2 背景跨门控一对；对照评分见末段）`,
);

// ── 展示 2：R2 因素提取（seed 1，逐对如实打印）────────────────────
out("");
out("【R2 神经化因素提取】（seed 1，逐对）：");
const probe = runChem({ label: "probe", gain: 3, learnFromQueries: false }, 1, true, chemCurriculum(false));
out(`教学后形成规则核 ${probe.ruleCount} 条（v2 课程另加数条，见末段对照）`);

// ── 展示 3：预测过程走查（3 个代表查询）──────────────────────────
out("");
out("【预测过程走查】钳制条件群体 → 局部退火选获胜核 → 读出结果（seed 1，完整模型）：");
const examples: { note: string; conditions: Conditions }[] = [
  {
    note: "教过的组合（基线）",
    conditions: { catalyst: 1, concentration: 1, temperature: 1, stirring: 1, vessel: 0 },
  },
  {
    note: "未见组合：高浓+高温+有催化剂（需要跨经验组合）",
    conditions: { catalyst: 1, concentration: 3, temperature: 3, stirring: 1, vessel: 0 },
  },
  {
    note: "未见组合：无催化剂+高浓+中温（门控应保持关闭）",
    conditions: { catalyst: 0, concentration: 3, temperature: 2, stirring: 1, vessel: 0 },
  },
];
{
  const mem = new PopRuleMemory(cm, om, { maxRules: 128 });
  teach(mem, new R2PopLayer(cm, om), 3, false, chemCurriculum(true));
  for (const ex of examples) {
    const t = chemQueries(chemCurriculum(true)).find(
      (q) => JSON.stringify(q.conditions) === JSON.stringify(ex.conditions),
    )!;
    const { decoded, activeNeurons, energy } = mem.predict(ex.conditions, 1);
    const coreIds = activeNeurons.filter((id) => id >= CORE_BASE);
    const coreBlocks = [...new Set(coreIds.map((id) => Math.floor((id - CORE_BASE) / CORE_SIZE)))];
    const verdict = score(decoded, t.truth);
    out(`  ${ex.note}`);
    out(`    条件 ${JSON.stringify(ex.conditions)}（钳制 ${cm.encode(ex.conditions).length} 神经元）`);
    out(
      `    退火后激活 ${activeNeurons.length} 神经元，获胜核块=${JSON.stringify(coreBlocks)}，能耗=${energy.toFixed(2)}`,
    );
    out(`    预测 ${JSON.stringify(decoded)}，真值 ${JSON.stringify(t.truth)} → ${verdict}`);
  }
}

// ── 展示 4：全量评分 + 消融（v1/v2 两版课程对照）──────────────────
out("");
out("【全量评分】64 查询 × 3 种子，按查询类型分档；v1/v2 两版课程对照（机制不变）：");
for (const variant of [
  { label: "v1 稀疏门控覆盖", curriculum: chemCurriculum(false) },
  { label: "v2 均衡门控覆盖", curriculum: chemCurriculum(true) },
] as const) {
  out(`\n  ══ 课程 ${variant.label} ══`);
  for (const cfg of [
    { label: "完整模型（G=3+否决，查询参与学习）", gain: 3, learnFromQueries: true },
    { label: "消融 断侧重（G=0；零增益通道自动不写否决——真断开）", gain: 0, learnFromQueries: true },
    { label: "消融 断否决（G=3，veto=false）", gain: 3, learnFromQueries: true, veto: false },
    { label: "消融 双断（G=0 + 无否决）", gain: 0, learnFromQueries: true, veto: false },
    { label: "消融 无查询学习", gain: 3, learnFromQueries: false },
  ] as const) {
    out(`\n  ── ${cfg.label} ──`);
    const agg: Record<string, { correct: number; wrong: number; ambiguous: number; total: number; coarse: number }> =
      {};
    let failures: string[] = [];
    let ms = 0;
    let act = 0;
    let rules = 0;
    for (const seed of [1, 2, 3]) {
      const r = runChem(cfg, seed, false, variant.curriculum);
      ms += r.msPerQuery / 3;
      act += r.activePerQuery / 3;
      rules = r.ruleCount;
      if (seed === 1) failures = r.failures;
      for (const [kind, b] of Object.entries(r.perKind)) {
        const a = (agg[kind] ??= { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarse: 0 });
        a.correct += b.correct;
        a.wrong += b.wrong;
        a.ambiguous += b.ambiguous;
        a.total += b.total;
        a.coarse += b.coarse;
      }
    }
    let correct = 0;
    let total = 0;
    let coarse = 0;
    for (const kind of ["seen-combo", "unseen-vessel", "unseen-combo", "no-reaction-control"] as const) {
      const s = agg[kind] ?? { correct: 0, wrong: 0, ambiguous: 0, total: 0, coarse: 0 };
      correct += s.correct;
      total += s.total;
      coarse += s.coarse;
      if (s.total === 0) continue;
      out(
        `    ${kind.padEnd(19)} 细粒度 ${String(s.correct).padStart(3)}/${s.total}（${((s.correct / s.total) * 100).toFixed(1)}%）` +
          `  粗粒度反应判定 ${((s.coarse / s.total) * 100).toFixed(1)}%  错 ${s.wrong}  歧义 ${s.ambiguous}`,
      );
    }
    out(
      `    总计：细粒度 ${correct}/${total}（${((correct / total) * 100).toFixed(1)}%），` +
        `粗粒度 ${((coarse / total) * 100).toFixed(1)}%；核 ${rules} 条；单次预测均时 ${ms.toFixed(1)}ms，平均激活 ${act.toFixed(0)} 神经元`,
    );
    if (failures.length > 0) {
      out("    失败样例（seed 1，至多 8 条）：");
      for (const f of failures) out(f);
    }
  }
}

mkdirSync("runs", { recursive: true });
out(`动力学质量与任一输出拒答（本日志全部 predict 调用，含训练期；答案质量另列）：${JSON.stringify(predictionQuality.snapshot())}`);
writeFileSync("runs/chem-pop-v1.log", lines.join("\n") + "\n");
predictionQuality.reset();
out(`\n日志已写入 runs/chem-pop-v1.log`);
