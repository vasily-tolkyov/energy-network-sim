import { mkdirSync, writeFileSync } from "node:fs";
import { PopChannelMap } from "../popmap.js";
import { PopRuleMemory } from "../popmemory.js";
import { R2PopLayer } from "../r2pop.js";
import { AttentionMonitor } from "./monitor.js";
import type { FrameObject } from "./monitor.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  groundTruth,
  type Conditions,
} from "../../prototype/world.js";

/**
 * 注意力世界流实验：三个持续存在的小球，按脚本改变条件；
 * 注意力监测器逐帧处理（预测→观察→三类比对→门控学习），
 * 对照三档：门控写观察 / 闭眼写预测（旧行为） / 不写。
 * 运行：npm run build && node dist/src/pop/attention/runner-attention.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};

const cm = new PopChannelMap(CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const OUTCOME_SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 3 };

/** 预训练：与 runner-pop 相同的课程教学（含换对互斥与神经化 R2 侧重） */
function pretrain(): PopRuleMemory {
  const mem = new PopRuleMemory(cm, om, { maxRules: 192 });
  const r2 = new R2PopLayer(cm, om);
  for (const group of curriculum()) {
    const magSum = new Map<string, number>();
    const magCount = new Map<string, number>();
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
      for (const spec of OUTCOME_SPECS) {
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

function ball(
  id: string,
  conditions: Conditions,
): FrameObject {
  return { id, conditions, outcomes: groundTruth(conditions) };
}

/** 脚本化世界流（确定性）：每只球的条件随时间演化，含未见组合 */
function* stream(): Generator<{ tick: number; objects: FrameObject[] }> {
  const A = { color: 1, direction: 0, damping: 0, hitWall: 1, speed: 1, wallStiffness: 1 };
  const B = { color: 2, direction: 0, damping: 0, hitWall: 1, speed: 2, wallStiffness: 1 };
  const C = { color: 3, direction: 0, damping: 0, hitWall: 1, speed: 1, wallStiffness: 2 };
  const events: Record<number, () => void> = {
    10: () => { A.speed = 3; }, // 已学组合，结果变化可预测
    20: () => { B.hitWall = 0; }, // 门控翻转：反弹 → 不弹
    30: () => { A.wallStiffness = 2; }, // (速度3,刚度2) 未见组合 → 未知变化
    45: () => { C.speed = 2; }, // 已学组合
    60: () => { B.hitWall = 1; }, // 门控恢复
  };
  for (let tick = 1; tick <= 70; tick++) {
    events[tick]?.();
    yield {
      tick,
      objects: [
        ball("A", { ...A }),
        ball("B", { ...B }),
        ball("C", { ...C }),
      ],
    };
  }
}

/** 探针集：定时评估全网预测准确率（含 t=30 后的未见组合） */
function probeSet(): { conditions: Conditions; truth: ReturnType<typeof groundTruth>; key: string }[] {
  const probes: { conditions: Conditions; truth: ReturnType<typeof groundTruth>; key: string }[] = [];
  for (const speed of [1, 2, 3]) {
    for (const stiffness of [1, 2]) {
      for (const hitWall of [1, 0]) {
        const c = { color: 1, direction: 0, damping: 0, hitWall, speed, wallStiffness: stiffness };
        probes.push({ conditions: c, truth: groundTruth(c), key: `s${speed}k${stiffness}h${hitWall}` });
      }
    }
  }
  return probes;
}

function accuracy(mem: PopRuleMemory, probes: ReturnType<typeof probeSet>): number {
  let correct = 0;
  for (const p of probes) {
    const { decoded } = mem.predict(p.conditions, 1);
    if (decoded.rebound === p.truth.rebound && (p.truth.rebound === 0 || decoded.reboundSpeed === p.truth.reboundSpeed)) {
      correct++;
    }
  }
  return correct / probes.length;
}

for (const mode of ["gated", "ungated", "none"] as const) {
  out(`\n── 模式 ${mode} ──`);
  const mem = pretrain();
  const monitor = new AttentionMonitor(mem, { learnRepeats: 2, learnMode: mode }, 1);
  const probes = probeSet();
  const baseAcc = accuracy(mem, probes);
  out(`  预训练基线准确率：${(baseAcc * 100).toFixed(1)}%（规则 ${mem.ruleCount} 条）`);
  const curve: string[] = [];
  for (const frame of stream()) {
    monitor.accept(frame);
    if (frame.tick % 10 === 0) {
      curve.push(`t${frame.tick}:${(accuracy(mem, probes) * 100).toFixed(0)}%`);
    }
  }
  out(`  滚动准确率：${curve.join(" → ")}`);
  const notices = monitor.notices.map(
    (n) => `t${n.tick} ${n.kind === "prediction-violation" ? "偏差" : "未知"}@${n.subjectId}(预测完成t${n.forecastCompletedTick ?? "无"})`,
  );
  out(`  通知（${notices.length}）：${notices.join("；") || "无"}`);
  out(`  焦点历史：切换 ${monitor.controller.snapshot().switchCount} 次，抢占 ${monitor.controller.snapshot().preemptionCount} 次`);
}

mkdirSync("runs", { recursive: true });
writeFileSync("runs/attention-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/attention-v1.log`);
