import { mkdirSync, writeFileSync } from "node:fs";
import { pretrainContinuous } from "../concept/pretrain.js";
import { continuousStream } from "../concept/stream-continuous.js";
import { captureClassify } from "./capture.js";
import { mismatchField } from "./mismatch.js";
import { NeuralFocusNet } from "./neural-focus.js";
import { AttentionController } from "./controller.js";
import { contQueries } from "../concept/world-continuous.js";

/**
 * M4：神经化注意力端到端 + 历史公式控制器对照。
 * 神经化：捕获判定（M2）→ 失配场（mismatchField）→ WTA 择焦（NeuralFocusNet）
 * → 门控学习（写观察）；公式版：历史 AttentionController + 容差分类。
 * 运行：npm run build && node dist/src/pop/attention/runner-neural-attention.js
 */

const lines: string[] = [];
const out = (s = "") => {
  console.log(s);
  lines.push(s);
};
const SEP = "─".repeat(64);

function accuracy(mem: ReturnType<typeof pretrainContinuous>["mem"]): number {
  const queries = contQueries().filter((_, i) => i % 3 === 0); // 抽样评测（成本控制）
  let ok = 0;
  for (const q of queries) {
    const p = mem.predict(q.conditions, 1);
    const rebOk = p.values.rebound !== null && p.values.rebound !== undefined && Math.abs(p.values.rebound - q.truth.rebound!) <= 0.5;
    if (
      rebOk &&
      (q.truth.rebound === 0 ||
        (p.values.reboundSpeed !== null &&
          p.values.reboundSpeed !== undefined &&
          Math.abs(p.values.reboundSpeed - q.truth.reboundSpeed!) <= 0.75))
    ) {
      ok++;
    }
  }
  return ok / queries.length;
}

// ── 神经化注意力 ─────────────────────────────────────────────────
out(SEP);
out("神经化注意力（捕获判定 + 失配场 WTA 择焦）");
out(SEP);
{
  const { mem, enc } = pretrainContinuous();
  const focus = new NeuralFocusNet(["A", "B", "C"], { inertia: 2.0 });
  const prevConditions = new Map<string, string>();
  // 评审 §8.2 修复（两条）：①时序纪律——预测在上一帧结束时生成并缓存
  // （含当时的规则核索引），本帧只比对已存在的缓存预测；修复前是到当前帧
  // 才用当前权重重算旧条件，不构成"变化发生前已完成预测"。
  // ②焦点因果作用——只有焦点对象获得下帧预测（可消融：焦点决定预测谁）。
  let cachedForecast: { subjectId: string; coreIdx: number | null; originTick: number } | null = null;
  const acc: string[] = [];
  for (const frame of continuousStream()) {
    // 焦点先择（基于截至上帧的失配场）
    const prevFocus = focus.focus;
    const winner = focus.select(frame.tick);
    if (winner !== prevFocus || frame.tick === 1) {
      out(`  t${frame.tick} 焦点 ${prevFocus ?? "∅"} → ${winner}`);
    }
    focus.beginFrame();
    for (const obj of frame.objects) {
      const key = JSON.stringify(obj.conditions);
      if (prevConditions.get(obj.id) === key) continue;
      prevConditions.set(obj.id, key);
      // 用缓存的上一帧预测比对（只对焦点对象存在缓存预测）
      const fcCore = cachedForecast && cachedForecast.subjectId === obj.id ? cachedForecast.coreIdx : null;
      // 捕获判定（观察结果 vs 已缓存预测核）
      const cap = captureClassify(mem, obj.conditions, obj.outcomes, fcCore, frame.tick);
      // 失配场 → 写入竞争
      // 失配必须对**预测核**（被违反的期望）测量——捕获核是吻合观察的，失配恒为 0
      focus.setMismatch(obj.id, mismatchField(mem, fcCore, obj.outcomes));
      // 门控学习：偏差/未知写观察
      if (cap.class !== "within-envelope") {
        mem.learnFromObservation(obj.conditions, obj.outcomes, 2);
        out(`  t${frame.tick} ${obj.id} ${cap.class === "prediction-violation" ? "偏差" : "未知"} → 写观察（核 ${fcCore ?? "无"}）`);
      }
    }
    // 帧末：为当前焦点对象生成下一帧预测并缓存（真正在变化前完成）
    const focusObj = frame.objects.find((o) => o.id === winner);
    if (focusObj) {
      const fc = mem.predict(focusObj.conditions, frame.tick);
      cachedForecast = {
        subjectId: winner,
        coreIdx: fc.winningCores.length > 0 ? fc.winningCores[0]! : null,
        originTick: frame.tick,
      };
    } else {
      cachedForecast = null;
    }
    if (frame.tick === 60) acc.push(`t${frame.tick}:${(accuracy(mem) * 100).toFixed(0)}%`);
    void enc;
  }
  out(`  门控学习后准确率曲线：${acc.join(" → ")}`);
}

// ── 公式控制器基线 ───────────────────────────────────────────────
out("");
out(SEP);
out("公式控制器基线（历史 AttentionController + 容差分类）");
out(SEP);
{
  const { mem, enc } = pretrainContinuous();
  const controller = new AttentionController(1);
  const prevOutcomes = new Map<string, Record<string, number>>();
  const sig = (d: string) => enc.sigma(d);
  for (const frame of continuousStream()) {
    const candidates = frame.objects.map((obj) => {
      const prev = prevOutcomes.get(obj.id);
      const changeMag =
        prev === undefined
          ? 0
          : Math.min(
              2,
              Object.keys(obj.outcomes).reduce(
                (sum, ch) => sum + Math.abs((obj.outcomes[ch] ?? 0) - (prev[ch] ?? 0)) / (2 * sig(ch)),
                0,
              ),
            );
      prevOutcomes.set(obj.id, obj.outcomes);
      const changed = prev !== undefined && JSON.stringify(prev) !== JSON.stringify(obj.outcomes);
      if (!changed) {
        return {
          targetId: obj.id, safe: true, changeMagnitude: 0, changeDerivative: 0,
          predictionDeviation: null, goalRelevance: 0, novelty: 0, actionTargetBinding: 0,
        };
      }
      const predicted = mem.predict(obj.conditions, frame.tick);
      const confident = Object.values(predicted.values).every((v) => v !== null);
      const deviation =
        !confident ? null
        : Object.entries(obj.outcomes).some(
            ([ch, v]) => Math.abs((predicted.values[ch] ?? -99) - v) > 2 * sig(ch),
          )
          ? 2
          : 0;
      return {
        targetId: obj.id,
        safe: true,
        changeMagnitude: changeMag,
        changeDerivative: 0,
        predictionDeviation: deviation,
        goalRelevance: 0,
        novelty: 0,
        actionTargetBinding: 0,
      };
    });
    const snap = controller.update(frame.tick, candidates);
    if (frame.tick === 1 || snap.focusTargetId !== controller.snapshot().focusTargetId) {
      out(`  t${frame.tick} 焦点 → ${snap.focusTargetId}`);
    }
  }
  out(`  切换 ${controller.snapshot().switchCount} 次，抢占 ${controller.snapshot().preemptionCount} 次`);
}

mkdirSync("runs", { recursive: true });
writeFileSync("runs/neural-attention-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/neural-attention-v1.log`);
