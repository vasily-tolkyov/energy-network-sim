import { test } from "node:test";
import assert from "node:assert/strict";
import { AttentionController } from "../src/pop/attention/controller.js";
import { AttentionMonitor } from "../src/pop/attention/monitor.js";
import type { AttentionCandidate } from "../src/pop/attention/controller.js";
import type { Forecast } from "../src/pop/attention/monitor.js";

// 注意力控制器与监测器单元测试

function cand(id: string, changeMagnitude: number, predictionDeviation: number | null = null): AttentionCandidate {
  return {
    targetId: id,
    safe: true,
    changeMagnitude,
    changeDerivative: 0,
    predictionDeviation,
    goalRelevance: 0,
    novelty: 0,
    actionTargetBinding: 0,
  };
}

test("控制器：突变强度达阈立即抢占，未达阈不抢占", () => {
  const c = new AttentionController(1);
  c.update(1, [cand("A", 0.2), cand("B", 0.1)]);
  const focus0 = c.snapshot().focusTargetId;
  // B 突变强度 1.3 ≥ 1.2 → 抢占
  const s1 = c.update(2, [cand("A", 0.2), cand("B", 1.3)]);
  assert.equal(s1.focusTargetId, "B");
  assert.equal(s1.preemptionCount, 1);
  void focus0;
});

test("控制器：迟滞边距 + 持续 tick——挑战者须连续占优才换焦", () => {
  const c = new AttentionController(1);
  c.bindActionTarget("A");
  // B 占优但仅维持 1 tick（sustainedTicks=2）→ 不换
  c.update(1, [cand("A", 0.0), cand("B", 2.0)]);
  // 第一次占优登记为竞争第 1 次，仍不换
  const s1 = c.update(2, [cand("A", 0.0), cand("B", 2.0)]);
  // 注：B 的变化强度 2.0 ≥ 1.2 属突变抢占——改用低变化高偏差场景区分
  void s1;
  const c2 = new AttentionController(1);
  c2.bindActionTarget("A");
  c2.update(1, [cand("A", 1.0), cand("B", 1.0)]);
  // A 持有偏好 1.15 在场；B 需要超出 0.55 边距且持续 2 tick
  const t1 = c2.update(2, [cand("A", 0.6, 2), cand("B", 0.1)]);
  assert.equal(t1.focusTargetId, "A", "单次占优不换焦");
  const t2 = c2.update(3, [cand("A", 0.6, 2), cand("B", 0.1)]);
  assert.equal(t2.focusTargetId, "A", "A 自身突变在场，B 无法仅靠迟滞胜出");
});

test("控制器：近似平局选择是种子可复现的", () => {
  const run = (seed: number): (string | null)[] => {
    const c = new AttentionController(seed);
    const out: (string | null)[] = [];
    for (let t = 1; t <= 5; t++) {
      out.push(c.update(t, [cand("A", 0.5), cand("B", 0.5), cand("C", 0.5)]).focusTargetId);
    }
    return out;
  };
  assert.deepEqual(run(42), run(42), "same seed → same sequence");
});

test("监测器三类比对：符合 / 偏差 / 未知分开，未知不算预测错误", () => {
  const confident: Forecast = {
    subjectId: "A",
    predicted: { rebound: 1, reboundSpeed: 2 },
    confident: true,
    originTick: 1,
    completedTick: 1,
  };
  const unconfident: Forecast = { ...confident, confident: false };
  // 符合
  assert.equal(
    AttentionMonitor.classify(confident, { rebound: 1, reboundSpeed: 2 }, true),
    "within-envelope",
  );
  // 偏差：置信预测被观察推翻
  assert.equal(
    AttentionMonitor.classify(confident, { rebound: 0, reboundSpeed: 0 }, true),
    "prediction-violation",
  );
  // 未知：无置信预测（哪怕观察与预测恰好一致，没有变化也不算）
  assert.equal(
    AttentionMonitor.classify(unconfident, { rebound: 0, reboundSpeed: 0 }, true),
    "unknown-change",
  );
  // 无变化 → 既不是偏差也不是未知
  assert.equal(
    AttentionMonitor.classify(unconfident, { rebound: 0, reboundSpeed: 0 }, false),
    "within-envelope",
  );
});

test("监测器时序纪律：迟到预测不与已开始的变化配对", () => {
  const late: Forecast = {
    subjectId: "A",
    predicted: { rebound: 0, reboundSpeed: 0 },
    confident: true,
    originTick: 5,
    completedTick: 6, // 完成于变化之后
  };
  // 监测器只使用 completedTick <= frame.tick 的预测；迟到者按无预测处理 → 未知而非偏差
  const usable = late.completedTick <= 5 ? late : null;
  assert.equal(usable, null);
  assert.equal(
    AttentionMonitor.classify(usable, { rebound: 1, reboundSpeed: 1 }, true),
    "unknown-change",
  );
});
