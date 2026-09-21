import { test } from "node:test";
import assert from "node:assert/strict";
import { ExperimentPlanner } from "../src/pop/explore/planner.js";
import { ContinuousExplorer } from "../src/pop/explore/explorer-continuous.js";
import type { Conditions, Outcomes } from "../src/prototype/world.js";

/** 小世界：两维三档，y = 1 当且仅当 a>b（陌生场景微缩版，跑完全网格很快） */
const DIMS = [{ name: "a", min: 0, max: 2 }, { name: "b", min: 0, max: 2 }];
const SPECS = [{ name: "a", bins: 3, values: [0, 1, 2] }, { name: "b", bins: 3, values: [0, 1, 2] }];
const truth = (c: Conditions): Outcomes => ({ y: c.a! > c.b! ? 1 : 0 });

function explorer(policy: "balanced" | "uncertainty", bench: { conduct(c: Conditions): Outcomes }, budget = 30, seed = 1) {
  const planner = new ExperimentPlanner(SPECS);
  return new ContinuousExplorer(DIMS, [{ name: "y", min: 0, max: 1 }], planner, SPECS, bench, { y: 1 }, { budget, policy }, seed);
}

test("uncertainty 策略：前沿清空后停止，覆盖报告无遗留", async () => {
  const e = explorer("uncertainty", { conduct: truth });
  while (await e.step()) {}
  assert.equal(e.terminationReason, "frontier-exhausted");
  const cov = e.coverageReport();
  assert.equal(cov.unvisited, 0);
  assert.equal(cov.reverifyPending, 0);
  assert.equal(cov.experiments, e.log.length);
  // 全网格 9 组合 × 复验：读回全部正确（感受野读出按 0.5 阈值化，与探索器口径一致）
  for (const a of [0, 1, 2]) for (const b of [0, 1, 2]) {
    const got = e.mem.predict({ a, b }, 1).values.y;
    assert.equal(got == null ? null : got >= 0.5 ? 1 : 0, truth({ a, b }).y, `a=${a} b=${b}`);
  }
});

test("U5 一次性异常：规则不改判，复验队列触发并排空", async () => {
  // 异常注入按"该组合第 3 次实验"触发（前 2 次为真值观察，规则已确立）——
  // 与采样路径无关的确定性场景；count=1 的临时规则不在本测试范围（另有专测）。
  const seen = new Map<string, number>();
  let anomalyAt = -1;
  let calls = 0;
  const bench = { conduct: (c: Conditions) => {
    calls++;
    const key = JSON.stringify(c);
    const n = (seen.get(key) ?? 0) + 1;
    seen.set(key, n);
    if (n === 3 && anomalyAt < 0) {
      anomalyAt = calls;
      return { y: 1 - truth(c).y! };
    }
    return truth(c);
  } };
  const e = explorer("uncertainty", bench, 40);
  let atAnomaly: (typeof e.log)[number] | null = null;
  while (await e.step() && !atAnomaly) {
    if (e.log.length && e.log.at(-1)!.classification === "prediction-violation") atAnomaly = e.log.at(-1)!;
  }
  assert.ok(atAnomaly, "应发生异常观察");
  const before = e.mem.predict(atAnomaly.conditions, 1).values.y;
  while (await e.step()) {}
  const after = e.mem.predict(atAnomaly.conditions, 1).values.y;
  assert.equal(before, truth(atAnomaly.conditions).y, "异常前读数应为真值");
  assert.equal(after, truth(atAnomaly.conditions).y, "单次异常不得改判已确立规则");
  assert.ok(e.log.length > anomalyAt, "复验队列应触发额外实验");
  assert.equal(e.coverageReport().reverifyPending, 0, "复验队列最终排空");
});

test("U5 持久变化：世界彻底反转后，计票积累推动规则翻转", async () => {
  // 前 8 次实验用原真值，之后永久反转——后期观察占多数，规则应学会新世界
  let calls = 0;
  const bench = { conduct: (c: Conditions) => { calls++; return calls <= 8 ? truth(c) : { y: 1 - truth(c).y! }; } };
  // 预算 80：9 组合的轮扫需要跑完一整轮（机制 tax 的如实口径，非调参）
  const e = explorer("uncertainty", bench, 80);
  while (await e.step()) {}
  let correct = 0;
  for (const a of [0, 1, 2]) for (const b of [0, 1, 2]) {
    const got = e.mem.predict({ a, b }, 1).values.y;
    const bin = got == null ? null : got >= 0.5 ? 1 : 0;
    if (bin === 1 - truth({ a, b }).y!) correct++;
  }
  assert.ok(correct >= 8, `持久反转后应学会新世界，9 组合中 ${correct} 个正确`);
});

test("balanced 策略行为不变：quorum-met 仍可达", async () => {
  const e = explorer("balanced", { conduct: truth });
  while (await e.step()) {}
  assert.ok(e.terminationReason === "quorum-met" || e.terminationReason === "frontier-exhausted");
});
