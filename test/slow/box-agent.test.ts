import { test } from "node:test";
import assert from "node:assert/strict";
import { MockBackend } from "../../src/perception/decision-backend.js";
import { BoxAgent } from "../../src/agent/box-agent.js";
import { BOX_START } from "../../src/topics/box-world.js";

test("箱子世界 agent：探索后听懂指令并绕行完成堆垛（Mock 后端）", async () => {
  const agent = new BoxAgent(new MockBackend());
  await agent.explore(1);

  // 指令解析 + 执行：摞两层
  const report = await agent.instruct("把箱子摞起来，摞两层", 1);
  assert.equal(report.planStatus, "found", `规划失败: ${report.note}`);
  assert.ok(report.reached, `执行未到达: ${report.note}`);
  assert.ok(report.steps >= 8, "绕行+搬运至少需要 8 步");
  // 命令链中应有取箱、绕行（下移/上移）、放箱
  assert.ok(report.commands.some(c => c.includes("取箱子")));
  assert.ok(report.commands.some(c => c.includes("放箱子")));
  assert.ok(report.commands.some(c => c.includes("下移")) || report.commands.some(c => c.includes("上移")), "必须有绕行");
});

test("指令含糊时如实拒动；无目标指令如实报无事可做", async () => {
  const agent = new BoxAgent(new MockBackend());
  await agent.explore(1);
  // Mock 对含糊文本命中不了"摞两层"标签 → 低置信 → unknownDims
  const vague = await agent.instruct("随便弄一下", 1);
  assert.equal(vague.planStatus, "unparsed");
  assert.ok(!vague.reached);
});

test("无指令时自设目标：诚实两端（未学过不编造路线；全学过则如实说没有新目标）", async () => {
  // 未探索：自设目标但无已知路线——诚实报告，不编造
  const fresh = new BoxAgent(new MockBackend());
  const r1 = await fresh.idleCuriosity(1);
  assert.notEqual(r1.planStatus, "found", "没学过就不该编出路线");
  assert.ok(!r1.reached);
  // 全探索后：所有状态都学过 → 如实说没有新目标可设
  const agent = new BoxAgent(new MockBackend());
  await agent.explore(1);
  const r2 = await agent.idleCuriosity(1);
  assert.equal(r2.planStatus, "nothing-new", `全探索后应如实报告: ${r2.note}`);
});
