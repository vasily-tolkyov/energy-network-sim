import { test } from "node:test";
import assert from "node:assert/strict";
import { SceneParser, type PerceptionSpec } from "../src/perception/scene-parser.js";
import { JevApiBackend, MockBackend } from "../src/perception/decision-backend.js";
import { ActionExecutor, type ExecutionSpec } from "../src/execution/action-executor.js";
import { TransitionMemory } from "../src/planning/transition-memory.js";
import { collectTransitions } from "../src/planning/collect.js";
import { planGoal } from "../src/planning/planner.js";
import { OrderedChainBench, CHAIN_SPACE } from "../src/topics/ordered-chain-world.js";

const CHAIN_PERCEPTION: PerceptionSpec = {
  dims: [
    {
      name: "node",
      values: [0, 1, 2, 3, 4].map(v => ({ value: v, label: `节点${["零", "一", "二", "三", "四"][v]!}` })),
      ask: "当前位于哪个节点？",
    },
  ],
  confidenceThreshold: 0.6,
};

const CHAIN_EXECUTION: ExecutionSpec = {
  environment: "有序链世界",
  actions: [
    { dimension: "advance", commands: { "0": "保持原地观察 {state}", "1": "向前推进一格（{state}）" } },
  ],
};

test("场景解析：过门槛维度进帧，低置信维度如实报 unknown", async () => {
  const parser = new SceneParser(new MockBackend());
  const got = await parser.parse("当前位于节点三。", CHAIN_PERCEPTION);
  assert.deepEqual(got.frame, { node: 3 });
  assert.deepEqual(got.unknownDims, []);
  assert.equal(got.details.node!.label, "节点三");
  // 场景不含任何节点标签：Mock 低置信 → 如实 unknown，不进帧
  const got2 = await parser.parse("这里有一盏灯亮着。", CHAIN_PERCEPTION);
  assert.deepEqual(got2.frame, {});
  assert.deepEqual(got2.unknownDims, ["node"]);
});

test("JevApiBackend 未配置 key 时响亮抛错（禁止静默降级）", async () => {
  const api = new JevApiBackend(null, null);
  await assert.rejects(() => api.ask("状态", { kind: "noul", text: "灯亮着吗？" }), /未配置/);
});

test("动作执行映射：命令渲染、状态代入、可行性失败如实上报", async () => {
  const executor = new ActionExecutor(new MockBackend());
  const spec: ExecutionSpec = {
    ...CHAIN_EXECUTION,
    actions: [{ ...CHAIN_EXECUTION.actions[0]!, feasibilityAsk: "推进一格" }],
  };
  const ok = await executor.map({ advance: 1 }, "节点二，可以推进一格", spec);
  assert.equal(ok.feasible, true);
  assert.ok(ok.command.includes("向前推进一格"));
  assert.ok(ok.command.includes("节点二"));
  // Mock 对"此路塌方无法推进"类文本无可行命中 → 可行性失败
  const bad = await executor.map({ advance: 1 }, "此路塌方", spec);
  assert.equal(bad.feasible, false);
  assert.ok(bad.reason);
});

test("端到端：文字场景 → 起始帧 → 学习 → 规划 → 逐步执行命令", async () => {
  // 文字场景选定动力学链的起始状态
  const scene = "当前位于节点一。";
  const parser = new SceneParser(new MockBackend());
  const perceived = await parser.parse(scene, CHAIN_PERCEPTION);
  assert.deepEqual(perceived.frame, { node: 1 });
  // 学习转移（与感知前端无关的既有链路）
  const model = new TransitionMemory(CHAIN_SPACE);
  collectTransitions(model, new OrderedChainBench(), 40, 1);
  const plan = planGoal(model, perceived.frame, { node: 4 }, 1);
  assert.equal(plan.status, "found");
  assert.deepEqual(plan.steps.map(s => s.next?.node), [2, 3, 4]);
  // 规划链逐步映射为可执行命令，全链可行
  const executor = new ActionExecutor(new MockBackend());
  const records = await executor.mapChain(
    plan.steps.map(s => s.action.values),
    i => `节点${["零", "一", "二", "三", "四"][(plan.steps[i]!.state.node ?? 0)]!}`,
    CHAIN_EXECUTION,
  );
  assert.equal(records.length, 3);
  assert.ok(records.every(r => r.feasible));
  assert.ok(records[0]!.command.includes("节点一"));
  assert.ok(records[2]!.command.includes("节点三"));
});
