/** Jev 真实 API 端到端：文字场景 → 起始帧 → 学习 → 规划 → 逐步执行。
 * 场景 A：清晰场景（应全链路走通）；场景 B：含糊场景（应如实 unknown）。
 * 打印全部 Jev 判定细节与本机耗时。 */
import { JevApiBackend } from "../dist/src/perception/decision-backend.js";
import { SceneParser } from "../dist/src/perception/scene-parser.js";
import { ActionExecutor } from "../dist/src/execution/action-executor.js";
import { TransitionMemory } from "../dist/src/planning/transition-memory.js";
import { collectTransitions } from "../dist/src/planning/collect.js";
import { planGoal } from "../dist/src/planning/planner.js";
import { OrderedChainBench, CHAIN_SPACE } from "../dist/src/topics/ordered-chain-world.js";

const key = process.env.TYPESAFE_API_KEY;
if (!key) throw new Error("set TYPESAFE_API_KEY");
const api = new JevApiBackend("https://api.typesafe.ai/v1/systemone", key);

const NODES = ["节点零", "节点一", "节点二", "节点三", "节点四"];
const perceptionSpec = {
  dims: [{ name: "node", values: NODES.map((label, v) => ({ value: v, label })), ask: "机器人当前位于哪个节点？" }],
  confidenceThreshold: 0.6,
  clarityAsk: (dim) => `场景对 ${dim} 的描述是否明确且无歧义？`,
};
const executionSpec = {
  environment: "有序链世界",
  actions: [
    { dimension: "advance", commands: { "0": "向前推进一格（{state}）", "1": "原地观察记录" },
      feasibilityAsk: "向前推进一格在当前条件下是否安全可行？" },
  ],
  confidenceThreshold: 0.5,
};

async function run(scene, label) {
  console.log(`\n══ ${label}：${scene}`);
  const t0 = performance.now();
  const parser = new SceneParser(api);
  const perceived = await parser.parse(scene, perceptionSpec);
  console.log("感知结果:", JSON.stringify(perceived.details), "unknown:", JSON.stringify(perceived.unknownDims));
  if (perceived.unknownDims.length) {
    console.log("→ 起始帧不完整，如实拒动（不强行选值）。耗时", (performance.now() - t0).toFixed(0), "ms");
    return;
  }
  const model = new TransitionMemory(CHAIN_SPACE);
  collectTransitions(model, new OrderedChainBench(), 40, 1);
  const goal = { node: 4 };
  const plan = planGoal(model, perceived.frame, goal, 1);
  console.log("规划:", plan.status, plan.steps.map(s => `${s.state.node}→${s.next ? s.next.node : "?"}`).join(" "));
  if (plan.status !== "found") return;
  const executor = new ActionExecutor(api);
  const records = await executor.mapChain(
    plan.steps.map(s => s.action.values),
    i => `机器人位于${NODES[plan.steps[i].state.node ?? 0]}，链上无障礙。`,
    executionSpec,
  );
  for (const [i, r] of records.entries()) {
    console.log(`  步${i}: [${r.feasible ? "可行" : "不可行"}] ${r.command}（置信 ${r.confidence.toFixed(2)}${r.reason ? "，" + r.reason : ""}）`);
  }
  console.log("总耗时", (performance.now() - t0).toFixed(0), "ms");
}

await run("机器人当前位于节点二，状态正常，准备向节点四前进。", "场景 A 清晰");
await run("机器人可能在一号位附近，也可能是二号位，报告不太一致。", "场景 B 含糊");
