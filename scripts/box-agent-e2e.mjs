/** 箱子 agent 真实 Jev e2e：探索 → 中文指令「把箱子摞起来」→ Jev 解析目标 → 规划执行。 */
import { JevApiBackend } from "../dist/src/perception/decision-backend.js";
import { BoxAgent } from "../dist/src/agent/box-agent.js";

const key = process.env.TYPESAFE_API_KEY;
if (!key) throw new Error("set TYPESAFE_API_KEY");
const api = new JevApiBackend("https://api.typesafe.ai/v1/systemone", key);

const agent = new BoxAgent(api);
console.time("探索学习");
agent.explore(1);
console.timeEnd("探索学习");

console.time("指令全流程");
const r = await agent.instruct("把箱子摞起来，摞两层", 1);
console.timeEnd("指令全流程");
console.log("指令:", r.instruction);
console.log("目标:", JSON.stringify(r.goal));
console.log("状态:", r.planStatus, "| 到达:", r.reached, "| 步数:", r.steps, "| 重规划:", r.replans);
console.log("命令链:");
r.commands.forEach((c, i) => console.log(`  ${i + 1}. ${c}`));

console.time("自设目标探索");
const idle = await agent.idleCuriosity(2);
console.timeEnd("自设目标探索");
console.log("无指令自设:", idle.note);
