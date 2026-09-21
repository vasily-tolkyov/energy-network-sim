/** 机房运维世界端到端（迁移性验证：同一段管道代码，只换世界配置）。 */
import { runWorldPipeline } from "./world-pipeline.mjs";
import { SR_COND_DIMS, SR_OUTCOME_DIMS, SR_GRID, srTruth, srDescribe, SR_PERCEPTION } from "../dist/src/topics/server-room-world.js";

const result = await runWorldPipeline({
  condDims: SR_COND_DIMS,
  outcomeDims: SR_OUTCOME_DIMS,
  condNames: SR_COND_DIMS.map(d => d.name),
  outcomeNames: SR_OUTCOME_DIMS.map(d => d.name),
  grid: SR_GRID,
  truth: srTruth,
  describe: srDescribe,
  perception: SR_PERCEPTION,
  openingScene: "我现在在机房里，负载处于中等水平，冷却开着强档，供电正常，有点灰尘。",
  sceneSpec: {
    dims: [
      { name: "load", values: [{ value: 0, label: "闲置" }, { value: 1, label: "低负载" }, { value: 2, label: "中等负载" }, { value: 3, label: "高负载" }], ask: "当前负载处于哪一档？" },
      { name: "cooling", values: [{ value: 0, label: "冷却关闭" }, { value: 1, label: "弱冷却" }, { value: 2, label: "强冷却" }], ask: "当前冷却处于哪一档？" },
      { name: "power", values: [{ value: 0, label: "欠压" }, { value: 1, label: "正常供电" }, { value: 2, label: "过压" }], ask: "当前供电处于哪一档？" },
    ],
    confidenceThreshold: 0.6,
    clarityAsk: (d) => `场景对${d === "load" ? "负载" : d === "cooling" ? "冷却" : "供电"}的描述是否明确无歧义？`,
  },
  outcomeSpan: { health: 2, uptime: 3 },
  budget: 120,
  seed: 1,
  allCombos() {
    const out = [];
    for (const load of SR_GRID.load) for (const cooling of SR_GRID.cooling)
      for (const power of SR_GRID.power) for (const dust of SR_GRID.dust) out.push({ load, cooling, power, dust });
    return out;
  },
  goals: [
    { label: "目标1：健康且中吞吐", target: [["health", 2], ["uptime", 2]], constraints: null },
    { label: "目标2：过压条件下不故障（必须避开冷却关闭）", target: [["health", 1], ["uptime", 1]], constraints: (c) => c.power === 2 },
  ],
});
console.log("\n══ 汇总:", JSON.stringify(result));
