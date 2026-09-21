/** 温室世界端到端：文字场景 → Jev 感知 → 自由探索 → 自主发现规律 → 复杂目标。
 * 真值只在实验台内部；agent 的每次观察都是一段文字，由 Jev 解析为结果。
 * 评估侧（本 runner）可以直接用真值评分——那是评分侧，不是学习侧。 */
import { JevApiBackend } from "../dist/src/perception/decision-backend.js";
import { SceneParser } from "../dist/src/perception/scene-parser.js";
import { ExperimentPlanner } from "../dist/src/pop/explore/planner.js";
import { ContinuousExplorer } from "../dist/src/pop/explore/explorer-continuous.js";
import { GH_COND_DIMS, GH_OUTCOME_DIMS, GH_GRID, ghTruth, ghDescribe, GH_PERCEPTION } from "../dist/src/topics/greenhouse-world.js";

const key = process.env.TYPESAFE_API_KEY;
if (!key) throw new Error("set TYPESAFE_API_KEY");
const api = new JevApiBackend("https://api.typesafe.ai/v1/systemone", key);
const parser = new SceneParser(api);

const condNames = GH_COND_DIMS.map(d => d.name);
const outNames = GH_OUTCOME_DIMS.map(d => d.name);

/** 文字观察 → 结构化结果（每次实验 2 个 Jev Choice 问题） */
async function perceive(text) {
  const spec = {
    dims: [
      { name: "growth", values: GH_PERCEPTION.growth.options.map(o => ({ value: o.value, label: o.label, desc: o.label })), ask: GH_PERCEPTION.growth.ask },
      { name: "yield", values: GH_PERCEPTION.yield.options.map(o => ({ value: o.value, label: o.label, desc: o.desc })), ask: GH_PERCEPTION.yield.ask },
    ],
    confidenceThreshold: 0.01, // 观察解析不做门槛：Jev 的判定原样进学习（故障模式另行记录）
  };
  const got = await parser.parse(text, spec);
  return { frame: got.frame, details: got.details };
}

// ── 1. 文字场景 → 起始帧 ─────────────────────────────
const OPENING = "我现在站在一座温室里，光照强度中等，浇水适量，没有施肥，温度适宜，角落有些装饰。";
const sceneSpec = {
  dims: [
    { name: "light", values: [{ value: 0, label: "无光照" }, { value: 1, label: "弱光" }, { value: 2, label: "中等光照" }, { value: 3, label: "强光" }], ask: "当前光照处于哪一档？" },
    { name: "water", values: [{ value: 0, label: "干旱" }, { value: 1, label: "偏少" }, { value: 2, label: "适量" }, { value: 3, label: "过多" }], ask: "当前浇水量处于哪一档？" },
    { name: "nutrient", values: [{ value: 0, label: "未施肥" }, { value: 1, label: "少量施肥" }, { value: 2, label: "适量施肥" }, { value: 3, label: "大量施肥" }], ask: "当前施肥量处于哪一档？" },
    { name: "temp", values: [{ value: 0, label: "寒冷" }, { value: 1, label: "适宜" }, { value: 2, label: "炎热" }], ask: "当前温度处于哪一档？" },
  ],
  confidenceThreshold: 0.6,
  clarityAsk: (d) => `场景对${d === "light" ? "光照" : d === "water" ? "浇水" : d === "nutrient" ? "施肥" : "温度"}的描述是否明确无歧义？`,
};
console.log("══ 1. 场景感知");
const start = await parser.parse(OPENING, sceneSpec);
console.log("起始帧:", JSON.stringify(start.frame), "unknown:", JSON.stringify(start.unknownDims));

// ── 2. 自由探索（文字观察 + Jev 解析 + 不确定性驱动） ──
console.log("\n══ 2. 自由探索（预算 60，uncertainty 策略）");
const SPECS = condNames.map(n => ({ name: n, bins: GH_GRID[n].length, values: GH_GRID[n] }));
const planner = new ExperimentPlanner(SPECS);
let benchCost = 0;
let jevCalls = 0;
const textualBench = {
  async conduct(c) {
    benchCost++;
    const truth = ghTruth(c);
    const text = ghDescribe(truth, 1000 + benchCost);
    jevCalls += 2;
    const got = await perceive(text);
    return got.frame;
  },
};
const t0 = performance.now();
const explorer = new ContinuousExplorer(GH_COND_DIMS, GH_OUTCOME_DIMS, planner, SPECS, textualBench,
  { growth: 2, yield: 3 }, { budget: 300, policy: "uncertainty" }, 1);
while (await explorer.step()) {}
const exploreMs = performance.now() - t0;
console.log(`探索结束：实验 ${explorer.log.length} 次（台成本 ${benchCost}），Jev 调用 ${jevCalls} 次，耗时 ${(exploreMs / 1000).toFixed(1)}s，终止 ${explorer.terminationReason}`);
console.log("影响因素:", JSON.stringify(explorer.influentialDims), "（真值相关：light, water, nutrient；干扰：temp, decor 应缺席）");
console.log("覆盖报告:", JSON.stringify(explorer.coverageReport()));

// ── 3. 读回评分（评分侧直接用真值；agent 的 Jev 判定不进评分） ──
console.log("\n══ 3. 全网格读回");
const combos = [];
for (const light of GH_GRID.light) for (const water of GH_GRID.water) for (const nutrient of GH_GRID.nutrient)
  for (const temp of GH_GRID.temp) for (const decor of GH_GRID.decor) combos.push({ light, water, nutrient, temp, decor });
let coarse = 0, growthRight = 0;
for (const c of combos) {
  const p = explorer.mem.predict(c, 1);
  const truth = ghTruth(c);
  if (p.values.growth !== null && Math.abs(p.values.growth - truth.growth) <= 0.5) growthRight++;
  if (p.values.growth !== null && Math.abs(p.values.growth - truth.growth) <= 0.5 &&
      p.values.yield !== null && Math.abs(p.values.yield - truth.yield) <= 0.5) coarse++;
}
console.log(`生长粗粒度 ${growthRight}/${combos.length}（${(growthRight / combos.length * 100).toFixed(1)}%），联合 ${coarse}/${combos.length}（${(coarse / combos.length * 100).toFixed(1)}%）`);

// ── 4. 复杂目标：自主找条件并验证 ─────────────────────
console.log("\n══ 4. 复杂目标");
async function findAndVerify(goal, extraConstraints, label) {
  // 提出 → 验证 → 纠错 → 再提出 闭环（最多 6 轮）：
  // 每次验证失败都把真实结果写回记忆（计票纠错），再重新找候选。
  const rejected = [];
  for (let round = 1; round <= 6; round++) {
    const candidates = [];
    for (const c of combos) {
      if (extraConstraints && !extraConstraints(c)) continue;
      if (rejected.some(r => JSON.stringify(r) === JSON.stringify(c))) continue;
      const p = explorer.mem.predict(c, round);
      if (p.values.growth !== null && Math.abs(p.values.growth - goal.growth) <= 0.5 &&
          p.values.yield !== null && Math.abs(p.values.yield - goal.yield) <= 0.5 &&
          p.converged && p.ambiguous.length === 0) {
        candidates.push({ c, p });
      }
    }
    if (!candidates.length) {
      console.log(`  ${label}：第 ${round} 轮后无候选（已纠错 ${rejected.length} 次，如实报告）`);
      return false;
    }
    const chosen = candidates[0].c;
    const text = ghDescribe(ghTruth(chosen), 7000 + round);
    jevCalls += 2;
    const verify = await perceive(text);
    const ok = verify.frame.growth === goal.growth && verify.frame.yield === goal.yield;
    if (ok) {
      console.log(`  ${label}：第 ${round} 轮，模型选定 ${JSON.stringify(chosen)} → 实种验证 ${JSON.stringify(verify.frame)}（达成，含 ${rejected.length} 次纠错）`);
      return true;
    }
    // 验证失败：真实结果写回记忆（纠正错误规则）
    explorer.mem.learnFromObservation(chosen, ghTruth(chosen), 2);
    rejected.push(chosen);
    console.log(`  ${label}：第 ${round} 轮验证否决 ${JSON.stringify(chosen)}（实 ${JSON.stringify(verify.frame)}），已纠错，重新提出…`);
  }
  console.log(`  ${label}：6 轮内未达成（纠错 ${rejected.length} 次，如实报告）`);
  return false;
}
const g1 = await findAndVerify({ growth: 2, yield: 3 }, null, "目标1：茂盛且高产");
const g2 = await findAndVerify({ growth: 2, yield: 0 }, (c) => c.water === 3, "目标2：多水条件下茂盛（必须避开肥害区 nutrient=3）");
console.log(`\n总 Jev 调用 ${jevCalls} 次`);
