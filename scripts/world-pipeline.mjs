/** 通用文字世界管道（迁移性证明的核心：同一段代码，只换世界配置）。
 * 文字场景 → Jev 感知 → 自由探索 → 自主发现规律 → 提出-验证-纠错闭环。
 * 温室世界（greenhouse-e2e）是它的第一个实例，机房运维世界是第二个。 */
import { JevApiBackend } from "../dist/src/perception/decision-backend.js";
import { SceneParser } from "../dist/src/perception/scene-parser.js";
import { ExperimentPlanner } from "../dist/src/pop/explore/planner.js";
import { ContinuousExplorer } from "../dist/src/pop/explore/explorer-continuous.js";

export async function runWorldPipeline(cfg) {
  const api = new JevApiBackend("https://api.typesafe.ai/v1/systemone", process.env.TYPESAFE_API_KEY ?? null);
  const parser = new SceneParser(api);
  let jevCalls = 0;

  async function perceive(text) {
    const spec = {
      dims: Object.entries(cfg.perception).map(([name, p]) => ({
        name, ask: p.ask,
        values: p.options.map(o => ({ value: o.value, label: o.label, desc: o.desc ?? o.label })),
      })),
      confidenceThreshold: 0.01,
    };
    const got = await parser.parse(text, spec);
    jevCalls += spec.dims.length;
    return got.frame;
  }

  // 1. 场景感知
  console.log("══ 1. 场景感知");
  const start = await parser.parse(cfg.openingScene, cfg.sceneSpec);
  console.log("起始帧:", JSON.stringify(start.frame), "unknown:", JSON.stringify(start.unknownDims));

  // 2. 自由探索
  console.log(`══ 2. 自由探索（预算 ${cfg.budget}，uncertainty）`);
  const specs = cfg.condNames.map(n => ({ name: n, bins: cfg.grid[n].length, values: cfg.grid[n] }));
  const planner = new ExperimentPlanner(specs);
  let cost = 0;
  const textualBench = {
    async conduct(c) {
      cost++;
      const truth = cfg.truth(c);
      return perceive(cfg.describe(truth, 1000 + cost));
    },
  };
  const t0 = performance.now();
  const explorer = new ContinuousExplorer(cfg.condDims, cfg.outcomeDims, planner, specs, textualBench,
    cfg.outcomeSpan, { budget: cfg.budget, policy: "uncertainty" }, cfg.seed ?? 1);
  while (await explorer.step()) {}
  const exploreMs = performance.now() - t0;
  console.log(`探索结束：实验 ${explorer.log.length} 次，Jev 调用 ${jevCalls} 次，耗时 ${(exploreMs / 1000).toFixed(1)}s，终止 ${explorer.terminationReason}`);
  console.log("影响因素:", JSON.stringify(explorer.influentialDims));
  console.log("覆盖报告:", JSON.stringify(explorer.coverageReport()));

  // 3. 全网格读回（评分侧直接用真值；agent 的 Jev 判定不进评分）
  console.log("══ 3. 全网格读回");
  const combos = cfg.allCombos();
  let joint = 0;
  for (const c of combos) {
    const p = explorer.mem.predict(c, 1);
    const truth = cfg.truth(c);
    if (cfg.outcomeNames.every(d => p.values[d] !== null && Math.abs(p.values[d] - truth[d]) <= 0.5)) joint++;
  }
  console.log(`联合读回 ${joint}/${combos.length}（${(joint / combos.length * 100).toFixed(1)}%）`);

  // 4. 复杂目标（提出 → 验证 → 纠错 → 再提出）
  console.log("══ 4. 复杂目标");
  const goalResults = [];
  for (const goal of cfg.goals) {
    const rejected = [];
    let done = false;
    for (let round = 1; round <= 6 && !done; round++) {
      const candidates = [];
      for (const c of combos) {
        if (goal.constraints && !goal.constraints(c)) continue;
        if (rejected.some(r => JSON.stringify(r) === JSON.stringify(c))) continue;
        const p = explorer.mem.predict(c, round);
        if (p.converged && p.ambiguous.length === 0 &&
            goal.target.every(([d, v]) => p.values[d] !== null && Math.abs(p.values[d] - v) <= 0.5)) {
          candidates.push(c);
        }
      }
      if (!candidates.length) {
        console.log(`  ${goal.label}：第 ${round} 轮后无候选（已纠错 ${rejected.length} 次，如实报告）`);
        break;
      }
      const chosen = candidates[0];
      const verify = await perceive(cfg.describe(cfg.truth(chosen), 7000 + round));
      const ok = goal.target.every(([d, v]) => verify[d] === v);
      if (ok) {
        console.log(`  ${goal.label}：第 ${round} 轮选定 ${JSON.stringify(chosen)} → 实种验证 ${JSON.stringify(verify)}（达成，含 ${rejected.length} 次纠错）`);
        done = true;
      } else {
        explorer.mem.learnFromObservation(chosen, cfg.truth(chosen), 2);
        rejected.push(chosen);
        console.log(`  ${goal.label}：第 ${round} 轮否决 ${JSON.stringify(chosen)}（实 ${JSON.stringify(verify)}），纠错后重新提出…`);
      }
    }
    goalResults.push({ label: goal.label, achieved: done });
  }
  console.log(`总 Jev 调用 ${jevCalls} 次`);
  return { factors: explorer.influentialDims, joint, total: combos.length, goalResults, jevCalls };
}
