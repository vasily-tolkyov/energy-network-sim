/** Jev 观察解析精度探针：对每个网格组合，渲染文字 → Jev 解析 → 与真值比对。
 * 只读评估（不学习），统计逐结果维的解析准确率与混淆点。 */
import { JevApiBackend } from "../dist/src/perception/decision-backend.js";
import { SceneParser } from "../dist/src/perception/scene-parser.js";
import { GH_GRID, ghTruth, ghDescribe, GH_PERCEPTION } from "../dist/src/topics/greenhouse-world.js";

const key = process.env.TYPESAFE_API_KEY;
if (!key) throw new Error("set TYPESAFE_API_KEY");
const api = new JevApiBackend("https://api.typesafe.ai/v1/systemone", key);
const parser = new SceneParser(api);

const spec = {
  dims: [
    { name: "growth", values: GH_PERCEPTION.growth.options.map(o => ({ value: o.value, label: o.label, desc: o.label })), ask: GH_PERCEPTION.growth.ask },
    { name: "yield", values: GH_PERCEPTION.yield.options.map(o => ({ value: o.value, label: o.label, desc: o.desc })), ask: GH_PERCEPTION.yield.ask },
  ],
  confidenceThreshold: 0.01,
};

const combos = [];
for (const light of GH_GRID.light) for (const water of GH_GRID.water) for (const nutrient of GH_GRID.nutrient)
  for (const temp of GH_GRID.temp) for (const decor of GH_GRID.decor) combos.push({ light, water, nutrient, temp, decor });

// 每组合取 1 个措辞样本（全量 384 组合 = 768 次 Jev 调用）
let gOk = 0, yOk = 0, both = 0;
const confusions = [];
for (const [i, c] of combos.entries()) {
  const truth = ghTruth(c);
  const text = ghDescribe(truth, 5000 + i);
  const got = await parser.parse(text, spec);
  const g = got.frame.growth === truth.growth;
  const y = got.frame.yield === truth.yield;
  if (g) gOk++;
  if (y) yOk++;
  if (g && y) both++;
  else if (confusions.length < 10) confusions.push({ text, truth, parsed: got.frame });
}
console.log(`growth 解析 ${gOk}/${combos.length}（${(gOk / combos.length * 100).toFixed(1)}%）`);
console.log(`yield 解析 ${yOk}/${combos.length}（${(yOk / combos.length * 100).toFixed(1)}%）`);
console.log(`联合 ${both}/${combos.length}（${(both / combos.length * 100).toFixed(1)}%）`);
for (const c of confusions) console.log("混淆:", JSON.stringify(c));
