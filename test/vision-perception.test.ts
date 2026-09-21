import { test } from "node:test";
import assert from "node:assert/strict";
import { MockVisionFrontEnd, OpenAiVisionFrontEnd, VisionPerceptionPipeline } from "../src/perception/vision-frontend.js";
import { SceneParser } from "../src/perception/scene-parser.js";
import { MockBackend } from "../src/perception/decision-backend.js";

const SPEC = {
  dims: [
    { name: "biome", values: [{ value: 0, label: "平原" }, { value: 1, label: "洞穴" }, { value: 2, label: "水域" }], ask: "当前处于哪种地形？" },
    { name: "threat", values: [{ value: 0, label: "无威胁", desc: "晴朗|安全|开阔" }, { value: 1, label: "有威胁", desc: "僵尸|怪物|袭击|追击" }], ask: "附近是否存在敌对威胁？" },
  ],
  confidenceThreshold: 0.6,
};

test("图像适配：图像 → 视觉描述 → 语义解析 → 条件帧（Mock 视觉）", async () => {
  const vision = new MockVisionFrontEnd({
    "screenshot-cave.png": "玩家在一个黑暗的洞穴里，前方有一只僵尸在游荡。",
    "screenshot-plain.png": "玩家站在开阔的平原上，天气晴朗，四周没有任何威胁。",
  });
  const pipe = new VisionPerceptionPipeline(vision, new SceneParser(new MockBackend()));
  const cave = await pipe.perceiveImage("screenshot-cave.png", SPEC);
  assert.deepEqual(cave.frame, { biome: 1, threat: 1 });
  assert.ok(cave.description.includes("洞穴"));
  const plain = await pipe.perceiveImage("screenshot-plain.png", SPEC);
  assert.deepEqual(plain.frame, { biome: 0, threat: 0 });
});

test("图像适配：未登记图像响亮抛错；视觉后端无 key 响亮抛错", async () => {
  const vision = new MockVisionFrontEnd({});
  const pipe = new VisionPerceptionPipeline(vision, new SceneParser(new MockBackend()));
  await assert.rejects(() => pipe.perceiveImage("unknown.png", SPEC), /未登记/);
  const api = new OpenAiVisionFrontEnd(null, null);
  await assert.rejects(() => api.describe("x.png"), /未配置/);
});
