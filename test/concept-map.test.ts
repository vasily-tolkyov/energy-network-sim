import { test } from "node:test";
import assert from "node:assert/strict";
import { SensoryEncoder } from "../src/pop/concept/sensory.js";
import { ConceptFormation } from "../src/pop/concept/formation.js";
import { EmergentMap, alignmentReport } from "../src/pop/concept/emergent-map.js";
import {
  CONT_CONDITION_DIMS,
  CONT_OUTCOME_DIMS,
  contCurriculum,
} from "../src/pop/concept/world-continuous.js";

// M3：EmergentMap——概念索引、resolve、对齐报告

function buildMap(): EmergentMap {
  const enc = new SensoryEncoder([...CONT_CONDITION_DIMS, ...CONT_OUTCOME_DIMS], 40);
  const formation = new ConceptFormation(enc);
  for (const group of contCurriculum()) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) {
        formation.presentExperiment({ ...e.conditions, ...e.outcomes }, 4);
      }
    }
  }
  return new EmergentMap(formation.extractConcepts(0.5), enc);
}

test("概念索引：按中心值排序、population 接口与 PopChannelMap 同签名", () => {
  const map = buildMap();
  const names = map.channelNames();
  assert.ok(names.includes("speed") && names.includes("hitWall"));
  const count = map.conceptCount("speed");
  assert.ok(count >= 2);
  const centers = Array.from({ length: count }, (_, i) => map.concept("speed", i).centerValue);
  const sorted = [...centers].sort((a, b) => a - b);
  assert.deepEqual(centers, sorted, "concepts sorted by center value");
  const pop = map.population("speed", 0);
  assert.ok(pop.length >= 2 && pop.every((id) => Number.isInteger(id)));
});

test("resolve：课程内的值解析到概念，中间值由合并概念的边缘认领", () => {
  const map = buildMap();
  assert.notEqual(map.resolve("speed", 0.9), null, "0.9 in taught low region");
  assert.notEqual(map.resolve("speed", 6.4), null, "6.4 in taught high region");
  // 速度 4.2 课程未覆盖，但合并的低值概念（中心 2.0）的感受野边缘覆盖到它——
  // 如实解析到该合并概念；若以后中间区独立成阱，归属会改变（如实记录）
  assert.equal(map.resolve("speed", 4.2), 0);
});

test("对齐报告：课程参照值全部解析到概念，分离区域的 IoU 高", () => {
  const map = buildMap();
  const entries = alignmentReport(map, {
    speed: [0.8, 1.5, 3.2, 6.5],
    hitWall: [0, 1],
  });
  const taught = entries.filter((e) => e.dimension === "speed");
  // 全部参照值都能解析（合并的低值区属于同一概念，IoU 低是合并的如实代价）
  assert.ok(taught.every((e) => e.conceptIndex !== null), "all taught references resolve");
  // 稀疏分离的高值区（6.5）应有高 IoU；合并的低值区 IoU 低（如实记录）
  const high = taught.find((e) => e.referenceValue === 6.5)!;
  assert.ok(high.iou >= 0.8, `separated region IoU ${high.iou.toFixed(2)} < 0.8`);
  const hit = entries.filter((e) => e.dimension === "hitWall");
  assert.ok(hit.every((e) => e.conceptIndex !== null), "hitWall regions formed");
});
