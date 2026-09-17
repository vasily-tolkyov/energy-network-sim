import { test } from "node:test";
import assert from "node:assert/strict";
import { SensoryEncoder } from "../src/pop/concept/sensory.js";
import { ConceptFormation } from "../src/pop/concept/formation.js";
import {
  CONT_CONDITION_DIMS,
  CONT_OUTCOME_DIMS,
  contCurriculum,
} from "../src/pop/concept/world-continuous.js";

// M2：概念形成循环——值区域成阱、替代区分离、部分线索补全

function buildFormation(fieldsPerDim = 40): ConceptFormation {
  const enc = new SensoryEncoder([...CONT_CONDITION_DIMS, ...CONT_OUTCOME_DIMS], fieldsPerDim);
  return new ConceptFormation(enc);
}

function presentCurriculum(formation: ConceptFormation): void {
  for (const group of contCurriculum()) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) {
        const frame = { ...e.conditions, ...e.outcomes };
        formation.presentExperiment(frame, 4);
      }
      // 换对互斥：被操纵维度的两个值区域
      const a = pair.e0.conditions[group.manipulated]!;
      const b = pair.e1.conditions[group.manipulated]!;
      formation.presentSwap(group.manipulated, a, b, 3.0);
    }
  }
}

test("值区域成阱：速度维上检测出分离的值概念，且中心值覆盖课程值域", () => {
  const formation = buildFormation();
  presentCurriculum(formation);
  const concepts = formation.extractConcepts(0.5);
  const speedConcepts = concepts.filter((c) => c.dimension === "speed");
  assert.ok(speedConcepts.length >= 2, `expected multiple speed value concepts, got ${speedConcepts.length}`);
  // 课程速度值 0.8/1.2/1.5/2.0/2.5/3.2/6.5：概念中心应覆盖低区与高区
  const centers = speedConcepts.map((c) => c.centerValue).sort((a, b) => a - b);
  assert.ok(centers[0]! < 2.5, `low region covered: ${centers}`);
  assert.ok(centers[centers.length - 1]! > 3.0, `high region covered: ${centers}`);
  // 概念内部强度为正
  for (const c of speedConcepts) assert.ok(c.meanInternalWeight > 0);
});

test("同值区域可被部分线索补全（成阱的动力学证据）", () => {
  const formation = buildFormation();
  presentCurriculum(formation);
  const concepts = formation.extractConcepts(0.5);
  const target = concepts.find((c) => c.dimension === "speed" && Math.abs(c.centerValue - 6.5) < 1.5);
  assert.ok(target, "high-speed concept should exist");
  // 钳制概念的一半成员 → settle 应补全其余成员
  const cue = target!.memberNeuronIds.slice(0, Math.max(1, Math.floor(target!.memberNeuronIds.length / 2)));
  const r = formation.net.settle(cue);
  const active = new Set(r.activeNeurons);
  const ratio = target!.memberNeuronIds.filter((id) => active.has(id)).length / target!.memberNeuronIds.length;
  assert.ok(ratio >= 0.9, `concept completion ratio ${(ratio * 100).toFixed(0)}% < 90%`);
});

test("替代值区域之间的边显著弱于同区域内部边（从不共激活的证据）", () => {
  const formation = buildFormation();
  presentCurriculum(formation);
  const enc = formation.encoder;
  const low = enc.encodeDimension("speed", 0.8);
  const high = enc.encodeDimension("speed", 6.5);
  let cross = 0;
  let crossCount = 0;
  for (const a of low) {
    for (const b of high) {
      cross += formation.net.getWeight(a, b);
      crossCount++;
    }
  }
  const crossMean = cross / crossCount;
  const intra = (ids: number[]): number => {
    let s = 0;
    let c = 0;
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        s += formation.net.getWeight(ids[i]!, ids[j]!);
        c++;
      }
    }
    return c === 0 ? 0 : s / c;
  };
  assert.ok(
    intra(low) > crossMean * 3 && intra(high) > crossMean * 3,
    `intra (${intra(low).toFixed(2)}/${intra(high).toFixed(2)}) should be >> cross (${crossMean.toFixed(2)})`,
  );
});

test("换对互斥：被对换的值区域之间出现抑制边", () => {
  const formation = buildFormation();
  presentCurriculum(formation);
  const enc = formation.encoder;
  const a = enc.encodeDimension("speed", 0.8)[0]!;
  const b = enc.encodeDimension("speed", 6.5)[0]!;
  assert.ok(formation.net.getInhibitoryWeight(a, b) > 0, "swap exclusion should exist");
});
