import { test } from "node:test";
import assert from "node:assert/strict";
import { PopChannelMap } from "../src/pop/popmap.js";
import { R2PopLayer } from "../src/pop/r2pop.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
} from "../src/prototype/world.js";

// R2A 神经化：全模式重合场分层，R2A/R2B 同一次动力学双侧读出

const cm = new PopChannelMap(CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })), 4);

test("R2A 神经化：结果变化标志与带符号档差从动力学读出", () => {
  const r2 = new R2PopLayer(cm, om);
  const byName = Object.fromEntries(curriculum().map((g) => [g.name, g]));
  // speed 对：rs 变化 +方向+数值（(0,1) rs0→(2,1) rs2）
  const speed = r2.analyzePair(byName["speed-group"]!.pairs[0]!);
  assert.equal(speed.outcomeChanged, true);
  assert.equal(speed.outcomeDelta.reboundSpeed, 1); // e0 rs=1 → e1 rs=2，带符号差 +1
  assert.equal(speed.outcomeDelta.rebound ?? 0, 0);
  assert.deepEqual(speed.diffChannels, ["speed"]);
  assert.deepEqual(speed.influentialChannels, ["speed"]);
  // 颜色对：结果不变 → 无影响（含退化对判定）
  const color = r2.analyzePair(byName["color-group"]!.pairs[0]!);
  assert.equal(color.outcomeChanged, false);
  assert.deepEqual(color.influentialChannels, []);
  // 碰墙对：反弹翻转，幅度最大
  const hit = r2.analyzePair(byName["hitwall-group"]!.pairs[0]!);
  assert.equal(hit.outcomeChanged, true);
  assert.equal(hit.outcomeDelta.rebound, -1);
  assert.deepEqual(hit.influentialChannels, ["hitWall"]);
});

test("R2A 神经化：退化对（刚度 0↔1 结果档相同）自动判为无变化", () => {
  const r2 = new R2PopLayer(cm, om);
  const g = curriculum().find((x) => x.name === "stiffness-group")!;
  const a1 = r2.analyzePair(g.pairs[0]!); // 0↔2，结果变
  const a2 = r2.analyzePair(g.pairs[1]!); // 0↔1，结果档恰好相同
  assert.equal(a1.outcomeChanged, true);
  assert.deepEqual(a1.influentialChannels, ["wallStiffness"]);
  assert.equal(a2.outcomeChanged, false);
  assert.deepEqual(a2.influentialChannels, []);
});

test("R2 全模式差分：条件共同项与结果共同项都在重合场中驻留", () => {
  const r2 = new R2PopLayer(cm, om);
  const g = curriculum().find((x) => x.name === "color-group")!;
  const a = r2.analyzePair(g.pairs[0]!);
  // 颜色对共享 5 个条件通道（20 群体成员）+ 全部结果（8 成员）
  assert.equal(a.conditionCommonNeurons.length, 20);
  assert.equal(a.outcomeChanged, false);
});
