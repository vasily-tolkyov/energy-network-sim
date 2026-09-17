import { test } from "node:test";
import assert from "node:assert/strict";
import { PopChannelMap } from "../src/pop/popmap.js";
import { PopRuleMemory } from "../src/pop/popmemory.js";
import { R2PopLayer } from "../src/pop/r2pop.js";
import {
  CONDITION_SPECS,
  OUTCOME_SPECS,
  curriculum,
  queries,
} from "../src/prototype/world.js";
import {
  CAP_CONDITION_SPECS,
  CAP_OUTCOME_SPECS,
  capCurriculum,
  capQueries,
  capTruth,
  sampleCombos,
} from "../src/topics/capacity-world.js";

// 群体编码 + 规则核版本锁定测试

const cm = new PopChannelMap(CONDITION_SPECS.map((s) => ({ ...s })), 4);
const om = new PopChannelMap(OUTCOME_SPECS.map((s) => ({ ...s })), 4);
const SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 3 };

test("群体编码：编码/归属往返一致，部分线索在足够重复后补全群体", () => {
  const map = new PopChannelMap([{ name: "x", bins: 3 }], 4);
  assert.equal(map.neuronCount, 12);
  assert.deepEqual(map.population("x", 1), [4, 5, 6, 7]);
  assert.deepEqual(map.binOf(6), { channel: "x", bin: 1 });
  // 部分线索补全：群内边 0.8，钳制 2/4 成员 → 其余 2 个被招募（0.4 边时不足以补全）
  const mem = new PopRuleMemory(map, new PopChannelMap([{ name: "y", bins: 1 }], 4), { maxRules: 4 });
  mem.teachExperience({ conditions: { x: 1 }, outcomes: { y: 0 } }, 8, 0.8);
  const cue = map.encodeCue({ x: 1 }, 2);
  const r = mem.net.settle(cue);
  const active = new Set(r.activeNeurons);
  for (const id of map.population("x", 1)) {
    assert.ok(active.has(id), `population member ${id} should be recruited`);
  }
});

function teachBall(gain: number): PopRuleMemory {
  const mem = new PopRuleMemory(cm, om, { maxRules: 128 });
  const r2 = new R2PopLayer(cm, om);
  for (const group of curriculum()) {
    const magSum = new Map<string, number>();
    const magCount = new Map<string, number>();
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
      for (const spec of OUTCOME_SPECS) {
        const a = pair.e0.outcomes[spec.name]!;
        const b = pair.e1.outcomes[spec.name]!;
        if (a !== b) mem.teachExclusion("outcome", spec.name, a, b, 3.0);
      }
      const analysis = r2.analyzePair(pair);
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
          m += Math.abs(delta) / (SPAN[och] ?? 1);
        }
        // 与读出层口径一致：按全部结果通道数取平均（2），不按变化通道数
        magSum.set(ch, (magSum.get(ch) ?? 0) + m / om.specs.length);
        magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
      }
    }
    const boost: Record<string, number> = {};
    for (const [ch, sum] of magSum) {
      boost[ch] = 0.1 * gain * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
    }
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  return mem;
}

test("群体+核端到端：粗粒度与错过墙门控 100%，细粒度 ≥ 65%", () => {
  const mem = teachBall(3);
  let correct = 0;
  let coarse = 0;
  let miss = 0;
  let missTotal = 0;
  const all = queries();
  for (const q of all) {
    const { decoded } = mem.predict(q.conditions, 1);
    if (decoded.rebound === q.truth.rebound) coarse++;
    if (decoded.rebound === q.truth.rebound && (q.truth.rebound === 0 || decoded.reboundSpeed === q.truth.reboundSpeed)) correct++;
    if (q.kind === "miss-wall") {
      missTotal++;
      if (decoded.rebound === q.truth.rebound) miss++;
    }
  }
  assert.equal(coarse, all.length);
  assert.equal(miss, missTotal);
  // 否决制去除直连短路后，细粒度实测 68.8%（代价如实记录：未见组合插值变弱）
  assert.ok(correct / all.length >= 0.65, `fine ${((correct / all.length) * 100).toFixed(1)}% < 65%`);
});

test("规则核消融：否决制下 G=0 同时断否决，门控随之崩溃（否决正是门控的机制）", () => {
  const pestGate = (gain: number): number => {
    const mem = teachBall(gain);
    const missQueries = queries().filter((q) => q.kind === "miss-wall");
    let ok = 0;
    for (const q of missQueries) {
      const { decoded } = mem.predict(q.conditions, 1);
      if (decoded.rebound === q.truth.rebound) ok++;
    }
    return ok / missQueries.length;
  };
  assert.equal(pestGate(3), 1, "G=3 门控应 100%");
  // 否决边随侧重写入：G=0 时无否决，门控退化为按原始匹配数猜测（实测 ~58%）
  assert.ok(pestGate(0) <= 0.75, `G=0 门控应显著退化，实得 ${(pestGate(0) * 100).toFixed(1)}%`);
});

test("互斥从经验学习：新建网络无预置抑制，换对观察后才出现抑制边", () => {
  const map = new PopChannelMap([{ name: "x", bins: 3 }], 4);
  const mem = new PopRuleMemory(map, map, { maxRules: 4 });
  const [a, b] = [map.population("x", 0)[0]!, map.population("x", 1)[0]!];
  assert.equal(mem.net.getInhibitoryWeight(a, b), 0, "no pre-wired exclusion");
  mem.teachExclusion("condition", "x", 0, 1, 1.0);
  assert.ok(mem.net.getInhibitoryWeight(a, b) > 0, "learned from swap");
  const c = map.population("x", 2)[0]!;
  assert.equal(mem.net.getInhibitoryWeight(a, c), 0, "unobserved pair stays zero");
});

test("侧重否决：门控不匹配被否决（100%），未见组合按相似兜底（≥20%，插值变弱如实记录）", () => {
  const mem = teachBall(3);
  const all = queries();
  let miss = 0;
  let missTotal = 0;
  let unseenComboCorrect = 0;
  let unseenComboTotal = 0;
  for (const q of all) {
    const { decoded } = mem.predict(q.conditions, 1);
    if (q.kind === "miss-wall") {
      missTotal++;
      if (decoded.rebound === q.truth.rebound) miss++;
    }
    if (q.kind === "unseen-combo") {
      unseenComboTotal++;
      if (decoded.rebound === q.truth.rebound && decoded.reboundSpeed === q.truth.reboundSpeed) unseenComboCorrect++;
    }
  }
  assert.equal(miss, missTotal, "gate must be absolute under influence veto");
  // 实测 25%：去除直连短路后未见组合的插值显著变弱——这是否决制换容量的
  // 真实代价（直连时代 58%，但那时容量只有 47.7%），如实锁定，不设虚高阈值
  assert.ok(
    unseenComboCorrect / unseenComboTotal >= 0.2,
    `unseen-combo fallback ${((unseenComboCorrect / unseenComboTotal) * 100).toFixed(1)}% < 20%`,
  );
});

test("未见颜色不触发否决（非影响通道无否决边，泛化保留）", () => {
  const mem = teachBall(3);
  const { decoded } = mem.predict(
    { color: 6, direction: 0, damping: 0, hitWall: 1, speed: 1, wallStiffness: 1 },
    1,
  );
  assert.equal(decoded.rebound, 1);
});

test("容量：mod-8 E=64 时否决制 taught ≥ 80%（此前 47.7%）", { timeout: 120_000 }, () => {
  const ccm = new PopChannelMap(CAP_CONDITION_SPECS.map((s) => ({ ...s })), 4);
  const com = new PopChannelMap(CAP_OUTCOME_SPECS.map((s) => ({ ...s })), 4);
  const combos = sampleCombos(64, 1);
  const mem = new PopRuleMemory(ccm, com, { maxRules: 192 });
  const r2 = new R2PopLayer(ccm, com);
  for (const group of capCurriculum(combos, capTruth)) {
    const magSum = new Map<string, number>();
    const magCount = new Map<string, number>();
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperience(e, 4);
      for (const spec of CAP_OUTCOME_SPECS) {
        const a = pair.e0.outcomes[spec.name]!;
        const b = pair.e1.outcomes[spec.name]!;
        if (a !== b) mem.teachExclusion("outcome", spec.name, a, b, 3.0);
      }
      const analysis = r2.analyzePair(pair);
      for (const ch of analysis.influentialChannels) {
        const m = Math.abs(analysis.outcomeDelta.out ?? 0) / 7;
        magSum.set(ch, (magSum.get(ch) ?? 0) + m);
        magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
      }
    }
    const boost: Record<string, number> = {};
    for (const [ch, sum] of magSum) {
      boost[ch] = 0.1 * 3 * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
    }
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e, boost, 4);
    }
  }
  const taughtSet = new Set(combos.map(([a, b]) => `${a},${b}`));
  let correct = 0;
  let total = 0;
  for (const q of capQueries(taughtSet, capTruth)) {
    if (q.kind !== "taught") continue;
    const { decoded } = mem.predict(q.conditions, 1);
    total++;
    if (decoded.out === q.truth.out) correct++;
  }
  assert.ok(correct / total >= 0.8, `veto capacity at E=64 ${((correct / total) * 100).toFixed(1)}% < 80%`);
});
