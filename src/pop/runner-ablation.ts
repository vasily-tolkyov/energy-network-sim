import { mkdirSync, writeFileSync } from "node:fs";
import assert from "node:assert/strict";
import { PopChannelMap } from "./popmap.js";
import { PopRuleMemory } from "./popmemory.js";

// Manipulation check, separate from performance scoring. Gain is not eligibility.
const rows = [];
for (const gain of [0, .2]) for (const veto of [false, true]) {
  const mem = new PopRuleMemory(new PopChannelMap([{ name: "x", bins: 2 }]),
    new PopChannelMap([{ name: "y", bins: 2 }]), { maxRules: 8 });
  const a = { conditions: { x: 0 }, outcomes: { y: 0 } };
  mem.teachExperience(a, 4);
  mem.teachExperience({ conditions: { x: 1 }, outcomes: { y: 1 } }, 4);
  mem.bindInfluence(a, { x: gain }, 4, 1.5, veto);
  const core = mem.conditionMap.neuronCount + mem.outcomeMap.neuronCount;
  rows.push({ gain, veto, matchedW: mem.net.getWeight(0, core),
    alternativeGamma: mem.net.getInhibitoryWeight(mem.conditionMap.population("x", 1)[0]!, core) });
}
assert.equal(new Set(rows.map(r => `${r.matchedW}:${r.alternativeGamma}`)).size, 4);
mkdirSync("runs", { recursive: true });
writeFileSync("runs/ablation-manipulation.json", JSON.stringify({ rows, distinctConfigurations: 4 }, null, 2));
console.log(JSON.stringify(rows));
