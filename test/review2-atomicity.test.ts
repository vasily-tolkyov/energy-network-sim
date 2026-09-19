import test from "node:test";
import assert from "node:assert/strict";
import { PopRuleMemory } from "../src/pop/popmemory.js";
import { PopChannelMap } from "../src/pop/popmap.js";
import { FieldRuleMemory } from "../src/pop/concept/field-memory.js";
import { SensoryEncoder } from "../src/pop/concept/sensory.js";
import { EmergentMap } from "../src/pop/concept/emergent-map.js";
import { ConceptFormation } from "../src/pop/concept/formation.js";
import { EnergyNetwork } from "../src/network.js";

test("F7 rejected writes preserve all edges, signatures, evidence and counters", () => {
  const mem = new PopRuleMemory(new PopChannelMap([{ name: "x", bins: 2 }]), new PopChannelMap([{ name: "y", bins: 2 }]), { maxRules: 1 });
  const enc = new SensoryEncoder(["x", "y"].map(name => ({ name, min: 0, max: 1 })));
  const field = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 1 });
  field.setOutcomeDimensions(["y"]);
  const check = (target: object, write: () => void) => {
    const before = structuredClone(target);
    assert.throws(write);
    assert.deepEqual(structuredClone(target), before);
  };
  check(mem, () => mem.teachExperience({ conditions: { x: 0 }, outcomes: { y: NaN } }, 2));
  check(mem, () => mem.bindInfluence({ conditions: { x: 0 }, outcomes: {} }, { unknown: .2 }, 2));
  check(mem, () => mem.learnFromQuery({ x: 0 }, { y: NaN }, 1));
  check(mem, () => mem.learnFromObservation({ x: 0 }, {}, 2));
  check(mem, () => mem.teachExclusion("outcome", "y", 0, 9, 1));
  check(field, () => field.teachExperiment({ x: .2 }, { y: NaN }));
  check(field, () => field.bindInfluence({ x: .2 }, { unknown: .2 }));
  check(field, () => field.learnFromObservation({ y: .2 }, { y: .2 }));
  check(field, () => field.setOutcomeDimensions(["y", "unknown"]));
  mem.learnFromObservation({ x: 0 }, { y: 0 }, 2);
  field.learnFromObservation({ x: .2 }, { y: .2 });
  check(mem, () => mem.bindInfluence({ conditions: { x: 1 }, outcomes: {} }, { x: .3 }, 2));
  check(field, () => field.bindInfluence({ x: .8 }, { x: .3 }));
});

test("F1 saturated retractable Γ retains other owners and structural inhibition", () => {
  const net = new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: .5, maxWeight: 3 });
  net.strengthenInhibitory(0, 1, .5);
  net.setInhibitionContribution(0, 1, "a", 3);
  net.setInhibitionContribution(0, 1, "b", 3);
  net.setInhibitionContribution(0, 1, "a", 0);
  assert.equal(net.getInhibitoryWeight(1, 0), 3);
  net.strengthenInhibitory(0, 1, .5);
  net.setInhibitionContribution(0, 1, "b", 0);
  assert.equal(net.getInhibitoryWeight(0, 1), 1);
});

test("F12 constructor clusterThresholdRatio actually controls default extraction", () => {
  const enc = new SensoryEncoder([{ name: "x", min: 0, max: 1 }], 4);
  const f = new ConceptFormation(enc, { clusterThresholdRatio: .9 });
  f.net.strengthen(0, 1, 1);
  f.net.strengthen(1, 2, .6);
  assert.equal(f.extractConcepts()[0]!.memberNeuronIds.length, 2);
  assert.equal(f.extractConcepts(.5)[0]!.memberNeuronIds.length, 3);
});
