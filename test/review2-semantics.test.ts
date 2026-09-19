import test from "node:test";
import assert from "node:assert/strict";
import { PopChannelMap } from "../src/pop/popmap.js";
import { PopRuleMemory } from "../src/pop/popmemory.js";
import { FieldRuleMemory } from "../src/pop/concept/field-memory.js";
import { SensoryEncoder } from "../src/pop/concept/sensory.js";
import { EmergentMap } from "../src/pop/concept/emergent-map.js";
import { captureClassify, type ForecastSnapshot } from "../src/pop/attention/capture.js";
import { compatibleSupports } from "../src/pop/evidence.js";
import { EnergyNetwork } from "../src/network.js";

function pop() {
  return new PopRuleMemory(new PopChannelMap([{ name: "x", bins: 2 }]),
    new PopChannelMap([{ name: "y", bins: 2 }, { name: "z", bins: 2 }]), { maxRules: 8 });
}
function field() {
  const enc = new SensoryEncoder(["x", "y", "z"].map(name => ({ name, min: 0, max: 1 })));
  const mem = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 8 });
  mem.setOutcomeDimensions(["y", "z"]);
  return mem;
}
for (const sequence of ["truth-guess", "guess-truth", "truth-guess-truth", "partial", "truth-change-return"]) {
  test(`F1 evidence sequence: ${sequence}`, () => {
    const mem = pop();
    if (sequence !== "guess-truth") mem.learnFromObservation({ x: 0 }, { y: 0, z: 0 }, 2);
    if (sequence.includes("guess")) mem.learnFromQuery({ x: 0 }, { y: 1 }, 2);
    if (sequence.endsWith("truth")) mem.learnFromObservation({ x: 0 }, { y: 0 }, 2);
    if (sequence === "partial" || sequence === "truth-change-return") mem.learnFromObservation({ x: 0 }, { y: 1 }, 2);
    if (sequence === "truth-change-return") mem.learnFromObservation({ x: 0 }, { y: 0 }, 2);
    for (let seed = 1; seed <= 5; seed++) {
      const got = mem.predict({ x: 0 }, seed).decoded;
      assert.equal(got.y, sequence === "partial" ? 1 : 0);
      if (sequence !== "guess-truth") assert.equal(got.z, 0);
    }
    if (sequence.startsWith("truth-guess")) assert.equal(mem.evidenceConflicts.length, 1);
  });
}
test("F2 aliases aggregate equally, overlap survives, disjoint correction reverses", () => {
  const mem = field();
  mem.learnFromObservation({ x: .2 }, { y: .5, z: .2 });
  mem.learnFromObservation({ x: .201 }, { y: .51 });
  assert.equal(mem.predict({ x: .2 }, 1).values.y, .505);
  const shared = mem.encoder.encodeDimension("y", .51).filter(id => mem.encoder.encodeDimension("y", .525).includes(id));
  mem.learnFromObservation({ x: .2 }, { y: .525 });
  for (const from of mem.ruleCore(0)) for (const to of shared) assert.equal(mem.net.getInhibitoryWeight(from, to), 0);
  assert.notEqual(mem.predict({ x: .2 }, 1).values.y, null);
  mem.learnFromObservation({ x: .2 }, { y: .9 });
  mem.learnFromObservation({ x: .2 }, { y: .5 });
  assert.ok(Math.abs(mem.predict({ x: .2 }, 1).values.y! - .5) < .04);
});
test("F3 joint capture: matching, contradiction, novel result, partial conflict", () => {
  for (const observed of [{ y: .2, z: .2 }, { y: .8, z: .2 }, { y: .95, z: .2 }, { y: .2, z: .8 }]) {
    const mem = field();
    mem.learnFromObservation({ x: .2 }, { y: .2, z: .2 }, 6);
    const p = mem.predict({ x: .2 }, 1);
    const cap = captureClassify(mem, { x: .2 }, observed, p.winningCores[0]!, 1);
    assert.equal(cap.class, observed.y === .2 && observed.z === .2 ? "within-envelope" : "prediction-violation");
  }
});
test("F3 unseen threshold correction uses independent memories for each seed (review N28)", () => {
  for (let seed = 1; seed <= 5; seed++) {
    // The original N28 reused a corrected memory, then asserted it was still wrong.
    const enc = new SensoryEncoder(["x", "y"].map(name => ({ name, min: 0, max: 1 })));
    const mem = new FieldRuleMemory(enc, new EmergentMap([], enc), { maxRules: 8 });
    mem.setOutcomeDimensions(["y"]);
    mem.learnFromObservation({ x: .2 }, { y: .2 }, 6);
    const p = mem.predict({ x: .3 }, seed);
    assert.equal(p.values.y, .2);
    const cap = captureClassify(mem, { x: .3 }, { y: .8 }, p.winningCores[0]!, seed);
    assert.notEqual(cap.class, "within-envelope");
    mem.learnFromObservation({ x: .3 }, { y: .8 });
    assert.ok(Math.abs(mem.predict({ x: .3 }, seed).values.y! - .8) < .04);
  }
});
test("F3 cached value snapshot survives changed core semantics", () => {
  const mem = field();
  mem.learnFromObservation({ x: .2 }, { y: .2, z: .2 }, 6);
  const p = mem.predict({ x: .2 }, 1);
  const snapshot: ForecastSnapshot = { coreIdx: p.winningCores[0]!, values: { ...p.values },
    generation: mem.evidenceGeneration, converged: p.converged, terminationReason: p.terminationReason };
  mem.learnFromObservation({ x: .2 }, { y: .8 }, 6);
  assert.ok(mem.evidenceGeneration > snapshot.generation);
  assert.equal(snapshot.values.y, .2);
  assert.notEqual(captureClassify(mem, { x: .2 }, { y: .8, z: .2 }, snapshot).class, "within-envelope");
});
test("F4 no legal quiet candidate returns structured failure, symmetric decay conserves Γ", () => {
  const net = new EnergyNetwork({ neuronCount: 3, activationEnergy: 1, maintenanceEnergy: .5 });
  net.strengthenDirectedInhibitory(0, 1, 2);
  const result = net.settleAnnealed([0], [], { fallbackQuietOnly: true, levels: 1, sweepsPerLevel: 1 });
  assert.equal(result.terminationReason, "no-quiet-candidate");
  assert.equal(result.converged, false);
  assert.deepEqual(result.activeNeurons, []);
  net.strengthenInhibitory(1, 2, 1);
  net.decayInhibition(1, 2, 2);
  assert.equal(net.getInhibitoryWeight(1, 2), 0);
  assert.equal(net.getInhibitoryWeight(2, 1), 0);
  assert.equal(compatibleSupports([1, 2], [2, 3]), true);
});
