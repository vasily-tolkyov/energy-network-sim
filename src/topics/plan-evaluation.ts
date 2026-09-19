/** Evaluation only. Neither planner nor transition memory imports this module. */
import { createHash } from "node:crypto";
import { mulberry32 } from "../prng.js";
import { frames, observedState, signature, actionsOf, type Frame, type TransitionBench, type TransitionSpace } from "../planning/space.js";
import type { GoalPlan } from "../planning/planner.js";
import type { TransitionMemory, StepPrediction } from "../planning/transition-memory.js";

export function shortestReference(space: TransitionSpace, bench: TransitionBench, start: Frame, goal: Frame): number | null {
  const queue = [{ state: start, depth: 0 }], visited = new Set([signature(start)]);
  for (let head = 0; head < queue.length; head++) {
    const { state, depth } = queue[head]!;
    if (signature(state) === signature(goal)) return depth;
    for (const action of actionsOf(space)) {
      const next = observedState(space, bench.conduct(state, action.values));
      const key = signature(next);
      if (!visited.has(key)) { visited.add(key); queue.push({ state: next, depth: depth + 1 }); }
    }
  }
  return null;
}
export function executeFrozen(space: TransitionSpace, bench: TransitionBench, plan: GoalPlan) {
  let state: Frame = { ...plan.start };
  const observations: { state: Frame; actionId: number; observation: Frame; actual: Frame }[] = [];
  for (const step of plan.steps) {
    const observation = bench.conduct(state, step.action.values);
    const actual = observedState(space, observation);
    observations.push({ state, actionId: step.action.id, observation, actual }); state = actual;
  }
  return { reached: plan.status === "found" && signature(state) === signature(plan.goal), finalState: state, observations };
}
export function randomActions(space: TransitionSpace, bench: TransitionBench, start: Frame, goal: Frame, budget: number, seed: number) {
  const rng = mulberry32(seed), actions = actionsOf(space);
  let state = start;
  const observations: { state: Frame; actionId: number; observation: Frame; actual: Frame }[] = [];
  while (signature(state) !== signature(goal) && observations.length < budget) {
    const action = actions[Math.floor(rng() * actions.length)]!;
    const observation = bench.conduct(state, action.values), actual = observedState(space, observation);
    observations.push({ state, actionId: action.id, observation, actual }); state = actual;
  }
  return { reached: signature(state) === signature(goal), finalState: state, observations, budget };
}
export function distribution(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const quantile = (q: number): number | null => {
    if (!n) return null;
    const i = (n - 1) * q, low = Math.floor(i), high = Math.ceil(i);
    return sorted[low]! + (sorted[high]! - sorted[low]!) * (i - low);
  };
  return { n, min: sorted[0] ?? null, median: quantile(.5), p90: quantile(.9), max: sorted.at(-1) ?? null,
    mean: n ? sorted.reduce((a, b) => a + b, 0) / n : null };
}
export function predictionMetrics(rows: readonly StepPrediction[]) {
  const n = rows.length;
  const reasons: Record<string, number> = {};
  for (const p of rows) reasons[p.snapshot.terminationReason] = (reasons[p.snapshot.terminationReason] ?? 0) + 1;
  return { calls: n, reasons, noQuietCandidate: rows.filter(p => p.kind === "infeasible").length,
    nonconverged: rows.filter(p => !p.snapshot.converged).length,
    nonconvergedRate: n ? rows.filter(p => !p.snapshot.converged).length / n : null,
    anyOutputRefusalRate: n ? rows.filter(p => p.ambiguous.length || Object.values(p.snapshot.values).some(v => v === null)).length / n : null,
    unknownDecodedRate: n ? rows.filter(p => p.kind === "unknown").length / n : null,
    ledgerMismatches: rows.filter(p => p.audit.mismatch).length,
    maxLedgerError: Math.max(0, ...rows.map(p => p.audit.error)),
    statusMissing: rows.filter(p => typeof p.snapshot.converged !== "boolean" || !p.snapshot.terminationReason).length,
    milliseconds: distribution(rows.map(p => p.milliseconds)) };
}
export function readback(model: TransitionMemory, seed: number): StepPrediction[] {
  return frames(model.space.states).flatMap(state => model.actions.map(action => model.predict(state, action, seed)));
}
export function stableReadback(row: StepPrediction): string {
  return JSON.stringify({ values: row.snapshot.values, ambiguous: row.ambiguous, next: row.next,
    snapshot: row.snapshot, energy: row.audit.energy, kind: row.kind });
}
/** Includes all synapses and registered evidence, excludes transient neural
 * activation, timing and passive global diagnostic counters. */
export function learningFingerprint(model: TransitionMemory): string {
  const hash = createHash("sha256"), net = model.mem.net;
  const row = new Float64Array(net.neuronCount * 4);
  for (let i = 0; i < net.neuronCount; i++) {
    for (let j = 0; j < net.neuronCount; j++) {
      row[j * 4] = net.getWeight(i, j); row[j * 4 + 1] = net.getInhibitoryWeight(i, j);
      row[j * 4 + 2] = net.getDirectedWeight(i, j); row[j * 4 + 3] = net.getDirectedInhibitoryWeight(i, j);
    }
    hash.update(Buffer.from(row.buffer));
  }
  hash.update(JSON.stringify({ generation: model.mem.evidenceGeneration, conflicts: model.mem.evidenceConflicts,
    rules: Array.from({ length: model.mem.ruleCount }, (_, i) => ({ core: model.mem.ruleCore(i), evidence: model.mem.ruleEvidence(i), fields: model.mem.ruleOutcomeFields(i) })) }));
  return hash.digest("hex");
}
