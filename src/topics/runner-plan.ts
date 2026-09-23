import { mkdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { mulberry32 } from "../prng.js";
import { TransitionMemory } from "../planning/transition-memory.js";
import { collectTransitions } from "../planning/collect.js";
import { planGoal } from "../planning/planner.js";
import { executeGoal } from "../planning/execute.js";
import type { Frame, TransitionBench } from "../planning/space.js";
import { PATH_SPACE, PathBench } from "./path-world.js";
import { CHAIN_SPACE, OrderedChainBench } from "./ordered-chain-world.js";
import { distribution, executeFrozen, learningFingerprint, predictionMetrics, randomActions, readback, shortestReference, stableReadback } from "./plan-evaluation.js";

const seeds = (process.env.SEEDS ?? "1,2,3").split(",").map(Number);
if (!seeds.length || seeds.some(s => !Number.isSafeInteger(s) || s < 0)) throw new Error("SEEDS must be nonnegative integers");
mkdirSync("runs", { recursive: true });
const revision = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const worktree = execFileSync("git", ["status", "--short"], { encoding: "utf8" }).trim();
const reports = [];
for (const seed of seeds) {
  const started = performance.now();
  const log: string[] = [];
  const out = (message: string) => { log.push(message); console.log(message); writeFileSync(`runs/plan-${seed}.log`, log.join("\n") + "\n"); };
  out(`PLAN-009 seed=${seed}, baseline=33f8b6a, source=${revision}; fixed collection budgets A=64/B=40, action budgets A=14/B=8`);
  const model = new TransitionMemory(PATH_SPACE);
  const training = await collectTransitions(model, new PathBench(), 64, seed);
  const before = readback(model, seed + 100000), fingerprintBefore = learningFingerprint(model);
  const pairs = Array.from({ length: 8 }, (_, start) => Array.from({ length: 8 }, (_, goal) => ({ start, goal }))).flat().filter(p => p.start !== p.goal);
  const rng = mulberry32(seed);
  for (let i = pairs.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [pairs[i], pairs[j]] = [pairs[j]!, pairs[i]!]; }
  const trials = pairs.map(({ start, goal }, i) => {
    const from = { pos: start }, to = { pos: goal }, trialSeed = seed * 1000000 + i * 100;
    const plan = planGoal(model, from, to, trialSeed);
    const actual = executeFrozen(PATH_SPACE, new PathBench(), plan);
    const bfs = shortestReference(PATH_SPACE, new PathBench(), from, to)!;
    const random = randomActions(PATH_SPACE, new PathBench(), from, to, 14, trialSeed);
    out(`A ${start}->${goal}: actions=${plan.steps.map(p => p.action.id)} reached=${actual.reached} predict=${plan.predictions.length} BFS=${bfs} random=${random.reached}`);
    return { start, goal, bfs, plan, actual, random, ratio: actual.observations.length / bfs };
  });
  const after = readback(model, seed + 100000), fingerprintAfter = learningFingerprint(model);
  const unchanged = before.filter((p, i) => stableReadback(p) === stableReadback(after[i]!)).length;
  // Separate monitored arm, reset to the same training corpus for every trial.
  // Thus neither previous test outcomes nor corrections leak into a later trial.
  const monitored = await Promise.all(pairs.map(async ({ start, goal }, i) => {
    const fresh = new TransitionMemory(PATH_SPACE);
    await collectTransitions(fresh, new PathBench(), 64, seed);
    const result = await executeGoal(fresh, new PathBench(), { pos: start }, { pos: goal }, seed * 1000000 + i * 100);
    out(`A monitored ${start}->${goal}: reached=${result.reached} steps=${result.steps.length} replans=${result.replans.length}`);
    return result;
  }));
  const chain = async (scenario: "normal" | "closed-gate" | "displacement", replan: boolean) => {
    const memory = new TransitionMemory(CHAIN_SPACE);
    const collection = await collectTransitions(memory, new OrderedChainBench(), 40, seed);
    const pre = readback(memory, seed + 200000), hashBefore = learningFingerprint(memory);
    const probe = planGoal(memory, { node: 0 }, { node: 4 }, seed);
    const reverse = planGoal(memory, { node: 3 }, { node: 2 }, seed);
    const post = readback(memory, seed + 200000), hashAfter = learningFingerprint(memory);
    const bench = new OrderedChainBench();
    let injected = false;
    const injections: { state: Frame; natural: Frame; actual: Frame }[] = [];
    const injecting: TransitionBench = { conduct(state, action) {
      if (scenario === "closed-gate" && state.node === 2) bench.setGateOpen(false);
      const natural = bench.conduct(state, action);
      if (scenario === "displacement" && !injected && natural.nextNode === 1) {
        injected = true; const actual = { nextNode: 3 }; injections.push({ state, natural, actual }); return actual;
      }
      if (scenario === "closed-gate" && state.node === 2 && action.advance === 0) injections.push({ state, natural, actual: natural });
      return natural;
    } };
    const execution = await executeGoal(memory, injecting, { node: 0 }, { node: 4 }, seed, { replan });
    const allPlans = [probe, reverse, ...execution.plans];
    const noReverseDoor = allPlans.every(p => p.steps.every(s => !(s.state.node === 3 && s.next?.node === 2)));
    const gateDetected = execution.steps.some(s => s.state.node === 2 && s.capture.class !== "within-envelope");
    const gateSafe = execution.finalState.node === 2 && execution.replans.some(r => r.from.node === 2)
      && execution.plans.slice(1).every(p => !p.steps.some(s => s.state.node === 2 && s.next?.node === 3));
    const success = scenario === "normal" ? execution.reached && JSON.stringify([0, ...execution.steps.map(s => s.actual.node)]) === "[0,1,2,3,4]"
      : scenario === "closed-gate" ? gateDetected && gateSafe
      : execution.reached && execution.replans.some(r => r.from.node === 3) && execution.steps[0]?.capture.class !== "within-envelope";
    out(`B ${scenario} replan=${replan}: reached=${execution.reached}, final=${execution.finalState.node}, reason=${execution.terminationReason}, replans=${execution.replans.length}, protocolSuccess=${success}`);
    return { scenario, replan, collection, probe, reverse, isolation: { before: pre, after: post,
      unchanged: pre.filter((p, i) => stableReadback(p) === stableReadback(post[i]!)).length, hashBefore, hashAfter },
      injections, execution, success, noReverseDoor,
      predictions: [...pre, ...post, ...allPlans.flatMap(p => p.predictions)] };
  };
  const chainTrials = [await chain("normal", true), await chain("closed-gate", true), await chain("displacement", true),
    await chain("closed-gate", false), await chain("displacement", false)];
  const randomChain = Array.from({ length: 56 }, (_, i) => randomActions(CHAIN_SPACE, new OrderedChainBench(), { node: 0 }, { node: 4 }, 8, seed * 1000000 + i * 100));
  const predictions = [...before, ...after, ...trials.flatMap(t => t.plan.predictions),
    ...monitored.flatMap(e => e.plans.flatMap(p => p.predictions)), ...chainTrials.flatMap(t => t.predictions)];
  const selected = [...trials.flatMap(t => t.plan.steps), ...monitored.flatMap(e => e.steps.map(s => s.forecast)),
    ...chainTrials.flatMap(t => t.execution.steps.map(s => s.forecast))];
  const metrics = predictionMetrics(predictions);
  const summary = {
    seed, path: { n: trials.length, reached: trials.filter(t => t.actual.reached).length,
      reachRate: trials.filter(t => t.actual.reached).length / trials.length,
      successfulStepRatio: distribution(trials.filter(t => t.actual.reached).map(t => t.ratio)),
      allStepRatio: distribution(trials.map(t => t.ratio)), predictCalls: distribution(trials.map(t => t.plan.predictions.length)),
      randomReached: trials.filter(t => t.random.reached).length, randomReachRate: trials.filter(t => t.random.reached).length / trials.length,
      monitoredReached: monitored.filter(t => t.reached).length, monitoredStepRatio: distribution(monitored.map((t, i) => t.steps.length / trials[i]!.bfs)),
      replans: monitored.reduce((n, t) => n + t.replans.length, 0),
      selectedFixedPointFraction: trials.flatMap(t => t.plan.steps).filter(p => p.snapshot.converged).length / trials.flatMap(t => t.plan.steps).length },
    chain: { ordered: chainTrials[0]!.success, noReverseDoor: chainTrials.every(t => t.noReverseDoor),
      closedGate: chainTrials[1]!.success, displacement: chainTrials[2]!.success,
      replanPredictIncrements: chainTrials.slice(0, 3).flatMap(t => t.execution.replans.map(r => r.predictIncrement)),
      noReplanClosedGateReached: chainTrials[3]!.execution.reached, noReplanDisplacementReached: chainTrials[4]!.execution.reached,
      noReplanFailureRate: chainTrials.slice(3).filter(t => !t.execution.reached).length / 2,
      randomReached: randomChain.filter(t => t.reached).length, randomN: randomChain.length },
    isolation: { pathUnchanged: unchanged, pathN: before.length, pathHashEqual: fingerprintBefore === fingerprintAfter,
      chainAllUnchanged: chainTrials.every(t => t.isolation.unchanged === 10 && t.isolation.hashBefore === t.isolation.hashAfter) },
    predictions: metrics, selectedFixedPointFraction: selected.filter(p => p.snapshot.converged).length / selected.length,
    totalMilliseconds: performance.now() - started,
  };
  const acceptance = {
    P1: summary.path.reachRate >= .9,
    P2: summary.path.allStepRatio.median !== null && summary.path.allStepRatio.median <= 1.5,
    P3: summary.chain.ordered, P4: summary.chain.noReverseDoor, P5: summary.chain.closedGate, P6: summary.chain.displacement,
    P7: unchanged === before.length && summary.isolation.pathHashEqual && summary.isolation.chainAllUnchanged,
    P8: metrics.ledgerMismatches === 0 && metrics.statusMissing === 0,
  };
  const failures = { openLoopPath: trials.filter(t => !t.actual.reached), monitoredPath: monitored.filter(t => !t.reached),
    chain: chainTrials.slice(0, 3).filter(t => !t.success), randomPath: trials.filter(t => !t.random.reached).map(t => ({ start: t.start, goal: t.goal, random: t.random })) };
  const report = { schema: "PLAN-009/v1", timestamp: new Date().toISOString(), source: { revision, worktree, node: process.version },
    summary, acceptance, worldA: { training, isolation: { before, after, fingerprintBefore, fingerprintAfter }, trials, monitored },
    worldB: { trials: chainTrials, random: randomChain }, failures };
  writeFileSync(`runs/plan-${seed}.json`, JSON.stringify(report, null, 2) + "\n");
  out(`SUMMARY ${JSON.stringify(summary)}`); out(`ACCEPTANCE ${JSON.stringify(acceptance)}`);
  reports.push({ seed, summary, acceptance });
}
writeFileSync("runs/plan-summary.json", JSON.stringify(reports, null, 2) + "\n");
if (reports.some(r => Object.values(r.acceptance).some(pass => !pass))) process.exitCode = 1;
