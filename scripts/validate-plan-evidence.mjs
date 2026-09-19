import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

// Recompute acceptance from the full persisted trials, not runner pass flags.
const seeds = (process.env.SEEDS ?? '1,2,3').split(',').map(Number);
const checked = [];
const reports = seeds.map(seed => JSON.parse(readFileSync(`runs/plan-${seed}.json`, 'utf8')));
for (let k = 0; k < reports.length; k++) {
  const r = reports[k], a = r.worldA, b = r.worldB.trials;
  assert.equal(a.trials.length, 56);
  assert.equal(new Set(a.trials.map(t => `${t.start},${t.goal}`)).size, 56);
  assert.ok(a.trials.filter(t => t.actual.reached).length / a.trials.length >= .9);
  const ratios = a.trials.map(t => t.actual.observations.length / t.bfs).sort((x,y) => x-y);
  assert.ok((ratios[27] + ratios[28]) / 2 <= 1.5);
  assert.deepEqual([0, ...b[0].execution.steps.map(s => s.actual.node)], [0,1,2,3,4]);
  const allPlans = [...a.trials.map(t => t.plan), ...a.monitored.flatMap(e => e.plans),
    ...b.flatMap(t => [t.probe, t.reverse, ...t.execution.plans])];
  for (const plan of allPlans) {
    assert.ok(plan.predictions.length <= plan.predictBudget);
    assert.ok(plan.steps.length <= plan.maxDepth);
    if (plan.status === 'found') {
      let state = plan.start;
      for (const step of plan.steps) { assert.deepEqual(step.state, state); state = step.next; }
      assert.deepEqual(state, plan.goal);
    }
    assert.ok(!plan.steps.some(s => s.state.node === 3 && s.next?.node === 2));
  }
  const gate = b[1].execution, displacement = b[2].execution;
  assert.ok(gate.steps.some(s => s.state.node === 2 && s.capture.class !== 'within-envelope'));
  assert.equal(gate.finalState.node, 2);
  assert.ok(gate.replans.some(x => x.from.node === 2));
  assert.ok(gate.plans.slice(1).every(p => !p.steps.some(s => s.state.node === 2 && s.next?.node === 3)));
  assert.equal(displacement.reached, true);
  assert.ok(displacement.replans.some(x => x.from.node === 3));
  const stable = p => ({ snapshot: p.snapshot, next: p.next, ambiguous: p.ambiguous, energy: p.audit.energy, kind: p.kind });
  assert.deepEqual(a.isolation.before.map(stable), a.isolation.after.map(stable));
  assert.equal(a.isolation.fingerprintBefore, a.isolation.fingerprintAfter);
  for (const t of b) {
    assert.deepEqual(t.isolation.before.map(stable), t.isolation.after.map(stable));
    assert.equal(t.isolation.hashBefore, t.isolation.hashAfter);
  }
  const rows = [...a.isolation.before, ...a.isolation.after, ...allPlans.flatMap(p => p.predictions),
    ...b.flatMap(t => [...t.isolation.before, ...t.isolation.after])];
  assert.equal(rows.length, r.summary.predictions.calls);
  for (const p of rows) {
    assert.equal(typeof p.snapshot.converged, 'boolean');
    assert.ok(['fixed-point','quiet-constraint','flip-budget','no-quiet-candidate'].includes(p.snapshot.terminationReason));
    assert.ok(Number.isFinite(p.audit.traceEnd));
    assert.ok(Math.abs(p.audit.traceEnd - p.audit.energy) < 1e-7);
    assert.ok(Math.abs(p.audit.recomputed - p.audit.energy) < 1e-7);
  }
  assert.equal(r.failures.openLoopPath.length, a.trials.filter(t => !t.actual.reached).length);
  for (const e of [...a.monitored, ...b.map(t => t.execution)]) {
    assert.ok(e.steps.length <= e.executionBudget);
    for (const [i, step] of e.steps.entries()) {
      assert.ok(step.evidenceGenerationAfter > step.forecast.snapshot.generation);
      if (e.replanningEnabled && step.capture.class !== 'within-envelope')
        assert.ok(e.replans.some(x => x.afterStep === i + 1));
    }
  }
  checked.push({ seed: seeds[k], P1: true, P2: true, P3: true, P4: true, P5: true, P6: true, P7: true, P8: true,
    predictCalls: rows.length, retainedOpenLoopFailures: r.failures.openLoopPath.length });
}
const verify = readFileSync('runs/plan-verify.log', 'utf8');
assert.equal(readFileSync('runs/plan-verify.exit.txt', 'utf8').trim(), '0');
const count = Number(verify.match(/(?:#|ℹ) tests (\d+)/)?.[1]);
const passes = Number(verify.match(/(?:#|ℹ) pass (\d+)/)?.[1]);
assert.ok(count > 179 && passes === count);
assert.match(verify, /(?:#|ℹ) fail 0/);
const walk = dir => readdirSync(dir, {withFileTypes:true}).flatMap(e => e.isDirectory() ? walk(join(dir,e.name)) : [join(dir,e.name)]);
const files = [...walk('src'), ...walk('test'), 'package.json', 'package-lock.json', 'tsconfig.json',
  'scripts/validate-plan-evidence.mjs', 'docs/PLAN-009-goal-chain-planning.zh-CN.md',
  ...seeds.flatMap(s => [`runs/plan-${s}.json`,`runs/plan-${s}.log`]), 'runs/plan-summary.json', 'runs/plan-verify.log'];
const manifest = { timestamp: new Date().toISOString(), baseline: '33f8b6a',
  parentRevision: execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
  node: process.version, seeds, validation: checked, P9: {tests:count,passed:passes,failed:0,exitCode:0},
  files: Object.fromEntries(files.sort().map(path => [path.replaceAll('\\','/'), createHash('sha256').update(readFileSync(path)).digest('hex')])) };
writeFileSync('runs/plan-source-manifest.json', JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({validation:checked,P9:manifest.P9,hashedFiles:files.length},null,2));
