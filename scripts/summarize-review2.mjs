// Derive a review index from saved evidence, never substitute missing results.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const load = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8').replace(/^\uFEFF/, ''));
const read = p => fs.readFileSync(path.join(root, p), 'utf8');
const exists = p => fs.existsSync(path.join(root, p));
const files = p => exists(p) ? fs.readdirSync(path.join(root, p)) : [];
const rate = (n, total) => total ? n / total : null;
const risk = 'Heuristic stopping and marginal coverage do not certify joint coverage or complete factor discovery.';
function counts(p) {
  const s = read(p), result = { file: p };
  for (const k of ['tests', 'pass', 'fail', 'cancelled', 'skipped']) {
    const m = [...s.matchAll(new RegExp(`(?:#|ℹ) ${k} (\\d+)`, 'g'))].at(-1);
    result[k] = m ? Number(m[1]) : null;
  }
  return result;
}
const testFiles = ['verify-final.log', 'stage4-mechanisms.tap', 'review-a-revision.tap', 'review-a-prior.tap',
  'review-b-revision-original.tap', 'review-b-revision-isolated.tap', 'review-b-prior.tap', 'new-worlds-verdict.tap'];
const officialNames = new Set([...files('review-artifacts/reproduction/runs').filter(n => n.endsWith('.log')),
  'attention-v1.log', 'capacity-v1.log']);
const official = [...officialNames].map(name => {
  const file = `runs/${name}`, s = read(file);
  const diagnostics = s.split('\n').filter(l => l.includes('confidencePolicy')).map(l => JSON.parse(l.slice(l.indexOf('{'))));
  return { name, file, answerMeasurements: s.split('\n').filter(l => /%|N\/A|准确|召回|实验 .*次|形成 .*次|影响因素/.test(l) && !l.includes('confidencePolicy')),
    dynamics: diagnostics, hypothesisFamily: 'Declared sensory dimensions, bounded rule cores, supplied candidate grid and local energy interactions.',
    coverageRisk: risk, rarePositiveRecall: null, rarePositiveNote: 'A designated rare-positive benchmark is reported separately as W04; ordinary accuracy is not rare-positive recall.' };
});
const external = 'runs/review2/reviewer-b';
const worlds = files(external).filter(n => /^W\d+-new-world.json$/.test(n)).map(name => {
  const { summary } = load(`${external}/${name}`);
  return { ...summary, file: `${external}/${name}`, nonconvergedRate: rate(summary.nonconverged, summary.n),
    anyOutputAbstentionRate: rate(summary.refused, summary.n), rarePositiveRecall: summary.positiveRecall ?? null,
    hypothesisFamily: summary.name, coverageRisk: risk };
});
const independentB = files(external).filter(n => /^phase-[ab]-(seed\d+|off-grid).json$/.test(n)).map(name => {
  const data = load(`${external}/${name}`);
  const metricScopes = {};
  const scopes = typeof data.summary.n === 'number' ? { all: data.summary } : data.summary;
  for (const [key, value] of Object.entries(scopes)) {
    if (value && typeof value === 'object' && typeof value.n === 'number') metricScopes[key] = {
      ...value, nonconvergedRate: typeof value.nonconverged === 'number' ? rate(value.nonconverged, value.n) : null,
      anyOutputAbstentionRate: typeof value.refused === 'number' ? rate(value.refused, value.n) : null,
    };
  }
  const training = (data.training ?? []).filter(r => typeof r.converged === 'boolean');
  const audits = data.audits ?? data.rows.map(r => r.audit);
  assert.equal(audits.length, training.length + data.rows.length);
  const nonconverged = audits.filter(r => !r.converged).length;
  const abstained = training.filter(r => r.anyOutputAbstained).length + data.rows.filter(r => r.refused).length;
  const completeDiagnostics = { predictions: audits.length, nonconverged, nonconvergedRate: rate(nonconverged, audits.length),
    anyOutputAbstained: abstained, anyOutputAbstentionRate: rate(abstained, audits.length),
    scope: training.length ? 'All recorded training and grid-evaluation predictions' : 'Off-grid evaluation only; training is recorded in the grid report' };
  return { file: `${external}/${name}`, ...data.summary, metricScopes, completeDiagnostics,
    hypothesisFamily: 'Independent literal circuit oracle; fixed grid or preselected off-grid sample.', coverageRisk: risk };
});
const independentA = files('review-artifacts').filter(n => /^(fresh|independent)-.*-results.json$/.test(n)).map(name => {
  const { summary, results } = load(`review-artifacts/${name}`);
  const trainingName = `review-artifacts/${name.replace('-results', '-training')}`;
  const training = load(trainingName), predicted = training.log.filter(r => r.converged !== null && r.converged !== undefined);
  const nonconverged = predicted.filter(r => !r.converged).length;
  const abstained = results.filter(r => Object.values(r.pred).some(v => typeof v !== 'number')).length;
  const diagnosticsFile = `runs/review2/${name.replace('-results.json', '-diagnostics.json')}`;
  let completeDiagnostics = null;
  if (exists(diagnosticsFile)) {
    const repeat = load(`review-artifacts/instrumented-${name}`);
    // This comparison excludes elapsed time but includes every prediction,
    // truth, nearest neighbour, partition, condition and score in order.
    assert.deepEqual(repeat.results, results, `Diagnostic repeat changed results: ${name}`);
    assert.deepEqual(repeat.summary.partitions, summary.partitions);
    const repeatTraining = load(`review-artifacts/instrumented-${name.replace('-results', '-training')}`);
    assert.deepEqual(repeatTraining.log, training.log, `Diagnostic repeat changed training predictions: ${name}`);
    assert.deepEqual(repeatTraining.episodes, training.episodes);
    completeDiagnostics = { ...load(diagnosticsFile), file: diagnosticsFile, exactPredictionRepeat: true };
    assert.equal(completeDiagnostics.exitCode, 0);
    assert.equal(completeDiagnostics.predictions, predicted.length + results.length);
  } else if (summary.dynamics.reportedNonConverged !== undefined) {
    const total = summary.dynamics.calls;
    const totalAbstained = abstained + predicted.filter(r => r.anyOutputAbstained).length;
    completeDiagnostics = { predictions: total, nonconverged: summary.dynamics.reportedNonConverged,
      nonconvergedRate: rate(summary.dynamics.reportedNonConverged, total),
      anyOutputAbstained: totalAbstained, anyOutputAbstentionRate: rate(totalAbstained, total),
      note: 'Exact aggregate convergence from original fresh audit; refusal from every original training/evaluation row.' };
  }
  assert.ok(completeDiagnostics, `Missing complete status audit: ${name}`);
  return { file: `review-artifacts/${name}`, ...summary,
    completeDiagnostics,
    trainingDiagnostics: { n: predicted.length, nonconverged, nonconvergedRate: rate(nonconverged, predicted.length),
      anyOutputAbstained: predicted.filter(r => r.anyOutputAbstained).length },
    evaluationDiagnostics: { n: results.length, anyOutputAbstained: abstained, anyOutputAbstentionRate: rate(abstained, results.length),
      reportedNonconvergence: completeDiagnostics.nonconverged - nonconverged,
      nonconvergedRate: rate(completeDiagnostics.nonconverged - nonconverged, results.length),
      note: 'Original A evaluation rows omit convergence. Subtract saved exact training counts from exact complete diagnostics; independent repeats assert identical ordered predictions.' },
    hypothesisFamily: 'Circuit oracle with supplied grid; independent off-grid and gate-boundary probes.', coverageRisk: risk };
});
const commandNames = ['discrete', 'continuous', 'extension', 'pop', 'chem', 'concept', 'neural-attention', 'demo', 'demo-seq',
  'independent-discrete-4', 'independent-continuous-4', 'independent-continuous-5', 'fresh-continuous-6', 'fresh-continuous-7'];
const reproductionCommands = commandNames.map(name => {
  const meta = load(`review-artifacts/reproduction/${name}-meta.json`);
  assert.equal(meta.exitCode, 0, `Incomplete/failed group: ${name}`);
  return { ...meta, originalHardcodedCommitIsBaseline: true, actualSourceManifest: 'runs/review2/source-manifest.json' };
});
assert.ok(exists('review-artifacts/reproduction/runs/explore-ext-frozen-seed1.log'));
assert.equal(Number(read('runs/review2/capacity-full.exit').trim()), 0);
const report = { sourceManifest: load('runs/review2/source-manifest.json'), generatedAt: new Date().toISOString(),
  note: 'All numbers derive from preserved outputs. Null means unavailable or N/A, never zero. Runtime measurements were concurrent and are not isolated performance benchmarks.',
  tests: testFiles.filter(n => exists(`runs/review2/${n}`)).map(n => counts(`runs/review2/${n}`)),
  beforeAfter: load('runs/review2/before-after-verdict.json'), ablation: load('runs/ablation-manipulation.json'),
  rareWorld: load('runs/review2/W04-budget-matched-baseline.json'), official, worlds, independentA, independentB,
  reproductionCommands, supplementaryCapacity: { ...load('runs/review2/capacity-full-start.json'), exitCode: 0,
    stdout: 'runs/review2/capacity-full.log' } };
fs.writeFileSync(path.join(root, 'runs/review2/summary.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ tests: report.tests, officialLogs: official.length, worlds: worlds.length, independentA: independentA.length, independentB: independentB.length }));
