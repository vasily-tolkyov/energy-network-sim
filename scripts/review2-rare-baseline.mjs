import { readFileSync, writeFileSync } from 'node:fs';
import { PopChannelMap } from '../dist/src/pop/popmap.js';
import { ExperimentPlanner } from '../dist/src/pop/explore/planner.js';
import { Explorer } from '../dist/src/pop/explore/explorer.js';
import { mulberry32 } from '../dist/src/prng.js';

const source = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const specs = ['a', 'b', 'c'].map(name => ({ name, bins: 4 }));
const truth = q => ({ y: Number(q.a === 0 && q.b === 0 && q.c === 0) });
const grid = specs.reduce((rows, s) => rows.flatMap(q => Array.from({ length: s.bins }, (_, v) => ({ ...q, [s.name]: v }))), [{}]);
const seed = 11, rand = mulberry32(seed), order = [...grid];
for (let k = order.length - 1; k > 0; k--) { const j = Math.floor(rand() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }
const planner = new ExperimentPlanner(specs), cm = new PopChannelMap(specs), om = new PopChannelMap([{ name: 'y', bins: 2 }]);
const ex = new Explorer(cm, om, planner, specs, { conduct: truth }, { y: 1 }, {}, seed);
const training = [];
for (const q of order.slice(0, source.summary.experiments)) {
  const outcomes = truth(q);
  ex.mem.learnFromObservation(q, outcomes, 2);
  const { pairs, newBins } = planner.register({ conditions: q, outcomes, classification: 'unknown-change' });
  // Audit-only reuse of the same learning pipeline, replacing just selection.
  ex.absorbPairs(pairs, newBins);
  training.push({ conditions: q, outcomes });
}
const rows = grid.map(q => { const p = ex.mem.predict(q, seed); return { q, truth: truth(q), ...p }; });
const summary = {
  selection: 'uniform without replacement, seed 11 fixed before scoring', budget: training.length,
  correct: rows.filter(r => r.decoded.y === r.truth.y).length, n: rows.length,
  positiveRecall: { n: 1, correct: rows.filter(r => r.truth.y === 1 && r.decoded.y === 1).length },
  foundFactors: ex.influentialDims,
  nonconverged: rows.filter(r => !r.converged).length,
  anyOutputAbstained: rows.filter(r => typeof r.decoded.y !== 'number').length,
  hypothesisFamily: 'finite 4×4×4 grid; arbitrary conjunctions including rare isolated positives',
  coverageRisk: 'quorum-met is heuristic; individual-dimension coverage does not certify joint coverage',
};
writeFileSync('runs/review2/W04-budget-matched-baseline.json', JSON.stringify({ explorer: source.summary, baseline: summary, training, rows }, null, 2));
console.log(JSON.stringify(summary));
