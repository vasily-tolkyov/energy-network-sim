import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const walk = dir => fs.readdirSync(path.join(root, dir), { withFileTypes: true }).flatMap(entry => {
  const name = `${dir}/${entry.name}`;
  return entry.isDirectory() ? walk(name) : [name];
});
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const paths = [...walk('src'), ...walk('test')].filter(p => p.endsWith('.ts') && p !== 'src/topics/path-world.ts');
paths.push('package.json', 'package-lock.json', 'tsconfig.json');
const files = Object.fromEntries(paths.sort().map(p => [p, digest(fs.readFileSync(path.join(root, p)))]));
const manifest = {
  baseline: 'e732f2b4ef46e0ac482480095a31cb5aa23ec6be',
  sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
  utc: new Date().toISOString(), node: process.version, files,
  sourceDigest: digest(JSON.stringify(files)), sourceDigestFormat: 'SHA256 of compact JSON with sorted path keys',
  scopeNote: 'Final source snapshot; path-world.ts excluded and untouched. Long native runs began with the model semantics committed as c399c11. Later sparse exchange optimization has exact complete-result equivalence checks. Affected attention/label/prototype runners were repeated. Raw audit scripts hard-code the baseline revision. Concurrent elapsed times are not isolated performance measurements.',
};
fs.writeFileSync(path.join(root, 'runs/review2/source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
const results = JSON.parse(fs.readFileSync(path.join(root, 'runs/review2/reviewer-b/before-after-regression.json'), 'utf8'));
const current = results.at(-1);
const all = (values, predicate) => values.length > 0 && values.every(predicate);
const checks = Object.fromEntries([
  ['confirmed-control', 'before'], ['truth-guess', 'afterGuess'], ['truth-guess-truth', 'afterTruth'],
].map(([name, field]) => [name, all(current.evidenceOrder[field], v => v === 1)]));
checks['continuous-equivalent'] = all(current.equivalentContinuous.after, v => Number.isFinite(v) && v >= .2 && v <= .201);
assert.ok(Object.values(checks).every(Boolean));
fs.writeFileSync(path.join(root, 'runs/review2/before-after-verdict.json'), JSON.stringify({
  sourceCommit: manifest.sourceCommit, checks, passed: Object.values(checks).filter(Boolean).length, total: 4,
  note: 'Four explicit nonempty readback predicates over the unmodified script output; the original script has no pass counter. Continuous alias reads must lie within the two observed values, not merely a broad accuracy tolerance.',
}, null, 2) + '\n');
console.log(JSON.stringify({ sourceCommit: manifest.sourceCommit, files: paths.length, beforeAfter: '4/4' }));
