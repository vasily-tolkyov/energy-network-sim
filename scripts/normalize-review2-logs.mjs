// Preserve original stdout. Correct only undefined empty-denominator displays
// in a derived report; do not rerun learning or alter any measured score.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sourceDir = path.join(root, 'review-artifacts/reproduction/runs');
const provenance = [];
for (const name of fs.readdirSync(sourceDir).filter(n => n.endsWith('.log'))) {
  const raw = fs.readFileSync(path.join(sourceDir, name), 'utf8');
  let correctedEmptyRows = 0;
  const derived = raw.split('\n').map(line => {
    if (!line.includes('NaN%')) return line;
    if (!/self-taught|unseen-gate-off|unseen-gate-on/.test(line)) throw new Error(`Unrecognized undefined metric: ${line}`);
    correctedEmptyRows++;
    return `${line.match(/^\s*\S+/)[0]} N/A (n=0; accuracy, refusal and constant baseline undefined)`;
  }).join('\n');
  fs.writeFileSync(path.join(root, 'runs', name), derived);
  provenance.push({ source: `review-artifacts/reproduction/runs/${name}`, output: `runs/${name}`,
    sourceSha256: createHash('sha256').update(raw).digest('hex'), correctedEmptyRows,
    note: 'Numerators, denominators, predictions and all nonempty scores unchanged. Original output retained.' });
}
fs.writeFileSync(path.join(root, 'runs/review2/log-rendering-provenance.json'), JSON.stringify(provenance, null, 2));
console.log(JSON.stringify(provenance.map(({output,correctedEmptyRows}) => ({output,correctedEmptyRows}))));
