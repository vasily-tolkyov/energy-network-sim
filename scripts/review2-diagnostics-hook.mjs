// Read-only exit reporting for original reviewers that omit status totals.
// Preloading imports only the passive counter; it does not change the model,
// random stream, inputs, stopping policy or scoring.
import fs from 'node:fs';
import path from 'node:path';
import { predictionQuality } from '../dist/src/pop/prediction-quality.js';
const output = process.env.REVIEW_DIAGNOSTICS_PATH;
if (!output) throw new Error('REVIEW_DIAGNOSTICS_PATH is required');
process.on('exit', code => {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify({ exitCode: code, ...predictionQuality.snapshot() }, null, 2) + '\n');
});
