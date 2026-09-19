import { readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
const paths=[1,2,3].flatMap(s=>[`runs/plan-${s}.json`,`runs/plan-${s}.log`]);
paths.push('runs/plan-summary.json');
writeFileSync('runs/plan-development/initial-formal-run.json.gz',gzipSync(JSON.stringify({
  note:'Initial formal run; rerun after P2 gate was made explicit for ALL trials, including failures. Learning and planner parameters unchanged.',
  files:Object.fromEntries(paths.map(p=>[p,readFileSync(p,'utf8')]))
})));
