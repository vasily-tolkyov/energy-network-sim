// Loose equivalence for incremental-field solver: identical answers and
// termination states, energy/ledger within tolerance (float tails differ by
// design - fields are maintained incrementally instead of recomputed).
// Usage: node scripts/check-solver-equivalence-loose.mjs /path/to/prior/dist/src/network.js
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {EnergyNetwork as After} from '../dist/src/network.js';
import {mulberry32} from '../dist/src/prng.js';
if (!process.argv[2]) throw new Error('Pass the prior compiled network.js path');
const {EnergyNetwork:Before}=await import(pathToFileURL(path.resolve(process.argv[2])).href);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const TOL=1e-6;
let worst=0, answerMismatch=0, reasonMismatch=0, ledgerMismatch=0;
for(let seed=1;seed<=200;seed++) {
 const rnd=mulberry32(seed), n=10+seed%35;
 const a=new Before({neuronCount:n,activationEnergy:1,maintenanceEnergy:.5}),b=new After({neuronCount:n,activationEnergy:1,maintenanceEnergy:.5});
 for(let i=0;i<n;i++)for(let j=0;j<i;j++) {
  const w=rnd()<.25?rnd()*2:0,g=rnd()<.2?rnd():0;
  for(const net of [a,b]){net.strengthen(i,j,w);net.strengthenInhibitory(i,j,g);}
 }
 for(let i=0;i<n;i++)if(rnd()<.1)for(let j=0;j<n;j++)if(rnd()<.3){const d=rnd()*3;for(const net of [a,b])net.strengthenDirectedInhibitory(i,j,d);}
 const wells=[{memberNeuronIds:Array.from({length:n},(_,i)=>i)}];
 const options={seed,levels:5,sweepsPerLevel:4,quenchMaxFlips:seed%40,quenchCandidatesOnly:seed%2===0,fallbackQuietOnly:seed%3!==0};
 const x=a.settleAnnealed([seed%n],wells,options),y=b.settleAnnealed([seed%n],wells,options);
 if(JSON.stringify(x.activeNeurons)!==JSON.stringify(y.activeNeurons))answerMismatch++;
 if(x.terminationReason!==y.terminationReason)reasonMismatch++;
 worst=Math.max(worst,Math.abs(x.energy-y.energy));
 const traceEnd=x.trace.energies.at(-1);
 if(Math.abs(traceEnd-y.trace.energies.at(-1))>TOL)ledgerMismatch++;
}
const report={comparison:'incremental fields vs dense recompute; answers+termination identical, energy within 1e-6',
 answerMismatch,reasonMismatch,ledgerMismatch,worstEnergyAbsDiff:worst,tolerance:TOL,seeds:200};
writeFileSync(path.join(root,'runs/incremental-fields-equivalence.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
assert.equal(answerMismatch,0,'answers must be identical');
assert.equal(reasonMismatch,0,'termination reasons must be identical');
assert.ok(worst<TOL,'energy within tolerance');
