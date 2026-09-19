// Compare complete solver outputs against a compiled prior checkout.
// Usage: node scripts/check-solver-equivalence.mjs /path/to/prior/dist/src/network.js
// Synthetic equivalence/performance evidence only; no capability claim.
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {EnergyNetwork as After} from '../dist/src/network.js';
import {mulberry32} from '../dist/src/prng.js';
if (!process.argv[2]) throw new Error('Pass the prior compiled network.js path');
const {EnergyNetwork:Before}=await import(pathToFileURL(path.resolve(process.argv[2])).href);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
{
const rows=[];
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
 assert.deepEqual(y,x,`seed ${seed}: entire result, including trace and all counters, must be bit-identical`);
 rows.push({seed,n,terminationReason:y.terminationReason,flips:y.trace.flipCount});
}
writeFileSync(path.join(root,'runs/review2/rank-view-equivalence.json'),JSON.stringify({comparison:'02507b1 sparse solver vs shuffled-rank active view; exact entire-result equality',pass:rows.length,fail:0,rows},null,2));
console.log('200/200 complete solver results bit-identical');


}
{
const n=1082,coreBase=56,coreCount=256,pool=1080;
const a=new Before({neuronCount:n,activationEnergy:1,maintenanceEnergy:.5,maxWeight:3}),b=new After({neuronCount:n,activationEnergy:1,maintenanceEnergy:.5,maxWeight:3});
for(let k=0;k<coreCount;k++){
 const core=Array.from({length:4},(_,i)=>coreBase+k*4+i);
 for(const i of core){for(const j of core)if(i<j)a.strengthen(i,j,.6);
  for(let d=0;d<4;d++)for(let z=0;z<4;z++)a.strengthen(i,d*8+((k>>d)&1)*4+z,.4);
  for(let z=0;z<4;z++)a.strengthen(i,32+(k%6)*4+z,.6);
  a.strengthen(i,pool,.3);a.strengthenDirectedInhibitory(pool,i,4,4);
 }
 for(let p=0;p<k;p++)for(const i of core)for(let z=0;z<4;z++)a.strengthenInhibitory(i,coreBase+p*4+z,3);
}
for(const key of ['weights','inhibitory','directed','directedInhibitory'])b[key].set(a[key]);
const rows=[];
for(let seed=1;seed<=5;seed++){
 const input=Array.from({length:4},(_,d)=>Array.from({length:4},(_,z)=>d*8+((seed>>d)&1)*4+z)).flat();
 const opts={seed,extraCandidates:Array.from({length:n-32},(_,i)=>i+32),levels:12,sweepsPerLevel:20,quenchCandidatesOnly:true,fallbackQuietOnly:true,quenchMaxFlips:8*n};
 let t=performance.now();const old=a.settleAnnealed(input,[],opts);const oldMs=performance.now()-t;
 t=performance.now();const current=b.settleAnnealed(input,[],opts);const newMs=performance.now()-t;
 assert.deepEqual(current,old);rows.push({seed,oldMs,newMs,proposals:old.proposals,converged:old.converged,flips:old.trace.flipCount});
}
console.log(JSON.stringify(rows));writeFileSync(path.join(root,'runs/review2/rank-large-equivalence.json'),JSON.stringify({note:'Synthetic performance/equivalence check only, not a learning capability demonstration. Old/new complete results are bit-identical; simultaneous process load affects timing.',n,coreCount,pass:5,rows},null,2));

}
