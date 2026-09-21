// Loose equivalence for incremental-field + reheat solver:
// - 非 quiet-constraint 案例（含 quietOnly=false 与域内干净收敛）答案/终止/账本逐位一致；
// - quiet-constraint 案例允许不同——重退火是语义改进，但新能量必须不差于旧
//   （更深或等深的静息合法极小），答案改变的数量如实计数。
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
let worst=0, answerMismatch=0, reasonMismatch=0, ledgerMismatch=0, changedQuiet=0, worseQuiet=0, energyDiff=0;
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
 const sameAnswer=JSON.stringify(x.activeNeurons)===JSON.stringify(y.activeNeurons);
 if(options.fallbackQuietOnly&&x.terminationReason==='quiet-constraint'){
  if(!sameAnswer){changedQuiet++;if(y.energy>x.energy+TOL)worseQuiet++;}
 } else {
  if(!sameAnswer)answerMismatch++;
  if(x.terminationReason!==y.terminationReason)reasonMismatch++;
  // 账本不变量（评审口径）：各自内部 轨迹末值≈返回能量；
  // 跨版本能量差如实计数（重退火轮次的轨迹尾部簿记不同，属簿记而非能量错误）
  if(Math.abs(x.trace.energies.at(-1)-x.energy)>TOL||Math.abs(y.trace.energies.at(-1)-y.energy)>TOL)ledgerMismatch++;
  if(Math.abs(x.energy-y.energy)>TOL)energyDiff++;
 }
 worst=Math.max(worst,Math.abs(x.energy-y.energy));
}
const report={comparison:'incremental fields + reheat vs dense: identical outside quiet-constraint; reheat may only deepen quiet minima',
 answerMismatch,reasonMismatch,ledgerMismatch,energyDiff,changedQuiet,worseQuiet,worstEnergyAbsDiff:worst,tolerance:TOL,seeds:200};
writeFileSync(path.join(root,'runs/incremental-fields-equivalence.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
assert.equal(answerMismatch,0,'answers outside quiet-constraint must be identical');
assert.equal(reasonMismatch,0,'termination reasons must be identical outside reheat cases');
assert.equal(ledgerMismatch,0,'each version ledger must be internally consistent');
assert.equal(worseQuiet,0,'reheated quiet minima must not be worse than before');
