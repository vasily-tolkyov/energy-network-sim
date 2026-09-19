import {writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {PopChannelMap} from '../repo/dist/src/pop/popmap.js';
import {ExperimentPlanner} from '../repo/dist/src/pop/explore/planner.js';
import {Explorer} from '../repo/dist/src/pop/explore/explorer.js';
import {LAB_CONDITION_SPECS as CS,LAB_OUTCOME_SPECS as OS,LabBench,labTruth} from '../repo/dist/src/topics/lab-world.js';
// Independently enumerated grid and literal scalar oracle; no repository score/bucket function used.
function grid(specs,i=0,q={}) {if(i===specs.length)return [{...q}];let out=[];let s=specs[i];for(let v=0;v<s.bins;v++)out.push(...grid(specs,i+1,{...q,[s.name]:v}));return out;}
function oracle(q){let lit=Number(q.switch===1 && q.voltage>=1);return {lit,brightness:lit?Math.max(0,Math.min(3,Math.floor(1+1.25*q.voltage-.75*q.resistance+.5*q.temperature))):0};}
const probes=grid(CS);assert.equal(probes.length,288);for(const p of probes)assert.deepEqual(oracle(p),labTruth(p));
const key=q=>CS.map(s=>q[s.name]).join(',');
const seeds=(process.env.SEEDS??'1,4,7').split(',').map(Number);
for(const seed of seeds){
 const started=performance.now(),cm=new PopChannelMap(CS,4),om=new PopChannelMap(OS,4),planner=new ExperimentPlanner(CS),bench=new LabBench();
 const ex=new Explorer(cm,om,planner,cm.specs,bench,{lit:1,brightness:3},{},seed);
 let stage='training',last=null;const audits=[];
 const original=ex.mem.net.settleAnnealed.bind(ex.mem.net);
 ex.mem.net.settleAnnealed=(...args)=>{const r=original(...args),tail=r.trace.energies.at(-1);last={stage,energy:r.energy,traceEnd:tail,energyError:Math.abs(tail-r.energy),converged:r.converged,terminationReason:r.terminationReason,residualFlips:r.residualFlips,flipCount:r.trace.flipCount,proposals:r.proposals};audits.push(last);return r;};
 while(ex.step()) {if(ex.log.length%20===0)console.log(`seed=${seed} training=${ex.log.length}`);}
 const trainedMs=performance.now()-started;console.log(`seed=${seed} training done experiments=${ex.log.length} reason=${ex.terminationReason} ms=${trainedMs.toFixed(0)}`);
 stage='evaluation';const conducted=new Set(planner.allEpisodes.map(ep=>key(ep.conditions))),rows=[];
 for(const [i,q] of probes.entries()){
  const p=ex.mem.predict(q,seed),truth=oracle(q),seen=conducted.has(key(q)),refused=Object.values(p.decoded).some(x=>typeof x!=='number');
  let nn=null,best=Infinity;for(const ep of planner.allEpisodes){const d=CS.reduce((s,c)=>s+Number(ep.conditions[c.name]!==q[c.name]),0);if(d<best){best=d;nn=ep.outcomes;}}
  rows.push({conditions:q,truth,decoded:p.decoded,seen,gate:truth.lit?'on':'off',strict:p.decoded.lit===truth.lit&&p.decoded.brightness===truth.brightness,coarse:p.decoded.lit===truth.lit,refused,nearestNeighbor:nn,nnStrict:nn.lit===truth.lit&&nn.brightness===truth.brightness,nnCoarse:nn.lit===truth.lit,audit:{...last}});
  if((i+1)%96===0)console.log(`seed=${seed} scored=${i+1}/288`);
 }
 function score(a){return {n:a.length,strict:a.filter(x=>x.strict).length,coarse:a.filter(x=>x.coarse).length,refused:a.filter(x=>x.refused).length,nonconverged:a.filter(x=>!x.audit.converged).length,residual:a.filter(x=>x.audit.residualFlips>0).length,ledgerMismatch:a.filter(x=>x.audit.energyError>1e-7).length,nnStrict:a.filter(x=>x.nnStrict).length,nnCoarse:a.filter(x=>x.nnCoarse).length,constantOff:a.filter(x=>x.truth.lit===0).length,constantOn:a.filter(x=>x.truth.lit===1).length};}
 const summary={seed,experiments:ex.log.length,rules:ex.mem.ruleCount,factors:ex.influentialDims,terminationReason:ex.terminationReason,trainedMs,totalMs:performance.now()-started,all:score(rows),seen:score(rows.filter(x=>x.seen)),unseen:score(rows.filter(x=>!x.seen)),unseenOff:score(rows.filter(x=>!x.seen&&x.truth.lit===0)),unseenOn:score(rows.filter(x=>!x.seen&&x.truth.lit===1)),trainingAudit:{calls:audits.filter(x=>x.stage==='training').length,nonconverged:audits.filter(x=>x.stage==='training'&&!x.converged).length,ledgerMismatch:audits.filter(x=>x.stage==='training'&&x.energyError>1e-7).length}};
 writeFileSync(new URL(`../results/phase-a-seed${seed}.json`,import.meta.url),JSON.stringify({summary,training:ex.log,rows,audits},null,2));console.log(JSON.stringify(summary));
}
