// Independent audit runner, unchanged production source; full grid, all three published seeds.
// Records original scoring, strict joint scoring, and actual quench-state diagnostics separately.
import {writeFileSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
import {Explorer} from '../repo/dist/src/pop/explore/explorer.js';
import {ExperimentPlanner} from '../repo/dist/src/pop/explore/planner.js';
import {PopChannelMap} from '../repo/dist/src/pop/popmap.js';
import {LAB_CONDITION_SPECS,LAB_OUTCOME_SPECS,LabBench,labProbes,labKey} from '../repo/dist/src/topics/lab-world.js';
const seeds=(process.env.AUDIT_SEEDS??'1,2,3').split(',').map(Number);
for(const seed of seeds){
 const begin=performance.now(),cm=new PopChannelMap(LAB_CONDITION_SPECS),om=new PopChannelMap(LAB_OUTCOME_SPECS);
 const planner=new ExperimentPlanner(LAB_CONDITION_SPECS),bench=new LabBench();
 const ex=new Explorer(cm,om,planner,cm.specs,bench,{lit:1,brightness:3},{},seed);
 // Transparent wrapper: original method is called with identical arguments and its result
 // returned unchanged. No state/weights are altered by the diagnostic calculations.
 let lastDiagnostic=null;
 const original=ex.mem.net.settleAnnealed.bind(ex.mem.net);
 ex.mem.net.settleAnnealed=(...args)=>{
  const r=original(...args),state=new Uint8Array(ex.mem.net.neuronCount),clamped=new Set(args[0]);
  for(const i of r.activeNeurons)state[i]=1;
  let improving=0,worst=0;
  for(const i of r.candidateSet){
   if(clamped.has(i))continue;
   const d=(state[i]===0?1:-1)*(ex.mem.net.threshold-ex.mem.net.symmetricField(i,state));
   if(d< -1e-4){improving++;worst=Math.min(worst,d);}
  }
  lastDiagnostic={improving,worst,energy:r.energy,traceEnd:r.trace.energies.at(-1),flipCount:r.trace.flipCount};
  return r;
 };
 while(ex.step())if(ex.log.length%10===0)console.log(JSON.stringify({seed,stage:'learn',experiments:ex.log.length}));
 const conducted=new Set(ex.log.map(s=>labKey(s.conditions))),probes=labProbes(conducted),rows=[];
 const groups={};
 for(let i=0;i<probes.length;i++){
  const p=probes[i],d=ex.mem.predict(p.conditions,seed).decoded;
  const coarse=d.lit===p.truth.lit;
  const ambiguous=d.lit==='ambiguous'||d.brightness==='ambiguous';
  const fine=!ambiguous&&coarse&&(p.truth.lit===0||d.brightness===p.truth.brightness);
  const strict=coarse&&d.brightness===p.truth.brightness;
  const g=groups[p.kind]??={total:0,fine:0,coarse:0,strictJoint:0,litOn:0,unstable:0,energyTraceMismatch:0};
  g.total++;g.fine+=Number(fine);g.coarse+=Number(coarse);g.strictJoint+=Number(strict);g.litOn+=Number(p.truth.lit===1);
  g.unstable+=Number(lastDiagnostic.improving>0);g.energyTraceMismatch+=Number(Math.abs(lastDiagnostic.energy-lastDiagnostic.traceEnd)>1e-6);
  rows.push({...p,predicted:d,coarse,fine,strictJoint:strict,diagnostic:lastDiagnostic});
  if((i+1)%48===0)console.log(JSON.stringify({seed,stage:'score',completed:i+1,total:probes.length}));
 }
 const result={seed,experiments:ex.log.length,rules:ex.mem.ruleCount,factors:ex.influentialDims,groups,elapsedMs:performance.now()-begin,log:ex.log,probes:rows};
 writeFileSync(new URL(`../results/phaseA-seed${seed}.json`,import.meta.url),JSON.stringify(result,null,2));
 console.log(JSON.stringify({seed,stage:'complete',experiments:result.experiments,rules:result.rules,factors:result.factors,groups,elapsedMs:result.elapsedMs}));
}
