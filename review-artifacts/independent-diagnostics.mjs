import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { PopChannelMap } from '../dist/src/pop/popmap.js';
import { ExperimentPlanner } from '../dist/src/pop/explore/planner.js';
import { Explorer } from '../dist/src/pop/explore/explorer.js';
import { ContinuousExplorer } from '../dist/src/pop/explore/explorer-continuous.js';
import { LAB_CONDITION_SPECS,LAB_OUTCOME_SPECS,LabBench,labProbes,labTruth,labKey } from '../dist/src/topics/lab-world.js';
import { LAB_CONT_CONDITION_DIMS,LAB_CONT_OUTCOME_DIMS,LAB_CONT_GRID,LabContBench,labContTruth,labContProbes,labContKey,lcScore } from '../dist/src/topics/lab-continuous-world.js';
import { mulberry32 } from '../dist/src/prng.js';

const mode=process.argv[2]??'discrete', seed=Number(process.argv[3]??4);
const scope=process.argv[4]??'all';
const prefix=new URL(`./instrumented-independent-${mode}-${seed}`,import.meta.url);
const save=(suffix,x)=>fs.writeFileSync(new URL(prefix.href+suffix),JSON.stringify(x,null,2));
let ex,planner,bench,dims,key,truth;
if(mode==='discrete'){
  const cm=new PopChannelMap(LAB_CONDITION_SPECS,4),om=new PopChannelMap(LAB_OUTCOME_SPECS,4);
  planner=new ExperimentPlanner(LAB_CONDITION_SPECS);bench=new LabBench();
  ex=new Explorer(cm,om,planner,cm.specs,bench,{lit:1,brightness:3},{},seed);
  dims=LAB_CONDITION_SPECS.map(s=>({name:s.name,min:0,max:s.bins-1}));key=labKey;truth=labTruth;
}else{
  const specs=LAB_CONT_CONDITION_DIMS.map(d=>({name:d.name,bins:LAB_CONT_GRID[d.name].length,values:LAB_CONT_GRID[d.name]}));
  planner=new ExperimentPlanner(specs);bench=new LabContBench();
  ex=new ContinuousExplorer(LAB_CONT_CONDITION_DIMS,LAB_CONT_OUTCOME_DIMS,planner,specs,bench,{lit:1,brightness:3},{},seed);
  dims=LAB_CONT_CONDITION_DIMS;key=labContKey;truth=labContTruth;
}
const dynamic={calls:0,energyTraceMismatch:0,nonFixedReturns:0,maxTraceError:0,maxQuenchFlips:0};
const raw=ex.mem.net.settleAnnealed.bind(ex.mem.net);
ex.mem.net.settleAnnealed=(input,wells,opts)=>{
  const clamp=new Set(input);const r=raw([...clamp],wells,opts);dynamic.calls++;
  const err=Math.abs(r.trace.energies.at(-1)-r.energy);
  if(err>1e-7)dynamic.energyTraceMismatch++;
  dynamic.maxTraceError=Math.max(dynamic.maxTraceError,err);
  dynamic.maxQuenchFlips=Math.max(dynamic.maxQuenchFlips,r.trace.flipCount);
  const active=new Set(r.activeNeurons),s=Uint8Array.from({length:ex.mem.net.neuronCount},(_,i)=>active.has(i)?1:0);
  if(r.candidateSet.some(i=>!clamp.has(i)&&(s[i]?-1:1)*(ex.mem.net.threshold-ex.mem.net.symmetricField(i,s)) < -1e-4))dynamic.nonFixedReturns++;
  return r;
};
const t0=performance.now();let steps=0;
while(ex.step()){
  if(++steps%20===0)console.log(JSON.stringify({stage:'train',experiments:planner.experimentCount,rules:ex.mem.ruleCount,seconds:(performance.now()-t0)/1000}));
}
const training={mode,seed,experiments:planner.experimentCount,cost:bench.experimentsUsed,rules:ex.mem.ruleCount,influential:ex.influentialDims,seconds:(performance.now()-t0)/1000,phase:ex.currentPhase??null,formation:ex.formationHistory??null,allocatedCores:ex.mem.coreCursor,registeredCores:ex.mem.rules?.length??ex.mem.cores?.length,dynamics:{...dynamic}};
save('-training.json',{...training,episodes:planner.allEpisodes,log:ex.log});console.log(JSON.stringify({stage:'training-complete',...training}));
const conducted=new Set(planner.allEpisodes.map(e=>key(e.conditions)));
const probes=mode==='discrete'?labProbes(conducted):labContProbes(conducted);
if(mode==='continuous'){
  const random=mulberry32(20260919);
  for(let i=0;i<128;i++){
    const conditions=Object.fromEntries(dims.map(d=>[d.name,d.min+random()*(d.max-d.min)]));
    probes.push({conditions,truth:truth(conditions),kind:'off-grid-uniform'});
  }
  for(const switchPos of [.35,.4,.45,.49,.499,.5,.501,.51,.55,.6,.65])for(const voltage of [.8,.95,.999,1,1.001,1.05,1.2]){
    const conditions={switchPos,voltage,resistance:1.2,temperature:1,material:1};
    probes.push({conditions,truth:truth(conditions),kind:'gate-boundary'});
  }
}
const nearest=c=>{
  let best=null,dist=Infinity;
  for(const ep of planner.allEpisodes){
    const d=dims.reduce((s,dim)=>s+Math.abs(c[dim.name]-ep.conditions[dim.name])/(dim.max-dim.min),0);
    if(d<dist){dist=d;best=ep;}
  }
  return best.outcomes;
};
const results=[];
if(scope==='off-grid-only')probes.splice(0,mode==='discrete'?288:432);
for(const [i,p] of probes.entries()){
  const r=ex.mem.predict(p.conditions,seed);const pred=mode==='discrete'?r.decoded:r.values;
  const nn=nearest(p.conditions);
  const metric=(v)=>{
    const coarse=mode==='discrete'?v.lit===p.truth.lit:(typeof v.lit==='number'?(v.lit>=.5?1:0):null)===p.truth.lit;
    const brightness=typeof v.brightness==='number'&&(mode==='discrete'?v.brightness===p.truth.brightness:Math.abs(v.brightness-p.truth.brightness)<=.5);
    return {coarse,strictFine:coarse&&brightness,officialFine:mode==='continuous'?lcScore(v,p.truth).fine:coarse&&(p.truth.lit===0||brightness)&&v.brightness!=='ambiguous'};
  };
  results.push({...p,seen:conducted.has(key(p.conditions)),pred,nn,modelScore:metric(pred),nnScore:metric(nn)});
  if((i+1)%96===0)console.log(JSON.stringify({stage:'score',done:i+1,total:probes.length}));
}
const summarize=arr=>({n:arr.length,model:Object.fromEntries(['coarse','strictFine','officialFine'].map(k=>[k,arr.filter(r=>r.modelScore[k]).length])),nearest:Object.fromEntries(['coarse','strictFine','officialFine'].map(k=>[k,arr.filter(r=>r.nnScore[k]).length]))});
const partitions={};
for(const kind of new Set(results.map(r=>r.kind)))partitions[kind]=summarize(results.filter(r=>r.kind===kind));
const grid=results.filter(r=>!['off-grid-uniform','gate-boundary'].includes(r.kind));
partitions['all-grid']=summarize(grid);partitions['all-unseen-grid']=summarize(grid.filter(r=>!r.seen));
const summary={...training,elapsedSeconds:(performance.now()-t0)/1000,dynamics:dynamic,partitions};
save('-results.json',{summary,results});console.log(JSON.stringify({stage:'complete',...summary}));
