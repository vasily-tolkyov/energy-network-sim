import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { EnergyNetwork, hebbianLearn, detectWells, ReadoutModule } from '../dist/src/index.js';
import { PopChannelMap } from '../dist/src/pop/popmap.js';
import { PopRuleMemory } from '../dist/src/pop/popmemory.js';
import { R2PopLayer } from '../dist/src/pop/r2pop.js';
import { SensoryEncoder } from '../dist/src/pop/concept/sensory.js';
import { ConceptFormation } from '../dist/src/pop/concept/formation.js';
import { EmergentMap } from '../dist/src/pop/concept/emergent-map.js';
import { FieldRuleMemory } from '../dist/src/pop/concept/field-memory.js';
import { LabContBench, lcScore } from '../dist/src/topics/lab-continuous-world.js';
import { NeuralFocusNet } from '../dist/src/pop/attention/neural-focus.js';

const network = (n=3) => new EnergyNetwork({neuronCount:n,activationEnergy:1,maintenanceEnergy:.5,maxWeight:8});
const fieldMemory = (maxRules=8) => {
  const enc=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}],40);
  const mem=new FieldRuleMemory(enc,new EmergentMap([],enc),{maxRules});
  mem.setOutcomeDimensions(['y']);
  return {enc,mem};
};
const oscillator = () => {
  const net=network();
  net.strengthen(0,1,2); net.strengthen(1,2,2);
  net.strengthenDirectedInhibitory(2,1,4,4);
  return net;
};
const diagnostic=(t,obj)=>t.diagnostic(JSON.stringify(obj));

test('A01 symmetric-only energy accounting and fixed point (positive control)',t=>{
  const n=network(5); hebbianLearn(n,[0,1,2,3],5,.4);
  const r=n.settle([0]); diagnostic(t,r);
  assert.ok(Math.abs(r.trace.energies.at(-1)-r.energy)<1e-8);
  assert.ok(r.trace.energies.every((e,i,a)=>i===0||e<a[i-1]));
  const s=Uint8Array.from({length:5},(_,i)=>r.activeNeurons.includes(i)?1:0);
  for(let i=1;i<5;i++) assert.ok((s[i]?-1:1)*(n.threshold-n.symmetricField(i,s))>=-1e-4);
});

test('A02 DI settle trace must equal actual energy',t=>{
  const n=oscillator(); const seen=[];
  const field=n.symmetricField.bind(n);
  n.symmetricField=(i,s)=>{seen.push(n.energy(s));return field(i,s);};
  const r=n.settle([0]);
  diagnostic(t,{flips:r.trace.flipCount,reported:r.trace.energies.at(-1),actual:r.energy,actualIncrease:seen.some((e,i)=>i&&e>seen[i-1]),active:r.activeNeurons});
  assert.ok(Math.abs(r.trace.energies.at(-1)-r.energy)<1e-8);
});

test('A03 bounded DI quench must report nonconvergence or return a fixed point',t=>{
  const n=oscillator();
  const r=n.settleAnnealed([0],[],{seed:7,extraCandidates:[1,2],levels:2,sweepsPerLevel:2,quenchMaxFlips:8,quenchCandidatesOnly:true});
  const s=Uint8Array.from({length:3},(_,i)=>r.activeNeurons.includes(i)?1:0);
  const deltas=[1,2].map(i=>(s[i]?-1:1)*(n.threshold-n.symmetricField(i,s)));
  diagnostic(t,{...r,deltas});
  assert.ok(r.converged===false||deltas.every(d=>d>=-1e-4),'No explicit nonconvergence flag and downhill moves remain');
});

test('A04 advertised candidate freeze survives default quench',t=>{
  const n=network(4); n.strengthen(0,1,2);n.strengthen(1,2,2);n.strengthen(2,3,2);
  const r=n.settleAnnealed([0],[{memberNeuronIds:[2,3]}],{seed:1}); diagnostic(t,r);
  assert.ok(r.activeNeurons.every(i=>r.candidateSet.includes(i)));
});

test('A05 invalid weight endpoint must be rejected without corrupting symmetry',t=>{
  const n=network(3); let rejected=false;
  try {n.strengthen(0,3,1);} catch {rejected=true;}
  diagnostic(t,{rejected,w01:n.getWeight(0,1),w10:n.getWeight(1,0)});
  assert.ok(rejected); assert.equal(n.getWeight(0,1),n.getWeight(1,0));
});

test('A06 nonfinite config and noninteger inputs must be rejected',t=>{
  const failures=[];
  for(const [label,fn] of [
    ['infinite activation',()=>new EnergyNetwork({neuronCount:2,activationEnergy:Infinity,maintenanceEnergy:0})],
    ['NaN learning rate',()=>new EnergyNetwork({neuronCount:2,activationEnergy:1,maintenanceEnergy:0,learningRate:NaN})],
    ['fractional input',()=>network().settle([.5])],
    ['NaN bin',()=>new PopChannelMap([{name:'x',bins:2}],4).population('x',NaN)]
  ]) {let thrown=false;try {fn();}catch{thrown=true;}if(!thrown)failures.push(label);}
  diagnostic(t,{acceptedInvalid:failures});assert.deepEqual(failures,[]);
});

test('A07 continuous bench rejects NaN without charging experiment',t=>{
  const b=new LabContBench(); let rejected=false;let output;
  try{output=b.conduct({switchPos:.9,voltage:NaN,resistance:1,temperature:1,material:1});}catch{rejected=true;}
  diagnostic(t,{rejected,output,cost:b.experimentsUsed});assert.ok(rejected);assert.equal(b.experimentsUsed,0);
});

test('A08 FieldRuleMemory enforces declared maxRules before pool overwrite',t=>{
  const {mem}=fieldMemory(1); mem.learnFromObservation({x:.1},{y:.1});
  let rejected=false;try{mem.learnFromObservation({x:.9},{y:.9});}catch{rejected=true;}
  diagnostic(t,{rejected,ruleCount:mem.ruleCount,neuronCount:mem.net.neuronCount,cores:mem.allCoreNeurons()});
  assert.ok(rejected,'second rule must not occupy pool neurons / out-of-range indices');
});

test('A09 influence-first field rule remains visible after later real observation',t=>{
  const {mem}=fieldMemory();
  mem.bindInfluence({x:.2},{x:.2},2);
  mem.learnFromObservation({x:.2},{y:.8},4);
  const r=mem.predict({x:.2},1);
  diagnostic(t,{ruleCount:mem.ruleCount,pred:r.values,coverage:mem.coreFieldCoverage({x:.2})});
  assert.equal(mem.ruleCount,1);assert.notEqual(r.values.y,null);
});

test('A10 Pop memory all teaching paths preserve two-stage binding (positive)',t=>{
  const cm=new PopChannelMap([{name:'x',bins:2}],4);
  const om=new PopChannelMap([{name:'a',bins:2},{name:'b',bins:2}],4);
  const mem=new PopRuleMemory(cm,om,{maxRules:4});
  mem.teachExperience({conditions:{x:0},outcomes:{a:0,b:1}},4);
  mem.bindInfluence({conditions:{x:0},outcomes:{a:0,b:1}},{x:.3},2);
  mem.learnFromQuery({x:1},{a:1,b:0},4);
  mem.learnFromObservation({x:1},{a:0,b:1},4);
  let forbidden=0;
  for(let i=0;i<cm.neuronCount;i++)for(let j=0;j<om.neuronCount;j++)if(mem.net.getWeight(i,cm.neuronCount+j)!==0)forbidden++;
  for(const a of om.population('a',0).concat(om.population('a',1)))for(const b of om.population('b',0).concat(om.population('b',1)))if(mem.net.getWeight(cm.neuronCount+a,cm.neuronCount+b)!==0)forbidden++;
  diagnostic(t,{forbidden,ruleCount:mem.ruleCount});assert.equal(forbidden,0);
});

test('A11 field teaching preserves star outputs (positive)',t=>{
  const {mem,enc}=fieldMemory();
  mem.teachExperiment({x:.2},{y:.3});mem.learnFromObservation({x:.4},{y:.7});
  let forbidden=0;
  for(let i=0;i<40;i++)for(let j=40;j<80;j++)if(mem.net.getWeight(i,j)!==0)forbidden++;
  for(let i=40;i<80;i++)for(let j=i+1;j<80;j++)if(mem.net.getWeight(i,j)!==0)forbidden++;
  diagnostic(t,{forbidden,neuronCount:enc.neuronCount});assert.equal(forbidden,0);
});

test('A12 concept ablation: swap inhibition does not affect extracted concepts',t=>{
  const enc=new SensoryEncoder([{name:'x',min:0,max:1}],40);
  const f=new ConceptFormation(enc);
  for(const x of [.4,.5,.6])f.presentExperiment({x},4);
  const before=f.extractConcepts();
  f.presentSwap('x',.4,.6,3);
  const after=f.extractConcepts();
  diagnostic(t,{before:before.map(c=>[c.centerValue,c.memberNeuronIds.length]),after:after.map(c=>[c.centerValue,c.memberNeuronIds.length])});
  assert.deepEqual(after,before,'ablation result: extraction only reads W');
});

test('A13 fine continuous score must reject wildly incorrect brightness even when off',t=>{
  const s=lcScore({lit:0,brightness:999},{lit:0,brightness:0});diagnostic(t,s);
  assert.equal(s.coarse,true);assert.equal(s.fine,false);
});

test('A14 R2 should not infer outcome change for a one-factor null intervention',t=>{
  const cm=new PopChannelMap([{name:'x',bins:2}],4);
  const om=new PopChannelMap([{name:'y',bins:2}],4);
  const r=new R2PopLayer(cm,om).analyzePair({e0:{conditions:{x:0},outcomes:{y:0}},e1:{conditions:{x:1},outcomes:{y:0}}});
  diagnostic(t,r);assert.equal(r.outcomeChanged,false);assert.deepEqual(r.influentialChannels,[]);
});

test('A15 WTA focus should activate exactly one marker under actual ignorance drive',t=>{
  const f=new NeuralFocusNet(['a','b','c'],{inertia:0}); f.beginFrame();
  for(const id of ['a','b','c'])f.setMismatch(id,2*4);
  const chosen=f.select(1); const markers=f.net.activeNeurons().filter(i=>i%2===1);
  diagnostic(t,{chosen,markers});assert.equal(markers.length,1);
});

test('A16 field outcome metadata includes newly learned observed fields',t=>{
  const {mem,enc}=fieldMemory();mem.learnFromObservation({x:.2},{y:.1});mem.learnFromObservation({x:.2},{y:.9},10);
  diagnostic(t,{metadata:mem.ruleOutcomeFields(0),latest:enc.encode({y:.9})});
  assert.ok(enc.encode({y:.9}).every(id=>mem.ruleOutcomeFields(0).includes(id)), 'metadata must represent the newly bound outcome, not only the first observation');
});

test('A17 claimed G=0 ablation must remove influence veto as well as excitation',t=>{
  const cm=new PopChannelMap([{name:'x',bins:2}],4),om=new PopChannelMap([{name:'y',bins:2}],4);
  const mem=new PopRuleMemory(cm,om,{maxRules:2});
  const a={conditions:{x:0},outcomes:{y:0}},b={conditions:{x:1},outcomes:{y:1}};
  mem.teachExperience(a,4);mem.teachExperience(b,4);mem.bindInfluence(a,{x:0},4);
  const veto=mem.net.getInhibitoryWeight(cm.population('x',1)[0],cm.neuronCount+om.neuronCount);
  diagnostic(t,{gain:0,veto});assert.equal(veto,0);
});

test('A18 concept centers alone cannot identify gate threshold (control)',t=>{
  const make=threshold=>{
    const enc=new SensoryEncoder([{name:'x',min:0,max:1},{name:'lit',min:0,max:1}],40);
    const f=new ConceptFormation(enc);
    for(const x of [.1,.5,.9])f.presentExperiment({x,lit:x>=threshold?1:0},4);
    return f.extractConcepts().filter(c=>c.dimension==='x');
  };
  const a=make(.5),b=make(.73);
  diagnostic(t,{thresholds:[.5,.73],centersA:a.map(c=>c.centerValue),centersB:b.map(c=>c.centerValue)});
  assert.deepEqual(a,b,'same centers despite different physical thresholds');
});

test('A19 README usage must detect and label its Hebbian clique memories',t=>{
  const net=new EnergyNetwork({neuronCount:24,activationEnergy:2,maintenanceEnergy:.5});
  hebbianLearn(net,[0,1,2,3],10);hebbianLearn(net,[12,13,14,15],10);
  const wells=detectWells(net);diagnostic(t,{wellCount:wells.length});
  const readout=new ReadoutModule(wells,net.config.readoutThreshold);
  readout.labelWell(0,'A');readout.labelWell(1,'B');
  assert.equal(readout.identify(net.settle([0,1]).activeNeurons)[0]?.label,'A');
});

test('A20 audit of exact C-minimum claim against exhaustive oracle without DI',t=>{
  const fixture=JSON.parse(fs.readFileSync(new URL('./annealing-global-counterexample.json',import.meta.url),'utf8'));
  const net=new EnergyNetwork({neuronCount:fixture.n,activationEnergy:1,maintenanceEnergy:.5,maxWeight:12});
  for(const [i,j,w,g] of fixture.edges){if(w)net.strengthen(i,j,w);if(g)net.strengthenInhibitory(i,j,g);}
  let best=Infinity;
  for(let mask=1;mask<1<<fixture.n;mask+=2){
    const s=Uint8Array.from({length:fixture.n},(_,i)=>(mask>>i)&1);
    best=Math.min(best,net.energy(s));
  }
  const result=net.settleAnnealed([0],[],{seed:1,extraCandidates:Array.from({length:fixture.n-1},(_,i)=>i+1),quenchCandidatesOnly:true});
  diagnostic(t,{oracleStates:2**(fixture.n-1),best,returned:result.energy,proposals:result.proposals,active:result.activeNeurons});
  assert.equal(result.energy,best,'finite annealing does not guarantee the exact minimum within C');
});
