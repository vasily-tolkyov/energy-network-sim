// @ts-nocheck -- verbatim independent JavaScript audit cases, imports relocated.
import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {EnergyNetwork,hebbianLearn,detectWells,ReadoutModule} from '../src/index.js';
import {PopChannelMap} from '../src/pop/popmap.js';
import {PopRuleMemory} from '../src/pop/popmemory.js';
import {SensoryEncoder} from '../src/pop/concept/sensory.js';
import {EmergentMap} from '../src/pop/concept/emergent-map.js';
import {FieldRuleMemory} from '../src/pop/concept/field-memory.js';
import {R2PopLayer} from '../src/pop/r2pop.js';
import {ConceptFormation} from '../src/pop/concept/formation.js';
import {EmergentChannelAdapter} from '../src/pop/concept/emergent-channel-adapter.js';
import {mulberry32} from '../src/prng.js';
const diag=(t,x)=>t.diagnostic(JSON.stringify(x));
const net=(n=3)=>new EnergyNetwork({neuronCount:n,activationEnergy:1,maintenanceEnergy:.5,maxWeight:8});
const osc=()=>{const n=net();n.strengthen(0,1,2);n.strengthen(1,2,2);n.strengthenDirectedInhibitory(2,1,4,4);return n;};
const pop=(config={})=>new PopRuleMemory(new PopChannelMap([{name:'x',bins:2}],4),new PopChannelMap([{name:'y',bins:2}],4),{maxRules:8,...config});
const field=(config={})=>{const e=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}],40);const m=new FieldRuleMemory(e,new EmergentMap([],e),{maxRules:8,...config});m.setOutcomeDimensions(['y']);return m;};

test('R01 DI accounting and reported residuals across 24 fresh random networks',t=>{
 const rand=mulberry32(20260921);let limited=0;
 for(let k=0;k<24;k++){
  const n=net(6);for(let i=0;i<6;i++)for(let j=i+1;j<6;j++){n.strengthen(i,j,Math.floor(rand()*5)/2);n.strengthenInhibitory(i,j,Math.floor(rand()*3)/2);if(rand()<.3)n.strengthenDirectedInhibitory(i,j,2);}
  const r=n.settleAnnealed([0],[],{seed:k+20,extraCandidates:[1,2,3,4,5],levels:3,sweepsPerLevel:3,quenchMaxFlips:30,quenchCandidatesOnly:true});
  assert.ok(Math.abs(r.energy-r.trace.energies.at(-1))<1e-7);
  const s=Uint8Array.from({length:6},(_,i)=>r.activeNeurons.includes(i)?1:0);
  const residual=[1,2,3,4,5].filter(i=>(s[i]?-1:1)*(n.threshold-n.symmetricField(i,s)) < -1e-4).length;
  assert.equal(residual,r.residualFlips);if(r.converged)assert.equal(residual,0);else limited++;
 }diag(t,{networks:24,limited});
});
test('R02 explicit candidate freeze blocks a connected outside chain',t=>{
 const n=net(4);for(let i=0;i<3;i++)n.strengthen(i,i+1,2);
 const r=n.settleAnnealed([0],[],{seed:1,quenchCandidatesOnly:true});diag(t,r);assert.deepEqual(r.activeNeurons,[0]);
});
test('R03 current documented peak-edge example is executable',t=>{
 const n=new EnergyNetwork({neuronCount:24,activationEnergy:2,maintenanceEnergy:.5});
 for(const ids of [[0,1,2,3],[12,13,14,15]]){hebbianLearn(n,ids,8);hebbianLearn(n,ids.slice(0,2),2);}
 const wells=detectWells(n),r=new ReadoutModule(wells,n.config.readoutThreshold);diag(t,{wells:wells.length});
 assert.equal(wells.length,2);r.labelWell(wells[0].wellId,'A');assert.equal(r.identify(n.settle([0,1]).activeNeurons)[0].label,'A');
});
test('R04 quiet-only fallback must not return an engaged driver when quiet state was visited',t=>{
 const n=osc(),r=n.settleAnnealed([0],[],{seed:1,extraCandidates:[1,2],levels:1,sweepsPerLevel:1,initialTemperature:1e-9,quenchMaxFlips:8,quenchCandidatesOnly:true,fallbackQuietOnly:true});
 const s=Uint8Array.from({length:3},(_,i)=>r.activeNeurons.includes(i)?1:0),engaged=s[2]===1||n.localField(2,s)-n.inhibitoryField(2,s)>n.threshold;
 diag(t,{...r,engaged,visitedQuietState:[0],quietEnergy:1.5});assert.equal(engaged,false);
});
test('R05 self-written guess cannot veto a previously confirmed observation',t=>{
 const m=pop();m.learnFromObservation({x:0},{y:0},4);assert.equal(m.predict({x:0},1).decoded.y,0);
 m.learnFromQuery({x:0},{y:1},2);const r=m.predict({x:0},1);diag(t,r);assert.equal(r.decoded.y,0);
});
test('R06 repeated same-world truth must recover after an erroneous self-written guess',t=>{
 const m=pop();m.learnFromObservation({x:0},{y:0},2);m.learnFromQuery({x:0},{y:1},2);m.learnFromObservation({x:0},{y:0},2);
 const r=m.predict({x:0},1);diag(t,r);assert.equal(r.decoded.y,0);
});
test('R07 field observations within identical receptive-field code cannot erase that code',t=>{
 const m=field();assert.deepEqual(m.encoder.encode({y:.5}),m.encoder.encode({y:.51}));
 m.learnFromObservation({x:.2},{y:.5});m.learnFromObservation({x:.2},{y:.51});const r=m.predict({x:.2},1);diag(t,r);assert.notEqual(r.values.y,null);assert.ok(Math.abs(r.values.y-.51)<.04);
});
test('R08 nearby but distinct inputs of a stationary function must retain a readable shared code',t=>{
 const m=field();assert.deepEqual(m.encoder.encode({x:.2}),m.encoder.encode({x:.21}));
 m.learnFromObservation({x:.2},{y:.5});m.learnFromObservation({x:.21},{y:.51});
 const a=m.predict({x:.2},1),b=m.predict({x:.21},1);diag(t,{rules:m.ruleCount,a:a.values,b:b.values});assert.notEqual(a.values.y,null);assert.notEqual(b.values.y,null);
});
test('R14 R2 unchanged outcomes across variable dimensions and population sizes',t=>{
 let checked=0;for(const size of [1,2,4,8])for(const dims of [1,2,3]){
  const specs=Array.from({length:dims},(_,i)=>({name:`x${i}`,bins:2}));const cm=new PopChannelMap(specs,size),om=new PopChannelMap([{name:'y',bins:2}],size);
  const a=Object.fromEntries(specs.map(s=>[s.name,0])),b={...a,x0:1};const r=new R2PopLayer(cm,om).analyzePair({e0:{conditions:a,outcomes:{y:0}},e1:{conditions:b,outcomes:{y:0}}});
  assert.equal(r.outcomeChanged,false);assert.equal(r.influentialChannels.length,0);checked++;
 }diag(t,{checked});
});
test('R18 actual observation can correct a prior wrong guess in the supported one-way order',t=>{
 const m=pop();m.learnFromQuery({x:0},{y:1},2);m.learnFromObservation({x:0},{y:0},2);const p=m.predict({x:0},1);diag(t,p);assert.equal(p.decoded.y,0);
});
test('R19 field R2 outcome change detection must be invariant to pair reversal',t=>{
 const enc=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}],40),f=new ConceptFormation(enc);
 f.presentExperiment({x:.1,y:0},4);f.presentExperiment({x:.9,y:.05},4);
 const em=new EmergentMap(f.extractConcepts(),enc),cm=new EmergentChannelAdapter(em,['x'],0),om=new EmergentChannelAdapter(em,['y'],40,'fields'),r2=new R2PopLayer(cm,om);
 const a={conditions:{x:em.resolve('x',.1)},outcomes:{y:0}},b={conditions:{x:em.resolve('x',.9)},outcomes:{y:.05}};
 const forward=r2.analyzePair({e0:a,e1:b}),reverse=r2.analyzePair({e0:b,e1:a});diag(t,{forward,reverse});
 assert.equal(forward.outcomeChanged,reverse.outcomeChanged);assert.ok(forward.outcomeChanged);assert.ok(Math.abs(forward.outcomeDelta.y+reverse.outcomeDelta.y)<1e-9);
});
test('R20 influence allocation and reverse observation order preserve registration identity',t=>{
 const m=field();m.bindInfluence({x:.2},{x:.2},2);m.bindInfluence({x:.8},{x:.2},2);
 m.learnFromObservation({x:.8},{y:.9},4);m.learnFromObservation({x:.2},{y:.1},4);
 const a=m.predict({x:.2},1),b=m.predict({x:.8},1);diag(t,{rules:m.ruleCount,a:a.values,b:b.values,winning:[a.winningCores,b.winningCores]});
 assert.equal(m.ruleCount,2);assert.ok(Math.abs(a.values.y-.1)<.05);assert.ok(Math.abs(b.values.y-.9)<.05);
});
