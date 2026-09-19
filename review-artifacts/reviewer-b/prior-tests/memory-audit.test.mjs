import test from 'node:test';
import assert from 'node:assert/strict';
import { EnergyNetwork, hebbianLearn, detectWells, ReadoutModule } from '../repo/dist/src/index.js';
import { PopChannelMap } from '../repo/dist/src/pop/popmap.js';
import { PopRuleMemory } from '../repo/dist/src/pop/popmemory.js';
function make(config={}){return new PopRuleMemory(new PopChannelMap([{name:'x',bins:2}]),new PopChannelMap([{name:'y',bins:2}]),{maxRules:8,...config});}
function answers(m,q={x:0},count=30){const a=[];for(let seed=1;seed<=count;seed++)a.push(m.predict(q,seed).decoded.y);return a;}
function hist(xs){const out={};for(const x of xs)out[String(x)]=(out[String(x)]??0)+1;return out;}

test('B01 learned two-state identity: taught examples remain retrievable (30 seeds each)',()=>{
  const m=make();for(let x=0;x<2;x++)m.teachExperience({id:`e${x}`,conditions:{x},outcomes:{y:x}},10);
  m.teachExclusion('outcome','y',0,1,1.5);
  for(let x=0;x<2;x++)m.bindInfluence({id:`e${x}`,conditions:{x},outcomes:{y:x}},{x:0.5},2);
  for(let x=0;x<2;x++){const got=answers(m,{x});console.log('B01 evidence',JSON.stringify({x,answers:hist(got)}));assert.ok(got.every(y=>y===x));}
});

test('B02 all four learning routes preserve condition/outcome and cross-output isolation',()=>{
  const c=new PopChannelMap([{name:'x',bins:2},{name:'d',bins:2}]);
  const o=new PopChannelMap([{name:'y',bins:2},{name:'z',bins:2}]);
  const m=new PopRuleMemory(c,o,{maxRules:8});
  m.teachExperience({id:'a',conditions:{x:0,d:0},outcomes:{y:0,z:1}},8);
  m.bindInfluence({id:'a',conditions:{x:0,d:0},outcomes:{y:0,z:1}},{x:0.5},2);
  m.learnFromQuery({x:1,d:0},{y:1,z:0},8);
  m.learnFromObservation({x:1,d:1},{y:0,z:1},8);
  for(let i=0;i<c.neuronCount;i++)for(let j=0;j<o.neuronCount;j++)assert.equal(m.net.getWeight(i,c.neuronCount+j),0);
  for(let b=0;b<2;b++)for(let d=0;d<2;d++)for(const i of o.population('y',b))for(const j of o.population('z',d))assert.equal(m.net.getWeight(c.neuronCount+i,c.neuronCount+j),0);
});

test('B03 no numeric prediction must not allocate an empty competing rule',()=>{
  const m=make();m.learnFromQuery({x:0},{y:null},1);
  console.log('B03 evidence',JSON.stringify({ruleCount:m.ruleCount,prediction:m.predict({x:0},1).decoded}));
  assert.equal(m.ruleCount,0,'unknown prediction consumes a core without outcome evidence');
});

test('B04 observation stronger than self-prediction should actually correct it without hidden pretraining',()=>{
  const m=make();m.learnFromQuery({x:0},{y:0},10);m.learnFromObservation({x:0},{y:1},20);
  const got=answers(m);console.log('B04 evidence',JSON.stringify({answers:hist(got)}));
  assert.ok(got.every(x=>x===1),'higher cap alone does not ensure correction when exclusion was never learned');
});

test('B05 the same correction with an explicit learned exclusion (positive control)',()=>{
  const m=make();m.teachExclusion('outcome','y',0,1,1.5);
  m.learnFromQuery({x:0},{y:0},10);m.learnFromObservation({x:0},{y:1},20);
  const got=answers(m);console.log('B05 evidence',JSON.stringify({answers:hist(got)}));assert.ok(got.every(x=>x===1));
});

test('B06 observed rule reversal: overwhelming new observations must replace obsolete outcome',()=>{
  const m=make();m.teachExclusion('outcome','y',0,1,1.5);
  m.learnFromObservation({x:0},{y:0},10);m.learnFromObservation({x:0},{y:1},1000);
  const got=answers(m);console.log('B06 evidence',JSON.stringify({oldObservationRepeats:10,newObservationRepeats:1000,answers:hist(got)}));
  assert.ok(got.every(x=>x===1),'monotone saturated edges cannot distinguish old from new observed evidence');
});

test('B07 gammaCore=0 must disable the configurable core inhibition',()=>{
  const m=make({gammaCore:0});for(let x=0;x<2;x++)m.teachExperience({id:`e${x}`,conditions:{x},outcomes:{y:x}},10);
  const base=m.conditionMap.neuronCount+m.outcomeMap.neuronCount;
  const actual=m.net.getInhibitoryWeight(base,base+m.coreSize);
  console.log('B07 evidence',JSON.stringify({gammaCoreRequested:0,gammaCoreActual:actual}));assert.equal(actual,0);
});

test('B08 uniform Hebbian plateau: an actual energy well is invisible to the strict peak detector (boundary)',()=>{
  const n=new EnergyNetwork({neuronCount:4,activationEnergy:1,maintenanceEnergy:0});
  hebbianLearn(n,[0,1,2,3],20);const e=n.energy(new Uint8Array([1,1,1,1]));
  const r=n.settle([0,1]);const wells=detectWells(n);
  console.log('B08 evidence',JSON.stringify({uniformPatternEnergy:e,settled:r.activeNeurons,detectedWells:wells.length}));
  assert.equal(e,-2);assert.deepEqual(r.activeNeurons,[0,1,2,3]);assert.equal(wells.length,0);
});

test('B09 learning and redetecting wells must not silently swap existing labels',()=>{
  const n=new EnergyNetwork({neuronCount:4,activationEnergy:1,maintenanceEnergy:0,maxWeight:3});
  n.strengthen(0,1,1);n.strengthen(2,3,0.8);
  const r=new ReadoutModule(detectWells(n),0.5);r.labelWell(0,'A');r.labelWell(1,'B');
  n.strengthen(2,3,1);r.updateWells(detectWells(n));
  const result=r.identify([0,1]);console.log('B09 evidence',JSON.stringify(result));assert.equal(result[0].label,'A');
});

test('B10 finite core capacity is an explicit boundary, not open-ended learning',()=>{
  const m=make({maxRules:1});m.learnFromObservation({x:0},{y:0},10);
  assert.throws(()=>m.learnFromObservation({x:1},{y:1},10),/rule capacity exhausted/);
  assert.equal(m.ruleCount,1);
});
