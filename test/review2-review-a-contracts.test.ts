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

test('R09 invalid observation must not consume a rule slot or write weights',t=>{
 const m=pop();let error;try{m.learnFromObservation({x:0},{y:NaN},2);}catch(e){error=e.message;}
 diag(t,{error,rules:m.ruleCount,allocated:m.coreCursor,conditionCoreWeight:m.net.getWeight(0,16)});assert.ok(error);assert.equal(m.ruleCount,0);assert.equal(m.net.getWeight(0,16),0);
});
test('R10 sensory NaN must be rejected before registration',t=>{
 const m=field();let rejected=false;try{m.learnFromObservation({x:NaN},{y:.5});}catch{rejected=true;}
 diag(t,{rejected,rules:m.ruleCount});assert.ok(rejected);assert.equal(m.ruleCount,0);
});
test('R11 finite integer annealing options must be validated',t=>{
 const accepted=[];for(const [label,options] of [['levels NaN',{levels:NaN}],['sweeps 1.5',{sweepsPerLevel:1.5}],['temperature NaN',{initialTemperature:NaN}],['budget -1',{quenchMaxFlips:-1}],['candidate .5',{extraCandidates:[.5]}]]){
  let rejected=false;try{net().settleAnnealed([0],[],{levels:1,sweepsPerLevel:1,...options});}catch{rejected=true;}if(!rejected)accepted.push(label);
 }diag(t,{accepted});assert.deepEqual(accepted,[]);
});
test('R12 infinite levels must reject promptly (isolated child with 2.5s hard timeout)',t=>{
 const url=new URL('../src/index.js',import.meta.url).href;
 const source=`import {EnergyNetwork} from ${JSON.stringify(url)};const n=new EnergyNetwork({neuronCount:3,activationEnergy:1,maintenanceEnergy:.5});try{n.settleAnnealed([0],[],{levels:Infinity,sweepsPerLevel:1});console.log('accepted')}catch{console.log('rejected')}`;
 let result;try{result=execFileSync(process.execPath,['--input-type=module','-e',source],{timeout:2500,encoding:'utf8',windowsHide:true}).trim();}catch(e){result=e.code??e.message;}
 diag(t,{result});assert.equal(result,'rejected');
});
test('R13 zero quench budget must reject or perform zero quench flips',t=>{
 const n=net(3);n.strengthen(0,1,2);n.strengthen(1,2,2);let r;try{r=n.settleAnnealed([0],[],{levels:1,sweepsPerLevel:1,quenchMaxFlips:0});}catch{return;}
 diag(t,r);assert.equal(r.trace.flipCount,0);
});
test('R15 fractional rule capacity must be rejected at construction',t=>{
 let rejected=false;try{field({maxRules:1.5});}catch{rejected=true;}diag(t,{rejected});assert.ok(rejected);
});
test('R17 allocator rejects ordinary overflow without changing registered state',t=>{
 const m=field({maxRules:1});m.learnFromObservation({x:.1},{y:.1});const n=m.coreCursor,count=m.ruleCount;
 assert.throws(()=>m.learnFromObservation({x:.9},{y:.9}),/capacity/);diag(t,{n,count,after:m.coreCursor});assert.equal(m.coreCursor,n);assert.equal(m.ruleCount,count);
});
