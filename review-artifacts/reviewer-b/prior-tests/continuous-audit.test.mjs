import test from 'node:test';
import assert from 'node:assert/strict';
import {FieldRuleMemory} from '../repo/dist/src/pop/concept/field-memory.js';
import {SensoryEncoder} from '../repo/dist/src/pop/concept/sensory.js';
import {EnergyNetwork,hebbianLearn,detectWells,ReadoutModule} from '../repo/dist/src/index.js';
function memory(maxRules=8){
 const enc=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}]);
 // The production class never dereferences emergent; use an inert, explicit fixture.
 const mem=new FieldRuleMemory(enc,{}, {maxRules});mem.setOutcomeDimensions(['y']);return{enc,mem};
}
test('D01 FieldRuleMemory must enforce capacity before touching the pool/out-of-range neurons',()=>{
 const {mem}=memory(1);mem.learnFromObservation({x:0.2},{y:0.2},4);
 let thrown=null;try{mem.learnFromObservation({x:0.8},{y:0.8},4);}catch(e){thrown=String(e);}
 const record={thrown,neuronCount:mem.net.neuronCount,ruleCount:mem.ruleCount,cores:Array.from({length:mem.ruleCount},(_,i)=>mem.ruleCore(i))};
 console.log('D01',JSON.stringify(record));
 assert.ok(thrown!==null,'second rule must be rejected or safely grow storage; neither happened');
});
test('D02 influence-first then real observation must register a readable rule',()=>{
 const {mem}=memory();mem.bindInfluence({x:0.2},{x:0.3},2);mem.learnFromObservation({x:0.2},{y:0.8},6);
 const p=mem.predict({x:0.2},1); console.log('D02',JSON.stringify({ruleCount:mem.ruleCount,allCoreNeurons:mem.allCoreNeurons(),values:p.values}));
 assert.equal(mem.ruleCount,1); assert.ok(Math.abs(p.values.y-0.8)<=0.05);
});
test('D03 rule outcome metadata must represent newly bound observed fields',()=>{
 const {enc,mem}=memory();mem.learnFromObservation({x:0.2},{y:0.2},6);mem.learnFromObservation({x:0.2},{y:0.8},6);
 const actual=mem.ruleOutcomeFields(0),latest=enc.encode({y:0.8});
 console.log('D03',JSON.stringify({actual,latest}));
 // This tests the advertised metadata accessor, not a claim that arbitrary concept drift is solved.
 assert.ok(latest.every(id=>actual.includes(id)),'metadata contains only first observation despite later binding');
});
test('D04 README usage snippet must run without throwing',()=>{
 const net=new EnergyNetwork({neuronCount:24,activationEnergy:2.0,maintenanceEnergy:0.5});
 hebbianLearn(net,[0,1,2,3],10);hebbianLearn(net,[12,13,14,15],10);
 const wells=detectWells(net);console.log('D04',JSON.stringify({wellCount:wells.length}));
 const readout=new ReadoutModule(wells,net.config.readoutThreshold);
 readout.labelWell(0,'A');
 const result=net.settle([0,1]);assert.ok(readout.identify(result.activeNeurons).some(r=>r.label==='A'));
});
test('D05 boundary characterization: out-of-range values collapse to the sensor boundary',()=>{
 const {enc}=memory();assert.deepEqual(enc.encode({x:1}),enc.encode({x:100}));
});
test('D06 field-memory positive control: fresh observed rule remains readable',()=>{
 const {mem}=memory();mem.learnFromObservation({x:0.2},{y:0.8},6);
 const predictions=Array.from({length:10},(_,i)=>mem.predict({x:0.2},i+1).values.y);
 console.log('D06',JSON.stringify({predictions}));
 assert.ok(predictions.every(v=>v!==null&&Math.abs(v-0.8)<=0.05));
});
test('D07 boundary characterization: nearby continuous values can have identical finite encodings',()=>{
 const {enc}=memory();assert.deepEqual(enc.encode({x:0.2}),enc.encode({x:0.201}));
});
