// @ts-nocheck -- verbatim independent audit cases, imports and diagnostics relocated.
import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {EnergyNetwork} from '../src/network.js';
import {hebbianLearn} from '../src/hebbian.js';
import {mulberry32} from '../src/prng.js';
import {PopChannelMap} from '../src/pop/popmap.js';
import {PopRuleMemory} from '../src/pop/popmemory.js';
import {FieldRuleMemory} from '../src/pop/concept/field-memory.js';
import {EmergentMap} from '../src/pop/concept/emergent-map.js';
import {captureClassify} from '../src/pop/attention/capture.js';
import {SensoryEncoder} from '../src/pop/concept/sensory.js';
import {NeuralFocusNet} from '../src/pop/attention/neural-focus.js';
import {ExperimentPlanner} from '../src/pop/explore/planner.js';
import {Explorer} from '../src/pop/explore/explorer.js';
import {LabBench,labProbes} from '../src/topics/lab-world.js';
const save = (_id: string, _value: unknown) => {};
const pop=(cfg={})=>new PopRuleMemory(new PopChannelMap([{name:'x',bins:2}],4),new PopChannelMap([{name:'y',bins:2}],4),{maxRules:8,...cfg});
function field(cfg={}){const enc=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}]);const mem=new FieldRuleMemory(enc,new EmergentMap([],enc), {maxRules:8,...cfg});mem.setOutcomeDimensions(['y']);return {enc,mem};}
const reads=m=>Array.from({length:5},(_,i)=>m.predict({x:0},i+1).decoded.y);
const freads=m=>Array.from({length:5},(_,i)=>m.predict({x:.2},i+1).values.y);
function snapshot(net){const vals=[];for(let i=0;i<net.neuronCount;i++)for(let j=0;j<net.neuronCount;j++)vals.push(net.getWeight(i,j),net.getInhibitoryWeight(i,j),net.getDirectedInhibitoryWeight(i,j));return JSON.stringify(vals);}

test('N18 planner history getter must not expose mutable internal conditions',()=>{const p=new ExperimentPlanner([{name:'x',bins:2}]);p.register({conditions:{x:0},outcomes:{y:0},classification:'unknown-change'});try{p.allEpisodes[0].conditions.x=1;}catch{}save('N18',{history:p.allEpisodes});assert.equal(p.allEpisodes[0].conditions.x,0);});
test('N19 returned auto-pairs must not mutate historical observations',()=>{const p=new ExperimentPlanner([{name:'x',bins:2}]);p.register({conditions:{x:0},outcomes:{y:0},classification:'unknown-change'});const r=p.register({conditions:{x:1},outcomes:{y:1},classification:'unknown-change'});try{r.pairs[0].e0.outcomes.y=99;}catch{}save('N19',{history:p.allEpisodes});assert.equal(p.allEpisodes[0].outcomes.y,0);});
test('N25 Field public rule views cannot corrupt internal registration',()=>{const {mem}=field();mem.learnFromObservation({x:.2},{y:.2},6);const original=mem.ruleCore(0)[0];try{mem.ruleCore(0)[0]=9999;}catch{}save('N25',{original,after:mem.ruleCore(0)[0]});assert.equal(mem.ruleCore(0)[0],original);});
test('N26 nominally valid small population configuration recalls learned identity or is rejected',()=>{let m;try{m=new PopRuleMemory(new PopChannelMap([{name:'x',bins:2}],1),new PopChannelMap([{name:'y',bins:2}],1),{maxRules:4,coreSize:1});}catch{return;}m.learnFromObservation({x:0},{y:0},2);const got=reads(m);save('N26',{got});assert.deepEqual(got,[0,0,0,0,0]);});
test('N27 claimed excitation-off/veto-on factorial cell retains veto',()=>{
 const rows=[];for(const gain of [0,.2])for(const veto of [false,true]){const m=pop();const a={conditions:{x:0},outcomes:{y:0}},b={conditions:{x:1},outcomes:{y:1}};m.teachExperience(a,4);m.teachExperience(b,4);m.bindInfluence(a,{x:gain},4,1.5,veto);const coreBase=m.conditionMap.neuronCount+m.outcomeMap.neuronCount;rows.push({gain,veto,alternativeVeto:m.net.getInhibitoryWeight(m.conditionMap.population('x',1)[0],coreBase),matchedWeight:m.net.getWeight(0,coreBase)});}save('N27',{rows});assert.ok(rows.find(r=>r.gain===0&&r.veto).alternativeVeto>0,'veto-on cell is actually veto-off');
});
