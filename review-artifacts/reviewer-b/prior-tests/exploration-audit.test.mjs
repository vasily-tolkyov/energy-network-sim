import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFileSync,mkdirSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
import {Explorer} from '../repo/dist/src/pop/explore/explorer.js';
import {ExperimentPlanner} from '../repo/dist/src/pop/explore/planner.js';
import {PopChannelMap} from '../repo/dist/src/pop/popmap.js';
import {R2PopLayer} from '../repo/dist/src/pop/r2pop.js';
import {NeuralFocusNet} from '../repo/dist/src/pop/attention/neural-focus.js';

// Five new, deterministic laboratory worlds, not copied from repository examples.
// Predeclared criterion: >=90% exact post-exploration predictions over the complete
// small grid and seeds 1..5, plus exact factor set. This is a small-world contract
// challenge, not an estimate of the published circuit benchmark's accuracy.
const worlds = [
 {id:'C01',name:'constant response; one irrelevant actuator',specs:[{name:'x',bins:2}],truth:()=>({y:0}),factors:[]},
 {id:'C02',name:'identity; one causal actuator',specs:[{name:'x',bins:2}],truth:c=>({y:c.x}),factors:['x']},
 {id:'C03',name:'XOR interaction; two causal actuators',specs:[{name:'a',bins:2},{name:'b',bins:2}],truth:c=>({y:c.a^c.b}),factors:['a','b']},
 {id:'C04',name:'identity with an irrelevant distractor',specs:[{name:'x',bins:2},{name:'d',bins:2}],truth:c=>({y:c.x}),factors:['x']},
 {id:'C05',name:'shifted threshold with AND gate',specs:[{name:'x',bins:3},{name:'gate',bins:2}],truth:c=>({y:Number(c.x>=2&&c.gate===1)}),factors:['gate','x']},
];
function grid(specs) { return specs.reduce((xs,s)=>xs.flatMap(c=>Array.from({length:s.bins},(_,v)=>({...c,[s.name]:v}))),[{}]); }
function runWorld(w,learnRepeats=2) {
 const start=performance.now();
 const cm=new PopChannelMap(w.specs),om=new PopChannelMap([{name:'y',bins:2}]);
 const planner=new ExperimentPlanner(w.specs);
 const learner=new Explorer(cm,om,planner,w.specs,{conduct:w.truth},{y:1},{learnRepeats,budget:30},1);
 let safety=0;
 while (learner.step()) { if (++safety>31) throw Error('Unexpected runaway beyond declared budget'); }
 const predictions=[];
 for (const c of grid(w.specs)) for (let seed=1;seed<=5;seed++) {
  const pred=learner.mem.predict(c,seed).decoded.y;
  predictions.push({conditions:c,seed,truth:w.truth(c).y,predicted:pred,correct:pred===w.truth(c).y});
 }
 const record={id:w.id,name:w.name,learnRepeats,gridSize:grid(w.specs).length,experiments:planner.experimentCount,rules:learner.mem.ruleCount,factors:learner.influentialDims,expectedFactors:w.factors,correct:predictions.filter(p=>p.correct).length,total:predictions.length,accuracy:predictions.filter(p=>p.correct).length/predictions.length,elapsedMs:performance.now()-start,log:learner.log,predictions};
 mkdirSync(new URL('../results/',import.meta.url),{recursive:true});
 writeFileSync(new URL(`../results/${w.id}-world.json`,import.meta.url),JSON.stringify(record,null,2));
 console.log(JSON.stringify({world:w.id,experiments:record.experiments,correct:record.correct,total:record.total,factors:record.factors,elapsedMs:record.elapsedMs}));
 return record;
}
for (const w of worlds) test(`${w.id} new-world acceptance: ${w.name}`,()=>{
 const r=runWorld(w);
 // Collect both criteria before asserting, so failed prediction does not hide factors.
 const failures=[];
 if(r.accuracy<0.90) failures.push(`exact accuracy ${r.correct}/${r.total}`);
 if(JSON.stringify(r.factors)!==JSON.stringify(w.factors)) failures.push(`factors ${JSON.stringify(r.factors)} expected ${JSON.stringify(w.factors)}`);
 assert.deepEqual(failures,[]);
});

test('C06 positive control: identity with six observation repeats',()=>{
 const w={...worlds[1],id:'C06'}; const r=runWorld(w,6);
 assert.ok(r.accuracy>=0.90,`accuracy ${r.correct}/${r.total}`);
 assert.deepEqual(r.factors,w.factors);
});

test('C07 R2 must not label an unchanged constant outcome as changed',()=>{
 const cm=new PopChannelMap([{name:'x',bins:2}]),om=new PopChannelMap([{name:'y',bins:2}]);
 const r=new R2PopLayer(cm,om).analyzePair({e0:{conditions:{x:0},outcomes:{y:0}},e1:{conditions:{x:1},outcomes:{y:0}}});
 console.log('C07',JSON.stringify(r));
 assert.equal(r.outcomeChanged,false);
 assert.deepEqual(r.influentialChannels,[]);
});

test('C08 WTA must produce one active marker at the default exploration drive 2*4',()=>{
 const f=new NeuralFocusNet(['a','b'],{inertia:0}); f.beginFrame(); f.setMismatch('a',8); f.setMismatch('b',8);
 const chosen=f.select(1),active=f.net.activeNeurons(),markers=active.filter(i=>i%2===1);
 console.log('C08',JSON.stringify({chosen,active,markers,gamma:f.net.getInhibitoryWeight(1,3)}));
 assert.equal(markers.length,1);
});

const ep=c=>({conditions:c,outcomes:{y:0},classification:'unknown-change'});
test('C09 planner positive control: frontier and auto-pairs obey Hamming-1',()=>{
 const p=new ExperimentPlanner([{name:'a',bins:3},{name:'b',bins:2}]); p.register(ep({a:0,b:0}));
 assert.deepEqual(p.candidates(),[{a:1,b:0},{a:2,b:0},{a:0,b:1}]);
 p.register(ep({a:1,b:0})); p.register(ep({a:1,b:1}));
 for(const {pair} of p.allAutoPairs()) assert.equal(Object.keys(pair.e0.conditions).filter(k=>pair.e0.conditions[k]!==pair.e1.conditions[k]).length,1);
});
test('C10 stored experimental history must not mutate through caller-owned objects',()=>{
 const p=new ExperimentPlanner([{name:'x',bins:2}]); const c={x:0};p.register(ep(c));c.x=1;
 console.log('C10',JSON.stringify({history:p.allEpisodes,candidates:p.candidates()}));
 assert.equal(p.allEpisodes[0].conditions.x,0);
});
test('C11 boundary characterization: completed grid cannot be re-experimented',()=>{
 const p=new ExperimentPlanner([{name:'x',bins:2}]);p.register(ep({x:0}));p.register(ep({x:1}));
 assert.deepEqual(p.candidates(),[]); // Static, noiseless-grid limitation; not a failed promise of noisy science.
});
test('C12 R2 common/difference contract must retain unchanged distractor members',()=>{
 const cm=new PopChannelMap([{name:'x',bins:2},{name:'d',bins:2}]),om=new PopChannelMap([{name:'y',bins:2}]);
 const r=new R2PopLayer(cm,om).analyzePair({e0:{conditions:{x:0,d:0},outcomes:{y:0}},e1:{conditions:{x:1,d:0},outcomes:{y:1}}});
 console.log('C12',JSON.stringify(r));
 assert.deepEqual(r.diffChannels,['x']); // The downstream AND may still produce the correct factor; distinguish these contracts.
});
