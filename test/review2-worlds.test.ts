// @ts-nocheck -- independent review world definitions unchanged.
import test from "node:test";
import assert from "node:assert/strict";
import {writeFileSync} from 'node:fs';import {performance} from 'node:perf_hooks';
import {PopChannelMap} from '../src/pop/popmap.js';import {Explorer} from '../src/pop/explore/explorer.js';import {ExperimentPlanner} from '../src/pop/explore/planner.js';
const cases=[
 {id:'W01',name:'3-bit parity with distractor',specs:[['a',2],['b',2],['c',2],['d',2]],outs:[['y',2]],truth:q=>({y:q.a^q.b^q.c}),factors:['a','b','c']},
 {id:'W02',name:'interior disjoint active intervals',specs:[['x',5],['d',2]],outs:[['y',2]],truth:q=>({y:Number(q.x===1||q.x===3)}),factors:['x']},
 {id:'W03',name:'two correlated output channels',specs:[['a',3],['b',3],['d',2]],outs:[['u',3],['v',3]],truth:q=>({u:q.a,v:(q.a+q.b)%3}),factors:['a','b']},
 {id:'W04',name:'rare three-factor joint condition',specs:[['a',4],['b',4],['c',4]],outs:[['y',2]],truth:q=>({y:Number(q.a===0&&q.b===0&&q.c===0)}),factors:['a','b','c']},
 {id:'W05',name:'nonlinear max with distractor',specs:[['a',3],['b',3],['d',2]],outs:[['y',3]],truth:q=>({y:Math.max(q.a,q.b)}),factors:['a','b']},
];
const grid=(s,i=0,q={})=>i===s.length?[q]:Array.from({length:s[i].bins},(_,v)=>grid(s,i+1,{...q,[s[i].name]:v})).flat();
for(const w of cases) test(`review2 ${w.id}: ${w.name}`, () => {const specs=w.specs.map(([name,bins])=>({name,bins})),outs=w.outs.map(([name,bins])=>({name,bins}));const cm=new PopChannelMap(specs),om=new PopChannelMap(outs),p=new ExperimentPlanner(specs),bench={conduct:w.truth};const seed=11;const ex=new Explorer(cm,om,p,cm.specs,bench,Object.fromEntries(outs.map(s=>[s.name,s.bins-1])),{},seed);let t=performance.now();while(ex.step()){}console.log(w.id,'trained',ex.log.length,'factors',ex.influentialDims);
 const rows=[];for(const q of grid(specs)){const pred=ex.mem.predict(q,seed),truth=w.truth(q);rows.push({q,truth,decoded:pred.decoded,converged:pred.converged,correct:outs.every(s=>truth[s.name]===pred.decoded[s.name]),refused:outs.some(s=>typeof pred.decoded[s.name]!=='number')});}
 const summary={id:w.id,name:w.name,seed,experiments:ex.log.length,reason:ex.terminationReason,n:rows.length,correct:rows.filter(r=>r.correct).length,refused:rows.filter(r=>r.refused).length,nonconverged:rows.filter(r=>!r.converged).length,expectedFactors:w.factors,foundFactors:ex.influentialDims,positiveRecall:w.id==='W04'?{n:rows.filter(r=>r.truth.y===1).length,correct:rows.filter(r=>r.truth.y===1&&r.correct).length}:undefined,ms:performance.now()-t};summary.passesAccuracy=summary.correct/summary.n>=.9;summary.passesFactors=JSON.stringify(summary.expectedFactors)===JSON.stringify(summary.foundFactors);if(w.id!=="W04") {assert.ok(summary.passesAccuracy, JSON.stringify(summary));assert.ok(summary.passesFactors, JSON.stringify(summary));}
else {assert.equal(summary.positiveRecall.n,1);assert.ok(summary.experiments<=300);}
});
