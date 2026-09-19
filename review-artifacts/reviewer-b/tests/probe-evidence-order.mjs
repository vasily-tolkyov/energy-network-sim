import {PopRuleMemory} from '../repo/dist/src/pop/popmemory.js';
import {PopChannelMap} from '../repo/dist/src/pop/popmap.js';
const mem=new PopRuleMemory(new PopChannelMap([{name:'x',bins:2}]),new PopChannelMap([{name:'y',bins:2}]),{maxRules:4});
const result={};const read=()=>Array.from({length:5},(_,i)=>mem.predict({x:0},i+1).decoded.y);
mem.learnFromObservation({x:0},{y:1},6);result.observed=read();
mem.learnFromQuery({x:0},{y:0},2);result.afterWrongGuess=read();
for(let i=0;i<10;i++)mem.learnFromObservation({x:0},{y:1},6);result.afterTenTrueObservations=read();
console.log(JSON.stringify(result,null,2));
