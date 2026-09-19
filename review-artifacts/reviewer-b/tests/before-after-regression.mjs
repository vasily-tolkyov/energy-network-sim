import {writeFileSync} from 'node:fs';
const results=[];
for(const [rev,root] of [['f25fefe','../baseline-f25fefe'],['e732f2b','../repo']]){
 const {PopRuleMemory}=await import(root+'/dist/src/pop/popmemory.js');const {PopChannelMap}=await import(root+'/dist/src/pop/popmap.js');const {FieldRuleMemory}=await import(root+'/dist/src/pop/concept/field-memory.js');const {SensoryEncoder}=await import(root+'/dist/src/pop/concept/sensory.js');
 const m=new PopRuleMemory(new PopChannelMap([{name:'x',bins:2}]),new PopChannelMap([{name:'y',bins:2}]),{maxRules:8});const read=()=>Array.from({length:5},(_,i)=>m.predict({x:0},i+1).decoded.y);m.learnFromObservation({x:0},{y:1},6);const before=read();m.learnFromQuery({x:0},{y:0},2);const afterGuess=read();for(let i=0;i<10;i++)m.learnFromObservation({x:0},{y:1},6);const afterTruth=read();
 // FieldRuleMemory's emergent reference is not used by either revision; identical inert argument.
 const enc=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}]),f=new FieldRuleMemory(enc,{}, {maxRules:8});f.setOutcomeDimensions(['y']);const fread=()=>Array.from({length:5},(_,i)=>f.predict({x:.2},i+1).values.y);f.learnFromObservation({x:.2},{y:.2},6);const fbefore=fread();f.learnFromObservation({x:.201},{y:.201},6);const fafter=fread();
 results.push({rev,evidenceOrder:{before,afterGuess,afterTruth},equivalentContinuous:{before:fbefore,after:fafter}});
}
console.log(JSON.stringify(results,null,2));writeFileSync(new URL('../results/before-after-regression.json',import.meta.url),JSON.stringify(results,null,2));
