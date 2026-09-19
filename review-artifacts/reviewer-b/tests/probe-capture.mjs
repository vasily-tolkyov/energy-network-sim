import {FieldRuleMemory} from '../repo/dist/src/pop/concept/field-memory.js';
import {SensoryEncoder} from '../repo/dist/src/pop/concept/sensory.js';
import {EmergentMap} from '../repo/dist/src/pop/concept/emergent-map.js';
import {captureClassify} from '../repo/dist/src/pop/attention/capture.js';
const enc=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}]),m=new FieldRuleMemory(enc,new EmergentMap([],enc),{maxRules:4});m.setOutcomeDimensions(['y']);
m.learnFromObservation({x:.2},{y:.2},6);const p=m.predict({x:.2},1),core=p.winningCores[0];
const cases=[];for(const y of [.2,.8])for(let seed=1;seed<=5;seed++)cases.push({observed:y,seed,...captureClassify(m,{x:.2},{y},core,seed)});
console.log(JSON.stringify({forecast:p.values,core,cases},null,2));
