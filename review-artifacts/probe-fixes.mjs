import {EnergyNetwork} from '../dist/src/index.js';
import {PopRuleMemory} from '../dist/src/pop/popmemory.js';
import {PopChannelMap} from '../dist/src/pop/popmap.js';
import {SensoryEncoder} from '../dist/src/pop/concept/sensory.js';
import {EmergentMap} from '../dist/src/pop/concept/emergent-map.js';
import {FieldRuleMemory} from '../dist/src/pop/concept/field-memory.js';
const pop=()=>new PopRuleMemory(new PopChannelMap([{name:'x',bins:2}],4),new PopChannelMap([{name:'y',bins:2}],4),{maxRules:8});
const field=()=>{const e=new SensoryEncoder([{name:'x',min:0,max:1},{name:'y',min:0,max:1}],40);const m=new FieldRuleMemory(e,new EmergentMap([],e),{maxRules:8});m.setOutcomeDimensions(['y']);return m;};
const out=(caseName,v)=>console.log(JSON.stringify({caseName,...v}));
{
 const m=pop();m.learnFromObservation({x:0},{y:0},2);const a=m.predict({x:0},1);m.learnFromQuery({x:0},{y:1},2);const b=m.predict({x:0},1);m.learnFromObservation({x:0},{y:0},2);const c=m.predict({x:0},1);out('observed-guess-observed',{before:a.decoded,afterGuess:b.decoded,afterRepeat:c.decoded});
}
for(const seed of [1,2,3,7,11]){
 const n=new EnergyNetwork({neuronCount:3,activationEnergy:1,maintenanceEnergy:.5,maxWeight:8});n.strengthen(0,1,2);n.strengthen(1,2,2);n.strengthenDirectedInhibitory(2,1,4,4);
 const r=n.settleAnnealed([0],[],{seed,extraCandidates:[1,2],levels:2,sweepsPerLevel:2,quenchMaxFlips:8,quenchCandidatesOnly:true,fallbackQuietOnly:true});
 out('quiet-fallback',{seed,energy:r.energy,active:r.activeNeurons,engaged:n.diDriveEngaged(),reason:r.terminationReason,flips:r.trace.flipCount,residual:r.residualFlips});
}
for(const [first,second] of [[.5,.525],[.5,.51],[.1,.9]]){
 const m=field();m.learnFromObservation({x:.2},{y:first});const a=m.predict({x:.2},1);m.learnFromObservation({x:.2},{y:second});const b=m.predict({x:.2},1);out('field-correction',{first,second,before:a.values,after:b.values,distribution:b.distribution});
}
{
 const m=pop();let error;try{m.learnFromObservation({x:0},{y:NaN},2);}catch(e){error=e.message;}out('invalid-observation',{error,rules:m.ruleCount,allocated:m.coreCursor});
 const f=field();let ferr;try{f.learnFromObservation({x:NaN},{y:.5});}catch(e){ferr=e.message;}out('nan-field',{error:ferr,rules:f.ruleCount,allocated:f.coreCursor,pred:f.predict({x:.5},1).values});
}
