import { SensoryEncoder } from '../../dist/src/pop/concept/sensory.js';
import { EmergentMap } from '../../dist/src/pop/concept/emergent-map.js';
import { FieldRuleMemory } from '../../dist/src/pop/concept/field-memory.js';
import { R2PopLayer } from '../../dist/src/pop/r2pop.js';
import { PopChannelMap } from '../../dist/src/pop/popmap.js';
const enc = new SensoryEncoder([{name:'pos',min:0,max:7},{name:'move',min:0,max:1},{name:'nextPos',min:0,max:7}]);
const mem = new FieldRuleMemory(enc,new EmergentMap([],enc),{maxRules:16});
mem.setOutcomeDimensions(['nextPos']);
const episodes=[];
for(let r=0;r<4;r++)for(let pos=0;pos<8;pos++)for(let move=0;move<2;move++) { const e={conditions:{pos,move},outcomes:{nextPos:Math.max(0,Math.min(7,pos+(move===1?1:-1)))}};mem.learnFromObservation(e.conditions,e.outcomes);if(r===0)episodes.push(e); }
const factors=new Set();
if(process.argv.includes('--r2')) {
 const r2=new R2PopLayer(new PopChannelMap([{name:'pos',bins:8},{name:'move',bins:2}]),new PopChannelMap([{name:'nextPos',bins:8}]));
 for(let i=0;i<episodes.length;i++) for(let j=i+1;j<episodes.length;j++) {
  const e0=episodes[i],e1=episodes[j];
  if(Object.keys(e0.conditions).filter(d=>e0.conditions[d]!==e1.conditions[d]).length!==1) continue;
  const a=r2.analyzePair({e0,e1});if(!a.undecidable)for(const d of a.influentialChannels)factors.add(d);
 }
 for(const e of episodes)mem.bindInfluence(e.conditions,Object.fromEntries([...factors].map(d=>[d,0])));
}
const rows=[];
for(let pos=0;pos<8;pos++)for(let move=0;move<2;move++) {const start=performance.now();const p=mem.predict({pos,move},1);rows.push({pos,move,values:p.values,cores:p.winningCores,converged:p.converged,reason:p.terminationReason,ms:performance.now()-start});}
console.log(JSON.stringify({configuration:'raw numeric ranges, defaults, 64 observations',factors:[...factors],veto:process.argv.includes('--r2'),rows},null,2));
