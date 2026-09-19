import {writeFileSync} from 'node:fs';import {ContinuousExplorer} from '../repo/dist/src/pop/explore/explorer-continuous.js';import {ExperimentPlanner} from '../repo/dist/src/pop/explore/planner.js';
const specs=[{name:'x',bins:2,values:[.2,.201]}],p=new ExperimentPlanner(specs),ex=new ContinuousExplorer([{name:'x',min:0,max:1}],[{name:'y',min:0,max:1}],p,specs,{conduct:q=>({y:q.x})},{y:1},{},1);
while(ex.step()){}
const rows=[];for(const x of specs[0].values)for(let seed=1;seed<=5;seed++){const pred=ex.mem.predict({x},seed);rows.push({x,truth:x,seed,values:pred.values,converged:pred.converged});}
const result={world:'static y=x; nearby but legal intervention values',experiments:ex.log.length,reason:ex.terminationReason,phase:ex.currentPhase,training:ex.log,rows};writeFileSync(new URL('../results/continuous-nearby-world.json',import.meta.url),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
