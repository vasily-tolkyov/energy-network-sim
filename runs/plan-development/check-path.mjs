import { TransitionMemory } from '../../dist/src/planning/transition-memory.js';
import { collectTransitions } from '../../dist/src/planning/collect.js';
import { planGoal } from '../../dist/src/planning/planner.js';
import { PathBench,PATH_SPACE } from '../../dist/src/topics/path-world.js';
const model=new TransitionMemory(PATH_SPACE);const training=collectTransitions(model,new PathBench(),64,1);
const readback=[];
for(let pos=0;pos<8;pos++)for(const action of model.actions)readback.push(model.predict({pos},action,100));
const failures=[];
for(let start=0;start<8;start++)for(let goal=0;goal<8;goal++) {
 const p=planGoal(model,{pos:start},{pos:goal},11+start*8+goal);let pos=start;const observations=[];
 for(const step of p.steps){ const observation=new PathBench().conduct({pos},step.action.values);observations.push({pos,observation});pos=observation.nextPos;}
 if(p.status!=='found'||pos!==goal)failures.push({start,goal,actual:pos,plan:p,observations});
}
console.log(JSON.stringify({training,readback,failures},null,2));
