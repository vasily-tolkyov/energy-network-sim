import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const base=path.dirname(fileURLToPath(import.meta.url)),root=path.dirname(base);
const groups={
  discrete:[['discrete','src/pop/explore/runner-explore.js','1,2,3']],
  continuous:[['continuous','src/pop/explore/runner-explore-continuous.js','1,2,3']],
  extension:[['extension','src/pop/explore/runner-explore-ext.js','1']],
  supporting:[['pop','src/pop/runner-pop.js','1'],['chem','src/topics/runner-chem-pop.js','1'],['concept','src/pop/concept/runner-formation.js','1'],['neural-attention','src/pop/attention/runner-neural-attention.js','1'],['demo','src/demo.js','1'],['demo-seq','src/demo-sequence.js','1']],
  independent:[['independent-discrete-4','../review-artifacts/independent-experiments.mjs','1',['discrete','4']],['independent-continuous-4','../review-artifacts/independent-experiments.mjs','1',['continuous','4']],['independent-continuous-5','../review-artifacts/independent-experiments.mjs','1',['continuous','5','off-grid-only']]],
  fresh:[['fresh-continuous-6','../review-artifacts/fresh-experiments.mjs','1',['continuous','6']],['fresh-continuous-7','../review-artifacts/fresh-experiments.mjs','1',['continuous','7','off-grid-only']]],
};
const group=process.argv[2];if(!groups[group])throw new Error(`unknown group ${group}`);
for(const [name,entry,seeds,args=[]] of groups[group]){
  const started=new Date().toISOString(),time=performance.now();
  const fd=fs.openSync(path.join(base,'reproduction',`${name}.stdout.log`),'w');
  console.log(`START ${name} ${started}`);
  const child=spawn(process.execPath,[path.join(root,'dist',entry),...args],{cwd:path.join(base,'reproduction'),env:{...process.env,SEEDS:seeds},stdio:['ignore',fd,fd],windowsHide:true});
  const code=await new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',resolve);});
  fs.closeSync(fd);
  const meta={name,entry,seeds,args,commit:'e732f2b4ef46e0ac482480095a31cb5aa23ec6be',started,finished:new Date().toISOString(),exitCode:code,seconds:(performance.now()-time)/1000};
  fs.writeFileSync(path.join(base,'reproduction',`${name}-meta.json`),JSON.stringify(meta,null,2));
  console.log(JSON.stringify(meta));
  if(code!==0){process.exitCode=1;break;}
}
