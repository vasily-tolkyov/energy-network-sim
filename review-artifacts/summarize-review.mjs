import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const base=path.dirname(fileURLToPath(import.meta.url));
const exists=p=>fs.existsSync(path.join(base,p));
const read=p=>fs.readFileSync(path.join(base,p),'utf8').replace(/^\uFEFF/,'');
const json=p=>JSON.parse(read(p));
const jobs=['discrete','continuous','extension','pop','chem','concept','neural-attention','demo','demo-seq','independent-discrete-4','independent-continuous-4','independent-continuous-5','fresh-continuous-6','fresh-continuous-7'];
const metas=Object.fromEntries(jobs.filter(n=>exists(`reproduction/${n}-meta.json`)).map(n=>[n,json(`reproduction/${n}-meta.json`)]));
const parseSeeds=(text,mode)=>text.split(/── 种子 /).slice(1).map(part=>{
 const seed=Number(part.match(/^\d+/)[0]),training=part.match(/终止：实验 (\d+) 次.*?规则核 (\d+) 条/);
 const discrete=part.match(/总计\s+门控任务分 ([\d.]+)%\s+严格联合 ([\d.]+)%\s+粗粒度 ([\d.]+)%/);
 const cont=part.match(/总计\s+细粒度 (\d+)\/(\d+)（([\d.]+)%）\s+粗粒度 ([\d.]+)%/);
 const total=mode==='discrete'?288:432;
 return {seed,experiments:training?Number(training[1]):null,rules:training?Number(training[2]):null,n:total,complete:mode==='discrete'?!!discrete:!!cont,fine:discrete?Math.round(Number(discrete[2])*total/100):cont?Number(cont[1]):null,finePct:discrete?Number(discrete[2]):cont?Number(cont[3]):null,coarse:discrete?Math.round(Number(discrete[3])*total/100):cont?Math.round(Number(cont[4])*total/100):null,coarsePct:discrete?Number(discrete[3]):cont?Number(cont[4]):null};
});
const official=Object.fromEntries(['discrete','continuous'].map(mode=>[mode,{actual:parseSeeds(read(`reproduction/${mode}.stdout.log`),mode),reference:[1,2,3].flatMap(s=>parseSeeds(read(`../runs/explore-${mode==='discrete'?'lab':'cont'}-seed${s}.log`),mode))}]));
const predictions={},distinctConditions=new Set(),queryRuns=[];let checkedQueries=0;
for(const key of ['independent-discrete-4','independent-continuous-4','independent-continuous-5','fresh-continuous-6','fresh-continuous-7']){
 if(!exists(`${key}-results.json`))continue;
 const data=json(`${key}-results.json`),cont=data.summary.mode==='continuous',seenQueries=new Map(),queryRun={name:key,records:data.results.length,duplicateConditions:0,differingPredictions:[]};
 assert.equal(data.summary.cost,data.summary.experiments);
 for(const r of data.results){
  const conditionKey=data.summary.mode+JSON.stringify(Object.entries(r.conditions).sort(([a],[b])=>a.localeCompare(b))),prediction=JSON.stringify(r.pred);distinctConditions.add(conditionKey);if(seenQueries.has(conditionKey)){queryRun.duplicateConditions++;if(seenQueries.get(conditionKey)!==prediction)queryRun.differingPredictions.push({conditions:r.conditions,before:seenQueries.get(conditionKey),after:prediction});}else seenQueries.set(conditionKey,prediction);
  for(const [pk,sk] of [['pred','modelScore'],['nn','nnScore']]){
   const v=r[pk];for(const value of Object.values(v))if(typeof value==='number')assert.ok(Number.isFinite(value));const coarse=(typeof v.lit==='number'?(cont?(v.lit>=.5?1:0):v.lit):null)===r.truth.lit;
   const brightness=typeof v.brightness==='number'&&(cont?Math.abs(v.brightness-r.truth.brightness)<=.5:v.brightness===r.truth.brightness);
   assert.equal(r[sk].coarse,coarse);assert.equal(r[sk].strictFine,coarse&&brightness);
  }checkedQueries++;
 }
 const partitions={};
 for(const [kind,summary] of Object.entries(data.summary.partitions)){
  const rows=data.results.filter(r=>kind==='all-grid'?!['off-grid-uniform','gate-boundary'].includes(r.kind):kind==='all-unseen-grid'?!['off-grid-uniform','gate-boundary'].includes(r.kind)&&!r.seen:r.kind===kind);
  assert.equal(rows.length,summary.n);
  for(const [mk,sk] of [['model','modelScore'],['nearest','nnScore']])for(const metric of ['coarse','strictFine','officialFine'])assert.equal(summary[mk][metric],rows.filter(r=>r[sk][metric]).length);
  const answered=rows.filter(r=>typeof r.pred.lit==='number').length,positive=rows.filter(r=>r.truth.lit===1).length;
  partitions[kind]={...summary,answered,refused:rows.length-answered,truthPositive:positive,predictedPositive:rows.filter(r=>typeof r.pred.lit==='number'&&r.pred.lit>=.5).length,majorityConstantCorrect:Math.max(positive,rows.length-positive)};
 }
 predictions[key]={...data.summary,partitions};queryRuns.push(queryRun);
}
fs.writeFileSync(path.join(base,'query-integrity.json'),JSON.stringify({records:checkedQueries,distinctConditionsAcrossModes:distinctConditions.size,byRun:queryRuns},null,2));
const clean=s=>s.replaceAll('\r\n','\n').replace(/单次预测均时 [\d.]+ms/g,'TIMING').replace(/耗时 [\d.]+ 分钟/g,'ELAPSED').trim();
const compared=[];
for(const name of fs.readdirSync(path.join(base,'reproduction','runs'))){
 if(exists(`../runs/${name}`))compared.push({file:name,equalIgnoringTimingsAndLineEndings:clean(read(`reproduction/runs/${name}`))===clean(read(`../runs/${name}`))});
}
const parseTap=p=>{const s=read(p);return {tests:Number(s.match(/^# tests (\d+)/m)[1]),pass:Number(s.match(/^# pass (\d+)/m)[1]),fail:Number(s.match(/^# fail (\d+)/m)[1])};};
const result={commit:'e732f2b4ef46e0ac482480095a31cb5aa23ec6be',baseline:json('baseline-meta.json'),legacy:parseTap('legacy-adversarial.tap'),revision:parseTap('revision-tests.tap'),oldControl:parseTap('new-regressions-old-control.tap'),r2OldControl:parseTap('r2-old-control.tap'),official,predictions,compared,metas,pending:jobs.filter(n=>!metas[n]),checkedQueries};
fs.writeFileSync(path.join(base,'results-summary.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify({pending:result.pending,checkedQueries,compared,official,predictions:Object.fromEntries(Object.entries(predictions).map(([k,v])=>[k,{experiments:v.experiments,dynamics:v.dynamics,partitions:v.partitions}]))},null,2));
