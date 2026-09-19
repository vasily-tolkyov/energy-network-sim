// Teaching helper copied from test/pop-memory.test.ts; no model parameters changed.
import {writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {PopChannelMap} from '../dist/src/pop/popmap.js';
import {PopRuleMemory} from '../dist/src/pop/popmemory.js';
import {R2PopLayer} from '../dist/src/pop/r2pop.js';
import {CONDITION_SPECS,OUTCOME_SPECS,curriculum,queries} from '../dist/src/prototype/world.js';
const cm=new PopChannelMap(CONDITION_SPECS,4),om=new PopChannelMap(OUTCOME_SPECS,4),SPAN={rebound:1,reboundSpeed:3};
function teachBall(gain, veto = true) {
    const mem = new PopRuleMemory(cm, om, { maxRules: 128 });
    const r2 = new R2PopLayer(cm, om);
    for (const group of curriculum()) {
        const magSum = new Map();
        const magCount = new Map();
        for (const pair of group.pairs) {
            for (const e of [pair.e0, pair.e1])
                mem.teachExperience(e, 4);
            for (const spec of OUTCOME_SPECS) {
                const a = pair.e0.outcomes[spec.name];
                const b = pair.e1.outcomes[spec.name];
                if (a !== b)
                    mem.teachExclusion("outcome", spec.name, a, b, 3.0);
            }
            const analysis = r2.analyzePair(pair);
            if (analysis.undecidable)
                continue; // 不可判定对不进幅度累计（评审 F04）
            for (const ch of analysis.influentialChannels) {
                let m = 0;
                for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
                    m += Math.abs(delta) / (SPAN[och] ?? 1);
                }
                // 与读出层口径一致：按全部结果通道数取平均（2），不按变化通道数
                magSum.set(ch, (magSum.get(ch) ?? 0) + m / om.specs.length);
                magCount.set(ch, (magCount.get(ch) ?? 0) + 1);
            }
        }
        const boost = {};
        for (const [ch, sum] of magSum) {
            boost[ch] = 0.1 * gain * (sum / Math.max(1, magCount.get(ch) ?? 1)) ** 2;
        }
        for (const pair of group.pairs) {
            for (const e of [pair.e0, pair.e1])
                mem.bindInfluence(e, boost, 4, 1.5, veto);
        }
    }
    return mem;
}

const hash=x=>createHash('sha256').update(Buffer.from(x.buffer)).digest('hex');
function score(mem){
 const rows=queries().map(q=>{const p=mem.predict(q.conditions,1);return {...q,prediction:p.decoded,converged:p.converged,reason:p.terminationReason,energy:p.energy,driveEngaged:mem.net.diDriveEngaged(),coarse:p.decoded.rebound===q.truth.rebound,fine:p.decoded.rebound===q.truth.rebound&&(q.truth.rebound===0||p.decoded.reboundSpeed===q.truth.reboundSpeed)};});
 const miss=rows.filter(r=>r.kind==='miss-wall'),unseen=rows.filter(r=>r.kind==='unseen-combo');
 return {n:rows.length,fine:rows.filter(r=>r.fine).length,coarse:rows.filter(r=>r.coarse).length,miss:{n:miss.length,correct:miss.filter(r=>r.coarse).length},unseen:{n:unseen.length,correct:unseen.filter(r=>r.fine).length},rows};
}
const configurations=[];
for(const gain of [0,3])for(const veto of [false,true]){const mem=teachBall(gain,veto);configurations.push({gain,veto,W:hash(mem.net.weights),Gamma:hash(mem.net.inhibitory),...score(mem)});}
let sameWeightsOldSolver=null;
if(process.argv[2]){const {EnergyNetwork:Old}=await import(pathToFileURL(process.argv[2]).href);const mem=teachBall(3,true);mem.net.settleAnnealed=(...args)=>Old.prototype.settleAnnealed.apply(mem.net,args);sameWeightsOldSolver=score(mem);}
const report={source:'current teaching weights, seed 1, original 48 small-ball queries',distinctConfigurations:new Set(configurations.map(c=>c.W+':'+c.Gamma)).size,configurations,sameWeightsOldSolver};
writeFileSync('runs/review2/pop-score-lock.json',JSON.stringify(report,null,2));
console.log(JSON.stringify({distinctConfigurations:report.distinctConfigurations,configurations:configurations.map(({rows,...c})=>c),sameWeightsOldSolver:sameWeightsOldSolver&&(({rows,...x})=>x)(sameWeightsOldSolver)}));
