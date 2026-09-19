import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const base=path.dirname(fileURLToPath(import.meta.url)),root=path.dirname(base);
const read=p=>fs.readFileSync(path.join(base,p),'utf8').replace(/^\uFEFF/,'');
const r=JSON.parse(read('results-summary.json'));
assert.deepEqual(r.pending,[]);assert.equal(r.checkedQueries,1972);
for(const m of Object.values(r.metas))assert.equal(m.exitCode,0,m.name);
assert.equal(r.baseline.ExitCode,0);assert.deepEqual(r.legacy,{tests:20,pass:15,fail:5});assert.deepEqual(r.revision,{tests:20,pass:7,fail:13});assert.deepEqual(r.oldControl,{tests:4,pass:4,fail:0});
assert.deepEqual(r.r2OldControl,{tests:1,pass:1,fail:0});
const identity=JSON.parse(read('script-identity.json'));assert.ok(identity.oldHarnessIdentical&&identity.legacyTestsIdentical&&identity.oldControlSameCodeExceptImportPathsAndTrailingNewlines);
const queries=JSON.parse(read('query-integrity.json'));assert.equal(queries.records,1972);assert.equal(queries.distinctConditionsAcrossModes,1052);assert.ok(queries.byRun.every(s=>s.differingPredictions.length===0));
const fmt=(n,d)=>`${n}/${d}=${(100*n/d).toFixed(1)}%`;
const official=['| 模式 / 种子 | 实验数 / 规则数 | 本机严格细粒度 | 本机粗粒度 | 与仓库本提交参考 |','|---|---:|---:|---:|---|'];
for(const [mode,label] of [['discrete','离散 A'],['continuous','连续 B']]){
 const a=r.official[mode].actual,b=r.official[mode].reference;assert.equal(a.length,3);assert.ok(a.every(s=>s.complete));
 for(const x of a){const y=b.find(z=>z.seed===x.seed);official.push(`| ${label} seed ${x.seed} | ${x.experiments} / ${x.rules} | ${fmt(x.fine,x.n)} | ${fmt(x.coarse,x.n)} | ${x.experiments===y.experiments&&x.rules===y.rules&&x.fine===y.fine&&x.coarse===y.coarse?'数值复现':'有差异，见原始日志'} |`);}
 const sum=k=>a.reduce((s,x)=>s+x[k],0);official.push(`| ${label} 合计/均值 | 平均 ${(sum('experiments')/3).toFixed(2)} 次 | ${fmt(sum('fine'),sum('n'))} | ${fmt(sum('coarse'),sum('n'))} | 三个种子等权 |`);
}
official.push('','连续细粒度使用本提交 lcScore，要求亮灭正确且亮度误差≤0.5，灭灯也检查亮度；离散严格联合要求两个结果都精确。计数从分母与一位小数百分比还原时，在本分母下是唯一的。');
const sums=mode=>r.official[mode].actual.reduce((s,x)=>({n:s.n+x.n,fine:s.fine+x.fine,coarse:s.coarse+x.coarse,experiments:s.experiments+x.experiments}),{n:0,fine:0,coarse:0,experiments:0});
const aSum=sums('discrete'),bSum=sums('continuous');
official.push('',`与上轮本机执行结果比较：离散旧门控任务分 88.3%、粗粒度 93.8%、平均 79.0 次实验；本次严格联合 ${fmt(aSum.fine,aSum.n)}、粗粒度 ${fmt(aSum.coarse,aSum.n)}、平均 ${(aSum.experiments/3).toFixed(2)} 次。新旧细粒度公式有变化，不能直接当作完全同口径差值。连续上轮细粒度 83.6%、粗粒度 96.5%、平均 118.7 次；本次 ${fmt(bSum.fine,bSum.n)} / ${fmt(bSum.coarse,bSum.n)}、平均 ${(bSum.experiments/3).toFixed(2)} 次；连续评分也收紧了灭灯亮度检查。对照应使用这些实际旧运行数值，而不是归属已不匹配的旧参考日志。`);
official.push('','| 审核指南重点验收的分区证据 | 识别因素 | 已观察：细 / 粗 | 未观察灭灯：细 / 粗 | 未观察亮灯：细 / 粗 |','|---|---|---:|---:|---:|');
for(const mode of ['lab','cont'])for(const seed of [1,2,3]){
 const log=read(`reproduction/runs/explore-${mode}-seed${seed}.log`);
 const factors=log.match(/R2 发现的影响因素：\[([^\]]+)\]/);assert.ok(factors);
 const scores=['self-taught','unseen-gate-off','unseen-gate-on'].map(part=>{
  const line=log.split('\n').find(s=>s.trim().startsWith(part));assert.ok(line);
  const fine=mode==='lab'?line.match(/严格联合 ([\d.]+)%/):line.match(/细粒度 \d+\/\d+（([\d.]+)%）/),coarse=line.match(/粗粒度亮灭 ([\d.]+)%/);assert.ok(fine&&coarse);
  return `${fine[1]}% / ${coarse[1]}%`;
 });
 official.push(`| ${mode==='lab'?'A':'B'} seed ${seed} | ${factors[1]} | ${scores[0]} | ${scores[1]} | ${scores[2]} |`);
}
official.push('','这些是当前互斥的未观察分区；它们避免把已教样本算进留出结果。概念中心出现 0.5 已复现，但训练候选本来包含 0.5，不能据此单独确认发现了精确物理门限；新的门限邻域探针见下一节。');
official.push('','**审核指南的局部指标并非全部满足。** A 的未观察分区粗粒度均不低于 93%，四个真实因素也在 A/B 的全部官方种子中识别成功。但 B seed 2 的未观察灭灯为 164/182=90.1%，低于指南仍列出的 gate-off 97.6–100%。对比源码确认，连续版这里仅把 gate-off 改名为 unseen-gate-off，分类逻辑未变；新旧训练样本不同使分母改变，却不能用改名或细粒度评分收紧来解释粗粒度下降。新三种子总体粗粒度 98.3% 也不能掩盖这一局部失败。');
official.push('','“每次观察都写入”也不等于“已观察样本都能准确回忆”：连续三个种子的已观察细粒度分别为 75/79、83/84、123/132，合计 281/295=95.3%，粗粒度均为 100%。这是完整学习与竞争读出管线的实际精度边界，不应与独立容量场景的 taught 100% 混为同一保证。');
const ext=read('reproduction/extension.stdout.log');
official.push('','| 扩展模式 | 总实验数 / 概念形成次数 | 扩展区细 / 粗 | 全网格细 / 粗 |','|---|---:|---:|---:|');
for(const label of ['概念更新','冻结（对照）']){
 const part=ext.split(`── 模式 ${label}，种子 1 ──`)[1]?.split('── 模式 ')[0];assert.ok(part);
 const train=part.match(/二阶段终止：实验 \+(\d+) 次（共 (\d+)），形成 (\d+) 次/),score=part.match(/扩展区探针（216 个）：细粒度 (\d+)（([\d.]+)%）\s+粗粒度 (\d+)（([\d.]+)%）/),all=part.match(/全网格（648 个）：细粒度 ([\d.]+)%\s+粗粒度 ([\d.]+)%/);assert.ok(train&&score&&all);
 official.push(`| ${label} | ${train[2]} / ${train[3]} | ${score[2]}% / ${score[4]}% | ${all[1]}% / ${all[2]}% |`);
}
official.push('','| 完整日志对照 | 与仓库参考文本比较 |','|---|---|');
for(const x of r.compared)official.push(`| ${x.file} | ${x.equalIgnoringTimingsAndLineEndings?'一致（只忽略换行和运行耗时）':'不一致，须查看原始日志'} |`);
official.push('','小球、容量、化学、概念形成、神经注意力及两个 demo 均有独立 stdout 和退出码。日志对照是本次新执行结果与该提交参考文件的比较，不是只阅读参考文件。扩展模式的实验数不同；更新模式结晶新概念不自动证明其预测精度更高。');
official.push('',
'| 配套实验 | 本机细粒度 / 粗粒度或主要结果 |',
'|---|---|',
'| 小球完整模型 | 68.1% / 88.9%；无查询学习为 86.8% / 100.0% |',
'| 小球断侧重 / 断否决 / 双断 | 70.8% / 83.3%；72.9% / 100.0%；70.8% / 83.3% |',
'| 容量，两种结构 E=8/16/24/32/48/64 | 已教样本全部 100%；E=48 留出 mod-8 93.8%、平滑 96.9%；E=64 留出无样本 |',
'| 化学 v1 完整模型 | 124/192=64.6% / 84.4%；无查询学习 123/192=64.1% / 96.4% |',
'| 化学 v2 完整模型 | 153/192=79.7% / 97.4%；无查询学习 126/192=65.6% / 94.8% |',
'| 概念形成 | 53/60=88.3% / 100.0%；歧义 0/60；规则 26 个 |',
'| 神经注意力 | t60 记录的准确率为 90%；不是整段平均准确率 |',
'| 静态 demo | 本例约束枚举最小 E=1.00；另一局部退火例 C 内最小 E=2.80，五种子命中 |',
'| 序列 demo | 有序 A→B→C 为 9/10；seed 3 出现 A→B、A→C |',
'',
'配套 runner 保留自己的内部种子：小球和化学汇总 1/2/3，容量汇总 1/2，不能把启动环境中的 SEEDS=1 误读为这些汇总只跑了一个种子。小球完整模型落后于无查询学习，说明开启更多机制不保证更好；化学 v2 的细粒度则从查询学习中获益。这些有限配置不能单独证明 N02 是所有性能差异的原因。');

const independent=['| 模型/种子/探针 | n | 模型粗粒度 | 模型严格细粒度 | 最近邻粗粒度 | 最近邻严格细粒度 | 模型未回答亮灭 |','|---|---:|---:|---:|---:|---:|---:|'];
for(const [key,s] of Object.entries(r.predictions)){
 for(const [part,label] of [['all-grid','原网格'],['all-unseen-grid','全部未观察网格'],['off-grid-uniform','同量程随机连续'],['gate-boundary','门限邻域']]){
  const p=s.partitions[part];if(!p||!p.n)continue;
  independent.push(`| ${key.replace('independent-','旧探针 ').replace('fresh-','新探针 ')} / ${label} | ${p.n} | ${fmt(p.model.coarse,p.n)} | ${fmt(p.model.strictFine,p.n)} | ${fmt(p.nearest.coarse,p.n)} | ${fmt(p.nearest.strictFine,p.n)} | ${p.refused} |`);
 }
}
independent.push('','“旧探针”表示沿用上轮探针数据，模型本身是本次新训练。种子 5/7 只测随机连续与门限探针，没有再测完整原网格；n=0 不是 0% 正确率。最近邻和被测模型使用相同观察，但新旧提交的自主探索可能获得不同数据，不能把它当作锁定训练样本的单因素干预。');
independent.push('','| 训练与动力学审计 | 实验 / 注册核 / 分配核 | 非固定返回 / 全部调用 | 能量轨迹失配 | 静息回退仍有驱动介入 |','|---|---:|---:|---:|---:|');
for(const [key,s] of Object.entries(r.predictions)){
 const d=s.dynamics;independent.push(`| ${key} | ${s.experiments} / ${s.registeredCores} / ${s.allocatedCores} | ${d.nonFixedReturns}/${d.calls} | ${d.energyTraceMismatch}/${d.calls} | ${d.quietFallbackEngaged===undefined?'未记录':d.quietFallbackEngaged} |`);
}
independent.push('','上表动力学调用涵盖训练和评分；比较最终能量与增量轨迹的误差阈值为 1e−7，按含 DI 的有效场重数残余可接受翻转，阈值为 −1e−4。这是动力学固定点检查，不是保守能量全局最优检查。静息回退列仅统计 `fallbackQuietOnly=true` 且预算终止的返回状态，按引擎自身相同的驱动介入定义检查。它不是所有预测的错误率。');
const totalDynamics=Object.values(r.predictions).reduce((a,s)=>({calls:a.calls+s.dynamics.calls,energy:a.energy+s.dynamics.energyTraceMismatch,nonFixed:a.nonFixed+s.dynamics.nonFixedReturns}),{calls:0,energy:0,nonFixed:0});
independent.push(`\n五组独立运行合计：${totalDynamics.calls} 次动力学调用，能量账本失配 ${totalDynamics.energy} 次，非固定返回 ${totalDynamics.nonFixed} 次（${(100*totalDynamics.nonFixed/totalDynamics.calls).toFixed(1)}%）。新 harness 中显式 converged 标志与独立残余翻转统计一致；这确认了“状态已如实报告”的修复，同时显示了上层仍需处理它。`);
for(const key of ['fresh-continuous-6','fresh-continuous-7']){const d=r.predictions[key].dynamics;assert.equal(d.reportedNonConverged,d.nonFixedReturns);assert.equal(d.convergenceFlagMismatch,0);}
const uniform=Object.entries(r.predictions).filter(([,s])=>s.partitions['off-grid-uniform']).map(([key,s])=>({key,p:s.partitions['off-grid-uniform']}));
independent.push(`\n四组随机连续探针中，被测模型粗粒度相对同观察最近邻 ${uniform.filter(({p})=>p.model.coarse>p.nearest.coarse).length} 组更高、${uniform.filter(({p})=>p.model.coarse<p.nearest.coarse).length} 组更低。这里报告观察到的计数，不把小差距解释为统计显著的优劣；现有结果未证明稳定超越最近邻。`);
for(const [key,s] of Object.entries(r.predictions)){
 const p=s.partitions['off-grid-uniform'];if(!p)continue;
 independent.push(`\n${key}：回答覆盖率 ${fmt(p.answered,p.n)}；已回答部分粗粒度正确率 ${p.answered?fmt(p.model.coarse,p.answered):'N/A'}；把拒答计错的总准确率 ${fmt(p.model.coarse,p.n)}。该探针集的事后最优常数基线为 ${fmt(p.majorityConstantCorrect,p.n)}（仅用于显示类别不平衡，不是额外训练出的预测器）。真实因素识别结果为 [${s.influential.join(', ')}]。`);
}
const p4=r.predictions['independent-continuous-4'].partitions['off-grid-uniform'];
independent.push(`\n同一旧随机探针上，seed 4 的粗粒度由上一轮 16/128=12.5% 变为 ${fmt(p4.model.coarse,p4.n)}。上轮同观察基线为 104/128=81.3%；本轮最近邻为 ${fmt(p4.nearest.coarse,p4.n)}。本模型的改善应明确承认，同时应以新种子、新探针和新的学习序列检验修复是否稳健。`);
const p4all=r.predictions['independent-continuous-4'];
independent.push(`\n修复说明对这份旧 harness 的记录为 79.7%、734 次动力学调用、原网格最近邻 93.3%；本机在所审提交得到 ${fmt(p4.model.coarse,p4.n)}、${p4all.dynamics.calls} 次调用、原网格最近邻 ${fmt(p4all.partitions['all-grid'].nearest.coarse,432)}。差异幅度不大，不推翻改善结论，但“同一固定脚本结果完全绑定最终提交”的证据仍需作者核对。这些差异不是用不同探针种子解释的：此组沿用了原脚本。`);
independent.push('\n门限探针固定 resistance=1.2、temperature=1、material=1，组合 11 个 switchPos（0.35 至 0.65，含 0.499/0.5/0.501）与 7 个 voltage（0.8 至 1.2，含 0.999/1/1.001）。它故意集中于训练分辨率最难区分的范围，不能解释为整个输入空间的平均性能；但可检验“0.5 概念中心”是否足以支持精确门限附近的判断。');
for(const [key,s] of Object.entries(r.predictions)){
 const p=s.partitions['gate-boundary'];if(!p)continue;
 independent.push(`\n${key} 的门限邻域：真实亮灯 ${p.truthPositive}/${p.n}，预测亮灯 ${p.predictedPositive}/${p.n}，拒答 ${p.refused}/${p.n}；总粗粒度 ${fmt(p.model.coarse,p.n)}，事后多数常数基线 ${fmt(p.majorityConstantCorrect,p.n)}。`);
}
const train6=JSON.parse(read('fresh-continuous-6-training.json'));independent.push(`\n新 seed 6 的训练阶段单独统计：${train6.experiments} 次实验、${train6.dynamics.calls} 次预测，${train6.dynamics.nonFixedReturns} 次非固定返回，${train6.dynamics.quietFallbackEngaged} 次静息态回退仍有驱动介入，保守能量账本失配 ${train6.dynamics.energyTraceMismatch} 次。`);
const synopsis=`本轮完成官方离散/连续三个种子、更新/冻结扩展、六个配套入口，以及五组独立训练与评分；逐条复核 **${r.checkedQueries.toLocaleString('en-US')} 条预测记录**，对应 **${queries.distinctConditionsAcrossModes.toLocaleString('en-US')} 个不同输入条件**（区分离散/连续；含跨种子复测，不宣称这些样本相互统计独立）。固定旧随机探针上的连续 seed 4 粗粒度从上轮 **12.5%** 提升到 **${(100*p4.model.coarse/p4.n).toFixed(1)}%**。完整数值、覆盖率、基线与非收敛统计见第 5–6 节。预测提升与学习序列正确性是不同验收项，不能互相替代。`;
let doc=read('REVIEW-REPORT.zh-CN.md');for(const [marker,text] of [['RESULTS_SYNOPSIS',synopsis],['OFFICIAL_RESULTS',official.join('\n')],['INDEPENDENT_RESULTS',independent.join('\n')]]){assert.ok(doc.includes(`<!-- ${marker} -->`));doc=doc.replace(`<!-- ${marker} -->`,text);}
doc=doc.replace(/`((?:src|test)\/[a-zA-Z0-9/_.-]+\.ts):(\d+)`/g,(_,file,line)=>{assert.ok(Number(line)<=fs.readFileSync(path.join(root,file),'utf8').split('\n').length);return `[${file}:${line}](https://github.com/vasily-tolkyov/energy-network-sim/blob/${r.commit}/${file}#L${line})`;});
assert.ok(!/<!--\s*[A-Z_]+\s*-->/.test(doc));fs.writeFileSync(path.join(base,'REVIEW-REPORT.zh-CN.md'),doc);
fs.writeFileSync(path.join(base,'evidence-check.json'),JSON.stringify({checkedQueries:r.checkedQueries,pending:[],reportHasPlaceholder:false,legacy:r.legacy,revision:r.revision,oldControl:r.oldControl,r2OldControl:r.r2OldControl},null,2));
console.log('Completed report and evidence validation.');
