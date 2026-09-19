# 第二轮证据目录

本轮基线 `e732f2b4ef46e0ac482480095a31cb5aa23ec6be`；模型与测试最终源码 `b25eb98`。结论与指标口径见 [修复报告](../../docs/REVIEW2-REMEDIATION.zh-CN.md)。

- `summary.json`：从已保存输出生成的结构化索引，空分母/不适用为 null；不是另一轮训练。
- `source-manifest.json`：最终源码、测试与依赖文件 SHA256；不包含另一任务的 `path-world.ts`。
- `audit-script-integrity.json`：27 份原评审脚本与交接材料逐字节一致。
- `verify-final.log` / `.exit`：最终源码 179/179、退出 0。
- `stage4-mechanisms.tap`：66 项新增永久回归的独立 TAP。
- `review-a-revision.tap`：20/20；`review-b-revision-original.tap`：28/29；`review-b-revision-isolated.tap`：29/29。原 N28 共用已纠错模型导致后续种子断言旧值；独立 fixture 版并排保留，不替换原版。
- `review-a-prior.tap`、`review-b-prior.tap`、`new-worlds-verdict.tap`：分别 14/20、40/43、4/5，例外逐条见报告。不能当作全绿证据。
- `reviewer-b/`：评审乙逐条 JSON、before-after、nearby、五世界及 phase-a/phase-b 完整输出。
- `*-instrumented.log` / `*-diagnostics.json`：甲的原脚本诊断补跑；仅改变落盘前缀并读取预测计数。汇总时断言全部预测与原样运行一致。
- `pop-score-lock.json`：固定教学权重比较新旧求解器，并记录四格完整课程边摘要及错误样本。
- `W04-budget-matched-baseline.json`：固定 seed=11、同 17 次预算基线；双方稀有阳性召回均 0/1。
- `log-rendering-provenance.json`：原 stdout 到根目录 `runs/*.log` 的空分母展示修正；只把未定义指标改为 N/A。

原评审甲逐条结果、分组 stdout 和起止元数据在 `../../review-artifacts/`。原 runner 和 before-after 数据中的写死 `e732f2b` 字段是基线标签，实际源码范围见报告说明。部分长实验启动于 `c399c11` 的待提交源码，后续纯实现优化有完整返回值等价检查；注意力、标签等受影响的运行器另行补跑。

## 故意保留的失败和中断

- `initial-failed-run/`：最初严格回退导致 W01 0/16、局部淬火方案 3/16 的失败。
- `pre-corpus-fix-interrupted/`：语料形成与置信验证循环依赖时的未完成运行。
- `verify.log`：首次完整 176/179，三条旧小球数值/消融契约失败。
- `verify-before-attention-close.log`：最终注意力补丁前的一次 179/179。
- `verify.tap`、`verify.exit`：早期/中断阶段遗留输出；不要据此替代 `verify-final.*`。早期 `.tap` 后缀未必代表 TAP reporter。
- `extension-milestone-insufficient-evidence.tap`：功能测试初版未等待足够的扩展归因证据。
- `capacity-control-interrupted.log`、`capacity-interruption.json`：并行时内存不足而中断的 N=8000 资源扫描。
- `review-a-prior-missing-fixture.tap`、`r2-old-control-incomplete-fixture.tap`：第一次复现布局缺文件；修正完整 checkout 后另存结果，不删除原始失败。

所有机器耗时处于共享负载，不能作为隔离性能对比。预测正确也可能非收敛，收敛也可能答错；两项在报告中分开统计。
