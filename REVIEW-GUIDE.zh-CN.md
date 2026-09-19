# 第三方复审与复现指南

本轮基线为 `e732f2b4ef46e0ac482480095a31cb5aa23ec6be`。当前验收结论、测试数与完整实测表仅以 [第二轮修复记录](docs/REVIEW2-REMEDIATION.zh-CN.md) 为准；原指南的旧数字和已撤回契约保存在 [基线快照](docs/history/e732f2b-REVIEW-GUIDE.zh-CN.md)。

## 工具链与永久测试

```bash
node --version           # >=24
npm ci
npm run verify           # tsc + node --test dist/test/*.test.js
node --test dist/test/review2-*.test.js
node dist/src/pop/runner-ablation.js
```

新增测试包括证据优先级五序列、同编码/重叠支持区、联合捕获和快照、静息资格、R2 正反对称、非法输入/原子写入/生命周期、任一输出拒答、防御性拷贝、点火可行域与五个独立世界。原断言与撤回条目均在复验记录逐项解释，不能把旧脚本失败总数等同当前实现缺陷数。

## 原样评审

评审甲的原始脚本和全局最优反例 fixture 位于仓库根 `review-artifacts/`：

```bash
node --test review-artifacts/adversarial.test.mjs
node --test review-artifacts/revision-tests.mjs
node review-artifacts/run-group.mjs discrete
node review-artifacts/run-group.mjs continuous
node review-artifacts/run-group.mjs extension
node review-artifacts/run-group.mjs supporting
node review-artifacts/run-group.mjs independent
node review-artifacts/run-group.mjs fresh
```

先创建 `review-artifacts/reproduction/`。这些原脚本的元数据内有写死的基线 SHA；实际受测源码身份以本轮 `runs/review2/source-manifest.json` 为准，不能把旧字段当当前提交。

评审乙按其原路径要求，建立隔离目录，将完整 checkout 命名为 `repo`，与原始 `tests/`、`prior-tests/`、`results/` 平级。原脚本副本在 `review-artifacts/reviewer-b/`；可使用目录联接而不复制当前工作区。`before-after-regression.mjs` 还需要同级 `baseline-f25fefe` 的已构建旧版。分别运行本轮报告列出的 revision、before-after、nearby、new-worlds/verdict、phase-a/phase-b 以及 prior-tests；完整 stdout、JSON 和源码哈希保留在本轮证据目录。

原 N28 会在已纠错的同一模型上继续断言旧错误预测。永久用例改为每种子独立模型，并增加反馈后读出正确值的断言；原脚本仍原样执行与保留失败，不冒写 29/29。另存 `review2-regressions-isolated.test.mjs`，只将 N28 fixture 移入每个种子并增加纠错后正确性断言；修正框架版实测 29/29，和原版分开记账。

在该隔离目录逐项执行（不要把 `before-after` 的打印输出当作它没有提供的断言通过数）：

```bash
node --test tests/review2-regressions.test.mjs
node --test tests/review2-regressions-isolated.test.mjs
node tests/before-after-regression.mjs
node tests/continuous-nearby-world.mjs
node tests/new-worlds.mjs
node --test tests/new-worlds-verdict.test.mjs
node tests/phase-a-independent.mjs
node tests/phase-b-independent.mjs
node --test prior-tests/*.test.mjs
```

旧版源码需在同一 Node 环境执行 `npm ci && npm run build`。评审甲 R19 老版对照另按原脚本路径提供同级 `energy-network-sim-f25fefe`，完整源码可用 `git worktree add` 从 `f25fefe` 建立。

甲的 `independent-experiments.mjs` 原版缺少正式收敛计数。并排的 `independent-diagnostics.mjs` 仅改变落盘前缀；用 `node --import ./scripts/review2-diagnostics-hook.mjs review-artifacts/independent-diagnostics.mjs continuous 4` 补记退出时的只读计数，环境变量 `REVIEW_DIAGNOSTICS_PATH` 指定 JSON 路径。同样运行 discrete 4、continuous 5 off-grid-only。汇总器断言补跑的全部逐条预测与原样运行完全一致，再合并动力学指标；原样文件不覆盖。

## 判读原则

1. 能量账本只检验保守项；DI 不能混入 energy。静息可行域下返回答案不等于无约束固定点。
2. within-envelope 进入置信连续验证必须收敛；评估答案正确率与动力学质量分开。
3. 覆盖分区是已观察、未观察灭灯、未观察亮灯，后两者并集必须覆盖全部未观察输入；n=0 为 N/A。
4. `quorum-met` 是启发式停止。标准报告包含假设族、联合覆盖风险与稀有阳性召回；W04 不能用改阈值或专用探针消除。
5. 原型无 LLM/回归器，不等于无工程先验；编码、动作网格、配对、竞争电路和读出都是具体设计。
6. 多个实验可能并行，耗时只用于复现诊断，不作为隔离性能或能效证据。

汇总已有输出（不重新学习或改分数）：

```bash
node scripts/review2-source-manifest.mjs
node scripts/normalize-review2-logs.mjs
node scripts/summarize-review2.mjs
```

`runs/review2/summary.json` 为结构化指标索引；汇总器要求 14 个分组条目全部退出 0、冻结日志与完整容量复跑存在，防止把中间状态当完成。原始 stdout 与派生展示之间的变化由 log-rendering-provenance.json 逐文件记录。求解器纯实现优化可用 `node scripts/check-solver-equivalence.mjs <旧 checkout 的 dist/src/network.js>` 重做完整返回值严格比较。早期以 .tap 命名的部分阶段输出实际为 Node spec 格式；本轮原评审和 66 项定向集均明确采用 TAP，完整 verify 保留实际默认日志格式。
