# energy-network-sim

二态神经元、赫布连接 W、对称抑制 Γ 与非平衡驱动 DI 的研究原型。经验由条件群体↔规则核、规则核↔各结果通道分段绑定，预测通过有限预算局部退火和读出完成。运行时无 LLM、无回归器；传感维度、动作网格、配对方式、竞争电路和停止规则是明确给定的工程先验。

## 运行

要求 Node.js ≥24。

```bash
npm ci
npm run verify
npm run exp:explore       # 离散电路，默认种子 1/2/3
npm run exp:explore-cont  # 连续电路，默认种子 1/2/3
npm run exp:explore-ext   # 扩展量程：更新与冻结
npm run exp:pop           # 小球与容量
npm run exp:chem
npm run exp:concept
```

完整测试数、逐种子指标、失败记录和原样复审结果集中在 [第二轮修复与复验记录](docs/REVIEW2-REMEDIATION.zh-CN.md)；基准原始日志在 `runs/review2/` 与 `review-artifacts/reproduction/`。历史数字见 [e732f2b README 快照](docs/history/e732f2b-README.md)，不作为当前能力声明。运行器会覆盖同名日志，比较前须另存历史输出。

## 当前契约

- 观察优先于自写假设；冲突假设被记录，不能撤销观察。观察可按代际纠正观察，纠错 Γ 可撤销。
- 编码支持区相交视为相容；完全同编码的观察等权聚合并保留样本数、范围。只对已激活的单峰支持区细化读出，空答案和歧义不会被登记表补全。
- 条件↔核、核↔结果必须分通道写入，只有有结果证据的核进入预测候选；写入前完成校验、编码和容量预检。非法写入整体不变，容量耗尽显式抛错。
- 保守能量 `E=(Ea+Em)Σs−ΣW·ss+ΣΓ·ss` 不含 DI，DI 做功独立记录。Γ 的对称增减仅改变学习后的能量地形。
- `fallbackQuietOnly` 在静息可行域中退火与局部交换搜索，淬火只从合格候选开始。`no-quiet-candidate` 返回空模式；`quiet-constraint` 表示可行域内已停止但域外仍有可翻转项，均不冒称收敛。
- `converged`、`terminationReason` 沿预测、注意力、探索和日志透传。答案正确率、非收敛率、任一输出拒答率分别报告。置信验证要求收敛。
- gain 与 veto 独立：前者控制加分 W，后者控制 R2 已识别维度的替代档 Γ。`node dist/src/pop/runner-ablation.js` 检查四格实际边配置。
- Pop 构造期检查观察封顶后的可达场：`min(0.6,maxWeight)×(coreSize+popSize−1) ≥ θ`，条件和结果两侧均须可行。cap 是绝对边权，不再额外乘 η。

## 能力边界

`quorum-met` 是启发式停止，不是全部因素已发现的证书。逐维覆盖不能证明联合覆盖；W04 的稀有阳性召回及同预算随机探索对照单列报告。固定候选点含 0.5 不证明精确物理门限已被辨识。连续网格外结果未稳定超越同观察最近邻。

概念重形成会重新计算 R2 证据并执行额外 R3 训练 epoch；它不是幂等更新，也不是增加独立观察。更新/冻结模式的训练次数和实验数必须同时比较。

每个网络持有四个密集 N×N Float64 矩阵（约 32N² 字节），另有稀疏的可撤销抑制来源记录。有限预算搜索不保证全局最优，非平衡驱动也不保证固定点收敛。这里没有证明硬件能效、长期无界学习或开放多阶段自主能力。

## 基本示例

```ts
import { EnergyNetwork, hebbianLearn, detectWells, ReadoutModule } from "./src/index.js";
const net = new EnergyNetwork({ neuronCount: 24, activationEnergy: 2, maintenanceEnergy: .5 });
hebbianLearn(net, [0, 1, 2, 3], 8);
hebbianLearn(net, [0, 1], 2); // 严格峰检测需要峰边，等权平台不够
hebbianLearn(net, [12, 13, 14, 15], 8);
hebbianLearn(net, [12, 13], 2);
const wells = detectWells(net);
const readout = new ReadoutModule(wells, net.config.readoutThreshold);
readout.labelWell(wells[0]!.wellId, "A");
console.log(readout.identify(net.settle([0, 1]).activeNeurons));
```

审查与复现入口：[REVIEW-GUIDE.zh-CN.md](REVIEW-GUIDE.zh-CN.md)。
