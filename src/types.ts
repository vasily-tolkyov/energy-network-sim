/** 网络配置。所有能耗单位为抽象能量单位。 */
export interface NetworkConfig {
  /** 神经元数量 */
  readonly neuronCount: number;
  /** 启动能耗：神经元从静息到激活的一次性成本（要求 2） */
  readonly activationEnergy: number;
  /** 维持能耗：激活状态每持续一步的成本（要求 3）。必须满足 Ea > Em（要求 4） */
  readonly maintenanceEnergy: number;
  /** 赫布学习步长 η（要求 6） */
  readonly learningRate: number;
  /** 连接强度上限 */
  readonly maxWeight: number;
  /** 有向通道强度上限（时序学习用，独立于对称连接），默认 1.0 */
  readonly maxDirectedWeight: number;
  /** 势阱邻域比例：边权 ≥ wellRatio × 极大值 的边属于该势阱（要求 8），默认 0.8 */
  readonly wellRatio: number;
  /** 读出激活率阈值 τ：势阱成员中被激活比例 ≥ τ 判为该势阱被激活（要求 9），默认 0.5 */
  readonly readoutThreshold: number;
}

export interface NetworkConfigInput {
  readonly neuronCount: number;
  readonly activationEnergy: number;
  readonly maintenanceEnergy: number;
  readonly learningRate?: number;
  readonly maxWeight?: number;
  readonly maxDirectedWeight?: number;
  readonly wellRatio?: number;
  readonly readoutThreshold?: number;
}

export function resolveConfig(input: NetworkConfigInput): NetworkConfig {
  const config: NetworkConfig = {
    neuronCount: input.neuronCount,
    activationEnergy: input.activationEnergy,
    maintenanceEnergy: input.maintenanceEnergy,
    learningRate: input.learningRate ?? 0.1,
    maxWeight: input.maxWeight ?? 1.0,
    maxDirectedWeight: input.maxDirectedWeight ?? 1.0,
    wellRatio: input.wellRatio ?? 0.8,
    readoutThreshold: input.readoutThreshold ?? 0.5,
  };
  if (!Number.isInteger(config.neuronCount) || config.neuronCount < 2) {
    throw new Error(`neuronCount must be an integer >= 2, got ${config.neuronCount}`);
  }
  // 评审 A06 修复：非有限值（NaN/Infinity）绕过一切比较检查，统一显式拒绝
  for (const [k, v] of Object.entries(config)) {
    if (!Number.isFinite(v)) throw new Error(`${k} must be finite, got ${v}`);
  }
  // 要求 4：Ea > Em，不满足直接拒绝构造。
  if (!(config.activationEnergy > config.maintenanceEnergy)) {
    throw new Error(
      `activationEnergy (Ea=${config.activationEnergy}) must be greater than maintenanceEnergy (Em=${config.maintenanceEnergy})`,
    );
  }
  if (config.maintenanceEnergy < 0) {
    throw new Error(`maintenanceEnergy must be >= 0, got ${config.maintenanceEnergy}`);
  }
  if (config.learningRate <= 0 || config.maxWeight <= 0 || config.maxDirectedWeight <= 0) {
    throw new Error(`learningRate, maxWeight and maxDirectedWeight must be > 0`);
  }
  if (config.wellRatio <= 0 || config.wellRatio > 1) {
    throw new Error(`wellRatio must be in (0, 1], got ${config.wellRatio}`);
  }
  if (config.readoutThreshold <= 0 || config.readoutThreshold > 1) {
    throw new Error(`readoutThreshold must be in (0, 1], got ${config.readoutThreshold}`);
  }
  return config;
}

/** 一次 settle 的能耗轨迹记录（要求 10：验证只凭局部规律单调下降）。 */
export interface SettleTrace {
  /**
   * 每一步翻转后的保守能量账本（≡ energy() 逐点差分累计）。
   * 第三方评审 F01 修复后：账本只含对称部分（θ/W/Γ），DI 非平衡驱动
   * 的做功不再混入（修复前闭环一周账本漂移 −γ，轨迹与真实能量脱钩）。
   */
  readonly energies: readonly number[];
  /** 执行的翻转次数（真实计数，回退写入不计） */
  readonly flipCount: number;
  /** DI 非平衡驱动的累计做功（决策场与保守账本之差；无 DI 时为 0） */
  readonly driveWork: number;
}

/** 终止原因：无翻转可降 = 固定点；翻转预算耗尽 = 非收敛（回退途中最低能态） */
export type SettleTermination = "fixed-point" | "flip-budget" | "no-quiet-candidate";

export interface SettleResult {
  /** 收敛后的激活神经元集合（含被输入钳制的神经元） */
  readonly activeNeurons: readonly number[];
  /** 返回态的真实总能量（θ/W/Γ） */
  readonly energy: number;
  readonly trace: SettleTrace;
  /** 是否以固定点终止（DI 存在时动力学可振荡，不再承诺无条件收敛） */
  readonly converged: boolean;
  readonly terminationReason: SettleTermination;
  /** 返回态在作用域内仍可翻转的神经元数（决策场口径；0 = 真固定点） */
  readonly residualFlips: number;
}

/** 时序记账账本（要求 1–4）：静息不计费，0→1 收 Ea，激活每步收 Em。 */
export interface EnergyLedger {
  /** 累计启动能耗 */
  readonly activationCost: number;
  /** 累计维持能耗 */
  readonly maintenanceCost: number;
  /** 累计启动次数 */
  readonly activationCount: number;
}

/** 局部退火只需要知道势阱的成员集合（结构化类型，兼容 PotentialWell） */
export interface WellMembership {
  readonly memberNeuronIds: readonly number[];
}

/** 局部退火参数。温度尺度默认取 θ=Ea+Em（启动势垒的尺度） */
export interface AnnealOptions {
  /** PRNG 种子，相同种子结果完全可复现 */
  readonly seed?: number;
  /** 初始温度，默认 θ */
  readonly initialTemperature?: number;
  /** 几何降温因子，默认 0.7 */
  readonly coolingFactor?: number;
  /** 温度层数，默认 20 */
  readonly levels?: number;
  /** 每层温度对候选集的扫描轮数，默认 30 */
  readonly sweepsPerLevel?: number;
  /**
   * 额外候选神经元：并入候选集 C（不经过势阱触及判定）。
   * 用于 R3 式读出——查询只含条件神经元时，把结果通道神经元纳入候选。
   */
  readonly extraCandidates?: Iterable<number>;
  /**
   * 淬火范围限制为候选集（默认 false = 全网扫尾）。
   * 读出场景必须为 true：对称 W 会把结果神经元的活动反传回未查询的
   * 条件档，全网扫尾会引起雪崩。
   */
  readonly quenchCandidatesOnly?: boolean;
  /**
   * 淬火尾巴的翻转上限（默认 100×N）。大核数下长尾爬降（每次翻转仅微幅
   * 改善）会把单次读出拖到秒级以上；调用方可按读出精度需求调低——
   * 超限时回退到途中最低能态（如实记录为非收敛终止），语义不变。
   */
  readonly quenchMaxFlips?: number;
  /**
   * 最优回退的选择域（默认 false = 全部访问态，保守系统/无 DI 的正确契约）：
   * true = 只在驱动静息态（无 DI 源活跃或将点燃）中选最低真实能态——
   * 带前馈抑制池的 WTA 电路必须开：多核共存态的真实能量反而更低
   * （耦合更多），普适回退会破坏池的单核语义（第三方评审 F01 修复
   * 引入的显式开关，池使用者（两个规则记忆）传 true）。
   */
  readonly fallbackQuietOnly?: boolean;
}

export interface AnnealResult extends SettleResult {
  /** 候选集 C = 输入 ∪ 被输入触及的势阱成员；C 外神经元全程冻结在静息 */
  readonly candidateSet: readonly number[];
  /** 静息 + 钳制输入的初始能量 */
  readonly initialEnergy: number;
  /** 退火结束（含最优状态恢复）、淬火前的能量 */
  readonly annealEndEnergy: number;
  /** 退火期间的翻转提议总数 */
  readonly proposals: number;
  /** 其中被接受的上坡翻转数（翻越势垒的证据） */
  readonly acceptedUphill: number;
}
