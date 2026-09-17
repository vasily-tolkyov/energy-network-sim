import { mulberry32 } from "./prng.js";
import { resolveConfig } from "./types.js";
import type {
  AnnealOptions,
  AnnealResult,
  EnergyLedger,
  NetworkConfig,
  NetworkConfigInput,
  SettleResult,
  WellMembership,
} from "./types.js";

/**
 * 能耗神经元网络。
 *
 * 模式能量函数（要求 6、7）：
 *   E(s) = (Ea+Em)·Σ_i s_i − Σ_{i<j} W_ij·s_i·s_j
 * 每个激活神经元付出启动+维持成本；连接越强，负耦合项把总能能耗压得越低。
 *
 * 极小能耗的求解（要求 10）不做任何配置枚举，只执行局部相互作用规律：
 * 神经元 i 的局部场 h_i = Σ_j W_ij·s_j，翻转仅当其使 ΔE < 0。
 * 因 W 对称，每次被接受的翻转严格降低 E；状态空间有限 ⇒ 有限步收敛到
 * 局部极小。这与连通器只靠局部压强规律收敛到最低重力势能同理。
 */
export class EnergyNetwork {
  readonly config: NetworkConfig;
  /** 对称权重矩阵，行优先展开，长度 N*N，对角线恒为 0 */
  private readonly weights: Float64Array;
  /**
   * 有向通道矩阵 D，行优先展开，D[from*N+to] 表示 from→to 的有向强度。
   * 与对称 W 是两种机制：W 由共激活（赫布）建立、承载关联与势阱；
   * D 由时间先后（前模式→后模式）建立、承载势阱间的转移通道。
   * D 不参与 settle 的能量函数——捕获保持对称动力学；
   * D 只在转移相作为非平衡驱动出现（类比材料实现的供能端口）。
   */
  private readonly directed: Float64Array;
  /**
   * 抑制矩阵 Γ，对称、非负，Γ_ij 越大神经元 i、j 越难同时激活。
   * 能量函数中的抑制项为 +Σ_{i<j} Γ_ij·s_i·s_j（同时激活付出额外能耗），
   * 对称性保持 ⇒ settle/退火的 Lyapunov 收敛论证不受影响。
   * 用途：竞争——同一前驱的候选后继之间互相抑制，噪声下先点燃者
   * 抬高对方势垒，实现"小球只进一个槽"的互斥概率选择。
   */
  private readonly inhibitory: Float64Array;
  /**
   * 有向抑制矩阵 DI，非对称、非负：DI[from*N+to] 表示 from→to 的前馈抑制。
   * 与 Γ 的本质区别：DI 是**非平衡驱动场**（供能电路，类比材料实现的
   * 供能比较器与生理前馈抑制），只进翻转规律的场、不进 energy()——
   * 静态能量耦合项必然对称惩罚两端，前馈抑制只能以驱动形式存在。
   * 用途：全局抑制池等广播电路——池压制核，而不被核反向压制。
   */
  private readonly directedInhibitory: Float64Array;
  /** 当前状态：0=静息，1=激活 */
  private readonly state: Uint8Array;
  private ledgerActivationCost = 0;
  private ledgerMaintenanceCost = 0;
  private ledgerActivationCount = 0;

  constructor(configInput: NetworkConfigInput) {
    this.config = resolveConfig(configInput);
    const n = this.config.neuronCount;
    this.weights = new Float64Array(n * n);
    this.directed = new Float64Array(n * n);
    this.inhibitory = new Float64Array(n * n);
    this.directedInhibitory = new Float64Array(n * n);
    this.state = new Uint8Array(n);
  }

  get neuronCount(): number {
    return this.config.neuronCount;
  }

  /** 模式选择阈值：从静息启动一个神经元，场强须覆盖启动+单位维持成本 */
  get threshold(): number {
    return this.config.activationEnergy + this.config.maintenanceEnergy;
  }

  getWeight(i: number, j: number): number {
    const n = this.neuronCount;
    return this.weights[i * n + j] ?? 0;
  }

  /** 对称地增加连接强度：只增不减，封顶 cap（默认 maxWeight）——已达 cap 的边不动 */
  strengthen(i: number, j: number, delta: number, cap?: number): void {
    if (i === j || delta <= 0) return;
    const n = this.neuronCount;
    const limit = cap ?? this.config.maxWeight;
    const cur = this.weights[i * n + j] ?? 0;
    if (cur >= limit) return;
    const next = Math.min(limit, cur + delta);
    this.weights[i * n + j] = next;
    this.weights[j * n + i] = next;
  }

  getDirectedWeight(from: number, to: number): number {
    const n = this.neuronCount;
    return this.directed[from * n + to] ?? 0;
  }

  /** 单向地增加有向通道强度并裁剪到 [0, maxDirectedWeight]（时序学习用） */
  strengthenDirected(from: number, to: number, delta: number): void {
    if (from === to || delta <= 0) return;
    const n = this.neuronCount;
    const idx = from * n + to;
    this.directed[idx] = Math.min(this.config.maxDirectedWeight, (this.directed[idx] ?? 0) + delta);
  }

  /** 有向场 g_i = Σ_j D_{j→i}·s_j：当前激活模式经由有向通道对 i 的驱动 */
  directedField(i: number, pattern: Uint8Array): number {
    const n = this.neuronCount;
    let g = 0;
    for (let j = 0; j < n; j++) {
      if (pattern[j] === 1) g += this.directed[j * n + i] ?? 0;
    }
    return g;
  }

  getInhibitoryWeight(i: number, j: number): number {
    const n = this.neuronCount;
    return this.inhibitory[i * n + j] ?? 0;
  }

  /** 对称地增加抑制强度并裁剪到 [0, maxWeight]（竞争学习用） */
  strengthenInhibitory(i: number, j: number, delta: number): void {
    if (i === j || delta <= 0) return;
    const n = this.neuronCount;
    const next = Math.min(this.config.maxWeight, (this.inhibitory[i * n + j] ?? 0) + delta);
    this.inhibitory[i * n + j] = next;
    this.inhibitory[j * n + i] = next;
  }

  /** 抑制场 γ_i = Σ_j Γ_ij·s_j：当前激活模式对激活 i 的惩罚 */
  inhibitoryField(i: number, pattern: Uint8Array): number {
    const n = this.neuronCount;
    let g = 0;
    for (let j = 0; j < n; j++) {
      if (pattern[j] === 1) g += this.inhibitory[i * n + j] ?? 0;
    }
    return g;
  }

  getDirectedInhibitoryWeight(from: number, to: number): number {
    const n = this.neuronCount;
    return this.directedInhibitory[from * n + to] ?? 0;
  }

  /** 单向地增加前馈抑制强度并裁剪到 [0, cap]（默认 maxDirectedWeight） */
  strengthenDirectedInhibitory(from: number, to: number, delta: number, cap?: number): void {
    if (from === to || delta <= 0) return;
    const n = this.neuronCount;
    const idx = from * n + to;
    const limit = cap ?? this.config.maxDirectedWeight;
    this.directedInhibitory[idx] = Math.min(limit, (this.directedInhibitory[idx] ?? 0) + delta);
  }

  /** 前馈抑制场：Σ_j DI_{j→i}·s_j——非平衡驱动，广播端不被反向压制 */
  directedInhibitoryField(i: number, pattern: Uint8Array): number {
    const n = this.neuronCount;
    let g = 0;
    for (let j = 0; j < n; j++) {
      if (pattern[j] === 1) g += this.directedInhibitory[j * n + i] ?? 0;
    }
    return g;
  }

  /**
   * 对称部分的有效场 = 吸引场(W) − 抑制场(Γ) − 前馈抑制场(DI)。
   * 翻转能量差：0→1 时 ΔE = θ − effectiveField；1→0 时取反。
   * 注意 DI 是非平衡驱动项：不进 energy()，能量报告只含 W/Γ/θ 部分。
   */
  symmetricField(i: number, pattern: Uint8Array): number {
    return this.localField(i, pattern) - this.inhibitoryField(i, pattern) - this.directedInhibitoryField(i, pattern);
  }

  isActive(i: number): boolean {
    return this.state[i] === 1;
  }

  /** 当前激活的神经元编号 */
  activeNeurons(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.state.length; i++) {
      if (this.state[i] === 1) out.push(i);
    }
    return out;
  }

  /** 全部回到静息（静息不耗能，要求 1） */
  reset(): void {
    this.state.fill(0);
  }

  /** 给定模式的局部场 h_i = Σ_j W_ij·s_j */
  localField(i: number, pattern: Uint8Array): number {
    const n = this.neuronCount;
    let h = 0;
    for (let j = 0; j < n; j++) {
      if (pattern[j] === 1) h += this.weights[i * n + j] ?? 0;
    }
    return h;
  }

  /** 模式总能量 E(s) = (Ea+Em)·Σ s_i − Σ_{i<j} W_ij·s_i·s_j + Σ_{i<j} Γ_ij·s_i·s_j */
  energy(pattern?: Uint8Array): number {
    const s = pattern ?? this.state;
    const n = this.neuronCount;
    const theta = this.threshold;
    let e = 0;
    for (let i = 0; i < n; i++) {
      if (s[i] !== 1) continue;
      e += theta;
      for (let j = i + 1; j < n; j++) {
        if (s[j] === 1) {
          e -= this.weights[i * n + j] ?? 0;
          e += this.inhibitory[i * n + j] ?? 0;
        }
      }
    }
    return e;
  }

  /**
   * 新输入到来（要求 5、7、10）：
   * - inputNeurons 中的神经元被钳制为激活（输入决定必须激活哪些）；
   * - 其余神经元从静息出发，仅按局部场规律翻转（ΔE<0 才执行）；
   * - 收敛后即为满足输入约束的能耗极小激活模式。
   * 全程确定性（固定扫描顺序），无随机数、无枚举。
   */
  settle(inputNeurons: Iterable<number>): SettleResult {
    const n = this.neuronCount;
    const clamped = new Uint8Array(n);
    for (const i of inputNeurons) {
      if (i < 0 || i >= n) throw new Error(`input neuron index out of range: ${i}`);
      clamped[i] = 1;
    }
    // 从静息出发：被钳制的激活，其余静息
    this.state.fill(0);
    for (let i = 0; i < n; i++) if (clamped[i] === 1) this.state[i] = 1;

    const theta = this.threshold;
    const energies: number[] = [this.energy()];
    let currentEnergy = energies[0]!;
    let flipCount = 0;
    // 浮点尘埃保护：能量差小于 EPS 的翻转会引发近平局无限循环（实测挂死），
    // 要求每次翻转都有实质降幅；另有总翻转数上限与最优回退双保险。
    const EPS = 1e-4;
    const maxFlips = 100 * n;
    let bestEnergy = currentEnergy;
    const bestState = Uint8Array.from(this.state);
    // 异步贪心扫描：只要某次完整扫描没有任何翻转，即为局部极小
    for (;;) {
      let flippedThisSweep = false;
      for (let i = 0; i < n; i++) {
        if (clamped[i] === 1) continue;
        const h = this.symmetricField(i, this.state);
        // 0→1: ΔE = theta − h；1→0: ΔE = −(theta − h)；仅在 ΔE<0 时翻转
        const dE = this.state[i] === 0 ? theta - h : -(theta - h);
        if (dE >= -EPS) continue;
        this.state[i] = this.state[i] === 0 ? 1 : 0;
        flipCount++;
        flippedThisSweep = true;
        currentEnergy += dE; // 增量记账，避免每翻转一次就 O(N²) 重算能量
        energies.push(currentEnergy);
        if (currentEnergy < bestEnergy) {
          bestEnergy = currentEnergy;
          bestState.set(this.state);
        }
      }
      if (!flippedThisSweep) break;
      if (flipCount > maxFlips) {
        // 非收敛终止：回退到途中最低能态（如实记录）
        this.state.set(bestState);
        energies.push(bestEnergy);
        break;
      }
    }
    return {
      activeNeurons: this.activeNeurons(),
      energy: this.energy(),
      trace: { energies, flipCount },
    };
  }

  /**
   * 局部退火模式选择（语义：只在"输入所及的范围"内求能耗极小）。
   *
   * 单一温度曲线，无独立贪心相（贪心只是本规律的 T=0 特例）：
   *   1. 从静息 + 钳制输入出发；
   *   2. 构造候选集 C = 输入 ∪ { 与输入有公共成员的势阱 }，C 之外的
   *      神经元结构性冻结在静息——不相干劲阱不是"大概率不被点燃"，
   *      而是根本没有翻转它的路径，读出选择性是硬保证；
   *   3. Metropolis 退火：T 从 initialTemperature 几何降温，仅翻转
   *      C\I 中的神经元，钳制神经元不动；带最优状态记忆（跟踪全程
   *      访问过的最低能状态，退火末恢复），消除冷却时停在浅盆地的
   *      运气成分；退火只用于翻越 C 内的协同势垒（单翻上坡、联合下坡）；
   *   4. 淬火尾巴：T=0 的全网贪心扫描至无翻转——既保证返回模式是
   *      局部极小，又顺手招募 C 外被场强直接推过阈值的普通连接神经元
   *      （无势垒），能量在此段重新严格单调下降。
   */
  settleAnnealed(
    inputNeurons: Iterable<number>,
    wells: readonly WellMembership[],
    options: AnnealOptions = {},
  ): AnnealResult {
    const n = this.neuronCount;
    const clamped = new Set<number>();
    for (const i of inputNeurons) {
      if (i < 0 || i >= n) throw new Error(`input neuron index out of range: ${i}`);
      clamped.add(i);
    }

    // 从静息 + 钳制出发（无独立贪心相）
    this.state.fill(0);
    for (const i of clamped) this.state[i] = 1;
    const initialEnergy = this.energy();

    // 候选集 C = I ∪ 被触及势阱
    const candidate = new Set<number>(clamped);
    for (const well of wells) {
      if (well.memberNeuronIds.some((id) => clamped.has(id))) {
        for (const id of well.memberNeuronIds) {
          if (id >= 0 && id < n) candidate.add(id);
        }
      }
    }
    if (options.extraCandidates) {
      for (const id of options.extraCandidates) {
        if (id >= 0 && id < n) candidate.add(id);
      }
    }
    const freeCandidates = [...candidate].filter((id) => !clamped.has(id)).sort((a, b) => a - b);

    // Metropolis 退火（仅 C\I）
    const theta = this.threshold;
    const coolingFactor = options.coolingFactor ?? 0.7;
    const levels = options.levels ?? 20;
    const sweepsPerLevel = options.sweepsPerLevel ?? 30;
    if (!(coolingFactor > 0 && coolingFactor < 1)) {
      throw new Error(`coolingFactor must be in (0, 1), got ${coolingFactor}`);
    }
    if (levels < 1 || sweepsPerLevel < 1) {
      throw new Error(`levels and sweepsPerLevel must be >= 1`);
    }
    let temperature = options.initialTemperature ?? theta;
    if (temperature <= 0) {
      throw new Error(`initialTemperature must be > 0, got ${temperature}`);
    }
    const rand = mulberry32(options.seed ?? 1);
    let proposals = 0;
    let acceptedUphill = 0;
    const flipDelta = (i: number): number => {
      const h = this.symmetricField(i, this.state);
      // 0→1: ΔE = theta − h；1→0: ΔE = −(theta − h)
      return this.state[i] === 0 ? theta - h : -(theta - h);
    };
    let currentEnergy = initialEnergy;
    let bestEnergy = initialEnergy;
    const bestState = Uint8Array.from(this.state);
    for (let level = 0; level < levels; level++, temperature *= coolingFactor) {
      for (let sweep = 0; sweep < sweepsPerLevel; sweep++) {
        // 每轮洗牌提议顺序（可复现种子）：打破固定下标顺序的点火偏置，
        // 为竞速动态（先点燃者压制其余）提供公平的时间结构
        for (let k = freeCandidates.length - 1; k > 0; k--) {
          const j = Math.floor(rand() * (k + 1));
          const tmp = freeCandidates[k]!;
          freeCandidates[k] = freeCandidates[j]!;
          freeCandidates[j] = tmp;
        }
        for (const i of freeCandidates) {
          const dE = flipDelta(i);
          proposals++;
          if (dE < 0) {
            this.state[i] = this.state[i] === 0 ? 1 : 0;
          } else if (rand() < Math.exp(-dE / temperature)) {
            this.state[i] = this.state[i] === 0 ? 1 : 0;
            acceptedUphill++;
          } else {
            continue;
          }
          currentEnergy += dE;
          if (currentEnergy < bestEnergy) {
            bestEnergy = currentEnergy;
            bestState.set(this.state);
          }
        }
      }
    }
    if (bestEnergy < currentEnergy) this.state.set(bestState);
    const annealEndEnergy = this.energy();

    // 淬火尾巴：T=0 贪心扫描至无翻转，能量严格单调下降（增量记账）。
    // 范围默认全网（招募无势垒的普通连接神经元）；读出场景限制在候选集内，
    // 防止对称 W 把结果活动反传回未查询条件档引发雪崩。
    // 引入非平衡驱动（DI 前馈抑制）后可能出现弛豫振荡（压制-熄火-复燃循环），
    // 因此淬火改为有界迭代 + 最优回退：记录途中最低能态，超限则回退到它。
    const quenchEnergies: number[] = [];
    let quenchEnergy = annealEndEnergy;
    let quenchBestEnergy = annealEndEnergy;
    const quenchBestState = Uint8Array.from(this.state);
    const quenchScope = options.quenchCandidatesOnly
      ? freeCandidates
      : Array.from({ length: n }, (_, i) => i);
    const maxFlips = 100 * n;
    const EPS = 1e-4; // 能量分辨率下限：近平局的尘埃翻转既慢又无意义（实测 >1e5 步仍不收敛）
    for (;;) {
      let flipped = false;
      for (const i of quenchScope) {
        if (clamped.has(i)) continue;
        const dE = flipDelta(i);
        if (dE < -EPS) {
          this.state[i] = this.state[i] === 0 ? 1 : 0;
          quenchEnergy += dE;
          quenchEnergies.push(quenchEnergy);
          if (quenchEnergy < quenchBestEnergy) {
            quenchBestEnergy = quenchEnergy;
            quenchBestState.set(this.state);
          }
          flipped = true;
        }
      }
      if (!flipped) break;
      if (quenchEnergies.length > maxFlips) {
        // 驱动振荡：回退到途中最低能态（如实记录为非收敛终止）
        this.state.set(quenchBestState);
        quenchEnergies.push(quenchBestEnergy);
        break;
      }
    }

    const energies = [annealEndEnergy, ...quenchEnergies];
    return {
      activeNeurons: this.activeNeurons(),
      energy: this.energy(),
      trace: { energies, flipCount: quenchEnergies.length },
      candidateSet: [...candidate].sort((a, b) => a - b),
      initialEnergy,
      annealEndEnergy,
      proposals,
      acceptedUphill,
    };
  }

  /**
   * 时序记账（要求 1–4）：把网络从当前状态推进到 nextActive 指定的激活集合，
   * - 静息 → 静息：0 耗能（要求 1）
   * - 静息 → 激活：收一次 Ea（要求 2）
   * - 激活 → 激活（维持一步）：收 Em（要求 3）
   * - 激活 → 静息：本步不计费
   */
  runStep(nextActive: Iterable<number>): EnergyLedger {
    const n = this.neuronCount;
    const next = new Uint8Array(n);
    for (const i of nextActive) {
      if (i < 0 || i >= n) throw new Error(`neuron index out of range: ${i}`);
      next[i] = 1;
    }
    for (let i = 0; i < n; i++) {
      const was = this.state[i] === 1;
      const will = next[i] === 1;
      if (!was && will) {
        this.ledgerActivationCost += this.config.activationEnergy;
        this.ledgerActivationCount++;
      } else if (was && will) {
        this.ledgerMaintenanceCost += this.config.maintenanceEnergy;
      }
    }
    this.state.set(next);
    return this.ledger();
  }

  ledger(): EnergyLedger {
    return {
      activationCost: this.ledgerActivationCost,
      maintenanceCost: this.ledgerMaintenanceCost,
      activationCount: this.ledgerActivationCount,
    };
  }

  resetLedger(): void {
    this.ledgerActivationCost = 0;
    this.ledgerMaintenanceCost = 0;
    this.ledgerActivationCount = 0;
  }
}
