import { integer, positive, nonnegative, neuronId } from "./validate.js";
import { mulberry32 } from "./prng.js";
import { resolveConfig } from "./types.js";
import type {
  AnnealOptions,
  AnnealResult,
  EnergyLedger,
  NetworkConfig,
  NetworkConfigInput,
  SettleResult,
  SettleTermination,
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
  private readonly inhibitionOwners = new Map<number, { base: number; values: Map<string, number> }>();
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

  /** 评审 A05/A06 修复：公开写边入口统一校验——非 [0,N) 整数索引/非有限权值
   *  直接抛错且零部分写入（修复前越界列经一维下标别名到下一行，破坏 W 对称性） */
  private checkEdge(i: number, j: number, value: number): void {
    for (const k of [i, j]) {
      if (!Number.isInteger(k) || k < 0 || k >= this.neuronCount) {
        throw new Error(`neuron index out of range: ${k}`);
      }
    }
    if (!Number.isFinite(value)) throw new Error(`weight delta must be finite, got ${value}`);
  }

  /** 对称地增加连接强度：只增不减，封顶 cap（默认 maxWeight）——已达 cap 的边不动 */
  strengthen(i: number, j: number, delta: number, cap?: number): void {
    this.checkEdge(i, j, delta);
    if (cap !== undefined) nonnegative(cap, "cap");
    if (i === j || delta <= 0) return;
    const n = this.neuronCount;
    const limit = cap ?? this.config.maxWeight;
    const cur = this.weights[i * n + j] ?? 0;
    if (cur >= limit) return;
    const next = Math.min(limit, cur + delta);
    this.weights[i * n + j] = next;
    this.weights[j * n + i] = next;
  }

  /** 绝对写入连接强度（对称）：用于需要"改写"而非"累加"的场（如失配场、惯性场） */
  setWeight(i: number, j: number, value: number): void {
    this.checkEdge(i, j, value);
    if (i === j) return;
    const n = this.neuronCount;
    const v = Math.max(0, Math.min(this.config.maxWeight, value));
    this.weights[i * n + j] = v;
    this.weights[j * n + i] = v;
  }

  getDirectedWeight(from: number, to: number): number {
    const n = this.neuronCount;
    return this.directed[from * n + to] ?? 0;
  }

  /** 单向地增加有向通道强度并裁剪到 [0, maxDirectedWeight]（时序学习用） */
  strengthenDirected(from: number, to: number, delta: number): void {
    this.checkEdge(from, to, delta);
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
    this.checkEdge(i, j, delta);
    if (i === j || delta <= 0) return;
    const n = this.neuronCount;
    const owned = this.inhibitionOwners.get(Math.min(i, j) * n + Math.max(i, j));
    if (owned) {
      owned.base = Math.min(this.config.maxWeight, owned.base + delta);
      this.writeOwnedInhibition(i, j, owned);
      return;
    }
    const next = Math.min(this.config.maxWeight, (this.inhibitory[i * n + j] ?? 0) + delta);
    this.inhibitory[i * n + j] = next;
    this.inhibitory[j * n + i] = next;
  }

  /** Reversible plasticity of symmetric Γ; Γ remains a conservative energy term.
   * Learning changes the landscape between settles, not the energy definition. */
  decayInhibition(i: number, j: number, delta: number): void {
    this.checkEdge(i, j, delta);
    if (delta < 0) throw new Error("inhibition decay must be nonnegative");
    const owned = this.inhibitionOwners.get(Math.min(i, j) * this.neuronCount + Math.max(i, j));
    if (owned) {
      owned.base = Math.max(0, owned.base - delta);
      this.writeOwnedInhibition(i, j, owned);
      return;
    }
    const next = Math.max(0, this.getInhibitoryWeight(i, j) - delta);
    this.inhibitory[i * this.neuronCount + j] = next;
    this.inhibitory[j * this.neuronCount + i] = next;
  }

  /** Independently retractable learning contributions. Saturation must not
   * erase ownership: withdrawing one cause cannot withdraw another cause. */
  setInhibitionContribution(i: number, j: number, owner: string, value: number): void {
    this.checkEdge(i, j, value);
    if (value < 0) throw new Error("inhibition contribution must be nonnegative");
    if (i === j) return;
    const key = Math.min(i, j) * this.neuronCount + Math.max(i, j);
    const entry = this.inhibitionOwners.get(key) ?? { base: this.getInhibitoryWeight(i, j), values: new Map<string, number>() };
    if (value === 0) entry.values.delete(owner); else entry.values.set(owner, value);
    this.writeOwnedInhibition(i, j, entry);
    if (entry.values.size) this.inhibitionOwners.set(key, entry); else this.inhibitionOwners.delete(key);
  }

  private writeOwnedInhibition(i: number, j: number, entry: { base: number; values: Map<string, number> }): void {
    const value = Math.min(this.config.maxWeight, entry.base + [...entry.values.values()].reduce((a, b) => a + b, 0));
    this.inhibitory[i * this.neuronCount + j] = value;
    this.inhibitory[j * this.neuronCount + i] = value;
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
    this.checkEdge(from, to, delta);
    if (cap !== undefined) nonnegative(cap, "cap");
    if (from === to || delta <= 0) return;
    this.diSourceCache = null; // DI 源缓存失效
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

  /** DI 源神经元缓存（有出向 DI 边的神经元）；写 DI 边时失效重建 */
  private diSourceCache: number[] | null = null;
  /** DI 驱动介入判定：源活跃，或源虽静息但场已越阈（即将点燃）——
   *  只有驱动真正静息的状态才有资格做最优回退候选。
   *  （修复瞬时快照漏洞：2 核共存+池恰熄灭的中瞬态曾被误记为最优态） */
  private diDriveEngaged(): boolean {
    if (this.diSourceCache === null) {
      const n = this.neuronCount;
      const src: number[] = [];
      for (let from = 0; from < n; from++) {
        for (let to = 0; to < n; to++) {
          if ((this.directedInhibitory[from * n + to] ?? 0) > 0) {
            src.push(from);
            break;
          }
        }
      }
      this.diSourceCache = src;
    }
    if (this.diSourceCache.length === 0) return false;
    const theta = this.threshold;
    for (const j of this.diSourceCache) {
      if (this.state[j] === 1) return true;
      const h = this.localField(j, this.state) - this.inhibitoryField(j, this.state);
      if (h > theta) return true; // 即将点燃
    }
    return false;
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
      if (!Number.isInteger(i) || i < 0 || i >= n) throw new Error(`input neuron index out of range: ${i}`);
      clamped[i] = 1;
    }
    // 从静息出发：被钳制的激活，其余静息
    this.state.fill(0);
    for (let i = 0; i < n; i++) if (clamped[i] === 1) this.state[i] = 1;

    const theta = this.threshold;
    const energies: number[] = [this.energy()];
    let currentEnergy = energies[0]!;
    let flipCount = 0;
    let driveWork = 0;
    // 浮点尘埃保护：能量差小于 EPS 的翻转会引发近平局无限循环（实测挂死），
    // 要求每次翻转都有实质降幅；另有总翻转数上限与最优回退双保险。
    const EPS = 1e-4;
    const maxFlips = 100 * n;
    let bestEnergy = currentEnergy;
    const bestState = Uint8Array.from(this.state);
    let terminated: SettleTermination = "fixed-point";
    // 异步贪心扫描：只要某次完整扫描没有任何翻转，即为局部极小
    outer: for (;;) {
      let flippedThisSweep = false;
      for (let i = 0; i < n; i++) {
        if (clamped[i] === 1) continue;
        // 第三方评审 F01 修复：决策场含 DI（非平衡驱动照常参与动力学），
        // 但账本只累计保守部分 ΔE_cons = θ − (W场 − Γ场)——它恰好是 energy()
        // 的差分，轨迹与真实能量逐点一致；DI 部分独立累计为 driveWork。
        // 修复前：账本混入驱动做功，闭环一圈漂移 −γ，最优回退按漂移账本失真。
        const hCons = this.localField(i, this.state) - this.inhibitoryField(i, this.state);
        const di = this.directedInhibitoryField(i, this.state);
        const active = this.state[i] === 1;
        const dEDecision = active ? -(theta - (hCons - di)) : theta - (hCons - di);
        if (dEDecision >= -EPS) continue;
        const dECons = active ? -(theta - hCons) : theta - hCons;
        this.state[i] = active ? 0 : 1;
        flipCount++;
        flippedThisSweep = true;
        currentEnergy += dECons;
        driveWork += dEDecision - dECons;
        energies.push(currentEnergy);
        if (currentEnergy < bestEnergy) {
          bestEnergy = currentEnergy;
          bestState.set(this.state);
        }
        // 预算逐翻转立即检查（原为整轮扫描后检查，会超出上限）
        if (flipCount >= maxFlips) {
          terminated = "flip-budget";
          // 非收敛终止：回退到途中最低能态（按真实保守能量）
          this.state.set(bestState);
          energies.push(bestEnergy);
          break outer;
        }
      }
      if (!flippedThisSweep) break;
    }
    return {
      activeNeurons: this.activeNeurons(),
      energy: this.energy(),
      trace: { energies, flipCount, driveWork },
      converged: terminated === "fixed-point",
      terminationReason: terminated,
      residualFlips: this.countResidualFlips((i) => clamped[i] === 1, null, EPS),
    };
  }

  /** 返回态在指定范围内仍可翻转的神经元数（决策场口径，含 DI；0 = 固定点） */
  private countResidualFlips(
    isClamped: (i: number) => boolean,
    scope: ReadonlySet<number> | null,
    eps: number,
  ): number {
    const theta = this.threshold;
    const n = this.neuronCount;
    let count = 0;
    for (let i = 0; i < n; i++) {
      if (scope !== null && !scope.has(i)) continue;
      if (isClamped(i)) continue;
      const h = this.symmetricField(i, this.state);
      const dE = this.state[i] === 0 ? theta - h : -(theta - h);
      if (dE < -eps) count++;
    }
    return count;
  }

  /**
   * 局部退火模式选择（语义：只在"输入所及的范围"内求能耗极小）。
   *
   * Metropolis 温度曲线跨越单点点火势垒，随后进行有限预算淬火。
   * 候选集为输入、被触及势阱和显式 extraCandidates 的并集。
   * 默认允许 DI 参与翻转规律，但最优态与轨迹只使用保守能量。
   * fallbackQuietOnly 时热提议与淬火均限制在 DI 静息可行域，允许
   * 单点/交换局部移动；无合法候选结构化失败，有残余无约束翻转则
   * 报 quiet-constraint，不能把受约束极小误报为全动力学固定点。
   */
  settleAnnealed(
    inputNeurons: Iterable<number>,
    wells: readonly WellMembership[],
    options: AnnealOptions = {},
  ): AnnealResult {
    const n = this.neuronCount;
    const clamped = new Set<number>();
    for (const i of inputNeurons) {
      if (!Number.isInteger(i) || i < 0 || i >= n) throw new Error(`input neuron index out of range: ${i}`);
      clamped.add(i);
    }

    const theta = this.threshold;
    const coolingFactor = options.coolingFactor ?? 0.7;
    const levels = options.levels ?? 20;
    const sweepsPerLevel = options.sweepsPerLevel ?? 30;
    if (!(coolingFactor > 0 && coolingFactor < 1)) {
      throw new Error(`coolingFactor must be in (0, 1), got ${coolingFactor}`);
    }
    integer(levels, "levels", 1);
    integer(sweepsPerLevel, "sweepsPerLevel", 1);
    integer(options.quenchMaxFlips ?? 100 * n, "quenchMaxFlips");
    const quietOnly = options.fallbackQuietOnly ?? false;
    let temperature = options.initialTemperature ?? theta;
    positive(temperature, "initialTemperature");
    const extraCandidates = options.extraCandidates ? [...options.extraCandidates] : [];
    for (const id of extraCandidates) neuronId(id, n);
    for (const well of wells) for (const id of well.memberNeuronIds) neuronId(id, n);
    // 从静息 + 钳制出发（无独立贪心相）
    this.state.fill(0);
    for (const i of clamped) this.state[i] = 1;
    const initialEnergy = this.energy();
    // Ordered sparse state view. Summation retains the original ascending id
    // order (including floating-point rounding); only zero terms are skipped.
    // The dense state remains authoritative and every local move updates both.
    let activeIds = this.activeNeurons();
    const flip = (id: number): void => {
      if (this.state[id] === 1) {
        this.state[id] = 0;
        activeIds.splice(activeIds.indexOf(id), 1);
      } else {
        this.state[id] = 1;
        const at = activeIds.findIndex(j => j > id);
        activeIds.splice(at < 0 ? activeIds.length : at, 0, id);
      }
    };
    const restore = (snapshot: Uint8Array): void => {
      this.state.set(snapshot); activeIds = this.activeNeurons();
    };
    const fields = (i: number): { cons: number; di: number } => {
      let w = 0, g = 0, di = 0;
      for (const j of activeIds) {
        w += this.weights[i * n + j]!;
        g += this.inhibitory[i * n + j]!;
        di += this.directedInhibitory[j * n + i]!;
      }
      return { cons: w - g, di };
    };
    // Populate the unchanged DI-source cache before using its sparse view.
    this.diDriveEngaged();
    const quietNow = (): boolean => this.diSourceCache!.every(j =>
      this.state[j] === 0 && fields(j).cons <= this.threshold);

    // 候选集 C = I ∪ 被触及势阱
    const candidate = new Set<number>(clamped);
    for (const well of wells) {
      if (well.memberNeuronIds.some((id) => clamped.has(id))) {
        for (const id of well.memberNeuronIds) {
          if (id >= 0 && id < n) candidate.add(id);
        }
      }
    }
    if (extraCandidates) {
      for (const id of extraCandidates) {
        if (id >= 0 && id < n) candidate.add(id);
      }
    }
    const freeCandidates = [...candidate].filter((id) => !clamped.has(id)).sort((a, b) => a - b);

    const rand = mulberry32(options.seed ?? 1);
    let proposals = 0;
    let acceptedUphill = 0;
    let driveWork = 0;
    let flipCount = 0;
    // F01 修复（同 settle）：决策场含 DI（动力学不变），账本只记保守部分
    // （≡ energy() 的差分），DI 部分独立累计为 driveWork。最优状态按真实能量。
    const deltas = (i: number): { decision: number; cons: number } => {
      const { cons: hCons, di } = fields(i);
      const active = this.state[i] === 1;
      return {
        decision: active ? -(theta - (hCons - di)) : theta - (hCons - di),
        cons: active ? -(theta - hCons) : theta - hCons,
      };
    };
    let currentEnergy = initialEnergy;
    let hasQuietCandidate = !quietOnly || quietNow();
    let bestEnergy = hasQuietCandidate ? initialEnergy : Infinity;
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
          const { decision, cons } = deltas(i);
          proposals++;
          let move = [i];
          let decisionDelta = decision;
          let conservativeDelta = cons;
          if (quietOnly) {
            // Metropolis on the eligible domain. If a single activation would
            // recruit DI, propose an exchange with an active free member. This
            // samples cooperative output ignition while keeping legal states;
            // no learned core ids, labels, truth or exact-match lookup is used.
            flip(i);
            const legal = quietNow();
            flip(i);
            if (!legal) {
              if (this.state[i] === 1) continue;
              const activeFree = freeCandidates.filter(j => this.state[j] === 1);
              if (!activeFree.length) continue;
              const j = activeFree[Math.floor(rand() * activeFree.length)]!;
              conservativeDelta += deltas(j).cons + this.getWeight(i, j) - this.getInhibitoryWeight(i, j);
              decisionDelta = conservativeDelta;
              flip(i); flip(j);
              const exchangeLegal = quietNow();
              flip(i); flip(j);
              if (!exchangeLegal) continue;
              move = [j, i];
            }
          }
          if (decisionDelta >= 0 && rand() >= Math.exp(-decisionDelta / temperature)) continue;
          if (decisionDelta >= 0) acceptedUphill++;
          for (const id of move) flip(id);
          flipCount += move.length;
          currentEnergy += conservativeDelta;
          driveWork += decisionDelta - conservativeDelta;
          // fallbackQuietOnly：带池 WTA 电路只认驱动静息期的最低真实能态
          // （多核共存态真实能量更低但破坏单核语义）；默认全访问态（评审 A04 契约）
          if (currentEnergy < bestEnergy && !(quietOnly && !quietNow())) {
            hasQuietCandidate = true;
            bestEnergy = currentEnergy;
            bestState.set(this.state);
          }
        }
      }
    }
    if (!hasQuietCandidate) {
      // No eligible answer: return no active pattern, never a DI-driven state.
      this.state.fill(0);
      return { activeNeurons: [], energy: 0, trace: { energies: [0], flipCount, driveWork },
        converged: false, terminationReason: "no-quiet-candidate", residualFlips: 0,
        candidateSet: [...candidate], initialEnergy, annealEndEnergy: 0, proposals, acceptedUphill };
    }
    if (quietOnly || bestEnergy < currentEnergy) restore(bestState);
    const annealEndEnergy = this.energy();

    // 淬火尾巴：T=0 贪心扫描至无翻转（保守账本随翻转变化；DI 驱动可致上坡）。
    // 范围默认全网（招募无势垒的普通连接神经元）；读出场景限制在候选集内，
    // 防止对称 W 把结果活动反传回未查询条件档引发雪崩。
    // 非平衡驱动（DI 前馈抑制）可能出现弛豫振荡（压制-熄火-复燃循环），
    // 因此淬火是有界迭代 + 最优回退（按真实能量），并如实报告终止原因与
    // 残余可翻转数——非平衡驱动下不再承诺无条件收敛。
    const quenchEnergies: number[] = [];
    let quenchEnergy = annealEndEnergy;
    let quenchBestEnergy = annealEndEnergy;
    const quenchBestState = Uint8Array.from(this.state);
    const quenchScope = options.quenchCandidatesOnly
      ? freeCandidates
      : Array.from({ length: n }, (_, i) => i);
    const quenchScopeSet = new Set(quenchScope);
    const maxFlips = options.quenchMaxFlips ?? 100 * n;
    const EPS = 1e-4; // 能量分辨率下限：近平局的尘埃翻转既慢又无意义（实测 >1e5 步仍不收敛）
    let terminated: SettleTermination = "fixed-point";
    let quenchFlips = 0; // 预算只管淬火尾巴（退火相按温度层数自然结束）
    if (quietOnly) {
      // Optimize within the same legal fallback domain, rather than repeatedly
      // leaving it under DI and returning a fragmented annealing snapshot.
      // Single-neuron descent plus 1-for-1 local exchanges cross the cardinality
      // boundary imposed by a quiet recruitment pool. No rule/core metadata is
      // consulted. Every actual flip consumes the unchanged quench budget.
      for (;;) {
        const quietAfter = (ids: readonly number[]) => {
          for (const id of ids) flip(id);
          const legal = quietNow();
          for (const id of ids) flip(id);
          return legal;
        };
        const proposals = quenchScope.filter(i => !clamped.has(i)).map(i => ({ i, delta: deltas(i).cons }));
        let move: number[] = [];
        let improvement = -EPS;
        for (const { i, delta } of proposals) if (delta < improvement && quietAfter([i])) {
          move = [i]; improvement = delta;
        }
        if (!move.length) {
          const on = proposals.filter(p => this.state[p.i] === 1);
          const off = proposals.filter(p => this.state[p.i] === 0);
          for (const x of on) for (const y of off) {
            const delta = x.delta + y.delta + this.getWeight(x.i, y.i) - this.getInhibitoryWeight(x.i, y.i);
            if (delta < improvement && quietAfter([x.i, y.i])) { move = [x.i, y.i]; improvement = delta; }
          }
        }
        if (!move.length) {
          terminated = this.countResidualFlips(i => clamped.has(i), quenchScopeSet, EPS) ? "quiet-constraint" : "fixed-point";
          break;
        }
        if (quenchFlips + move.length > maxFlips) { terminated = "flip-budget"; break; }
        for (const i of move) {
          const { decision, cons } = deltas(i);
          flip(i);
          flipCount++; quenchFlips++;
          quenchEnergy += cons; driveWork += decision - cons;
          quenchEnergies.push(quenchEnergy);
        }
        quenchBestEnergy = quenchEnergy;
        quenchBestState.set(this.state);
      }
    } else {
    outer: for (;;) {
      let flipped = false;
      for (const i of quenchScope) {
        if (clamped.has(i)) continue;
        const { decision, cons } = deltas(i);
        if (decision < -EPS) {
          if (quenchFlips >= maxFlips) {
            terminated = "flip-budget";
            restore(quenchBestState);
            quenchEnergies.push(quenchBestEnergy);
            break outer;
          }
          flip(i);
          flipCount++;
          quenchFlips++;
          quenchEnergy += cons;
          driveWork += decision - cons;
          quenchEnergies.push(quenchEnergy);
          if (quenchEnergy < quenchBestEnergy && !(quietOnly && !quietNow())) {
            quenchBestEnergy = quenchEnergy;
            quenchBestState.set(this.state);
          }
          flipped = true;
          // 预算逐翻转立即检查（原为整轮扫描后检查，会超出上限）；
          // 回退写入轨迹但不计为翻转（评审 A12 反例）
          if (quenchFlips >= maxFlips) {
            terminated = "flip-budget";
            restore(quenchBestState);
            quenchEnergies.push(quenchBestEnergy);
            break outer;
          }
        }
      }
      if (!flipped) break;
    }
    }

    if (quietOnly && !quietNow()) {
      restore(quenchBestState);
      quenchEnergies.push(quenchBestEnergy);
      terminated = "flip-budget";
    }
    const energies = [annealEndEnergy, ...quenchEnergies];
    return {
      activeNeurons: this.activeNeurons(),
      energy: this.energy(),
      trace: { energies, flipCount, driveWork },
      converged: terminated === "fixed-point",
      terminationReason: terminated,
      residualFlips: this.countResidualFlips((i) => clamped.has(i), quenchScopeSet, EPS),
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
      if (!Number.isInteger(i) || i < 0 || i >= n) throw new Error(`neuron index out of range: ${i}`);
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
