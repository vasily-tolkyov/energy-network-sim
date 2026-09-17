import { mulberry32 } from "./prng.js";
import type { EnergyNetwork } from "./network.js";
import type { WellMembership } from "./types.js";

/**
 * 势阱间的有向通道与噪声推动的转移。
 *
 * 机制分离（对应架构文档 §9）：
 * - 对称 W：共激活建立，承载关联与势阱（settle 的能量函数只用它）；
 * - 有向 D：时间先后建立（前模式 → 后模式），承载势阱间的转移通道。
 *
 * 转移动力学：Metropolis 噪声翻转，有效翻转移量
 *   ΔE′ = ΔE_W(疲劳修正) − β·g_i   （0→1；1→0 时驱动项取 +β·g_i）
 * 其中 g_i = Σ_j D_{j→i}·s_j 是当前模式经有向通道对 i 的驱动。
 * 有向通道把后继势阱的势垒压低但不归零，第一下点火仍由噪声完成——
 * 噪声与驱动协同。D 是非平衡驱动（类比材料实现的供能端口），driveWork
 * 记录它对系统做的功。
 *
 * 驻留疲劳：维持激活的能耗随连续激活步数线性增长（φ = κ·dwell），
 * 仍在能耗语言内、神经元保持两态；旧势阱疲劳崩塌，活动向有向通道
 * 指向的下一势阱迁移，形成 A→B→C 的行波。回合结束恢复静息（reset），
 * 疲劳随之归零。
 */

export interface TransitionOptions {
  /** 转移动力学的总步数，默认 200 */
  readonly steps?: number;
  /** 噪声温度 T，默认 0.3；T=0 时退化为纯确定性（噪声必要性对照） */
  readonly temperature?: number;
  /** 有向驱动系数 β，默认 0.5；β=0 用于无驱动对照 */
  readonly driveBeta?: number;
  /** 驻留疲劳速率 κ（每激活一步维持能耗增量），默认 0.05 */
  readonly fatigueRate?: number;
  /**
   * 疲劳恢复速率 κ_r：静息时每步 φ 减少 κ_r，默认 0（静息立即清零）。
   * κ_r>0 时疲劳缓慢恢复——刚死亡的神经元短期内难以再点火，
   * 阻止"成员周转"造成的势阱亚稳态churn（大簇异步死亡时的再点火回填）。
   */
  readonly fatigueRecoveryRate?: number;
  /** PRNG 种子，默认 1 */
  readonly seed?: number;
  /** 势阱占优判据：成员激活率 ≥ 此值，默认 0.5 */
  readonly dominanceThreshold?: number;
}

export interface TransitionEvent {
  /** 离开的主导势阱 */
  readonly from: number;
  /** 新进入激活集的势阱 */
  readonly to: number;
  readonly step: number;
}

export interface TransitionRunResult {
  /** 每一步的主导势阱（无占优时为 null） */
  readonly wellTimeline: readonly (number | null)[];
  /** 每个势阱处于激活集（成员激活率 ≥ 阈值）的累计步数——共存期间各算各的，不被占优掩盖 */
  readonly wellActiveSteps: readonly number[];
  /** 确认的势阱转移事件（按时间序） */
  readonly transitions: readonly TransitionEvent[];
  /** 有向驱动对系统做的累计功（接受的 0→1 翻转的 β·g 之和） */
  readonly driveWork: number;
  readonly acceptedFlips: number;
  readonly acceptedUphill: number;
  /** 结束时仍激活的神经元（疲劳作用下通常为空：归于静息） */
  readonly finalActive: readonly number[];
}

/**
 * 时序赫布学习：patterns 按时间顺序呈现，对每对相邻模式 (prev, next)
 * 建立 prev→next 的有向通道（i∈prev, j∈next, D_ij += eta），单向。
 * 势阱本身的对称连接由 hebbianLearn 另行建立。
 */
export function learnSequence(
  net: EnergyNetwork,
  patterns: readonly (readonly number[])[],
  repeats = 1,
  eta?: number,
): void {
  const step = eta ?? net.config.learningRate;
  for (let r = 0; r < repeats; r++) {
    for (let p = 0; p + 1 < patterns.length; p++) {
      for (const i of patterns[p]!) {
        for (const j of patterns[p + 1]!) {
          net.strengthenDirected(i, j, step);
        }
      }
    }
  }
}

/**
 * 分支学习：predecessor 的多个候选后继。
 * - 对每个后继建立 predecessor→successor 的有向通道（eta 可为每个后继
 *   单独指定强度，实现不等强通道）；
 * - 在候选后继两两之间建立对称抑制边（强度 gamma）：先被噪声点燃的
 *   后继通过抑制抬高竞争对手的点火势垒，实现"小球只进一个槽"的
 *   互斥概率选择——进入哪条通道不确定，概率随通道强度倾斜。
 */
export function learnBranches(
  net: EnergyNetwork,
  predecessor: readonly number[],
  successors: readonly (readonly number[])[],
  options: { repeats?: number; eta?: number | readonly number[]; gamma?: number } = {},
): void {
  const repeats = options.repeats ?? 1;
  const gamma = options.gamma ?? 0.1;
  successors.forEach((succ, k) => {
    const eta = Array.isArray(options.eta) ? options.eta[k] : options.eta;
    learnSequence(net, [predecessor, succ], repeats, eta);
  });
  for (let a = 0; a < successors.length; a++) {
    for (let b = a + 1; b < successors.length; b++) {
      for (const i of successors[a]!) {
        for (const j of successors[b]!) {
          net.strengthenInhibitory(i, j, gamma);
        }
      }
    }
  }
}

/**
 * 噪声推动的转移运行。从 net 当前状态（通常由 settle 捕获的势阱）出发，
 * 在局部状态副本上演化，不修改 net 的神经元状态——回合结束由调用方
 * reset() 恢复静息。
 */
export function runTransitions(
  net: EnergyNetwork,
  wells: readonly WellMembership[],
  options: TransitionOptions = {},
): TransitionRunResult {
  const n = net.neuronCount;
  const steps = options.steps ?? 200;
  const temperature = options.temperature ?? 0.3;
  const beta = options.driveBeta ?? 0.5;
  const kappa = options.fatigueRate ?? 0.05;
  const kappaRecovery = options.fatigueRecoveryRate ?? 0;
  const dominance = options.dominanceThreshold ?? 0.5;
  if (steps < 1) throw new Error(`steps must be >= 1, got ${steps}`);
  if (temperature < 0) throw new Error(`temperature must be >= 0, got ${temperature}`);
  if (kappaRecovery < 0) throw new Error(`fatigueRecoveryRate must be >= 0, got ${kappaRecovery}`);

  const rand = mulberry32(options.seed ?? 1);
  const state = new Uint8Array(n);
  for (const i of net.activeNeurons()) state[i] = 1;
  // 疲劳量 φ：激活时每步 += κ；静息时每步 −= κ_r（κ_r=0 时立即清零）
  const fatigueLevel = new Float64Array(n);
  const theta = net.threshold;

  const wellTimeline: (number | null)[] = [];
  const activeSteps = new Array<number>(wells.length).fill(0);
  const transitions: TransitionEvent[] = [];
  let activeWellSet = new Set<number>();
  let driveWork = 0;
  let acceptedFlips = 0;
  let acceptedUphill = 0;

  const observe = (step: number): void => {
    // 各势阱成员激活率
    let dominantWell: number | null = null;
    let dominantRatio = 0;
    const current = new Set<number>();
    wells.forEach((well, k) => {
      const members = well.memberNeuronIds;
      if (members.length === 0) return;
      let hit = 0;
      for (const id of members) if (state[id] === 1) hit++;
      const ratio = hit / members.length;
      if (ratio >= dominance) {
        current.add(k);
        activeSteps[k]!++;
      }
      if (ratio > dominantRatio) {
        dominantRatio = ratio;
        dominantWell = ratio >= dominance ? k : null;
      }
    });
    wellTimeline.push(dominantWell);
    // 新势阱进入激活集 → 记录转移事件。
    // 归因：在当前激活的势阱中，取对新势阱有向驱动总量最强者（物理归因），
    // 无驱动来源时回退到成员重叠率最高者。
    for (const k of current) {
      if (activeWellSet.has(k)) continue;
      let from: number | null = null;
      let bestDrive = 0;
      for (const prev of activeWellSet) {
        let drive = 0;
        for (const j of wells[prev]!.memberNeuronIds) {
          for (const i of wells[k]!.memberNeuronIds) {
            drive += net.getDirectedWeight(j, i);
          }
        }
        if (drive > bestDrive) {
          bestDrive = drive;
          from = prev;
        }
      }
      if (from === null) {
        let best = 0;
        for (const prev of activeWellSet) {
          const members = wells[prev]!.memberNeuronIds;
          let hit = 0;
          for (const id of members) if (state[id] === 1) hit++;
          const ratio = members.length === 0 ? 0 : hit / members.length;
          if (ratio > best) {
            best = ratio;
            from = prev;
          }
        }
      }
      if (from !== null) transitions.push({ from, to: k, step });
    }
    activeWellSet = current;
  };
  observe(0);

  for (let step = 1; step <= steps; step++) {
    // 每步 N 次随机位点提议（Metropolis）
    for (let probe = 0; probe < n; probe++) {
      const i = Math.floor(rand() * n);
      const hW = net.symmetricField(i, state); // 吸引场(W) − 抑制场(Γ)
      const thetaI = theta + fatigueLevel[i]!;
      const dEW = state[i] === 0 ? thetaI - hW : -(thetaI - hW);
      const g = net.directedField(i, state);
      // 0→1 受驱动助力；1→0 受驱动阻抗
      const dE = state[i] === 0 ? dEW - beta * g : dEW + beta * g;
      const accept = dE < 0 || (temperature > 0 && rand() < Math.exp(-dE / temperature));
      if (!accept) continue;
      if (state[i] === 0) {
        state[i] = 1;
        driveWork += beta * g;
      } else {
        state[i] = 0;
      }
      acceptedFlips++;
      if (dEW > 0) acceptedUphill++; // 热力翻越对称能量势垒的证据
    }
    // 疲劳更新：激活积累 κ；静息以 κ_r 缓慢恢复（κ_r=0 立即清零）
    for (let i = 0; i < n; i++) {
      if (state[i] === 1) {
        fatigueLevel[i] = fatigueLevel[i]! + kappa;
      } else if (kappaRecovery === 0) {
        fatigueLevel[i] = 0;
      } else {
        fatigueLevel[i] = Math.max(0, fatigueLevel[i]! - kappaRecovery);
      }
    }
    observe(step);
  }

  const finalActive: number[] = [];
  for (let i = 0; i < n; i++) if (state[i] === 1) finalActive.push(i);
  return { wellTimeline, wellActiveSteps: activeSteps, transitions, driveWork, acceptedFlips, acceptedUphill, finalActive };
}
