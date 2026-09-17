import type { EnergyNetwork } from "../network.js";

/**
 * 神经差分引擎：用重合场分层从两个激活模式中筛出共同项与差分项。
 *
 * 机制（不查表、纯动力学）：把模式 A、B 的成员全部钳制激活后，
 * 以高于单阱自维持场的高阈值 θHigh 松弛：
 * - 只属于一个模式的成员：场 ≈ h（自己阱内互连）< θHigh → 熄灭；
 * - 交集成员：同时在两阱中，交集互连边被两次赫布加强（≈2h）> θHigh → 驻留。
 * 松弛后的驻留集合即共同项；A\共同项、B\共同项即两侧的差分候选。
 * 变化标志与变化数值保持分离：标志 = 是否出现在差分侧，
 * 数值 = 该神经元自身的档位身份（从 1 变 0 仍是变化）。
 */

export interface DifferentialResult {
  /** 共同项（交集且在高阈值下驻留的神经元） */
  readonly common: readonly number[];
  /** 仅在 A 侧（A 的差分候选） */
  readonly onlyA: readonly number[];
  /** 仅在 B 侧（B 的差分候选） */
  readonly onlyB: readonly number[];
  /** 高阈值松弛后仍驻留的全部神经元（分层有效性的直接证据） */
  readonly persisted: readonly number[];
}

/**
 * thetaHigh 的选取：须满足 h_single < thetaHigh < h_common，
 * 其中 h_single 是单阱成员的自维持场，h_common 是交集成员的互连场。
 * 对本库的簇结构（簇内 0.8、8 成员）：h_single ≈ 0.8×7 = 5.6，h_common ≈ 11.2，取 8.0 合适；
   // 一般原则：θHigh 须落在单阱自维持场与交集互连场之间，随阱大小缩放。
 */
export function neuralDifferential(
  net: EnergyNetwork,
  patternA: readonly number[],
  patternB: readonly number[],
  thetaHigh: number,
): DifferentialResult {
  const n = net.neuronCount;
  const state = new Uint8Array(n);
  for (const i of patternA) state[i] = 1;
  for (const i of patternB) state[i] = 1;
  // 与 settle 同一条局部规律，仅阈值抬高（高阈值分层）
  for (;;) {
    let flipped = false;
    for (let i = 0; i < n; i++) {
      const h = net.symmetricField(i, state);
      const dE = state[i] === 0 ? thetaHigh - h : -(thetaHigh - h);
      if (dE < 0) {
        state[i] = state[i] === 0 ? 1 : 0;
        flipped = true;
      }
    }
    if (!flipped) break;
  }
  const setA = new Set(patternA);
  const setB = new Set(patternB);
  const common: number[] = [];
  const onlyA: number[] = [];
  const onlyB: number[] = [];
  for (const i of setA) {
    if (setB.has(i) && state[i] === 1) common.push(i);
    else onlyA.push(i);
  }
  for (const i of setB) {
    if (setA.has(i) && state[i] === 1) continue; // 已计入共同项
    onlyB.push(i);
  }
  const persisted: number[] = [];
  for (let i = 0; i < n; i++) if (state[i] === 1) persisted.push(i);
  return {
    common: common.sort((a, b) => a - b),
    onlyA: onlyA.sort((a, b) => a - b),
    onlyB: onlyB.sort((a, b) => a - b),
    persisted,
  };
}
