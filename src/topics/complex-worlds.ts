import { validateFrame, type Frame, type TransitionSpace } from "../planning/space.js";
import type { MultistageWorld } from "./multistage-worlds.js";

/**
 * 复杂度升级的三组陌生场景：更深的分阶段依赖、动态世界规则、大状态网格。
 * 真值只存在于实验台与评分侧，不进学习核。
 */

/** W6 双资源工厂：矿（pos1 取料）+ 充电站（pos3）+ 工厂（pos6 合成）。
 * 深层分阶段：取料 → 保能赶路 → 合成，能量与矿石双重约束。 */
export const W6: MultistageWorld = {
  name: "factory",
  space: {
    states: [
      { name: "pos", outcome: "nextPos", bins: 7 },
      { name: "ore", outcome: "nextOre", bins: 2 },
      { name: "energy", outcome: "nextEnergy", bins: 4 },
      { name: "made", outcome: "nextMade", bins: 2 },
    ],
    actions: [{ name: "act", bins: 3 }], // 0=移动 1=就地取材 2=合成
    diameter: 10,
  },
  truth: (s, a) => {
    if (a.act === 1) {
      if (s.pos === 1) return { pos: 1, ore: 1, energy: s.energy!, made: s.made! };
      if (s.pos === 3) return { pos: 3, ore: s.ore!, energy: 3, made: s.made! };
      return { pos: s.pos!, ore: s.ore!, energy: s.energy!, made: s.made! };
    }
    if (a.act === 2) {
      // 合成只耗矿（能量约束由赶路承担——耗能合成会让终点能量恒为 0 的任务不可达，实测教训）
      if (s.pos === 6 && s.ore === 1) {
        return { pos: 6, ore: 1, energy: s.energy!, made: 1 };
      }
      return { pos: s.pos!, ore: s.ore!, energy: s.energy!, made: s.made! };
    }
    if (s.energy! > 0) return { pos: Math.min(6, s.pos! + 1), ore: s.ore!, energy: s.energy! - 1, made: s.made! };
    return { pos: s.pos!, ore: s.ore!, energy: 0, made: s.made! };
  },
  tasks: [
    { start: { pos: 0, ore: 0, energy: 3, made: 0 }, goal: { pos: 6, ore: 1, energy: 0, made: 1 } },
    { start: { pos: 1, ore: 0, energy: 2, made: 0 }, goal: { pos: 6, ore: 1, energy: 0, made: 1 } },
  ],
  factors: ["pos", "ore", "energy", "act"],
  distractors: [],
};

/** W7 世界规则中途翻转：原链 0..9；翻转后 4→5 关闭成墙、2→7 捷径开启。
 * 完整动态世界循环：失配检测 → 行为层回避（诚实报无路线）→ 再探索 →
 * 计票翻转旧规则 + 学到捷径 → 重规划走捷径到达。 */
export const W7: MultistageWorld & {
  flippedTruth: (state: Frame, action: Frame) => Frame;
} = {
  name: "world-flip",
  space: {
    states: [{ name: "pos", outcome: "nextPos", bins: 10 }],
    actions: [{ name: "act", bins: 2 }], // 0=停留 1=前进
    diameter: 14,
  },
  truth: (s, a) => (a.act === 0 ? { pos: s.pos! } : { pos: Math.min(9, s.pos! + 1) }),
  // 翻转后：4→5 被墙挡住（原地）；2→7 捷径开启
  flippedTruth: (s, a) => {
    if (a.act === 0) return { pos: s.pos! };
    if (s.pos === 4) return { pos: 4 };
    if (s.pos === 2) return { pos: 7 };
    return { pos: Math.min(9, s.pos! + 1) };
  },
  tasks: [{ start: { pos: 0 }, goal: { pos: 9 } }],
  factors: ["pos", "act"],
  distractors: [],
};

/** W8 双门长走廊：24 节点、两把钥匙（4/12）、两道门（8/16 一一对应）。
 * 大网格下的长程多阶段规划。
 * 规模注记：32 节点版（512 规则、核度 ~2000）实测单预测 ~40s——
 * 核间互斥全图的度随规则数线性增长，是当前实用规模的主要成本源；
 * 本世界取 24 节点/192 规则以保持验收可在合理时间完成。 */
export const W8: MultistageWorld = {
  name: "two-gates",
  space: {
    states: [
      { name: "pos", outcome: "nextPos", bins: 24 },
      { name: "k1", outcome: "nextK1", bins: 2 },
      { name: "k2", outcome: "nextK2", bins: 2 },
    ],
    actions: [{ name: "move", bins: 2 }],
    diameter: 28,
  },
  truth: (s, a) => {
    let np = s.pos!;
    if (a.move === 1 && s.pos! < 23) np = s.pos! + 1;
    if (a.move === 0 && s.pos! > 0) np = s.pos! - 1;
    if (a.move === 1) {
      if (s.pos === 7 && s.k1 === 0) np = 7;
      if (s.pos === 15 && s.k2 === 0) np = 15;
    }
    const nk1 = np === 4 ? 1 : s.k1!;
    const nk2 = np === 12 ? 1 : s.k2!;
    return { pos: np, k1: nk1, k2: nk2 };
  },
  tasks: [
    { start: { pos: 0, k1: 0, k2: 0 }, goal: { pos: 23, k1: 1, k2: 1 } },
  ],
  factors: ["pos", "move", "k1", "k2"],
  distractors: [],
};

export const COMPLEX_WORLDS: readonly MultistageWorld[] = [W6, W7, W8];
