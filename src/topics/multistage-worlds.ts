import { validateFrame, type Frame, type TransitionSpace } from "../planning/space.js";

/**
 * 多阶段陌生场景验收：五个转移结构互不相同的小世界。
 * 每个世界都要求"先探索学习转移，再规划并执行多阶段目标"：
 * 取钥匙/冷却节拍/不可逆传送带/能量预算/相位变速。
 * 真值只存在于实验台与评分侧，不进学习核。
 */

export interface MultistageWorld {
  readonly name: string;
  readonly space: TransitionSpace;
  readonly truth: (state: Frame, action: Frame) => Frame;
  /** 验收任务：起点 → 目标（完整状态帧） */
  readonly tasks: readonly { start: Frame; goal: Frame }[];
  /** 诚实性负对照：不可达目标应如实报无路线 */
  readonly unreachable?: { start: Frame; goal: Frame };
  /** 应识别的影响维 / 应排除的干扰维 */
  readonly factors: readonly string[];
  readonly distractors: readonly string[];
}

export class MultistageBench {
  private cost = 0;
  constructor(private readonly world: MultistageWorld) {}
  conduct(state: Frame, action: Frame): Frame {
    validateFrame(state, this.world.space.states);
    validateFrame(action, this.world.space.actions);
    this.cost++;
    return out(this.world.space, this.world.truth(state, action));
  }
  get experimentsUsed(): number {
    return this.cost;
  }
}

const out = (space: TransitionSpace, next: Frame): Frame =>
  Object.fromEntries(space.states.map(d => [d.outcome, next[d.name]!]));

/** W1 钥匙门：先经过位置 2 拿到钥匙，才能通过 4→5 的门。decoy 是干扰维。 */
export const W1: MultistageWorld = {
  name: "key-door",
  space: {
    states: [
      { name: "pos", outcome: "nextPos", bins: 6 },
      { name: "key", outcome: "nextKey", bins: 2 },
      { name: "decoy", outcome: "nextDecoy", bins: 2 },
    ],
    actions: [{ name: "move", bins: 2 }],
    diameter: 8,
  },
  truth: (s, a) => {
    let np = s.pos!;
    if (a.move === 1 && s.pos! < 5) np = s.pos! + 1;
    if (a.move === 0 && s.pos! > 0) np = s.pos! - 1;
    if (s.pos === 4 && a.move === 1 && s.key === 0) np = 4; // 无钥匙被门挡回
    const nk = np === 2 ? 1 : s.key!;
    return { pos: np, key: nk, decoy: 0 }; // 干扰维恒定输出，无任何结果耦合
  },
  tasks: [
    { start: { pos: 0, key: 0, decoy: 0 }, goal: { pos: 5, key: 1, decoy: 0 } },
    { start: { pos: 0, key: 0, decoy: 1 }, goal: { pos: 5, key: 1, decoy: 0 } },
  ],
  factors: ["pos", "move", "key"],
  distractors: ["decoy"],
};

/** W2 冷却节拍：冷却为 1 时前进无效且复位——前进必须与冷却交替。 */
export const W2: MultistageWorld = {
  name: "cooldown",
  space: {
    states: [
      { name: "pos", outcome: "nextPos", bins: 8 },
      { name: "cool", outcome: "nextCool", bins: 2 },
    ],
    actions: [{ name: "act", bins: 2 }], // 0=等待 1=前进
    diameter: 13,
  },
  truth: (s, a) => {
    if (a.act === 1 && s.cool === 0) return { pos: Math.min(7, s.pos! + 1), cool: 1 };
    return { pos: s.pos!, cool: 0 };
  },
  tasks: [
    { start: { pos: 0, cool: 0 }, goal: { pos: 7, cool: 1 } },
    { start: { pos: 2, cool: 0 }, goal: { pos: 7, cool: 1 } },
  ],
  factors: ["pos", "cool", "act"],
  distractors: [],
};

/** W3 不可逆传送带：ride 前进一格，brake 原地；只能向前，过头不可回头。 */
export const W3: MultistageWorld = {
  name: "conveyor",
  space: {
    states: [{ name: "pos", outcome: "nextPos", bins: 8 }],
    actions: [{ name: "cmd", bins: 2 }], // 0=brake 1=ride
    diameter: 7,
  },
  truth: (s, a) => ({ pos: a.cmd === 1 ? Math.min(7, s.pos! + 1) : s.pos! }),
  tasks: [
    { start: { pos: 0 }, goal: { pos: 5 } },
    { start: { pos: 1 }, goal: { pos: 7 } },
  ],
  unreachable: { start: { pos: 5 }, goal: { pos: 2 } },
  factors: ["pos", "cmd"],
  distractors: [],
};

/** W4 能量预算：移动耗 1 能量，充电回满 3；0 能量寸步难行。 */
export const W4: MultistageWorld = {
  name: "energy-budget",
  space: {
    states: [
      { name: "pos", outcome: "nextPos", bins: 7 },
      { name: "energy", outcome: "nextEnergy", bins: 4 },
    ],
    actions: [{ name: "act", bins: 2 }], // 0=移动 1=充电
    diameter: 9,
  },
  truth: (s, a) => {
    if (a.act === 1) return { pos: s.pos!, energy: 3 };
    if (s.energy! > 0) return { pos: Math.min(6, s.pos! + 1), energy: s.energy! - 1 };
    return { pos: s.pos!, energy: 0 };
  },
  tasks: [
    { start: { pos: 0, energy: 3 }, goal: { pos: 6, energy: 0 } },
    { start: { pos: 0, energy: 2 }, goal: { pos: 6, energy: 0 } },
  ],
  factors: ["pos", "energy", "act"],
  distractors: [],
};

/** W5 相位变速：越过位置 3 的拉杆后进入高速相位（步幅 2），不可逆。 */
export const W5: MultistageWorld = {
  name: "phase-lever",
  space: {
    states: [
      { name: "pos", outcome: "nextPos", bins: 7 },
      { name: "phase", outcome: "nextPhase", bins: 2 },
    ],
    actions: [{ name: "act", bins: 2 }], // 0=停留 1=前进
    diameter: 9,
  },
  truth: (s, a) => {
    if (a.act === 0) return { pos: s.pos!, phase: s.phase! };
    const np = Math.min(6, s.pos! + (s.phase === 1 ? 2 : 1));
    const np2 = s.phase === 0 && np >= 3 ? 1 : s.phase!;
    return { pos: np, phase: np2 };
  },
  tasks: [
    { start: { pos: 0, phase: 0 }, goal: { pos: 6, phase: 1 } },
    { start: { pos: 1, phase: 0 }, goal: { pos: 6, phase: 1 } },
  ],
  factors: ["pos", "phase", "act"],
  distractors: [],
};

export const MULTISTAGE_WORLDS: readonly MultistageWorld[] = [W1, W2, W3, W4, W5];
