import { validateFrame, type Frame, type TransitionSpace } from "../planning/space.js";

/**
 * 箱子世界（交互 agent 的第一座虚拟世界）：
 * 两行走廊，(row=0, col=2) 有一堵墙——去目标点必须绕行到 row=1。
 * 箱源在 col 0，堆垛点在 (row=0, col=4)。
 * 状态维：row、col、carrying（是否拿着箱子）、stack（堆垛高度 0..2）。
 * 动作：move（0左 1右 2上 3下）、handle（0无 1取 2放）。
 * 真值只在实验台内。
 */

export const BOX_SPACE: TransitionSpace = {
  states: [
    { name: "row", outcome: "nextRow", bins: 2 },
    { name: "col", outcome: "nextCol", bins: 5 },
    { name: "carry", outcome: "nextCarry", bins: 2 },
    { name: "stack", outcome: "nextStack", bins: 3 },
  ],
  actions: [{ name: "move", bins: 5 }, { name: "handle", bins: 3 }], // move: 0左 1右 2上 3下 4原地
  diameter: 22, // 堆垛两次往返 ~44 步；maxDepth=2×diameter 覆盖（实测 24 不够：预算在第二程耗尽）
};

export function boxTruth(s: Frame, a: Frame): Frame {
  let { row, col, carry, stack } = { row: s.row!, col: s.col!, carry: s.carry!, stack: s.stack! };
  // 移动（撞墙原地；stack 位可站）
  if (a.move === 0 && col > 0) col--;
  if (a.move === 1 && col < 4) col++;
  if (a.move === 2 && row > 0) row--;
  if (a.move === 3 && row < 1) row++;
  // move=4 原地
  if (row === 0 && col === 2) { row = s.row!; col = s.col!; } // 墙
  // 取放：箱源在 col0；堆垛点在 (0,4)
  if (a.handle === 1 && col === 0 && carry === 0) carry = 1;
  if (a.handle === 2 && row === 0 && col === 4 && carry === 1 && stack < 2) {
    carry = 0;
    stack++;
  } else if (a.handle === 2 && carry === 1) {
    carry = 0; // 别处放下箱子（丢失在原地——世界规则的一部分）
  }
  return { row, col, carry, stack };
}

export class BoxBench {
  private cost = 0;
  conduct(state: Frame, action: Frame): Frame {
    validateFrame(state, BOX_SPACE.states);
    validateFrame(action, BOX_SPACE.actions);
    this.cost++;
    const n = boxTruth(state, action);
    return Object.fromEntries(BOX_SPACE.states.map(d => [d.outcome, n[d.name]!]));
  }
  get experimentsUsed(): number {
    return this.cost;
  }
}

/** 初始状态：agent 在左下角，空手，无堆垛 */
export const BOX_START: Frame = { row: 1, col: 0, carry: 0, stack: 0 };
/** "把箱子摞起来"的目标态（两层）：位置无关，只看 stack */
export const BOX_GOAL_STACK2: Frame = { row: 0, col: 4, carry: 0, stack: 2 };

/** 指令 → 目标的词汇规格（Jev 用）：目标维 stack 的自然语言标签 */
export const BOX_GOAL_PERCEPTION: { dims: readonly { name: string; values: readonly { value: number; label: string }[]; ask: string }[]; confidenceThreshold: number; clarityAsk: () => string } = {
  dims: [
    { name: "stack", values: [
      { value: 0, label: "不需要堆垛" },
      { value: 1, label: "摞一层" },
      { value: 2, label: "摞两层" },
    ], ask: "指令要求箱子堆到几层？" },
  ],
  confidenceThreshold: 0.6,
  clarityAsk: () => "指令对堆垛目标层数的要求是否明确？",
};
