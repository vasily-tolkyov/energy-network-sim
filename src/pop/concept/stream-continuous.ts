import { contTruth, type CValues } from "./world-continuous.js";

/**
 * 连续世界流（M1）：三只小球的连续条件演化。
 * 事件脚本含：已学值域变化、门控翻转、未见组合、未形成区域边缘试探。
 * 每帧给出各对象的连续条件与连续观察结果（教师侧 contTruth 只供生成与评分）。
 */

export interface ContinuousFrameObject {
  readonly id: string;
  readonly conditions: CValues;
  readonly outcomes: CValues;
}

export interface ContinuousFrame {
  readonly tick: number;
  readonly objects: readonly ContinuousFrameObject[];
}

function obj(id: string, conditions: CValues): ContinuousFrameObject {
  return { id, conditions, outcomes: contTruth(conditions) };
}

export function* continuousStream(): Generator<ContinuousFrame> {
  const A: CValues = { color: 0.5, direction: 0.2, damping: 0.2, hitWall: 1, speed: 1.5, wallStiffness: 1.0 };
  const B: CValues = { color: 2.5, direction: 0.2, damping: 0.2, hitWall: 1, speed: 2.5, wallStiffness: 1.0 };
  const C: CValues = { color: 4.5, direction: 0.2, damping: 0.2, hitWall: 1, speed: 1.2, wallStiffness: 2.6 };
  const events: Record<number, () => void> = {
    // 已学值域内的连续变化（不应触发偏差）
    8: () => { A.speed = 1.7; },
    10: () => { A.speed = 3.2; },
    // 门控翻转：反弹 → 不弹
    20: () => { B.hitWall = 0; },
    // 未见组合（速度3.2 × 刚度2.6）
    30: () => { A.wallStiffness = 2.6; },
    // 未形成区域边缘（速度 4.5 介于已形成 3.2 与 6.5 之间）
    40: () => { C.speed = 4.5; },
    // 门控恢复
    55: () => { B.hitWall = 1; },
  };
  for (let tick = 1; tick <= 65; tick++) {
    events[tick]?.();
    yield {
      tick,
      objects: [obj("A", { ...A }), obj("B", { ...B }), obj("C", { ...C })],
    };
  }
}
