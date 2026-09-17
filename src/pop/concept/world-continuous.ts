/**
 * 连续版小球世界：条件与结果为实数，地面真理由连续函数计算。
 * 课程结构与离散版一致（控制变量对），但数值连续——
 * 值的分档不存在，概念必须由共现自形成。
 * 地面真理只用于课程生成与评分，不进学习核。
 */

export interface ContinuousDim {
  readonly name: string;
  readonly min: number;
  readonly max: number;
}

export const CONT_CONDITION_DIMS: ContinuousDim[] = [
  { name: "color", min: 0, max: 7 },
  { name: "speed", min: 0, max: 8 },
  { name: "direction", min: 0, max: 1 },
  { name: "wallStiffness", min: 0, max: 3 },
  { name: "damping", min: 0, max: 1 },
  { name: "hitWall", min: 0, max: 1 },
];

export const CONT_OUTCOME_DIMS: ContinuousDim[] = [
  { name: "rebound", min: 0, max: 1 },
  { name: "reboundSpeed", min: 0, max: 4 },
];

export type CValues = Record<string, number>;

/** 连续地面真理：反弹 = 碰墙（>0.5）；反弹速度 = clip(速度×刚度/2, 0..4) */
export function contTruth(c: CValues): CValues {
  if (c.hitWall! < 0.5) return { rebound: 0, reboundSpeed: 0 };
  return {
    rebound: 1,
    reboundSpeed: Math.min(4, Math.max(0.5, (c.speed! * c.wallStiffness!) / 2)),
  };
}

export interface ContExperiment {
  readonly conditions: CValues;
  readonly outcomes: CValues;
}

export interface ContPair {
  readonly e0: ContExperiment;
  readonly e1: ContExperiment;
}

export interface ContGroup {
  readonly name: string;
  readonly manipulated: string;
  readonly pairs: readonly ContPair[];
}

function exp(speed: number, stiffness: number, overrides: Partial<CValues> = {}): ContExperiment {
  const conditions: CValues = {
    color: 0.5,
    direction: 0.2,
    damping: 0.2,
    hitWall: 1,
    ...overrides,
    speed,
    wallStiffness: stiffness,
  };
  return { conditions, outcomes: contTruth(conditions) };
}

/** 连续控制变量课程：与离散版同构，数值取量程内的分散实数 */
export function contCurriculum(): ContGroup[] {
  const pair = (a: ContExperiment, b: ContExperiment): ContPair => ({ e0: a, e1: b });
  return [
    {
      name: "color-group",
      manipulated: "color",
      pairs: [
        pair(exp(1.2, 1.0, { color: 0.5 }), exp(1.2, 1.0, { color: 2.5 })),
        pair(exp(1.2, 1.0, { color: 3.5 }), exp(1.2, 1.0, { color: 5.5 })),
        pair(exp(1.2, 1.0, { color: 1.0 }), exp(1.2, 1.0, { color: 6.0 })),
      ],
    },
    {
      name: "speed-group",
      manipulated: "speed",
      pairs: [
        pair(exp(0.8, 1.0), exp(3.2, 1.0)),
        pair(exp(0.8, 1.0), exp(6.5, 1.0)),
        pair(exp(2.0, 1.0), exp(6.5, 1.0)),
      ],
    },
    {
      name: "stiffness-group",
      manipulated: "wallStiffness",
      pairs: [pair(exp(1.5, 0.4), exp(1.5, 2.6)), pair(exp(1.5, 0.4), exp(1.5, 1.2))],
    },
    {
      name: "hitwall-group",
      manipulated: "hitWall",
      pairs: [
        pair(exp(1.5, 1.0, { hitWall: 1 }), exp(1.5, 1.0, { hitWall: 0 })),
        pair(exp(2.5, 1.0, { hitWall: 1 }), exp(2.5, 1.0, { hitWall: 0 })),
      ],
    },
    {
      // miss 经验的背景覆盖（文档"背景覆盖"职责）：多种速度/刚度/颜色背景下的
      // 碰墙↔错过墙对照，弥补 yes/no 经验的课程级失衡
      name: "misswall-coverage-group",
      manipulated: "hitWall",
      pairs: [
        pair(exp(0.8, 0.4, { color: 2.5, hitWall: 1 }), exp(0.8, 0.4, { color: 2.5, hitWall: 0 })),
        pair(exp(3.2, 2.6, { color: 4.5, hitWall: 1 }), exp(3.2, 2.6, { color: 4.5, hitWall: 0 })),
        pair(exp(6.5, 1.0, { color: 0.5, hitWall: 1 }), exp(6.5, 1.0, { color: 0.5, hitWall: 0 })),
        pair(exp(1.5, 2.6, { color: 3.5, hitWall: 1 }), exp(1.5, 2.6, { color: 3.5, hitWall: 0 })),
      ],
    },
    {
      name: "damping-group",
      manipulated: "damping",
      pairs: [pair(exp(1.5, 1.0, { damping: 0.2 }), exp(1.5, 1.0, { damping: 0.9 }))],
    },
    {
      name: "direction-group",
      manipulated: "direction",
      pairs: [pair(exp(1.5, 1.0, { direction: 0.2 }), exp(1.5, 1.0, { direction: 0.8 }))],
    },
  ];
}

export interface ContQuery {
  readonly conditions: CValues;
  readonly truth: CValues;
  readonly kind: "seen-region" | "novel-value" | "novel-combo" | "miss-wall";
}

/** 查询：连续网格 × 见过/未见值域 × 碰墙/错过墙 */
export function contQueries(): ContQuery[] {
  const out: ContQuery[] = [];
  const seenSpeeds = new Set([0.8, 1.2, 1.5, 2.0, 2.5, 3.2, 6.5]);
  for (const speed of [0.8, 1.5, 2.5, 4.0, 6.5]) {
    for (const stiffness of [0.4, 1.0, 2.6]) {
      for (const hitWall of [1, 0]) {
        for (const color of [0.5, 4.5]) {
          const conditions: CValues = { color, direction: 0.2, damping: 0.2, hitWall, speed, wallStiffness: stiffness };
          const kind: ContQuery["kind"] =
            hitWall === 0 ? "miss-wall"
            : color === 4.5 ? "novel-value"
            : !seenSpeeds.has(speed) || stiffness === 2.6 ? "novel-combo"
            : "seen-region";
          out.push({ conditions, truth: contTruth(conditions), kind });
        }
      }
    }
  }
  return out;
}
