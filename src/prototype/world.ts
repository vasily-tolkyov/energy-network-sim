/**
 * 教师侧世界模型与课程（独立地面真理，只用于生成课程和评分，不进学习核）。
 *
 * 小球碰墙世界的离散版：
 * - 条件通道：颜色 color(8)、速度 speed(4 档)、方向 direction(2)、
 *   墙刚度 wallStiffness(3)、阻尼 damping(2)、是否碰墙 hitWall(2)；
 * - 结果通道：是否反弹 rebound(2)、反弹速度 reboundSpeed(4 档)。
 * 地面真理：rebound = hitWall；reboundSpeed 档 = f(speed, wallStiffness)。
 * 颜色/方向/阻尼不影响结果（干扰因素）；是否碰墙是门控因素。
 */

export const CONDITION_SPECS = [
  { name: "color", bins: 8 },
  { name: "speed", bins: 4 },
  { name: "direction", bins: 2 },
  { name: "wallStiffness", bins: 3 },
  { name: "damping", bins: 2 },
  { name: "hitWall", bins: 2 },
] as const;

export const OUTCOME_SPECS = [
  { name: "rebound", bins: 2 },
  { name: "reboundSpeed", bins: 4 },
] as const;

export type Conditions = Record<string, number>;
export type Outcomes = Record<string, number>;

/** 地面真理（教师私有）：反弹 = 碰墙；反弹速度档 = clip(floor(speed×stiffness/2), 1..4)，档位从 1 起 */
export function groundTruth(c: Conditions): Outcomes {
  if (c.hitWall === 0) return { rebound: 0, reboundSpeed: 0 }; // 0 档保留给"无反弹"
  const rs = Math.min(3, Math.max(1, Math.floor(((c.speed! + 1) * (c.wallStiffness! + 1)) / 2) - 1));
  return { rebound: 1, reboundSpeed: rs };
}

export interface Experiment {
  readonly conditions: Conditions;
  readonly outcomes: Outcomes;
}

export interface Pair {
  readonly e0: Experiment;
  readonly e1: Experiment;
}

export interface Group {
  readonly name: string;
  /** 本组操纵的条件通道（教师课程设计信息，只用于同步与评分，不进学习核） */
  readonly manipulated: string;
  readonly pairs: readonly Pair[];
}

function exp(speed: number, stiffness: number, overrides: Partial<Conditions> = {}): Experiment {
  const conditions: Conditions = {
    color: 0,
    direction: 0,
    damping: 0,
    hitWall: 1,
    ...overrides,
    speed,
    wallStiffness: stiffness,
  };
  return { conditions, outcomes: groundTruth(conditions) };
}

/** 控制变量课程：每组只操纵一个条件通道，其余固定在共同背景上 */
export function curriculum(): Group[] {
  const pair = (a: Experiment, b: Experiment): Pair => ({ e0: a, e1: b });
  return [
    {
      name: "color-group",
      manipulated: "color",
      pairs: [
        pair(exp(1, 1, { color: 0 }), exp(1, 1, { color: 1 })),
        pair(exp(1, 1, { color: 2 }), exp(1, 1, { color: 3 })),
        pair(exp(1, 1, { color: 4 }), exp(1, 1, { color: 5 })),
      ],
    },
    {
      name: "speed-group",
      manipulated: "speed",
      pairs: [
        pair(exp(0, 1), exp(2, 1)),
        pair(exp(0, 1), exp(3, 1)),
        pair(exp(1, 1), exp(3, 1)),
      ],
    },
    {
      name: "stiffness-group",
      manipulated: "wallStiffness",
      pairs: [pair(exp(1, 0), exp(1, 2)), pair(exp(1, 0), exp(1, 1))],
    },
    {
      name: "hitwall-group",
      manipulated: "hitWall",
      pairs: [
        pair(exp(1, 1, { hitWall: 1 }), exp(1, 1, { hitWall: 0 })),
        pair(exp(2, 1, { hitWall: 1 }), exp(2, 1, { hitWall: 0 })),
      ],
    },
    {
      name: "damping-group",
      manipulated: "damping",
      pairs: [pair(exp(1, 1, { damping: 0 }), exp(1, 1, { damping: 1 }))],
    },
    {
      name: "direction-group",
      manipulated: "direction",
      pairs: [pair(exp(1, 1, { direction: 0 }), exp(1, 1, { direction: 1 }))],
    },
  ];
}

export interface Query {
  readonly conditions: Conditions;
  readonly truth: Outcomes;
  readonly kind: "seen-combo" | "unseen-color" | "unseen-combo" | "miss-wall";
}

/** 未教学查询：网格化组合 × 见过/未见颜色 × 碰墙/错过墙 */
export function queries(): Query[] {
  const taught = new Set<string>();
  for (const g of curriculum()) {
    for (const p of g.pairs) {
      for (const e of [p.e0, p.e1]) {
        taught.add(`${e.conditions.speed},${e.conditions.wallStiffness},${e.conditions.hitWall}`);
      }
    }
  }
  const out: Query[] = [];
  for (let speed = 0; speed < 4; speed++) {
    for (let stiffness = 0; stiffness < 3; stiffness++) {
      for (const hitWall of [1, 0]) {
        for (const color of [0, 6]) {
          const conditions: Conditions = { color, direction: 0, damping: 0, hitWall, speed, wallStiffness: stiffness };
          const seenCombo = taught.has(`${speed},${stiffness},${hitWall}`);
          const kind: Query["kind"] =
            hitWall === 0 ? "miss-wall" : !seenCombo ? "unseen-combo" : color === 6 ? "unseen-color" : "seen-combo";
          out.push({ conditions, truth: groundTruth(conditions), kind });
        }
      }
    }
  }
  return out;
}
