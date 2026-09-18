/**
 * 第三主题（独立测试）：化学反应实验。
 * 与小球碰墙（单门控）、植物生长（乘性交互+单门控）、mod-8/平滑（纯数学结构）
 * 都不同的地面真理：
 * - 门控是析取：催化剂 OR 高温（temp=3）都能使反应发生；
 * - 速率是浓度/温度/搅拌三个因素的可加组合；
 * - 容器材质 vessel 是干扰因素（课程操纵但不影响结果）。
 *
 * 本文件只是教师侧世界模型：生成课程与评分，不进学习核。
 *
 * 课程有两个变体（机制无关，纯教师侧设计）：
 * - v1 稀疏门控覆盖：催化剂组只在 conc=1 背景做实验——"不反应"经验背景单一；
 * - v2 均衡门控覆盖：催化剂组在 conc 全档、temp=0 背景各做一次对换，
 *   温度组补一对 conc=2 背景跨越门控边界的实验。
 *   动机与小球世界 miss-wall 覆盖组相同：析取门控的"关闭"分支必须在
 *   多个背景上被教师演示过，最近匹配才有可比的支持面（实测 v1 下
 *   no-reaction 查询在所有候选核上等距，退火近似扔硬币）。
 */

import type { Conditions, Experiment, Group, Outcomes, Pair } from "../prototype/world.js";

export const CHEM_CONDITION_SPECS = [
  { name: "catalyst", bins: 2 },
  { name: "concentration", bins: 4 },
  { name: "temperature", bins: 4 },
  { name: "stirring", bins: 3 },
  { name: "vessel", bins: 3 },
] as const;

export const CHEM_OUTCOME_SPECS = [
  { name: "reacted", bins: 2 },
  { name: "rate", bins: 4 },
] as const;

/** 地面真理（教师私有）：反应 = 催化剂 OR 高温；速率档 = 浓度×0.75 + 温度×0.5 + 搅拌×0.5 */
export function chemTruth(c: Conditions): Outcomes {
  if (c.catalyst === 0 && c.temperature! < 3) return { reacted: 0, rate: 0 };
  const raw = c.concentration! * 0.75 + c.temperature! * 0.5 + c.stirring! * 0.5;
  return { reacted: 1, rate: Math.min(3, Math.floor(raw)) };
}

function exp(overrides: Partial<Conditions>): Experiment {
  const conditions: Conditions = {
    catalyst: 1,
    concentration: 1,
    temperature: 1,
    stirring: 1,
    vessel: 0,
    ...overrides,
  };
  return { conditions, outcomes: chemTruth(conditions) };
}

/** 教师课程：5 组控制变量实验，每组只操纵一个因素（容器组换两种背景） */
export function chemCurriculum(balancedGate = false): Group[] {
  const pair = (a: Experiment, b: Experiment): Pair => ({ e0: a, e1: b });
  const catalystPairs: Pair[] = [
    pair(exp({ catalyst: 0 }), exp({ catalyst: 1 })),
    pair(exp({ catalyst: 0, temperature: 2 }), exp({ catalyst: 1, temperature: 2 })),
  ];
  const temperaturePairs: Pair[] = [
    pair(exp({ temperature: 0 }), exp({ temperature: 2 })),
    // 跨越析取门控边界：无催化剂时高温独自打开门
    pair(exp({ temperature: 2, catalyst: 0 }), exp({ temperature: 3, catalyst: 0 })),
  ];
  if (balancedGate) {
    // 门控关闭分支的多背景覆盖（v2）：每个浓度档、temp=0 各演示一次催化剂对换
    catalystPairs.push(
      pair(exp({ catalyst: 0, concentration: 0 }), exp({ catalyst: 1, concentration: 0 })),
      pair(exp({ catalyst: 0, concentration: 2 }), exp({ catalyst: 1, concentration: 2 })),
      pair(exp({ catalyst: 0, concentration: 3 }), exp({ catalyst: 1, concentration: 3 })),
      pair(exp({ catalyst: 0, temperature: 0 }), exp({ catalyst: 1, temperature: 0 })),
    );
    // 门控开启分支（高温侧）也补一个背景
    temperaturePairs.push(
      pair(exp({ temperature: 2, catalyst: 0, concentration: 2 }), exp({ temperature: 3, catalyst: 0, concentration: 2 })),
    );
  }
  return [
    {
      name: "vessel-group",
      manipulated: "vessel",
      pairs: [
        pair(exp({ vessel: 0 }), exp({ vessel: 1 })),
        pair(
          exp({ vessel: 0, concentration: 2, temperature: 2 }),
          exp({ vessel: 1, concentration: 2, temperature: 2 }),
        ),
      ],
    },
    { name: "catalyst-group", manipulated: "catalyst", pairs: catalystPairs },
    {
      name: "concentration-group",
      manipulated: "concentration",
      pairs: [
        pair(exp({ concentration: 0 }), exp({ concentration: 2 })),
        pair(exp({ concentration: 1 }), exp({ concentration: 3 })),
        pair(exp({ concentration: 2 }), exp({ concentration: 3 })),
      ],
    },
    { name: "temperature-group", manipulated: "temperature", pairs: temperaturePairs },
    {
      name: "stirring-group",
      manipulated: "stirring",
      pairs: [pair(exp({ stirring: 0 }), exp({ stirring: 2 }))],
    },
  ];
}

export interface CQuery {
  readonly conditions: Conditions;
  readonly truth: Outcomes;
  readonly kind: "seen-combo" | "unseen-vessel" | "unseen-combo" | "no-reaction-control";
}

/** 查询全集：浓度×温度×催化剂 全网格 × 容器{0=教过, 2=从未出现}，搅拌固定中档 */
export function chemQueries(curriculum: Group[] = chemCurriculum()): CQuery[] {
  const taught = new Set<string>();
  for (const g of curriculum) {
    for (const p of g.pairs) {
      for (const e of [p.e0, p.e1]) {
        taught.add(`${e.conditions.concentration},${e.conditions.temperature},${e.conditions.catalyst}`);
      }
    }
  }
  const out: CQuery[] = [];
  for (let concentration = 0; concentration < 4; concentration++) {
    for (let temperature = 0; temperature < 4; temperature++) {
      for (const catalyst of [0, 1]) {
        for (const vessel of [0, 2]) {
          const conditions: Conditions = { catalyst, concentration, temperature, stirring: 1, vessel };
          const seenCombo = taught.has(`${concentration},${temperature},${catalyst}`);
          const kind: CQuery["kind"] =
            catalyst === 0 && temperature < 3
              ? "no-reaction-control"
              : vessel === 2
                ? "unseen-vessel"
                : seenCombo
                  ? "seen-combo"
                  : "unseen-combo";
          out.push({ conditions, truth: chemTruth(conditions), kind });
        }
      }
    }
  }
  return out;
}
