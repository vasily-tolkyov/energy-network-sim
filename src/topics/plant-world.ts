/**
 * 第二主题：植物生长实验（冻结原型机制下的迁移测试）。
 * 与小球碰墙结构不同：双因素交互 + 肥料/温度加成 + 虫害门控。
 *
 * - 条件通道：品种 species(4)、光照 light(4)、水量 water(4)、肥料 fertilizer(3)、
 *   温度 temperature(3)、虫害 pest(2)；
 * - 结果通道：存活 alive(2)、生长量档 growth(4)。
 * 地面真理：存活 = 无虫害；生长量 = f(光照×水量 + 肥料 + 温度)。
 * 品种是干扰因素（课程操纵但不影响结果）。
 */

export const PLANT_CONDITION_SPECS = [
  { name: "species", bins: 4 },
  { name: "light", bins: 4 },
  { name: "water", bins: 4 },
  { name: "fertilizer", bins: 3 },
  { name: "temperature", bins: 3 },
  { name: "pest", bins: 2 },
] as const;

export const PLANT_OUTCOME_SPECS = [
  { name: "alive", bins: 2 },
  { name: "growth", bins: 4 },
] as const;

export type PConditions = Record<string, number>;
export type POutcomes = Record<string, number>;

export function plantTruth(c: PConditions): POutcomes {
  if (c.pest === 1) return { alive: 0, growth: 0 };
  const raw =
    ((c.light! + 1) * (c.water! + 1)) / 4 +
    c.fertilizer! * 0.75 +
    [0, 0.5, 1][c.temperature!]!;
  return { alive: 1, growth: Math.min(3, Math.floor(raw)) };
}

export interface PExperiment {
  readonly conditions: PConditions;
  readonly outcomes: POutcomes;
}
export interface PPair {
  readonly e0: PExperiment;
  readonly e1: PExperiment;
}
export interface PGroup {
  readonly name: string;
  readonly manipulated: string;
  readonly pairs: readonly PPair[];
}

function exp(light: number, water: number, overrides: Partial<PConditions> = {}): PExperiment {
  const conditions: PConditions = {
    species: 0,
    fertilizer: 1,
    temperature: 1,
    pest: 0,
    ...overrides,
    light,
    water,
  };
  return { conditions, outcomes: plantTruth(conditions) };
}

export function plantCurriculum(): PGroup[] {
  const pair = (a: PExperiment, b: PExperiment): PPair => ({ e0: a, e1: b });
  return [
    {
      name: "species-group",
      manipulated: "species",
      pairs: [
        pair(exp(1, 1, { species: 0 }), exp(1, 1, { species: 1 })),
        pair(exp(1, 1, { species: 0 }), exp(1, 1, { species: 2 })),
        pair(exp(1, 1, { species: 1 }), exp(1, 1, { species: 2 })),
      ],
    },
    {
      name: "light-group",
      manipulated: "light",
      pairs: [
        pair(exp(0, 1), exp(2, 1)),
        pair(exp(0, 1), exp(3, 1)),
        pair(exp(1, 1), exp(3, 1)),
      ],
    },
    {
      name: "water-group",
      manipulated: "water",
      pairs: [pair(exp(1, 0), exp(1, 2)), pair(exp(1, 0), exp(1, 3))],
    },
    {
      name: "fertilizer-group",
      manipulated: "fertilizer",
      pairs: [pair(exp(1, 1, { fertilizer: 0 }), exp(1, 1, { fertilizer: 2 }))],
    },
    {
      name: "temperature-group",
      manipulated: "temperature",
      pairs: [pair(exp(1, 1, { temperature: 0 }), exp(1, 1, { temperature: 2 }))],
    },
    {
      name: "pest-group",
      manipulated: "pest",
      pairs: [
        pair(exp(1, 1, { pest: 0 }), exp(1, 1, { pest: 1 })),
        pair(exp(2, 2, { pest: 0 }), exp(2, 2, { pest: 1 })),
      ],
    },
  ];
}

export interface PQuery {
  readonly conditions: PConditions;
  readonly truth: POutcomes;
  readonly kind: "seen-combo" | "unseen-species" | "unseen-combo" | "pest-control";
}

export function plantQueries(): PQuery[] {
  const taught = new Set<string>();
  for (const g of plantCurriculum()) {
    for (const p of g.pairs) {
      for (const e of [p.e0, p.e1]) {
        taught.add(`${e.conditions.light},${e.conditions.water},${e.conditions.pest}`);
      }
    }
  }
  const out: PQuery[] = [];
  for (let light = 0; light < 4; light++) {
    for (let water = 0; water < 4; water++) {
      for (const pest of [0, 1]) {
        for (const species of [0, 3]) {
          const conditions: PConditions = { species, light, water, fertilizer: 1, temperature: 1, pest };
          const seenCombo = taught.has(`${light},${water},${pest}`);
          const kind: PQuery["kind"] =
            pest === 1 ? "pest-control" : !seenCombo ? "unseen-combo" : species === 3 ? "unseen-species" : "seen-combo";
          out.push({ conditions, truth: plantTruth(conditions), kind });
        }
      }
    }
  }
  return out;
}
