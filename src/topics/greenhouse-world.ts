import { mulberry32 } from "../prng.js";

/**
 * 温室世界（陌生场景）：多因素交互的规则发现对象。
 * 真值只存在于实验台内部——对外只暴露**文字观察**，
 * 由决策后端（Jev）把文字解析为结构化结果，学习核永远看不到真值。
 */

export const GH_COND_DIMS = [
  { name: "light", min: 0, max: 3 },    // 0无 1弱 2中 3强
  { name: "water", min: 0, max: 3 },    // 0干 1少 2适 3多
  { name: "nutrient", min: 0, max: 3 }, // 0无 1少 2适 3多
  { name: "temp", min: 0, max: 2 },     // 0冷 1适 2热
  { name: "decor", min: 0, max: 1 },    // 干扰维：装饰
] as const;
export const GH_OUTCOME_DIMS = [
  { name: "growth", min: 0, max: 2 },   // 0枯萎 1存活 2茂盛
  { name: "yield", min: 0, max: 3 },    // 0..3 产量档
] as const;

export const GH_GRID: Readonly<Record<string, readonly number[]>> = {
  light: [0, 1, 2, 3],
  water: [0, 1, 2, 3],
  nutrient: [0, 1, 2, 3],
  temp: [0, 1, 2],
  decor: [0, 1],
};

export function ghTruth(c: Record<string, number>): Record<string, number> {
  const { light, water, nutrient } = c as { light: number; water: number; nutrient: number };
  let growth: number;
  if (water === 0) growth = 0;
  else if (nutrient === 3 && water === 3) growth = 0; // 肥害：多肥+多水
  else if (light >= 2 && (water === 1 || water === 2)) growth = 2;
  else if (light === 3 && water === 3 && nutrient !== 3) growth = 2;
  else growth = 1;
  let yield_: number;
  if (growth === 0) yield_ = 0;
  else {
    yield_ = growth === 2 ? 2 : 1;
    if (nutrient === 1 || nutrient === 2) yield_ = Math.min(3, yield_ + 1);
  }
  return { growth, yield: yield_ };
}

const PROSE: Record<string, readonly string[]> = {
  "g0": ["植株叶片枯黄下垂，茎秆软塌，毫无生气。", "整株蔫萎发黄，叶缘焦枯，眼看活不成了。"],
  "g1": ["植株维持基本存活，叶色偏淡，生长缓慢但没有恶化。", "植株状态平平，活着但不见起色。"],
  "g2": ["植株叶色浓绿，茎秆挺拔，新芽不断冒出。", "植株长势旺盛，叶片肥厚油亮，一派生机。"],
  // 产量四档措辞刻意拉开距离（"中产被误读为高产"的实测教训）
  "y0": ["枝头空空，颗粒无收，一个果也没有。", "完全没有坐果。"],
  "y1": ["枝头只挂着零星两三个小果，收成很少。", "坐果寥寥无几，产量偏低。"],
  "y2": ["枝头大约一半位置挂着果，中等收成，不算多也不算少。", "挂果约五成，产量中等。"],
  "y3": ["果实压弯了枝条，密密麻麻，产量极高，明显是丰收。", "挂果多到把枝条压垂，产量惊人。"],
};

/** 把真值渲染为文字观察（带措辞变体；决策后端只收到这段文字） */
export function ghDescribe(outcomes: Record<string, number>, seed: number): string {
  const rng = mulberry32(seed);
  const g = PROSE[`g${outcomes.growth}`]!;
  const y = PROSE[`y${outcomes.yield}`]!;
  return `${g[Math.floor(rng() * g.length)]}${y[Math.floor(rng() * y.length)]}`;
}

/** 温室感知规格：Jev 把文字观察解析回结构化结果 */
export const GH_PERCEPTION = {
  growth: {
    ask: "这段描述中植株的生长状态是哪一种？",
    options: [
      { value: 0, label: "枯萎" },
      { value: 1, label: "存活" },
      { value: 2, label: "茂盛" },
    ],
  },
  yield: {
    ask: "这段描述中产量水平是哪一档？",
    options: [
      { value: 0, label: "无产", desc: "完全没有果实，颗粒无收" },
      { value: 1, label: "低产", desc: "只有零星几个小果，收成很少" },
      { value: 2, label: "中产", desc: "约一半位置挂果，中等收成" },
      { value: 3, label: "高产", desc: "果实多到压弯枝条，明显丰收" },
    ],
  },
} as const;
