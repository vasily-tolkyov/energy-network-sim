import { mulberry32 } from "../prng.js";

/**
 * 机房运维世界（迁移性测试的第二主题）：与温室世界完全不同的主题与规则结构。
 * 真值只存在于实验台内部——对外只暴露文字观察。
 * 规则含交互：过压+无冷却 = 烧毁故障；高负载+弱冷却 = 过热故障。
 */

export const SR_COND_DIMS = [
  { name: "load", min: 0, max: 3 },    // 0闲置 1低 2中 3高
  { name: "cooling", min: 0, max: 2 }, // 0关 1弱 2强
  { name: "power", min: 0, max: 2 },   // 0欠压 1正常 2过压
  { name: "dust", min: 0, max: 1 },    // 干扰维：灰尘
] as const;
export const SR_OUTCOME_DIMS = [
  { name: "health", min: 0, max: 2 },  // 0故障 1亚稳 2健康
  { name: "uptime", min: 0, max: 3 },  // 0..3 产出档
] as const;

export const SR_GRID: Readonly<Record<string, readonly number[]>> = {
  load: [0, 1, 2, 3],
  cooling: [0, 1, 2],
  power: [0, 1, 2],
  dust: [0, 1],
};

export function srTruth(c: Record<string, number>): Record<string, number> {
  const { load, cooling, power } = c as { load: number; cooling: number; power: number };
  let health: number;
  if (power === 2 && cooling === 0) health = 0;      // 过压无冷却 → 烧毁
  else if (load === 3 && cooling <= 1) health = 0;   // 高负载散热不足 → 过热
  else if (power === 1 && cooling >= 1 && load <= 2) health = 2;
  else health = 1;
  let uptime: number;
  if (health === 0) uptime = 0;
  else {
    uptime = health === 2 ? 2 : 1;
    if (health === 2 && load === 3) uptime = 3;
    if (power === 0) uptime = Math.min(uptime, 1); // 欠压封顶
  }
  return { health, uptime };
}

const PROSE: Record<string, readonly string[]> = {
  "h0": ["机柜冒烟报警，服务全部中断，指示灯全灭。", "机器过热宕机，风扇狂转后停转，一片死寂。"],
  "h1": ["机器勉强运行，偶尔卡顿，告警灯时亮时灭。", "服务断断续续，性能不稳但没有彻底停摆。"],
  "h2": ["机器运行平稳，温度正常，指示灯常绿。", "服务流畅无卡顿，一切指标健康。"],
  "u0": ["完全没有服务产出，请求全部失败。", "输出为零。"],
  "u1": ["服务产出稀少，只能处理零星请求。", "吞吐量很低。"],
  "u2": ["服务产出中等，吞吐稳定在五成左右。", "请求处理量中规中矩。"],
  "u3": ["服务满负荷高效运转，吞吐量拉满。", "产出极高，所有请求秒回。"],
};

export function srDescribe(outcomes: Record<string, number>, seed: number): string {
  const rng = mulberry32(seed);
  const h = PROSE[`h${outcomes.health}`]!;
  const u = PROSE[`u${outcomes.uptime}`]!;
  return `${h[Math.floor(rng() * h.length)]}${u[Math.floor(rng() * u.length)]}`;
}

export const SR_PERCEPTION = {
  health: {
    ask: "这段描述中机房的运行状态是哪一种？",
    options: [
      { value: 0, label: "故障", desc: "宕机、冒烟、服务中断" },
      { value: 1, label: "亚稳", desc: "勉强运行、卡顿、不稳定" },
      { value: 2, label: "健康", desc: "平稳、流畅、指标正常" },
    ],
  },
  uptime: {
    ask: "这段描述中服务产出的吞吐水平是哪一档？",
    options: [
      { value: 0, label: "零产出", desc: "请求全部失败" },
      { value: 1, label: "低吞吐", desc: "只能处理零星请求" },
      { value: 2, label: "中吞吐", desc: "稳定五成左右" },
      { value: 3, label: "高吞吐", desc: "满负荷高效" },
    ],
  },
} as const;
