import { SceneParser } from "../perception/scene-parser.js";
import type { DecisionBackend } from "../perception/decision-backend.js";
import { ActionExecutor } from "../execution/action-executor.js";
import { TransitionMemory } from "../planning/transition-memory.js";
import { collectTransitions } from "../planning/collect.js";
import { planGoal } from "../planning/planner.js";
import { executeGoal, type GoalExecution } from "../planning/execute.js";
import { frames, signature, type Frame } from "../planning/space.js";
import { BOX_SPACE, BOX_START, BOX_GOAL_STACK2, BOX_GOAL_PERCEPTION, BoxBench } from "../topics/box-world.js";

/**
 * 交互 Agent（箱子世界）：自然语言指令 → Jev 解析为目标帧 →
 * 自主探索出的规律支撑规划执行；无指令时自设目标探索。
 * 全部规律（墙挡、取放、堆垛）由能量网络从经验学——不预教。
 */

export interface AgentReport {
  readonly kind: "instruction" | "idle";
  readonly instruction?: string;
  readonly goal: Frame;
  readonly planStatus: string;
  readonly reached: boolean;
  readonly steps: number;
  readonly replans: number;
  readonly commands: readonly string[];
  readonly note: string;
}

export class BoxAgent {
  readonly mem = new TransitionMemory(BOX_SPACE);
  private readonly bench = new BoxBench();
  private readonly parser: SceneParser;
  private readonly executor: ActionExecutor;
  /** 已观察过的状态签名集（自设目标用：只挑没去过的） */
  private readonly visited = new Set<string>();

  // 默认预算 = 世界全网格（状态×动作）——部分覆盖会导致规划在未知区耗尽预算
  // （实测 200/900 时规划报 prediction-budget，首次完整覆盖后 found）
  constructor(
    private readonly backend: DecisionBackend,
    private readonly collectBudget = [...BOX_SPACE.states, ...BOX_SPACE.actions].reduce((n, d) => n * d.bins, 1),
  ) {
    this.parser = new SceneParser(backend);
    this.executor = new ActionExecutor(backend);
  }

  /** 自由探索：语料式覆盖全部状态×动作（方法先验），把转移规律学进网络 */
  explore(seed = 1): void {
    collectTransitions(this.mem, this.bench, this.collectBudget, seed);
    for (const s of frames(BOX_SPACE.states)) {
      // 记录"去过哪些状态"：以读回能给出确定答案为准
      if (this.mem.mem.coreFieldCoverage(s) >= this.mem.mem.net.threshold) this.visited.add(signature(s));
    }
  }

  /** 指令模式："把箱子摞起来" → 目标帧 → 规划 → 执行 */
  async instruct(text: string, seed = 1): Promise<AgentReport> {
    const got = await this.parser.parse(text, BOX_GOAL_PERCEPTION);
    if (got.unknownDims.length > 0) {
      return {
        kind: "instruction", instruction: text, goal: {}, planStatus: "unparsed",
        reached: false, steps: 0, replans: 0, commands: [],
        note: `指令的目标层数不明确（unknownDims=${got.unknownDims.join(",")}），如实拒动`,
      };
    }
    const stack = got.frame.stack ?? 0;
    if (stack === 0) {
      return {
        kind: "instruction", instruction: text, goal: {}, planStatus: "no-goal",
        reached: false, steps: 0, replans: 0, commands: [],
        note: "指令不含堆垛目标，无事可做",
      };
    }
    const goal: Frame = { ...BOX_GOAL_STACK2, stack };
    const plan = planGoal(this.mem, BOX_START, goal, seed);
    if (plan.status !== "found") {
      return {
        kind: "instruction", instruction: text, goal, planStatus: plan.status,
        reached: false, steps: 0, replans: 0, commands: [],
        note: `没有已知路线（${plan.status}）——需要先探索`,
      };
    }
    const exec = executeGoal(this.mem, this.bench, BOX_START, goal, seed);
    const commands = await this.executor.mapChain(
      exec.steps.map(s => s.forecast.action.values),
      i => `在 ${JSON.stringify(exec.steps[i]?.state ?? BOX_START)}`,
      {
        environment: "箱子世界",
        actions: [
          { dimension: "move", commands: { "0": "左移", "1": "右移", "2": "上移", "3": "下移" } },
          { dimension: "handle", commands: { "0": "无操作", "1": "取箱子", "2": "放箱子" } },
        ],
      },
    );
    return {
      kind: "instruction", instruction: text, goal, planStatus: plan.status,
      reached: exec.reached, steps: exec.steps.length, replans: exec.replans.length,
      commands: records_(commands), note: exec.terminationReason,
    };
  }

  /** 无指令时：自设目标 = 一个尚未去过的状态，规划并前往（内驱探索） */
  async idleCuriosity(seed = 1): Promise<AgentReport> {
    const all = frames(BOX_SPACE.states);
    const unvisited = all.filter(s => !this.visited.has(signature(s)));
    if (unvisited.length === 0) {
      return {
        kind: "idle", goal: {}, planStatus: "nothing-new",
        reached: false, steps: 0, replans: 0, commands: [],
        note: "所有状态都探索过了，没有新目标可设",
      };
    }
    const goal = unvisited[0]!;
    const plan = planGoal(this.mem, BOX_START, goal, seed);
    if (plan.status !== "found") {
      // 没学过的区域可能暂时不可达——如实报告，不编造
      return {
        kind: "idle", goal, planStatus: plan.status,
        reached: false, steps: 0, replans: 0, commands: [],
        note: `自设目标 ${JSON.stringify(goal)} 暂无已知路线`,
      };
    }
    const exec = executeGoal(this.mem, this.bench, BOX_START, goal, seed);
    this.visited.add(signature(exec.finalState));
    return {
      kind: "idle", goal, planStatus: plan.status,
      reached: exec.reached, steps: exec.steps.length, replans: exec.replans.length,
      commands: [], note: `自设目标：前往未探索状态 ${JSON.stringify(goal)}（${exec.terminationReason}）`,
    };
  }
}
const records_ = (recs: { command: string }[]) => recs.map(r => r.command);
