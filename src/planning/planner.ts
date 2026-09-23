import { signature, validateFrame, type Frame } from "./space.js";
import type { StepPrediction, TransitionReader } from "./transition-memory.js";

export type PlanStatus = "found" | "no-known-route" | "unexplored" | "depth-limit" | "prediction-budget" | "interrupted";
export interface GoalPlan {
  start: Frame; goal: Frame; status: PlanStatus; steps: StepPrediction[];
  predictions: StepPrediction[]; unknownFrontier: StepPrediction[];
  maxDepth: number; predictBudget: number;
}
export function candidateTier(p: StepPrediction, visited: ReadonlySet<string>): number {
  if (p.next && visited.has(signature(p.next))) return 3;
  if (p.kind === "usable" && p.observed) return 0;
  if (p.kind === "usable" && p.snapshot.converged) return 1;
  return 2;
}

interface SearchCtx {
  model: TransitionReader; goalKey: string; maxDepth: number; predictBudget: number;
  predictions: StepPrediction[]; unknownFrontier: StepPrediction[];
  cache: Map<string, StepPrediction[]>; visited: Set<string>; bestDepth: Map<string, number>;
  avoid?: ReadonlySet<string>; seed: number;
  depthLimited: boolean; budgetLimited: boolean; interrupted: boolean;
}

/** Bounded forward depth-first rollout with backtracking; no distance heuristic,
 * inverse transitions, bench access or BFS. Cache exists for this plan only.
 * Backtracking consumes the SAME global prediction budget (2×diameter×actions).
 * bestDepth 换位表：记录每个状态已被完整探索的最浅深度；确定性转移下，
 * 从不更深的位置重复探索同一状态不可能得到新结果，剪掉。否则预算未耗尽时
 * DFS 会在已缓存状态间枚举简单路径（可达图 < 预算/动作数 时永不完结）。
 * options.avoid：本次执行中已被捕获判为失配的条件签名（可疑转移）——
 * 行为层的即时回避，不消耗预测预算，不修改记忆中的规则（规则改判走计票）。
 *
 * 生成器形态：每产生一条预测 yield 一次——同步驱动（planGoal）直接跑到底，
 * 与重构前行为完全一致；异步驱动（planGoalAsync）据此周期性让出事件循环
 * （实体宿主的需要：MC 服务器 30s 收不到 keepalive 就踢人，纯同步规划会杀死
 * 连接），并支持 shouldAbort 中断（用户指令抢占自主目标的规划）。 */
function* searchGen(ctx: SearchCtx, state: Frame, path: StepPrediction[]): Generator<void, StepPrediction[] | null> {
  if (ctx.interrupted) return null;
  if (signature(state) === ctx.goalKey) return path;
  if (path.length >= ctx.maxDepth) { ctx.depthLimited = true; return null; }
  const key = signature(state);
  let candidates = ctx.cache.get(key);
  if (!candidates) {
    candidates = [];
    for (const action of [...ctx.model.actions].sort((a, b) => a.id - b.id)) {
      if (ctx.avoid?.has(ctx.model.conditionKey ? ctx.model.conditionKey(state, action) : signature({ ...state, ...action.values }))) continue;
      if (ctx.predictions.length >= ctx.predictBudget) { ctx.budgetLimited = true; break; }
      const p = ctx.model.predict(state, action, (ctx.seed + ctx.predictions.length) >>> 0);
      ctx.predictions.push(p); candidates.push(p);
      if (p.kind === "unknown") ctx.unknownFrontier.push(p);
      yield;
      if (ctx.interrupted) return null;
    }
    ctx.cache.set(key, candidates);
  }
  const ranked = [...candidates].filter(p => p.kind !== "infeasible")
    .sort((a, b) => candidateTier(a, ctx.visited) - candidateTier(b, ctx.visited) || a.action.id - b.action.id);
  for (const p of ranked) {
    if (p.kind !== "usable" || !p.next) continue;
    const nextKey = signature(p.next);
    if (ctx.visited.has(nextKey)) continue;
    const seenDepth = ctx.bestDepth.get(nextKey);
    if (seenDepth !== undefined && seenDepth <= path.length + 1) continue;
    ctx.visited.add(nextKey);
    ctx.bestDepth.set(nextKey, path.length + 1);
    const result = yield* searchGen(ctx, p.next, [...path, p]);
    ctx.visited.delete(nextKey);
    if (result) return result;
    if (ctx.budgetLimited || ctx.interrupted) break;
  }
  return null;
}

function makeCtx(model: TransitionReader, start: Frame, goal: Frame, seed: number,
  avoid?: ReadonlySet<string>): SearchCtx {
  validateFrame(start, model.space.states); validateFrame(goal, model.space.states);
  const maxDepth = 2 * model.space.diameter;
  return { model, goalKey: signature(goal), maxDepth, predictBudget: maxDepth * model.actions.length,
    predictions: [], unknownFrontier: [], cache: new Map(),
    visited: new Set([signature(start)]), bestDepth: new Map([[signature(start), 0]]),
    avoid, seed, depthLimited: false, budgetLimited: false, interrupted: false };
}

function finish(ctx: SearchCtx, start: Frame, goal: Frame, steps: StepPrediction[] | null): GoalPlan {
  return { start: { ...start }, goal: { ...goal },
    status: steps ? "found" : ctx.interrupted ? "interrupted" : ctx.budgetLimited ? "prediction-budget"
      : ctx.depthLimited ? "depth-limit" : ctx.unknownFrontier.length ? "unexplored" : "no-known-route",
    steps: steps ?? [], predictions: ctx.predictions, unknownFrontier: ctx.unknownFrontier,
    maxDepth: ctx.maxDepth, predictBudget: ctx.predictBudget };
}

/** 同步规划（语义与行为不变；用于测试与无事件循环保活需求的环境）。 */
export function planGoal(model: TransitionReader, start: Frame, goal: Frame, seed: number,
  options: { avoid?: ReadonlySet<string> } = {}): GoalPlan {
  const ctx = makeCtx(model, start, goal, seed, options.avoid);
  const gen = searchGen(ctx, start, []);
  let r = gen.next();
  while (!r.done) r = gen.next();
  return finish(ctx, start, goal, r.value);
}

/** 异步规划：每 yieldEvery 条预测让出一次事件循环；shouldAbort 触发后按 interrupted 收尾。 */
export async function planGoalAsync(model: TransitionReader, start: Frame, goal: Frame, seed: number,
  options: { avoid?: ReadonlySet<string>; yieldEvery?: number; shouldAbort?: () => boolean } = {}): Promise<GoalPlan> {
  const ctx = makeCtx(model, start, goal, seed, options.avoid);
  const gen = searchGen(ctx, start, []);
  const every = Math.max(1, options.yieldEvery ?? 5);
  let n = 0;
  let r = gen.next();
  while (!r.done) {
    if (options.shouldAbort?.()) ctx.interrupted = true;
    if (++n % every === 0) await new Promise((res) => setImmediate(res));
    r = gen.next();
  }
  return finish(ctx, start, goal, r.value);
}
