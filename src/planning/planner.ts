import { signature, validateFrame, type Frame } from "./space.js";
import type { StepPrediction, TransitionReader } from "./transition-memory.js";

export type PlanStatus = "found" | "no-known-route" | "unexplored" | "depth-limit" | "prediction-budget";
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

/** Bounded forward depth-first rollout with backtracking; no distance heuristic,
 * inverse transitions, bench access or BFS. Cache exists for this plan only.
 * Backtracking consumes the SAME global prediction budget (2×diameter×actions).
 * options.avoid：本次执行中已被捕获判为失配的条件签名（可疑转移）——
 * 行为层的即时回避，不消耗预测预算，不修改记忆中的规则（规则改判走计票）。 */
export function planGoal(model: TransitionReader, start: Frame, goal: Frame, seed: number,
  options: { avoid?: ReadonlySet<string> } = {}): GoalPlan {
  validateFrame(start, model.space.states); validateFrame(goal, model.space.states);
  const maxDepth = 2 * model.space.diameter;
  const predictBudget = maxDepth * model.actions.length;
  const predictions: StepPrediction[] = [], unknownFrontier: StepPrediction[] = [];
  const cache = new Map<string, StepPrediction[]>();
  const visited = new Set([signature(start)]);
  const goalKey = signature(goal);
  let depthLimited = false, budgetLimited = false;
  const search = (state: Frame, path: StepPrediction[]): StepPrediction[] | null => {
    if (signature(state) === goalKey) return path;
    if (path.length >= maxDepth) { depthLimited = true; return null; }
    const key = signature(state);
    let candidates = cache.get(key);
    if (!candidates) {
      candidates = [];
      for (const action of [...model.actions].sort((a, b) => a.id - b.id)) {
        if (options.avoid?.has(model.conditionKey ? model.conditionKey(state, action) : signature({ ...state, ...action.values }))) continue;
        if (predictions.length >= predictBudget) { budgetLimited = true; break; }
        const p = model.predict(state, action, (seed + predictions.length) >>> 0);
        predictions.push(p); candidates.push(p);
        if (p.kind === "unknown") unknownFrontier.push(p);
      }
      cache.set(key, candidates);
    }
    const ranked = [...candidates].filter(p => p.kind !== "infeasible")
      .sort((a, b) => candidateTier(a, visited) - candidateTier(b, visited) || a.action.id - b.action.id);
    for (const p of ranked) {
      if (p.kind !== "usable" || !p.next) continue;
      const nextKey = signature(p.next);
      if (visited.has(nextKey)) continue;
      visited.add(nextKey);
      const result = search(p.next, [...path, p]);
      visited.delete(nextKey);
      if (result) return result;
      if (budgetLimited) break;
    }
    return null;
  };
  const steps = search(start, []);
  return { start: { ...start }, goal: { ...goal }, status: steps ? "found" : budgetLimited ? "prediction-budget"
    : depthLimited ? "depth-limit" : unknownFrontier.length ? "unexplored" : "no-known-route",
    steps: steps ?? [], predictions, unknownFrontier, maxDepth, predictBudget };
}
