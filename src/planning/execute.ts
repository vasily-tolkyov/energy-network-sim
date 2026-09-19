import { captureClassify } from "../pop/attention/capture.js";
import { integer } from "../validate.js";
import { observedState, signature, type Frame, type TransitionBench } from "./space.js";
import { planGoal, type GoalPlan } from "./planner.js";
import { TransitionMemory, type StepPrediction } from "./transition-memory.js";

export interface ExecutionStep {
  state: Frame; forecast: StepPrediction; observation: Frame; actual: Frame;
  capture: ReturnType<typeof captureClassify>; discreteStateDrift: boolean;
  evidenceGenerationAfter: number;
}
export interface ReplanEvent {
  afterStep: number; from: Frame; cause: "capture" | "state-drift";
  predictIncrement: number; planIndex: number;
}
export interface GoalExecution {
  start: Frame; goal: Frame; reached: boolean; finalState: Frame;
  terminationReason: "goal-reached" | "no-known-route" | "unexplored" | "depth-limit" | "prediction-budget" | "execution-budget" | "chain-ended";
  executionBudget: number; replanningEnabled: boolean;
  plans: GoalPlan[]; steps: ExecutionStep[]; replans: ReplanEvent[];
}

/** Only real bench responses reach observe(). A returned plan is read-only;
 * cache values and generations are never rewritten by subsequent learning. */
export function executeGoal(model: TransitionMemory, bench: TransitionBench, start: Frame, goal: Frame,
  seed: number, options: { replan?: boolean; executionBudget?: number } = {}): GoalExecution {
  const executionBudget = options.executionBudget ?? 2 * model.space.diameter;
  integer(executionBudget, "execution budget");
  const replanningEnabled = options.replan ?? true;
  const plans = [planGoal(model, start, goal, seed)];
  const steps: ExecutionStep[] = [], replans: ReplanEvent[] = [];
  let state: Frame = { ...start }, current = plans[0]!, index = 0;
  const atGoal = () => signature(state) === signature(goal);
  let terminationReason: GoalExecution["terminationReason"] = "chain-ended";
  while (steps.length < executionBudget) {
    // The no-replanning control deliberately executes its original chain to end.
    if (atGoal() && (replanningEnabled || index >= current.steps.length)) { terminationReason = "goal-reached"; break; }
    if (current.status !== "found") { terminationReason = current.status; break; }
    const forecast = current.steps[index++];
    if (!forecast) { terminationReason = atGoal() ? "goal-reached" : "chain-ended"; break; }
    const observation = bench.conduct(state, forecast.action.values);
    const actual = observedState(model.space, observation);
    const conditions = model.conditions(state, forecast.action);
    // Compare against the old value snapshot BEFORE updating the evidence.
    const capture = captureClassify(model.mem, conditions, { ...observation }, forecast.snapshot, (seed + steps.length + 10000) >>> 0);
    // Encoding compatibility is coarser than the declared discrete state
    // alphabet. A route whose next input changed must also be regenerated.
    const discreteStateDrift = forecast.next === null || signature(actual) !== signature(forecast.next);
    model.observe(state, forecast.action, observation);
    steps.push({ state: { ...state }, forecast, observation: { ...observation }, actual, capture,
      discreteStateDrift, evidenceGenerationAfter: model.mem.evidenceGeneration });
    state = actual;
    if (replanningEnabled && (capture.class !== "within-envelope" || discreteStateDrift)) {
      current = planGoal(model, state, goal, (seed + plans.length * 1000) >>> 0);
      plans.push(current); index = 0;
      replans.push({ afterStep: steps.length, from: { ...state }, cause: capture.class !== "within-envelope" ? "capture" : "state-drift",
        predictIncrement: current.predictions.length, planIndex: plans.length - 1 });
    }
  }
  if (atGoal()) terminationReason = "goal-reached";
  else if (steps.length >= executionBudget) terminationReason = "execution-budget";
  return { start: { ...start }, goal: { ...goal }, reached: atGoal(), finalState: state, terminationReason,
    executionBudget, replanningEnabled, plans, steps, replans };
}
