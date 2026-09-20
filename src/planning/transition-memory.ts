import { FieldRuleMemory } from "../pop/concept/field-memory.js";
import { SensoryEncoder } from "../pop/concept/sensory.js";
import { EmergentMap } from "../pop/concept/emergent-map.js";
import { sameSupport, compatibleSupports } from "../pop/evidence.js";
import type { ForecastSnapshot } from "../pop/attention/capture.js";
import { actionsOf, signature, validateFrame, validateSpace, observedState, type TransitionSpace, type Frame, type Action } from "./space.js";

export type StepKind = "usable" | "unknown" | "infeasible";
export interface EnergyAudit {
  energy: number; traceEnd: number; recomputed: number; error: number; mismatch: boolean;
}
export interface StepPrediction {
  state: Frame; action: Action; seed: number; observed: boolean;
  snapshot: ForecastSnapshot; ambiguous: readonly string[];
  next: Frame | null; kind: StepKind; approximate: boolean;
  audit: EnergyAudit; milliseconds: number;
}
/** Search receives this read-only surface, never a bench or observed answers. */
export interface TransitionReader {
  readonly space: TransitionSpace;
  readonly actions: readonly Action[];
  predict(state: Frame, action: Action, seed: number): StepPrediction;
  /** 条件签名（可疑转移回避用）；缺省时规划器自行拼装 */
  conditionKey?(state: Frame, action: Action): string;
}

export class TransitionMemory implements TransitionReader {
  readonly space: TransitionSpace;
  readonly actions: readonly Action[];
  readonly mem: FieldRuleMemory;
  /** Eligibility metadata only: there is intentionally no transition lookup table. */
  private readonly observed = new Set<string>();

  constructor(space: TransitionSpace) {
    validateSpace(space);
    this.space = structuredClone(space);
    this.actions = actionsOf(space);
    // 离散世界的编码分辨率跟着档位数走（长链渗色修复）：
    // - σ = 0.225、激活半径 2σ = 0.45 < 半档距——相邻档感受野**不相交**：
    //   离散档是符号而非连续量，相邻档必须是可判别的不同答案（重叠会让编码
    //   等价关系把不同结果误判为"同一感知类"，对比否决随之失效——实测教训）；
    // - 量程向两侧各外扩半档：边缘档与中间档拿到同样多的感受野（修复前
    //   端点档只剩 2 个感受野，支持场比共享动作维的对手核还弱，实测跳读）；
    // - 场密度按每档约 5 个活跃感受野配（点火不等式的群体下限）。
    const maxBins = Math.max(...[...space.states, ...space.actions].map(d => d.bins));
    const pad = (bins: number) => ({ min: -0.5, max: bins - 0.5, sigma: 0.225 });
    const dimensions = [...space.states.map(d => ({ name: d.name, ...pad(d.bins) })),
      ...space.actions.map(d => ({ name: d.name, ...pad(d.bins) })),
      ...space.states.map(d => ({ name: d.outcome, ...pad(d.bins) }))];
    const encoder = new SensoryEncoder(dimensions, Math.max(2, 5 * maxBins));
    const maxRules = [...space.states, ...space.actions].reduce((n, d) => n * d.bins, 1);
    this.mem = new FieldRuleMemory(encoder, new EmergentMap([], encoder), { maxRules });
    this.mem.setOutcomeDimensions(space.states.map(d => d.outcome));
  }
  observe(state: Frame, action: Action, outcomes: Frame): void {
    const conditions = this.conditions(state, action);
    observedState(this.space, outcomes);
    this.mem.learnFromObservation(conditions, { ...outcomes });
    this.observed.add(signature(conditions));
  }
  conditions(state: Frame, action: Action): Record<string, number> {
    validateFrame(state, this.space.states);
    const canonical = this.actions.find(a => a.id === action.id);
    if (!canonical || signature(canonical.values) !== signature(action.values)) throw new Error("invalid action");
    return { ...state, ...action.values };
  }
  conditionKey(state: Frame, action: Action): string {
    return signature(this.conditions(state, action));
  }
  /** All output dimensions are required; aliases cannot be forced to one state. */
  decode(values: Readonly<Record<string, number | null>>): Frame | null {
    const next: Record<string, number> = {};
    for (const d of this.space.states) {
      const v = values[d.outcome];
      if (typeof v !== "number" || !Number.isFinite(v)) return null;
      const support = this.mem.encoder.encodeDimension(d.outcome, v);
      const alphabet = Array.from({ length: d.bins }, (_, i) => i);
      const exact = alphabet.filter(i => sameSupport(support, this.mem.encoder.encodeDimension(d.outcome, i)));
      const matches = exact.length ? exact : alphabet.filter(i => compatibleSupports(support, this.mem.encoder.encodeDimension(d.outcome, i)));
      if (matches.length !== 1) return null;
      next[d.name] = matches[0]!;
    }
    return next;
  }
  predict(state: Frame, action: Action, seed: number): StepPrediction {
    const conditions = this.conditions(state, action);
    const started = performance.now();
    // Passive, synchronous audit. Forward all arguments unchanged and restore
    // even on error. No prediction-core settings are bypassed or modified.
    const original = this.mem.net.settleAnnealed;
    let settlement: ReturnType<typeof original> | undefined;
    this.mem.net.settleAnnealed = (...args) => {
      const options = args[2];
      if (!options?.fallbackQuietOnly || !options.quenchCandidatesOnly || options.quenchMaxFlips !== 8 * this.mem.net.neuronCount)
        throw new Error("planning prediction lost memory solver constraints");
      settlement = original.apply(this.mem.net, args);
      return settlement;
    };
    let p;
    try { p = this.mem.predict(conditions, seed); }
    finally { this.mem.net.settleAnnealed = original; }
    if (!settlement) throw new Error("prediction missing solver audit");
    // Independent θ/W/Γ sum on the returned pattern; DI is absent.
    const ids = [...p.activeNeurons].sort((a, b) => a - b);
    let recomputed = ids.length * this.mem.net.threshold;
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++)
      recomputed += this.mem.net.getInhibitoryWeight(ids[i]!, ids[j]!) - this.mem.net.getWeight(ids[i]!, ids[j]!);
    const traceEnd = settlement.trace.energies.at(-1)!;
    const error = Math.max(Math.abs(traceEnd - p.energy), Math.abs(recomputed - p.energy));
    const audit = { energy: p.energy, traceEnd, recomputed, error, mismatch: !Number.isFinite(error) || error > 1e-7 };
    const snapshot = Object.freeze({ coreIdx: p.winningCores[0] ?? null, values: Object.freeze({ ...p.values }),
      generation: this.mem.evidenceGeneration, converged: p.converged, terminationReason: p.terminationReason });
    const next = p.ambiguous.length ? null : this.decode(p.values);
    const kind: StepKind = p.terminationReason === "no-quiet-candidate" ? "infeasible" : next === null ? "unknown" : "usable";
    return { state: { ...state }, action: structuredClone(action), seed, observed: this.observed.has(signature(conditions)),
      snapshot, ambiguous: [...p.ambiguous], next: kind === "usable" ? next : null, kind,
      approximate: kind === "usable" && !p.converged, audit, milliseconds: performance.now() - started };
  }
}
