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
    const dimensions = [...space.states.map(d => ({ name: d.name, min: 0, max: d.bins - 1 })),
      ...space.actions.map(d => ({ name: d.name, min: 0, max: d.bins - 1 })),
      ...space.states.map(d => ({ name: d.outcome, min: 0, max: d.bins - 1 }))];
    const encoder = new SensoryEncoder(dimensions);
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
