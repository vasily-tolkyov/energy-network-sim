import { integer } from "../validate.js";

export type Frame = Readonly<Record<string, number>>;
export interface DiscreteDimension { readonly name: string; readonly bins: number }
export interface StateDimension extends DiscreteDimension { readonly outcome: string }
export interface TransitionSpace {
  readonly states: readonly StateDimension[];
  readonly actions: readonly DiscreteDimension[];
  /** Declared task bound, not an oracle accessible to the planner. */
  readonly diameter: number;
}
export interface Action { readonly id: number; readonly values: Frame }
export interface TransitionBench { conduct(state: Frame, action: Frame): Frame }

export function validateSpace(space: TransitionSpace): void {
  integer(space.diameter, "diameter", 1);
  if (!space.states.length || !space.actions.length) throw new Error("state and action dimensions required");
  const names = [...space.states.flatMap(d => [d.name, d.outcome]), ...space.actions.map(d => d.name)];
  if (names.some(n => !n) || new Set(names).size !== names.length) throw new Error("dimension names must be distinct");
  for (const d of [...space.states, ...space.actions]) integer(d.bins, `${d.name}.bins`, 2);
}
export function validateFrame(frame: Frame, dims: readonly DiscreteDimension[]): void {
  if (Object.keys(frame).length !== dims.length) throw new Error("frame must have exactly the declared dimensions");
  for (const d of dims) {
    integer(frame[d.name]!, d.name);
    if (frame[d.name]! >= d.bins) throw new Error(`${d.name} out of range`);
  }
}
export function frames(dims: readonly DiscreteDimension[]): Frame[] {
  return dims.reduce<Frame[]>((rows, d) => rows.flatMap(row =>
    Array.from({ length: d.bins }, (_, value) => ({ ...row, [d.name]: value }))), [{}]);
}
export function signature(frame: Frame): string {
  return JSON.stringify(Object.entries(frame).sort(([a], [b]) => a.localeCompare(b)));
}
export function actionsOf(space: TransitionSpace): Action[] {
  return frames(space.actions).map((values, id) => ({ id, values }));
}
export function observedState(space: TransitionSpace, outcomes: Frame): Frame {
  validateFrame(outcomes, space.states.map(d => ({ name: d.outcome, bins: d.bins })));
  return Object.fromEntries(space.states.map(d => [d.name, outcomes[d.outcome]!]));
}
