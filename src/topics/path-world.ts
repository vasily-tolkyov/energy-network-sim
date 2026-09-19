import { validateFrame, type TransitionSpace, type Frame } from "../planning/space.js";

/** The alphabet is public; the transition function belongs only to the bench. */
export const PATH_SPACE: TransitionSpace = {
  states: [{ name: "pos", outcome: "nextPos", bins: 8 }],
  actions: [{ name: "move", bins: 2 }],
  diameter: 7,
};

export class PathBench {
  private cost = 0;
  conduct(state: Frame, action: Frame): Frame {
    validateFrame(state, PATH_SPACE.states);
    validateFrame(action, PATH_SPACE.actions);
    this.cost++;
    return { nextPos: Math.min(7, Math.max(0, state.pos! + (action.move === 1 ? 1 : -1))) };
  }
  get experimentsUsed(): number { return this.cost; }
}
