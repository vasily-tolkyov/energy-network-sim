import { validateFrame, type TransitionSpace, type Frame } from "../planning/space.js";

export const CHAIN_SPACE: TransitionSpace = {
  states: [{ name: "node", outcome: "nextNode", bins: 5 }],
  actions: [{ name: "advance", bins: 2 }],
  diameter: 4, // maximum finite shortest path; reverse pairs are unreachable
};

/** A=0 ... E=4. Only this bench knows the directed transition rule. */
export class OrderedChainBench {
  private cost = 0;
  private gateOpen = true;
  setGateOpen(open: boolean): void { this.gateOpen = open; }
  conduct(state: Frame, action: Frame): Frame {
    validateFrame(state, CHAIN_SPACE.states); validateFrame(action, CHAIN_SPACE.actions);
    this.cost++;
    const node = state.node!;
    return { nextNode: action.advance === 1 || (node === 2 && !this.gateOpen) ? node : Math.min(4, node + 1) };
  }
  get experimentsUsed(): number { return this.cost; }
}
