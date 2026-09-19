import type { EnergyNetwork } from "../network.js";

/** One encoding-level compatibility rule throughout learning and capture:
 * nonempty intersection is compatible; identical supports are aliases.
 * This deliberately makes no claim about sub-receptive-field resolution. */
export function compatibleSupports(a: readonly number[], b: readonly number[]): boolean {
  const bs = new Set(b);
  return a.some(id => bs.has(id));
}
export function sameSupport(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every(id => b.includes(id));
}
export type EvidenceSource = "observation" | "hypothesis";
export interface Evidence {
  value: number; source: EvidenceSource; generation: number;
  support: number[]; samples: number; min: number; max: number;
}

/** Metadata controls plasticity, never chooses a winning core. Supersede edges
 * are owned here separately from structural alternative-condition inhibition. */
export class OutcomeEvidence {
  readonly records = new Map<number, Map<string, Evidence>>();
  private readonly history = new Map<string, number[][]>();
  private readonly vetoes = new Map<string, { from: number; to: number; delta: number; generation: number }[]>();
  private version = 0;
  private conflicts: { core: number; dimension: string; value: number; generation: number }[] = [];
  get generation(): number { return this.version; }
  get conflictLog() { return this.conflicts.map(x => ({ ...x })); }

  accept(net: EnergyNetwork, index: number, core: readonly number[], values: Readonly<Record<string, number>>,
    source: EvidenceSource, encode: (dim: string, value: number) => number[]): Record<string, number> {
    // Encode the whole request before touching evidence or inhibition.
    const plan = Object.entries(values).map(([dim, value]) => ({ dim, value, support: encode(dim, value) }));
    const generation = ++this.version;
    const dims = this.records.get(index) ?? new Map<string, Evidence>();
    const accepted: Record<string, number> = {};
    for (const { dim, value, support } of plan) {
      const previous = dims.get(dim);
      if (source === "hypothesis" && previous?.source === "observation") {
        if (!compatibleSupports(previous.support, support)) this.conflicts.push({ core: index, dimension: dim, value, generation });
        continue;
      }
      accepted[dim] = value;
      const alias = previous?.source === source && sameSupport(previous.support, support);
      const samples = alias ? previous.samples + 1 : 1;
      dims.set(dim, { value: alias ? (previous.value * previous.samples + value) / samples : value,
        source, generation, support, samples, min: alias ? Math.min(previous.min, value) : value,
        max: alias ? Math.max(previous.max, value) : value });
      const key = `${index}:${dim}`;
      const history = this.history.get(key) ?? [];
      if (!history.some(old => sameSupport(old, support))) history.push(support);
      this.history.set(key, history);
      // Remove only the actual contribution we installed (including saturation).
      for (const edge of this.vetoes.get(key) ?? []) net.setInhibitionContribution(edge.from, edge.to, `supersede:${key}`, 0);
      const targets = new Set(history.filter(old => !compatibleSupports(old, support)).flat().filter(id => !support.includes(id)));
      const edges: { from: number; to: number; delta: number; generation: number }[] = [];
      for (const from of core) for (const to of targets) {
        const before = net.getInhibitoryWeight(from, to);
        net.setInhibitionContribution(from, to, `supersede:${key}`, 3);
        edges.push({ from, to, delta: net.getInhibitoryWeight(from, to) - before, generation });
      }
      this.vetoes.set(key, edges);
    }
    this.records.set(index, dims);
    return accepted;
  }
  snapshot(index: number): Record<string, Evidence> {
    return Object.fromEntries([...(this.records.get(index) ?? [])].map(([dim, e]) => [dim, { ...e, support: [...e.support] }]));
  }
}
