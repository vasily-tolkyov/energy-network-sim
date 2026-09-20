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

interface Candidate {
  support: number[]; value: number; samples: number; min: number; max: number; lastGen: number;
}
interface DimState {
  source: EvidenceSource;
  incumbent: string | null;
  candidates: Map<string, Candidate>;
}

/** Metadata controls plasticity, never chooses a winning core. Supersede edges
 * are owned here separately from structural alternative-condition inhibition.
 *
 * 改判按计票（与成核同一赫布原则：重复才构成规则）：
 * - 同支持区的观察合并计数（别名聚合）；不相容观察开新候选，各记各的票。
 * - 在位者（规则的答案）只有在挑战者票数 ≥ min(在位者票数, switchQuorum)
 *   时才更换——单次异常永不改判已确立的规则；世界真变了，重复确认照样学会。
 * - 否决边始终跟随在位者（压制与在位者不相容的候选支持区，保持读出干净），
 *   但否决 ≠ 改判：被压制的候选只是不读出，票数保留，日后翻转即可复活。
 * - 观察 > 假设是来源级优先级，不走计票：单次真观察照样纠正错误猜测。 */
export class OutcomeEvidence {
  private readonly records = new Map<number, Map<string, DimState>>();
  private readonly vetoes = new Map<string, { from: number; to: number; delta: number; generation: number }[]>();
  private version = 0;
  private conflicts: { core: number; dimension: string; value: number; generation: number }[] = [];
  /** 翻转定额：在位者票数超过此数时按此数计——世界变化的确认成本有界 */
  readonly switchQuorum: number;
  constructor(switchQuorum = 3) {
    if (!Number.isSafeInteger(switchQuorum) || switchQuorum < 1) throw new Error(`switchQuorum must be an integer >= 1, got ${switchQuorum}`);
    this.switchQuorum = switchQuorum;
  }
  get generation(): number { return this.version; }
  get conflictLog() { return this.conflicts.map(x => ({ ...x })); }

  accept(net: EnergyNetwork, index: number, core: readonly number[], values: Readonly<Record<string, number>>,
    source: EvidenceSource, encode: (dim: string, value: number) => number[]): Record<string, number> {
    // Encode the whole request before touching evidence or inhibition.
    const plan = Object.entries(values).map(([dim, value]) => ({ dim, value, support: encode(dim, value) }));
    const generation = ++this.version;
    const dims = this.records.get(index) ?? new Map<string, DimState>();
    const accepted: Record<string, number> = {};
    for (const { dim, value, support } of plan) {
      const previous = dims.get(dim);
      if (source === "hypothesis" && previous?.source === "observation") {
        if (previous.incumbent !== null && !compatibleSupports(previous.candidates.get(previous.incumbent)!.support, support))
          this.conflicts.push({ core: index, dimension: dim, value, generation });
        continue;
      }
      accepted[dim] = value;
      // 首次观察取代假设档：来源优先级不走计票。假设候选的支持区保留
      // （供否决压制其读出碎片），但票数清零——假设不得为真实观察垫票。
      if (source === "observation" && previous?.source === "hypothesis") {
        for (const c of previous.candidates.values()) c.samples = 0;
        previous.incumbent = null;
      }
      const state = dims.get(dim) ?? { source, incumbent: null, candidates: new Map<string, Candidate>() };
      const key = [...state.candidates.keys()].find(k => sameSupport(state.candidates.get(k)!.support, support));
      const sKey = key ?? support.join(",");
      const cand = state.candidates.get(sKey) ?? { support: [...support], value, samples: 0, min: value, max: value, lastGen: 0 };
      cand.value = (cand.value * cand.samples + value) / (cand.samples + 1);
      cand.samples++; cand.min = Math.min(cand.min, value); cand.max = Math.max(cand.max, value);
      cand.lastGen = generation;
      state.candidates.set(sKey, cand);
      // 在位者判定：假设档即时（脚手架语义）；观察档按计票
      const incumbent = state.incumbent !== null ? state.candidates.get(state.incumbent)! : null;
      const quorum = source === "hypothesis" ? 1 : Math.min(incumbent?.samples ?? 0, this.switchQuorum);
      if (incumbent === null || (sKey !== state.incumbent && cand.samples >= quorum)) state.incumbent = sKey;
      // 否决始终跟随在位者（压制不相容候选的读出碎片），与是否改判无关
      this.installVeto(net, index, dim, core, state);
      state.source = source;
      dims.set(dim, state);
    }
    this.records.set(index, dims);
    return accepted;
  }

  /** 否决只打与在位者不相容的候选支持区（去掉共享成员）；非翻转不动边 */
  private installVeto(net: EnergyNetwork, index: number, dim: string, core: readonly number[], state: DimState): void {
    const key = `${index}:${dim}`;
    this.retractVeto(net, index, dim);
    const incumbent = state.candidates.get(state.incumbent!)!;
    const targets = new Set<number>();
    for (const [k, c] of state.candidates) {
      if (k === state.incumbent || compatibleSupports(c.support, incumbent.support)) continue;
      for (const id of c.support) if (!incumbent.support.includes(id)) targets.add(id);
    }
    const edges: { from: number; to: number; delta: number; generation: number }[] = [];
    for (const from of core) for (const to of targets) {
      const before = net.getInhibitoryWeight(from, to);
      net.setInhibitionContribution(from, to, `supersede:${key}`, 3);
      edges.push({ from, to, delta: net.getInhibitoryWeight(from, to) - before, generation: incumbent.lastGen });
    }
    this.vetoes.set(key, edges);
  }

  private retractVeto(net: EnergyNetwork, index: number, dim: string): void {
    const key = `${index}:${dim}`;
    for (const edge of this.vetoes.get(key) ?? []) net.setInhibitionContribution(edge.from, edge.to, `supersede:${key}`, 0);
  }

  /** 在位者视图（读出/捕获用）：返回的是当前在位候选，不是最新写入 */
  snapshot(index: number): Record<string, Evidence> {
    const out: Record<string, Evidence> = {};
    for (const [dim, state] of this.records.get(index) ?? []) {
      if (state.incumbent === null) continue;
      const c = state.candidates.get(state.incumbent)!;
      out[dim] = { value: c.value, source: state.source, generation: c.lastGen, support: [...c.support], samples: c.samples, min: c.min, max: c.max };
    }
    return out;
  }

  /** 各结果维的候选票数（不确定性地图的"证据竞争"信号用；只读元数据） */
  candidateCounts(index: number): Record<string, number[]> {
    const out: Record<string, number[]> = {};
    for (const [dim, state] of this.records.get(index) ?? []) {
      if (state.source !== "observation") continue;
      out[dim] = [...state.candidates.values()].map(c => c.samples).sort((a, b) => b - a);
    }
    return out;
  }

  /** 遗忘一个核槽：撤销它施加的全部 supersede 否决贡献（只撤自己安装的
   * 那份），删除证据记录。其他核的证据与其他来源的边不动。 */
  remove(net: EnergyNetwork, index: number): void {
    const prefix = `${index}:`;
    for (const [key, edges] of [...this.vetoes]) {
      if (!key.startsWith(prefix)) continue;
      for (const edge of edges) net.setInhibitionContribution(edge.from, edge.to, `supersede:${key}`, 0);
      this.vetoes.delete(key);
    }
    this.records.delete(index);
    this.conflicts = this.conflicts.filter(c => c.core !== index);
  }
}
