import { mulberry32 } from "../prng.js";
import { integer } from "../validate.js";
import { R2PopLayer } from "../pop/r2pop.js";
import { PopChannelMap } from "../pop/popmap.js";
import { frames, type TransitionBench } from "./space.js";
import { TransitionMemory } from "./transition-memory.js";

export interface ObservedEpisode { conditions: Record<string, number>; outcomes: Record<string, number> }
export interface Collection {
  budget: number; experiments: number; terminationReason: "budget-exhausted";
  episodes: ObservedEpisode[]; distinctQueries: number; possibleQueries: number;
  factors: string[]; r2Pairs: number; undecidablePairs: number;
}
/** Resettable-bench corpus collection, not an embodied random walk. A seeded,
 * balanced schedule knows the alphabet and visit counts, never the answers. */
export function collectTransitions(model: TransitionMemory, bench: TransitionBench, budget: number, seed: number): Collection {
  integer(budget, "collection budget", 1);
  const rng = mulberry32(seed);
  const queries = frames(model.space.states).flatMap(state => model.actions.map(action => ({ state, action })));
  const episodes: ObservedEpisode[] = [];
  const distinct = new Map<number, ObservedEpisode>();
  while (episodes.length < budget) {
    const order = queries.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j]!, order[i]!]; }
    for (const index of order) {
      if (episodes.length === budget) break;
      const { state, action } = queries[index]!;
      const outcomes = bench.conduct(state, action.values);
      model.observe(state, action, outcomes);
      const episode = { conditions: model.conditions(state, action), outcomes: { ...outcomes } };
      episodes.push(episode); distinct.set(index, episode);
    }
  }
  // Existing neural differential identifies factors from single-coordinate
  // observed contrasts. No transition formula or numeric outcome comparator.
  const cond = [...model.space.states, ...model.space.actions];
  const out = model.space.states.map(d => ({ name: d.outcome, bins: d.bins }));
  const r2 = new R2PopLayer(new PopChannelMap(cond), new PopChannelMap(out));
  const unique = [...distinct.entries()].sort(([a], [b]) => a - b).map(([, e]) => e);
  const factors = new Set<string>();
  let r2Pairs = 0, undecidablePairs = 0;
  for (let i = 0; i < unique.length; i++) for (let j = i + 1; j < unique.length; j++) {
    const e0 = unique[i]!, e1 = unique[j]!;
    if (cond.filter(d => e0.conditions[d.name] !== e1.conditions[d.name]).length !== 1) continue;
    const analysis = r2.analyzePair({ e0, e1 }); r2Pairs++;
    if (analysis.undecidable) { undecidablePairs++; continue; }
    for (const d of analysis.influentialChannels) factors.add(d);
  }
  // One structural veto pass; gain=0, no extra excitation. Eligibility comes
  // only from R2 above. Existing default veto strength is unchanged.
  const eligibility = Object.fromEntries([...factors].map(d => [d, 0]));
  if (factors.size) for (const e of unique) model.mem.bindInfluence(e.conditions, eligibility);
  return { budget, experiments: episodes.length, terminationReason: "budget-exhausted", episodes,
    distinctQueries: distinct.size, possibleQueries: queries.length, factors: [...factors].sort(), r2Pairs, undecidablePairs };
}
