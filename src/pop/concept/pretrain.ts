import { SensoryEncoder } from "./sensory.js";
import { ConceptFormation } from "./formation.js";
import { EmergentMap } from "./emergent-map.js";
import { EmergentChannelAdapter } from "./emergent-channel-adapter.js";
import { R2PopLayer } from "../r2pop.js";
import { FieldRuleMemory } from "./field-memory.js";
import {
  CONT_CONDITION_DIMS,
  CONT_OUTCOME_DIMS,
  contCurriculum,
  type CValues,
} from "./world-continuous.js";

/**
 * 连续课程教学管线（runner-formation 与神经注意力共用）：
 * 形成 → EmergentMap → R2 神经差分 → 场级教学 + 换对互斥 + 全局侧重否决。
 */
export interface PretrainedContinuous {
  readonly mem: FieldRuleMemory;
  readonly enc: SensoryEncoder;
  readonly em: EmergentMap;
}

export function pretrainContinuous(fieldsPerDim = 40): PretrainedContinuous {
  const enc = new SensoryEncoder([...CONT_CONDITION_DIMS, ...CONT_OUTCOME_DIMS], fieldsPerDim);
  const formation = new ConceptFormation(enc);
  for (const group of contCurriculum()) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) {
        formation.presentExperiment({ ...e.conditions, ...e.outcomes }, 4);
      }
      formation.presentSwap(
        group.manipulated,
        pair.e0.conditions[group.manipulated]!,
        pair.e1.conditions[group.manipulated]!,
        3.0,
      );
    }
  }
  const em = new EmergentMap(formation.extractConcepts(0.5), enc);
  const condAdapter = new EmergentChannelAdapter(em, CONT_CONDITION_DIMS.map((d) => d.name), 0);
  const outAdapter = new EmergentChannelAdapter(
    em,
    CONT_OUTCOME_DIMS.map((d) => d.name),
    enc.dimensionOffset(CONT_OUTCOME_DIMS[0]!.name),
    "fields",
  );
  const r2 = new R2PopLayer(condAdapter, outAdapter);
  const mem = new FieldRuleMemory(enc, em, { maxRules: 192 });
  mem.setOutcomeDimensions(CONT_OUTCOME_DIMS.map((d) => d.name));
  const SPAN: Record<string, number> = { rebound: 1, reboundSpeed: 4 };
  const globalMag = new Map<string, { sum: number; count: number }>();
  const bin = (v: CValues): CValues => {
    const o: CValues = {};
    for (const [d, x] of Object.entries(v)) o[d] = em.resolve(d, x) ?? -1;
    return o;
  };
  for (const group of contCurriculum()) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.teachExperiment(e.conditions, e.outcomes, 4);
      for (const dim of CONT_OUTCOME_DIMS) {
        const a = pair.e0.outcomes[dim.name]!;
        const b = pair.e1.outcomes[dim.name]!;
        if (Math.abs(a - b) > 1e-9) mem.learnExclusion(dim.name, a, b, 3.0);
      }
      const analysis = r2.analyzePair({
        e0: { conditions: bin(pair.e0.conditions), outcomes: pair.e0.outcomes },
        e1: { conditions: bin(pair.e1.conditions), outcomes: pair.e1.outcomes },
      });
      for (const ch of analysis.influentialChannels) {
        let m = 0;
        for (const [och, delta] of Object.entries(analysis.outcomeDelta)) {
          m += Math.abs(delta) / (SPAN[och] ?? 1);
        }
        const e0 = globalMag.get(ch) ?? { sum: 0, count: 0 };
        e0.sum += m / CONT_OUTCOME_DIMS.length;
        e0.count++;
        globalMag.set(ch, e0);
      }
    }
  }
  const globalBoost: Record<string, number> = {};
  for (const [ch, { sum, count }] of globalMag) {
    globalBoost[ch] = 0.1 * 3 * (sum / Math.max(1, count)) ** 2;
  }
  for (const group of contCurriculum()) {
    for (const pair of group.pairs) {
      for (const e of [pair.e0, pair.e1]) mem.bindInfluence(e.conditions, globalBoost, 4);
    }
  }
  return { mem, enc, em };
}
