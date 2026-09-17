import { mulberry32 } from "../../prng.js";

/**
 * 注意力控制器（从历史版本 attention-controller.ts 忠实移植）。
 * 职责单一：在给定候选对象及其信号后，决定当前焦点对象。
 * 不参与预测、不碰网络动力学（注意力 ≠ 捕获）。
 *
 * 移植保留：评分构成、突变抢占、迟滞边距+持续 tick、近似平局 softmax。
 * 如实标注（历史文档 §7.2）：goalRelevance/novelty/changeDerivative 在历史
 * 实际调用中恒为 0，这里保留接口默认 0；focusHeldTicks 疲劳扣分与
 * 近似平局规则属历史实现细节，沿用但标记"未在新模型验证"。
 * PRNG 换成 mulberry32（种子可复现）。
 */

export interface AttentionCandidate {
  readonly targetId: string;
  readonly safe: boolean;
  readonly changeMagnitude: number;
  readonly changeDerivative: number;
  readonly predictionDeviation: number | null;
  readonly goalRelevance: number;
  readonly novelty: number;
  readonly actionTargetBinding: number;
}

export interface AttentionScoreBreakdown {
  readonly targetBinding: number;
  readonly change: number;
  readonly derivative: number;
  readonly predictionDeviation: number;
  readonly predictionDeviationKnown: boolean;
  readonly goalRelevance: number;
  readonly novelty: number;
  readonly holdBias: number;
  readonly fatigue: number;
  readonly switchCost: number;
  readonly total: number;
}

export interface AttentionSnapshot {
  readonly tick: number;
  readonly focusTargetId: string | null;
  readonly boundActionTargetId: string | null;
  readonly focusHeldTicks: number;
  readonly switchCount: number;
  readonly preemptionCount: number;
  readonly scores: readonly { readonly targetId: string; readonly score: AttentionScoreBreakdown }[];
}

export interface AttentionConfiguration {
  readonly changeWeight: number;
  readonly derivativeWeight: number;
  readonly predictionDeviationWeight: number;
  readonly goalWeight: number;
  readonly noveltyWeight: number;
  readonly actionTargetWeight: number;
  readonly holdBias: number;
  readonly fatiguePerTick: number;
  readonly switchingCost: number;
  readonly hysteresisMargin: number;
  readonly sustainedTicks: number;
  readonly mutationPreemptionThreshold: number;
  readonly closeCallWindow: number;
}

/** 与历史 DEFAULT_ATTENTION_CONFIGURATION 一致 */
export const DEFAULT_ATTENTION_CONFIGURATION: AttentionConfiguration = Object.freeze({
  changeWeight: 1.75,
  derivativeWeight: 1.3,
  predictionDeviationWeight: 1.25,
  goalWeight: 0.9,
  noveltyWeight: 0.65,
  actionTargetWeight: 2.4,
  holdBias: 1.15,
  fatiguePerTick: 0.025,
  switchingCost: 0.8,
  hysteresisMargin: 0.55,
  sustainedTicks: 2,
  mutationPreemptionThreshold: 1.2,
  closeCallWindow: 0.08,
});

interface MutableAttentionState {
  tick: number;
  focusTargetId: string | null;
  boundActionTargetId: string | null;
  focusHeldTicks: number;
  switchCount: number;
  preemptionCount: number;
  scores: AttentionSnapshot["scores"];
}

export class AttentionController {
  readonly #random: () => number;
  readonly #configuration: AttentionConfiguration;
  readonly #competitionDuration = new Map<string, number>();
  #state: MutableAttentionState = {
    tick: 0,
    focusTargetId: null,
    boundActionTargetId: null,
    focusHeldTicks: 0,
    switchCount: 0,
    preemptionCount: 0,
    scores: [],
  };

  constructor(seed = 1, configuration: AttentionConfiguration = DEFAULT_ATTENTION_CONFIGURATION) {
    this.#random = mulberry32(seed);
    this.#configuration = { ...configuration };
  }

  bindActionTarget(targetId: string | null): void {
    this.#state.boundActionTargetId = targetId;
    // 新提交的行动立即拥有主焦点（普通目标绑定切换，非突变抢占）。
    if (targetId !== null && this.#state.focusTargetId !== targetId) {
      const previous = this.#state.focusTargetId;
      this.#state.focusTargetId = targetId;
      this.#state.focusHeldTicks = 0;
      this.#state.switchCount += previous === null ? 0 : 1;
      this.#competitionDuration.clear();
    }
  }

  update(tick: number, candidates: readonly AttentionCandidate[]): AttentionSnapshot {
    const safeCandidates = candidates.filter((candidate) => candidate.safe);
    const scored = safeCandidates.map((candidate) => ({
      candidate,
      score: this.#score(candidate),
    }));
    this.#state.tick = tick;
    this.#state.scores = scored.map(({ candidate, score }) => ({ targetId: candidate.targetId, score }));
    if (scored.length === 0) return this.snapshot();

    const current = scored.find(({ candidate }) => candidate.targetId === this.#state.focusTargetId) ?? null;
    const strongestMutation = scored
      .filter(({ candidate }) => candidate.targetId !== this.#state.focusTargetId)
      .filter(({ candidate }) => this.#mutationStrength(candidate) >= this.#configuration.mutationPreemptionThreshold)
      .sort((left, right) => right.score.total - left.score.total || left.candidate.targetId.localeCompare(right.candidate.targetId))[0];

    let selectedTarget = current?.candidate.targetId ?? this.#selectCloseCall(scored).candidate.targetId;
    let preempted = false;
    if (strongestMutation !== undefined) {
      selectedTarget = strongestMutation.candidate.targetId;
      preempted = true;
    } else if (current !== null) {
      const competitors = scored
        .filter(({ candidate }) => candidate.targetId !== current.candidate.targetId)
        .filter(({ score }) => score.total >= current.score.total + this.#configuration.hysteresisMargin);
      const best = competitors.length === 0 ? null : this.#selectCloseCall(competitors);
      if (best !== null) {
        const duration = (this.#competitionDuration.get(best.candidate.targetId) ?? 0) + 1;
        this.#competitionDuration.set(best.candidate.targetId, duration);
        if (duration >= this.#configuration.sustainedTicks) selectedTarget = best.candidate.targetId;
      }
    }

    const previous = this.#state.focusTargetId;
    if (previous === selectedTarget) {
      this.#state.focusHeldTicks += 1;
    } else {
      this.#state.focusTargetId = selectedTarget;
      this.#state.focusHeldTicks = 1;
      this.#state.switchCount += previous === null ? 0 : 1;
      if (preempted) this.#state.preemptionCount += 1;
      this.#competitionDuration.clear();
    }
    for (const targetId of [...this.#competitionDuration.keys()]) {
      if (!scored.some(({ candidate }) => candidate.targetId === targetId)) this.#competitionDuration.delete(targetId);
    }
    return this.snapshot();
  }

  snapshot(): AttentionSnapshot {
    return structuredClone(this.#state);
  }

  restore(snapshot: AttentionSnapshot): void {
    this.#state = structuredClone(snapshot);
    this.#competitionDuration.clear();
  }

  #score(candidate: AttentionCandidate): AttentionScoreBreakdown {
    const isCurrent = candidate.targetId === this.#state.focusTargetId;
    const cfg = this.#configuration;
    const targetBinding = cfg.actionTargetWeight * candidate.actionTargetBinding;
    const change = cfg.changeWeight * candidate.changeMagnitude;
    const derivative = cfg.derivativeWeight * candidate.changeDerivative;
    const predictionDeviationKnown = candidate.predictionDeviation !== null;
    const predictionDeviation = cfg.predictionDeviationWeight * (candidate.predictionDeviation ?? 0);
    const goalRelevance = cfg.goalWeight * candidate.goalRelevance;
    const novelty = cfg.noveltyWeight * candidate.novelty;
    const holdBias = isCurrent ? cfg.holdBias : 0;
    const fatigue = isCurrent ? cfg.fatiguePerTick * this.#state.focusHeldTicks : 0;
    const switchCost = isCurrent || this.#state.focusTargetId === null ? 0 : cfg.switchingCost;
    return {
      targetBinding,
      change,
      derivative,
      predictionDeviation,
      predictionDeviationKnown,
      goalRelevance,
      novelty,
      holdBias,
      fatigue,
      switchCost,
      total: targetBinding + change + derivative + predictionDeviation + goalRelevance + novelty
        + holdBias - fatigue - switchCost,
    };
  }

  #mutationStrength(candidate: AttentionCandidate): number {
    return candidate.changeMagnitude + candidate.changeDerivative + 0.5 * (candidate.predictionDeviation ?? 0);
  }

  #selectCloseCall<T extends { readonly score: AttentionScoreBreakdown }>(candidates: readonly T[]): T {
    const ordered = [...candidates].sort((left, right) => right.score.total - left.score.total);
    const maximum = ordered[0]!.score.total;
    const close = ordered.filter(({ score }) => maximum - score.total <= this.#configuration.closeCallWindow);
    if (close.length === 1) return close[0]!;
    const weights = close.map(({ score }) => Math.exp(score.total - maximum));
    let draw = this.#random() * weights.reduce((sum, value) => sum + value, 0);
    for (let index = 0; index < close.length; index += 1) {
      draw -= weights[index]!;
      if (draw <= 0) return close[index]!;
    }
    return close[close.length - 1]!;
  }
}
