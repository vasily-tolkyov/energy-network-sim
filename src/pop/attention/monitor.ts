import { AttentionController } from "./controller.js";
import type { AttentionCandidate } from "./controller.js";
import type { PopRuleMemory } from "../popmemory.js";
import type { PopDecoded } from "../popmap.js";
import type { Conditions, Outcomes } from "../../prototype/world.js";

/**
 * 注意力监测器（从历史 monitor.ts 适配移植）：
 * 持续接收真实观察 → 对焦点对象生成预测 → 后续真实变化到来时与
 * 已完成且时序对应的预测做三类比对 → 按类别驱动门控学习。
 *
 * 三条硬规矩（与历史一致）：
 * 1. 三类情况分开：符合已有预测 / 预测偏差 / 未知变化——未知不等于预测错误；
 * 2. 时序纪律：迟到的预测（完成时刻晚于变化窗口起点）不与已开始的变化配对，
 *    不伪造"提前预测"；
 * 3. 注意力只调度"当前重点处理谁"（控制器），不命令网络进入任何答案势阱。
 *
 * 门控学习（写观察、不写预测——响应模式修正）：
 * - 预测偏差：观察与已完成的置信预测矛盾 → 强写入（纠错）；
 * - 未知变化：无足够支持的预测 → 分配新核写观察（新经验）；
 * - 符合：不写（无新东西）。
 */

export interface FrameObject {
  readonly id: string;
  readonly conditions: Conditions;
  readonly outcomes: Outcomes;
}

export interface ObservationFrame {
  readonly tick: number;
  readonly objects: readonly FrameObject[];
}

export interface Forecast {
  readonly subjectId: string;
  readonly predicted: Record<string, PopDecoded>;
  readonly confident: boolean;
  readonly originTick: number;
  readonly completedTick: number;
}

export type ChangeClassification = "within-envelope" | "prediction-violation" | "unknown-change";

export interface AttentionNotice {
  readonly kind: "prediction-violation" | "unknown-change";
  readonly subjectId: string;
  readonly tick: number;
  readonly forecastCompletedTick: number | null;
  readonly detail: string;
}

export interface AttentionMonitorConfig {
  /** 门控学习写入轮数（纠错/新经验），默认 2 */
  readonly learnRepeats: number;
  /** 关闭门控学习（对照用） */
  readonly learnMode: "gated" | "ungated" | "none";
}

export class AttentionMonitor {
  readonly controller: AttentionController;
  readonly notices: AttentionNotice[] = [];
  #forecast: Forecast | null = null;
  #prevOutcomes = new Map<string, Outcomes>();
  #learnRepeats: number;
  #learnMode: "gated" | "ungated" | "none";
  #seedCursor = 1;

  constructor(
    readonly memory: PopRuleMemory,
    config: AttentionMonitorConfig = { learnRepeats: 2, learnMode: "gated" },
    seed = 1,
  ) {
    this.controller = new AttentionController(seed);
    this.#learnRepeats = config.learnRepeats;
    this.#learnMode = config.learnMode;
  }

  /** 三类比对（纯判定，不写学习） */
  static classify(
    forecast: Forecast | null,
    observed: Outcomes,
    changed: boolean,
  ): ChangeClassification {
    if (!changed) return "within-envelope";
    if (!forecast || !forecast.confident) return "unknown-change";
    for (const [ch, obs] of Object.entries(observed)) {
      if (forecast.predicted[ch] !== obs) return "prediction-violation";
    }
    return "within-envelope";
  }

  accept(frame: ObservationFrame): void {
    const candidates: AttentionCandidate[] = [];
    for (const obj of frame.objects) {
      const prev = this.#prevOutcomes.get(obj.id);
      const changed =
        prev !== undefined &&
        Object.keys(obj.outcomes).some((ch) => obj.outcomes[ch] !== prev[ch]);
      const changeMagnitude =
        prev === undefined
          ? 0
          : Math.min(
              2,
              Object.keys(obj.outcomes).reduce(
                (sum, ch) => sum + Math.abs((obj.outcomes[ch] ?? 0) - (prev[ch] ?? 0)),
                0,
              ),
            );

      // 时序纪律：只有完成时刻不晚于本帧的预测才参与比对；迟到的丢弃且不配对
      const forecast =
        this.#forecast && this.#forecast.subjectId === obj.id && this.#forecast.completedTick <= frame.tick
          ? this.#forecast
          : null;
      const classification = AttentionMonitor.classify(forecast, obj.outcomes, changed);

      // 门控学习：写观察、不写预测
      if (this.#learnMode === "gated") {
        if (classification === "prediction-violation" || classification === "unknown-change") {
          this.memory.learnFromObservation(obj.conditions, obj.outcomes, this.#learnRepeats);
          if (changed) {
            this.#notice({
              kind: classification,
              subjectId: obj.id,
              tick: frame.tick,
              forecastCompletedTick: forecast?.completedTick ?? null,
              detail: `${prev ? JSON.stringify(prev) : "?"} → ${JSON.stringify(obj.outcomes)}`,
            });
          }
        }
      } else if (this.#learnMode === "ungated") {
        // 旧行为对照：闭眼写预测（自强化错误）
        const predicted = this.memory.predict(obj.conditions, this.#seedCursor++).decoded;
        this.memory.learnFromQuery(obj.conditions, predicted, 1);
      }

      candidates.push({
        targetId: obj.id,
        safe: true,
        changeMagnitude,
        changeDerivative: 0, // 历史实际调用恒为 0
        predictionDeviation:
          classification === "prediction-violation" ? 2 : classification === "within-envelope" ? 0 : null,
        goalRelevance: 0, // 历史实际调用恒为 0
        novelty: 0, // 历史实际调用恒为 0
        actionTargetBinding:
          this.controller.snapshot().boundActionTargetId === obj.id ? 1 : 0,
      });

      this.#prevOutcomes.set(obj.id, obj.outcomes);
    }

    const snapshot = this.controller.update(frame.tick, candidates);

    // 至多一个未完成预测；只对焦点对象生成预测，完成时刻打戳
    this.#forecast = null;
    if (snapshot.focusTargetId) {
      const focus = frame.objects.find((o) => o.id === snapshot.focusTargetId);
      if (focus) {
        const predicted = this.memory.predict(focus.conditions, this.#seedCursor++).decoded;
        const confident = Object.values(predicted).every(
          (v) => typeof v === "number",
        );
        this.#forecast = {
          subjectId: focus.id,
          predicted,
          confident,
          originTick: frame.tick,
          completedTick: frame.tick, // 预测在观察处理内同步完成
        };
      }
    }
  }

  #notice(notice: AttentionNotice): void {
    if (
      this.notices.some(
        (n) => n.kind === notice.kind && n.subjectId === notice.subjectId && n.tick === notice.tick,
      )
    ) {
      return;
    }
    this.notices.push(notice);
  }
}
