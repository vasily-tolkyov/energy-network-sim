import { finite } from "../validate.js";
import type { Frame } from "../planning/space.js";
import type { DecisionBackend } from "../perception/decision-backend.js";

/**
 * 动作执行映射（输出端）：把规划器选出的抽象动作帧映射为环境可执行的
 * 命令记录。每个环境声明自己的动作词汇与可选可行性检查；可行性判定
 * 交给决策后端（带置信度），不可行如实上报——不强行执行。
 * 边界：这里只做"抽象动作 → 躯体命令"的翻译，不改变动作的选择本身
 * （选择永远来自能量动力学与已学规则）。
 */

export interface ExecutionActionSpec {
  /** 动作维名（与 TransitionSpace 的动作维一致） */
  readonly dimension: string;
  /** 动作档 → 该档的可执行命令模板；{state} 为状态占位符 */
  readonly commands: Readonly<Record<string, string>>;
  /** 可选可行性检查：对当前状态提一个 Noul 问题；返回 false 则该档不可执行 */
  readonly feasibilityAsk?: string;
}

export interface ExecutionSpec {
  readonly environment: string;
  readonly actions: readonly ExecutionActionSpec[];
  /** 可行性置信的最低门槛（默认 0.5） */
  readonly confidenceThreshold?: number;
}

export interface ExecutionRecord {
  readonly environment: string;
  readonly action: Frame;
  readonly command: string;
  readonly feasible: boolean;
  readonly confidence: number;
  readonly reason?: string;
}

export class ActionExecutor {
  constructor(private readonly backend: DecisionBackend) {}

  async map(action: Frame, stateText: string, spec: ExecutionSpec): Promise<ExecutionRecord> {
    const threshold = spec.confidenceThreshold ?? 0.5;
    finite(threshold, "confidenceThreshold");
    const parts: string[] = [];
    let feasible = true;
    let confidence = 1;
    let reason: string | undefined;
    for (const dimSpec of spec.actions) {
      const raw = action[dimSpec.dimension];
      if (raw === undefined) throw new Error(`action frame missing dimension: ${dimSpec.dimension}`);
      const cmd = dimSpec.commands[String(raw)];
      if (cmd === undefined) throw new Error(`no command template for ${dimSpec.dimension}=${raw}`);
      parts.push(cmd.replace(/\{state\}/g, stateText));
      if (dimSpec.feasibilityAsk) {
        const answer = await this.backend.ask(stateText, { kind: "noul", text: dimSpec.feasibilityAsk });
        if (answer.kind !== "noul") throw new Error(`backend returned ${answer.kind} for a noul question`);
        confidence = Math.min(confidence, answer.confidence);
        if (!answer.value || answer.confidence < threshold) {
          feasible = false;
          reason = `可行性检查未通过（${dimSpec.feasibilityAsk} → ${answer.value} @${answer.confidence.toFixed(2)}）`;
        }
      }
    }
    return { environment: spec.environment, action: { ...action }, command: parts.join("；"), feasible, confidence, reason };
  }

  /** 执行一条规划链：逐步映射，遇不可行即停并如实报告（不跳过继续） */
  async mapChain(actions: readonly Frame[], stateTextOf: (step: number) => string, spec: ExecutionSpec): Promise<ExecutionRecord[]> {
    const out: ExecutionRecord[] = [];
    for (let i = 0; i < actions.length; i++) {
      const rec = await this.map(actions[i]!, stateTextOf(i), spec);
      out.push(rec);
      if (!rec.feasible) break;
    }
    return out;
  }
}
