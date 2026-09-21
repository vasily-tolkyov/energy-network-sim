import { integer, finite } from "../validate.js";
import type { DecisionBackend } from "./decision-backend.js";

/**
 * 场景解析（感知前端的核心）：把自由文本的世界场景描述映射为条件帧——
 * 动力学链的起始状态。对每个声明的条件维，按其候选值的自然语言标签
 * 向后端提类型化问题；低置信维度如实报 unknown，绝不强行填入。
 * 边界：这里只决定"世界现在长什么样"，不决定"世界遵循什么规则"。
 */

export interface PerceptionDimSpec {
  /** 条件维名（与 TransitionSpace/记忆的条件维一致） */
  readonly name: string;
  /** 候选值：数值档 → 该档的自然语言标签（用于提问与文本命中） */
  readonly values: readonly { value: number; label: string }[];
  /** 提问模板（可选）：如 "开关现在是什么状态？"；缺省按维名生成 */
  readonly ask?: string;
}

export interface PerceptionSpec {
  readonly dims: readonly PerceptionDimSpec[];
  /** 维度置信的最低门槛（默认 0.6）；低于它进 unknownDims，不进帧 */
  readonly confidenceThreshold?: number;
}

export interface PerceivedFrame {
  /** 通过置信门槛的条件帧（动力学链的可钳制起始状态） */
  readonly frame: Record<string, number>;
  /** 各维判定值与置信度（含未过门槛的） */
  readonly details: Record<string, { label: string; confidence: number }>;
  /** 未达置信门槛的维度——如实上报，调用方决定跳过、追问或拒答 */
  readonly unknownDims: string[];
}

export class SceneParser {
  constructor(private readonly backend: DecisionBackend) {}

  async parse(sceneText: string, spec: PerceptionSpec): Promise<PerceivedFrame> {
    const threshold = spec.confidenceThreshold ?? 0.6;
    finite(threshold, "confidenceThreshold");
    if (!(threshold > 0 && threshold <= 1)) throw new Error(`confidenceThreshold must be in (0,1], got ${threshold}`);
    const frame: Record<string, number> = {};
    const details: PerceivedFrame["details"] = {};
    const unknownDims: string[] = [];
    for (const dim of spec.dims) {
      if (!dim.values.length) throw new Error(`dimension ${dim.name} has no candidate values`);
      for (const v of dim.values) integer(v.value, `${dim.name}.value`);
      const text = dim.ask ?? `${dim.name} 当前是哪种状态？`;
      const answer = await this.backend.ask(sceneText, {
        kind: "choice",
        text,
        options: dim.values.map(v => v.label),
      });
      if (answer.kind !== "choice") throw new Error(`backend returned ${answer.kind} for a choice question`);
      const hit = dim.values.find(v => v.label === answer.value);
      if (!hit) throw new Error(`backend returned a label outside the candidate set: ${answer.value}`);
      details[dim.name] = { label: hit.label, confidence: answer.confidence };
      if (answer.confidence >= threshold) frame[dim.name] = hit.value;
      else unknownDims.push(dim.name);
    }
    return { frame, details, unknownDims };
  }
}
