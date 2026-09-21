/**
 * 决策后端（感知/执行层的可插拔判定服务）。
 * 三种类型化问题（与 TypeSafe Jev 的题型对齐）：
 * - Noul：是/否判定，返回带校准概率的布尔；
 * - Choice：有限多选，返回选中项与置信度；
 * - Score：量表打分（0..1）与置信度。
 * 边界声明：此后端只服务于感知接地与动作映射（躯体层），
 * 不进学习核、不做规则来源——规则全部由能量网络从观察中学。
 */

export interface NoulAnswer {
  readonly kind: "noul";
  readonly value: boolean;
  readonly confidence: number;
}
export interface ChoiceAnswer {
  readonly kind: "choice";
  readonly value: string;
  readonly confidence: number;
}
export interface ScoreAnswer {
  readonly kind: "score";
  readonly value: number;
  readonly confidence: number;
}
export type DecisionAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export interface DecisionBackend {
  /** 对一段状态描述提出一个类型化问题，返回带置信度的判定 */
  ask(state: string, question: DecisionQuestion): Promise<DecisionAnswer>;
}

export type DecisionQuestion =
  | { kind: "noul"; text: string }
  | { kind: "choice"; text: string; options: readonly string[] }
  | { kind: "score"; text: string };

/** Jev API 后端（HTTP 桩）：URL 与 key 由构造注入；无 key 时响亮抛错，
 * 绝不静默降级为本地启发式——感知链路要么真实，要么明确不可用。 */
export class JevApiBackend implements DecisionBackend {
  constructor(
    private readonly endpoint: string | null,
    private readonly apiKey: string | null,
    private readonly model = "typesafe/jev-latest",
  ) {}
  async ask(state: string, question: DecisionQuestion): Promise<DecisionAnswer> {
    if (!this.endpoint || !this.apiKey) {
      throw new Error("JevApiBackend 未配置 endpoint/apiKey——感知后端不可用（禁止静默降级）");
    }
    const body = {
      model: this.model,
      state,
      questions: [
        question.kind === "noul"
          ? { type: "noul", text: question.text }
          : question.kind === "choice"
            ? { type: "choice", text: question.text, options: question.options }
            : { type: "score", text: question.text },
      ],
    };
    const res = await fetch(this.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Jev API ${res.status}: ${await res.text()}`);
    const data = (await res.json()) as { answers: { type: string; value: unknown; confidence: number }[] };
    const a = data.answers[0]!;
    if (a.type === "noul") return { kind: "noul", value: Boolean(a.value), confidence: a.confidence };
    if (a.type === "choice") return { kind: "choice", value: String(a.value), confidence: a.confidence };
    return { kind: "score", value: Number(a.value), confidence: a.confidence };
  }
}

/** Mock 后端（测试用）：确定性的关键词匹配——候选值标签出现在状态文本中
 * 即给出高置信判定；多个命中取最长标签；无命中给低置信。只允许测试使用。 */
export class MockBackend implements DecisionBackend {
  constructor(private readonly hitConfidence = 0.9, private readonly missConfidence = 0.2) {}
  ask(state: string, question: DecisionQuestion): Promise<DecisionAnswer> {
    if (question.kind === "choice") {
      const hits = question.options.filter(o => state.includes(o)).sort((a, b) => b.length - a.length);
      if (hits.length) {
        return Promise.resolve({ kind: "choice", value: hits[0]!, confidence: this.hitConfidence });
      }
      return Promise.resolve({ kind: "choice", value: question.options[0]!, confidence: this.missConfidence });
    }
    if (question.kind === "noul") {
      const hit = question.text.replace(/[？?]/g, "").split(/，|。/).some(seg => seg && state.includes(seg));
      return Promise.resolve({ kind: "noul", value: hit, confidence: hit ? this.hitConfidence : this.missConfidence });
    }
    return Promise.resolve({ kind: "score", value: 0.5, confidence: this.missConfidence });
  }
}
