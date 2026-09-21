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

/** Jev API 后端（官方 HTTP 形状：POST https://api.typesafe.ai/v1/systemone，
 * Authorization: Bearer；body = { model, state, questions: { id: { type, instructions, criteria? } } }）。
 * URL 与 key 由构造注入；无 key 时响亮抛错，绝不静默降级为本地启发式。
 * 映射注记：Jev 的 Noul 直接返回"命题为真"的概率 p（无独立置信字段），
 * 这里映射为 value = p ≥ 0.5、confidence = |2p − 1|（距 0.5 的校准距离）。 */
export class JevApiBackend implements DecisionBackend {
  constructor(
    private readonly endpoint: string | null,
    private readonly apiKey: string | null,
    private readonly model = "jev-latest",
  ) {}
  async ask(state: string, question: DecisionQuestion): Promise<DecisionAnswer> {
    if (!this.endpoint || !this.apiKey) {
      throw new Error("JevApiBackend 未配置 endpoint/apiKey——感知后端不可用（禁止静默降级）");
    }
    const q: Record<string, unknown> = question.kind === "choice"
      ? {
          type: "choice",
          instructions: question.text,
          criteria: Object.fromEntries(question.options.map(o => [o, o])),
        }
      : { type: question.kind, instructions: question.text };
    const res = await fetch(this.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify({ model: this.model, state, questions: { q0: q } }),
    });
    if (!res.ok) throw new Error(`Jev API ${res.status}: ${await res.text()}`);
    const data = (await res.json()) as Record<string, unknown>;
    if (process.env.JEV_DEBUG) console.error("[jev raw]", JSON.stringify(data).slice(0, 800));
    const answers = (data.answers ?? data.questions ?? data) as Record<string, Record<string, unknown>>;
    const a = answers.q0 ?? Object.values(answers)[0];
    if (!a) throw new Error(`Jev API 响应缺少 answers.q0：${JSON.stringify(data).slice(0, 300)}`);
    if (question.kind === "noul") {
      const p = Number(a.noul ?? a.probability);
      if (!Number.isFinite(p)) throw new Error(`Jev noul 响应缺少概率字段：${JSON.stringify(a)}`);
      return { kind: "noul", value: p >= 0.5, confidence: Math.abs(2 * p - 1) };
    }
    if (question.kind === "choice") {
      const winner = String(a.choice ?? a.value ?? a.answer);
      return { kind: "choice", value: winner, confidence: Number(a.confidence ?? 0.5) };
    }
    return { kind: "score", value: Number(a.score ?? a.value ?? a.answer), confidence: Number(a.confidence ?? 0.5) };
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
