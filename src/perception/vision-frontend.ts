import type { SceneParser, PerceivedFrame, PerceptionSpec } from "./scene-parser.js";

/**
 * 视觉前端（图像识别的可插拔换能层）：图像 → 文字/结构化描述。
 * Jev 当前只接受文本/结构化状态，图像必须先经视觉模型转述——
 * 这一层就是"视网膜色素层"：把光子变成神经可读的电信号。
 * 感知链路：图像 → 视觉前端 → 场景描述 → SceneParser（决策后端）→ 条件帧。
 */

export interface VisionFrontEnd {
  /** 把图像转述为文字描述。imageRef = 本地路径、URL 或 base64。 */
  describe(imageRef: string, prompt?: string): Promise<string>;
}

/** OpenAI 兼容视觉后端（任何支持 chat/completions + image_url 的视觉模型）。
 * 无端点/key 时响亮抛错——感知链路要么真实要么明确不可用。 */
export class OpenAiVisionFrontEnd implements VisionFrontEnd {
  constructor(
    private readonly endpoint: string | null,
    private readonly apiKey: string | null,
    private readonly model = "gpt-4o-mini",
  ) {}
  async describe(imageRef: string, prompt = "客观描述这张图片中的场景、物体状态与可见文字。"): Promise<string> {
    if (!this.endpoint || !this.apiKey) {
      throw new Error("OpenAiVisionFrontEnd 未配置 endpoint/apiKey——视觉后端不可用（禁止静默降级）");
    }
    const res = await fetch(this.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: "user", content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: imageRef } },
        ] }],
      }),
    });
    if (!res.ok) throw new Error(`vision API ${res.status}: ${await res.text()}`);
    const data = (await res.json()) as { choices: { message: { content: string } }[] };
    return data.choices[0]!.message.content;
  }
}

/** Mock 视觉后端（测试用）：图片路径 → 预设描述的映射表；未登记的路径响亮抛错。 */
export class MockVisionFrontEnd implements VisionFrontEnd {
  constructor(private readonly table: Readonly<Record<string, string>>) {}
  describe(imageRef: string): Promise<string> {
    const hit = this.table[imageRef];
    if (!hit) return Promise.reject(new Error(`MockVision 未登记的图像：${imageRef}`));
    return Promise.resolve(hit);
  }
}

/** 完整感知管道：图像 → 视觉描述 → 语义解析 → 条件帧 */
export class VisionPerceptionPipeline {
  constructor(
    private readonly vision: VisionFrontEnd,
    private readonly parser: SceneParser,
  ) {}
  async perceiveImage(imageRef: string, spec: PerceptionSpec, prompt?: string): Promise<PerceivedFrame & { description: string }> {
    const description = await this.vision.describe(imageRef, prompt);
    const frame = await this.parser.parse(description, spec);
    return { ...frame, description };
  }
}
