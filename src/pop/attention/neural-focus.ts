import { EnergyNetwork } from "../../index.js";

/**
 * 神经化焦点竞争（M3）：焦点由 WTA 动力学择出，无评分公式。
 *
 * 结构（每个候选对象一对神经元）：
 * - driver_X：钳制激活，driver_X → marker_X 的边强 = 该对象的失配场
 *   （mismatchField，网络量）+ 持有焦点时的惯性边——失配越大挑战越强，
 *   迟滞（惯性边）是连接性质，不是迟滞公式；
 * - 边强每帧**绝对重写**（setWeight，不累加）：失配场随观察更新；
 * - 竞争：全部 marker 互抑（Γ）——WTA 择出最强候选；
 * - 抢占（非阈值参数）：失配场大到直接越过点火阈值时挑战核自然点燃。
 */

export class NeuralFocusNet {
  readonly net: EnergyNetwork;
  private readonly ids: string[] = [];
  private currentFocus: string | null = null;
  private readonly inertia: number;
  private readonly mismatches = new Map<string, number>();

  constructor(objectIds: readonly string[], options: { inertia?: number } = {}) {
    this.ids = [...objectIds];
    this.inertia = options.inertia ?? 2.0;
    this.net = new EnergyNetwork({
      neuronCount: this.ids.length * 2,
      activationEnergy: 1.0,
      maintenanceEnergy: 0.5,
      learningRate: 0.1,
      maxWeight: 40.0,
    });
    // 评审 A15/C08 修复（WTA 诚实化）：
    // ① 互抑只写无序对一次（修复前 (i,j)/(j,i) 双写，Γ 意外翻倍）；
    // ② Γ=6 的语义：切换要求挑战者失配 > 在位者 + 惯性 + Γ − θ 量级——
    //    惯性持焦与换焦都由动力学完成（既有测试锁定）；
    // ③ 诚实边界：完全等强的候选可以共存（8−6=2>θ），此时 select 的择一是
    //    **读出层**操作（argmax）——机制叙述应为"神经候选筛选 + 读出择优"，
    //    唯一性不靠静默兜底，平局由 lastSelectionTied 如实上报。
    const GAMMA = 6;
    for (let i = 0; i < this.ids.length; i++) {
      for (let j = i + 1; j < this.ids.length; j++) {
        this.net.strengthenInhibitory(2 * i + 1, 2 * j + 1, GAMMA);
      }
    }
    for (const id of this.ids) this.mismatches.set(id, 0);
  }

  private driver(id: string): number {
    return 2 * this.ids.indexOf(id);
  }

  private marker(id: string): number {
    return 2 * this.ids.indexOf(id) + 1;
  }

  /** 帧开始：失配场是事件信号不是状态——每帧清零后由本帧事件重写 */
  beginFrame(): void {
    for (const id of this.ids) this.mismatches.set(id, 0);
  }

  /** 更新某对象的失配场（driver→marker 边强，绝对重写） */
  setMismatch(id: string, field: number): void {
    this.mismatches.set(id, field);
  }

  /** 上一次择选是否出现多 marker 平局（如实上报；Γ 严格化后应为否） */
  lastSelectionTied = false;

  /** 焦点动力学择选：重写边强 → 钳制全部 driver → 退火 marker 竞争 → 胜者 id */
  select(seed = 1): string {
    for (const id of this.ids) {
      const m = (this.mismatches.get(id) ?? 0) + (id === this.currentFocus ? this.inertia : 0);
      this.net.setWeight(this.driver(id), this.marker(id), m);
    }
    const clamped = this.ids.map((_, i) => 2 * i);
    const markers = this.ids.map((_, i) => 2 * i + 1);
    const result = this.net.settleAnnealed(clamped, [], {
      seed,
      extraCandidates: markers,
      quenchCandidatesOnly: true,
      levels: 8,
      sweepsPerLevel: 10,
    });
    // 读出：激活 marker 中边最强者；多激活 = 平局如实标记（不静默兜底）
    const active = this.ids.filter((id) => result.activeNeurons.includes(this.marker(id)));
    this.lastSelectionTied = active.length > 1;
    let best: { id: string; field: number } | null = null;
    for (const id of active) {
      const f = this.net.getWeight(this.driver(id), this.marker(id));
      if (!best || f > best.field) best = { id, field: f };
    }
    if (best) this.currentFocus = best.id;
    return best ? best.id : (this.currentFocus ?? this.ids[0]!);
  }

  get focus(): string | null {
    return this.currentFocus;
  }
}
