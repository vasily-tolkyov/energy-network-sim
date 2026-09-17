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
    this.ids.forEach((_, i) => {
      const marker = 2 * i + 1;
      this.ids.forEach((_, j) => {
        if (j !== i) this.net.strengthenInhibitory(marker, 2 * j + 1, 3.0);
      });
    });
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
    // 胜者：激活 marker 中边最强者
    let best: { id: string; field: number } | null = null;
    for (const id of this.ids) {
      if (!result.activeNeurons.includes(this.marker(id))) continue;
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
