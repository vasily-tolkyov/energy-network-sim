import type { PotentialWell } from "./wells.js";

export interface WellReadout {
  readonly wellId: number;
  /** 教师或模型自身赋予的标签；未标注时为 null */
  readonly label: string | null;
  /** 势阱成员中被激活的比例 */
  readonly activationRatio: number;
}

/**
 * 读出模块（要求 9）：新输入导致网络收敛后，识别激活模式中被激活的势阱，
 * 读出对应标签。标签由教师程序（labelWell）或模型自身（autoLabel）赋予。
 */
export class ReadoutModule {
  private wells: PotentialWell[];
  private readonly labels = new Map<number, string>();
  private readonly threshold: number;

  constructor(wells: PotentialWell[], readoutThreshold: number) {
    this.wells = wells;
    this.threshold = readoutThreshold;
  }

  /** 势阱快照更新（学习导致连接变化后重新检测） */
  updateWells(wells: PotentialWell[]): void {
    this.wells = wells;
  }

  /** 教师程序接口：给势阱赋标签（要求 8 后半句） */
  labelWell(wellId: number, label: string): void {
    if (!this.wells.some((w) => w.wellId === wellId)) {
      throw new Error(`unknown wellId: ${wellId}`);
    }
    this.labels.set(wellId, label);
  }

  /** 模型自身赋标签：为未标注势阱生成占位标签 */
  autoLabel(prefix = "well"): void {
    for (const well of this.wells) {
      if (!this.labels.has(well.wellId)) {
        this.labels.set(well.wellId, `${prefix}-${well.wellId}`);
      }
    }
  }

  /**
   * 识别激活集合 A 中被激活的势阱（要求 9）：
   * 激活率 r_k = |A ∩ S_k| / |S_k| ≥ 阈值判为被激活；
   * 返回按激活率降序排列的读出结果。
   */
  identify(activeNeurons: Iterable<number>): WellReadout[] {
    const active = new Set(activeNeurons);
    const readouts: WellReadout[] = [];
    for (const well of this.wells) {
      const members = well.memberNeuronIds;
      if (members.length === 0) continue;
      let hit = 0;
      for (const id of members) if (active.has(id)) hit++;
      const ratio = hit / members.length;
      if (ratio >= this.threshold) {
        readouts.push({
          wellId: well.wellId,
          label: this.labels.get(well.wellId) ?? null,
          activationRatio: ratio,
        });
      }
    }
    readouts.sort((a, b) => b.activationRatio - a.activationRatio || a.wellId - b.wellId);
    return readouts;
  }
}
