/**
 * 稀疏边存储（#4 规模墙的物理层）：N×N 稠密矩阵（32N² 字节/网）换成
 * 按行组织的邻接 Map。语义与稠密版逐位一致：
 * - 读：缺失键 ≡ 0（与稠密版的 `?? 0` 相同）；
 * - 写 0 = 删除该键（稠密版显式写 0 读出来也是 0）；
 * - 求和顺序由调用方的升序循环决定（activeIds/pattern 驱动），
 *   存储本身不参与求和顺序——等价性由 check-solver-equivalence 逐位验证。
 */

export class SparseMatrix {
  private readonly rows = new Map<number, Map<number, number>>();

  get(i: number, j: number): number {
    return this.rows.get(i)?.get(j) ?? 0;
  }

  /** 写单元素；v===0 时删除（读 0 与不存在等价） */
  set(i: number, j: number, v: number): void {
    if (v === 0) {
      const r = this.rows.get(i);
      if (r) {
        r.delete(j);
        if (r.size === 0) this.rows.delete(i);
      }
      return;
    }
    let r = this.rows.get(i);
    if (!r) {
      r = new Map();
      this.rows.set(i, r);
    }
    r.set(j, v);
  }

  /** 清空某神经元的全部入边与出边（槽位回收用；O(边数)，非常态路径） */
  clearNeuron(id: number): void {
    this.rows.delete(id);
    for (const r of this.rows.values()) r.delete(id);
  }

  /** 非零边迭代（顺序不保证——只用于与顺序无关的扫描，如 DI 源枚举） */
  *entries(): Generator<[i: number, j: number, w: number]> {
    for (const [i, r] of this.rows) {
      for (const [j, w] of r) yield [i, j, w];
    }
  }

  get edgeCount(): number {
    let n = 0;
    for (const r of this.rows.values()) n += r.size;
    return n;
  }
}
