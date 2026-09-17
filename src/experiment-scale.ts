import { mkdirSync, writeFileSync } from "node:fs";
import {
  EnergyNetwork,
  hebbianLearn,
  detectWells,
  learnSequence,
  learnBranches,
  runTransitions,
  type PotentialWell,
} from "./index.js";

/**
 * 较大规模单网络实机测试：经验学习 + 基于经验的链式生成。
 *
 * 规模：N=2000（稠密矩阵 32MB，本机安全范围）。
 * 势阱：16 神经元/簇，簇内 0.8 + 峰边 1.0，θ=1.5。
 * 有向通道 d=0.1：完整前驱（16 成员）给出 β·g=0.8 的助力，点火仍需
 * 噪声翻越 ΔE′=0.7 的势垒；κ=0.7 使 16 深势阱在 ~15 步驻留后崩塌交接。
 *
 * 运行：npm run build && node dist/src/experiment-scale.js
 */

const N = 2000;
const WELL_SIZE = 16;
const THETA = 1.5;
const SEQ = { steps: 80, temperature: 0.12, driveBeta: 0.5, fatigueRate: 0.7, fatigueRecoveryRate: 0.15 } as const;

const lines: string[] = [];
function out(s = ""): void {
  console.log(s);
  lines.push(s);
}

function buildNet(): EnergyNetwork {
  return new EnergyNetwork({
    neuronCount: N,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 1.0,
    maxDirectedWeight: 1.0,
  });
}

/** 不相交簇：第 k 簇 = [k×size, (k+1)×size) */
function makeClusters(count: number, size = WELL_SIZE): number[][] {
  return Array.from({ length: count }, (_, k) =>
    Array.from({ length: size }, (_, i) => k * size + i),
  );
}

function trainWells(net: EnergyNetwork, clusters: readonly number[][]): void {
  for (const c of clusters) {
    hebbianLearn(net, c, 8);
    hebbianLearn(net, [c[0]!, c[1]!], 2);
  }
}

/** 检出率：训练簇中能被某个检测势阱覆盖（Jaccard ≥ 0.8）的比例 */
function detectionRecall(clusters: readonly number[][], wells: readonly PotentialWell[]): number {
  let hit = 0;
  for (const c of clusters) {
    const cs = new Set(c);
    const found = wells.some((w) => {
      const inter = w.memberNeuronIds.filter((id) => cs.has(id)).length;
      const union = new Set([...c, ...w.memberNeuronIds]).size;
      return inter / union >= 0.8;
    });
    if (found) hit++;
  }
  return hit / clusters.length;
}

/** 捕获：cue 簇的 25% → settle → 成员完成率；同时统计误激活（簇外神经元） */
function captureStats(
  net: EnergyNetwork,
  clusters: readonly number[][],
): { completion: number; stray: number } {
  let completionSum = 0;
  let strayTotal = 0;
  for (const c of clusters) {
    const cue = c.slice(0, WELL_SIZE / 4);
    const r = net.settle(cue);
    const active = new Set(r.activeNeurons);
    const cs = new Set(c);
    let hit = 0;
    for (const id of c) if (active.has(id)) hit++;
    completionSum += hit / c.length;
    for (const id of active) if (!cs.has(id)) strayTotal++;
  }
  return { completion: completionSum / clusters.length, stray: strayTotal / clusters.length };
}

/** 链生成：cue 首簇 → settle 捕获 → 转移；返回按序命中前缀长度与完整率 */
function chainRun(
  net: EnergyNetwork,
  wells: readonly PotentialWell[],
  chainClusters: readonly number[][],
  chainWellIds: readonly number[],
  seed: number,
): { prefix: number; events: string } {
  const cue = chainClusters[0]!.slice(0, WELL_SIZE / 4);
  net.settle(cue);
  const r = runTransitions(net, wells, { ...SEQ, seed });
  // 期望顺序：chainWellIds[0]→[1]→[2]…；从转移事件中提取有序前缀
  let prefix = 0;
  let want = 1;
  for (const t of r.transitions) {
    if (want < chainWellIds.length && t.from === chainWellIds[want - 1] && t.to === chainWellIds[want]) {
      want++;
      prefix = want - 1;
      if (want === chainWellIds.length) break;
    }
  }
  return { prefix, events: r.transitions.map((t) => `${t.from}→${t.to}`).join(",") };
}

function timed<T>(label: string, fn: () => T): T {
  const t0 = performance.now();
  const v = fn();
  out(`  [${label} ${(performance.now() - t0).toFixed(0)}ms]`);
  return v;
}

// ── S1 容量与检出 ────────────────────────────────────────────────
out("═".repeat(60));
out("S1 经验学习：容量与检出（簇大小 16，cue 25%）");
out("═".repeat(60));
for (const count of [20, 50, 100]) {
  const net = buildNet();
  const clusters = makeClusters(count);
  timed("learn", () => trainWells(net, clusters));
  const wells = timed("detect", () => detectWells(net));
  const recall = detectionRecall(clusters, wells);
  const cap = timed("capture×" + count, () => captureStats(net, clusters));
  out(
    `  势阱数 ${String(count).padStart(3)}：检测势阱 ${String(wells.length).padStart(3)}，` +
      `检出率 ${(recall * 100).toFixed(0)}%，捕获完成率 ${(cap.completion * 100).toFixed(1)}%，` +
      `平均误激活 ${cap.stray.toFixed(2)} 个`,
  );
}

// ── S2 链式生成 ──────────────────────────────────────────────────
out("");
out("═".repeat(60));
out(`S2 链式生成：10 条不相交 5 节链（50 势阱），T=${SEQ.temperature} β=${SEQ.driveBeta} κ=${SEQ.fatigueRate}`);
out("═".repeat(60));
{
  const net = buildNet();
  const clusters = makeClusters(50);
  trainWells(net, clusters);
  const wells = detectWells(net);
  // 检测势阱成员 → 训练簇索引映射
  const clusterOfWell = wells.map((w) =>
    clusters.findIndex((c) => w.memberNeuronIds.includes(c[0]!)),
  );
  const chains = Array.from({ length: 10 }, (_, k) => clusters.slice(k * 5, k * 5 + 5));
  let linkOk = 0;
  let linkTotal = 0;
  let fullChains = 0;
  for (let c = 0; c < chains.length; c++) {
    const chainClusters = chains[c]!;
    timed(`learn-seq${c}`, () => learnSequence(net, chainClusters, 1, 0.15)); // d = 0.15
    const chainWellIds = chainClusters.map((cl) => {
      const wi = wells.findIndex((w) => w.memberNeuronIds.includes(cl[0]!));
      return wi;
    });
    if (chainWellIds.includes(-1)) {
      out(`  链 ${c}：首簇未被检测为势阱，跳过`);
      continue;
    }
    for (let seed = 1; seed <= 3; seed++) {
      const r = timed(`chain${c}-s${seed}`, () => chainRun(net, wells, chainClusters, chainWellIds, seed));
      linkOk += r.prefix;
      linkTotal += chainWellIds.length - 1;
      if (r.prefix === chainWellIds.length - 1) fullChains++;
      out(
        `  链 ${c} seed=${seed}：前缀 ${r.prefix}/${chainWellIds.length - 1} 棒  ` +
          `事件[${r.events}]（簇序 ${chainWellIds.map((wi) => clusterOfWell[wi]).join("→")}）`,
      );
    }
  }
  out(`  汇总：逐棒交接成功率 ${((linkOk / Math.max(1, linkTotal)) * 100).toFixed(1)}%，完整链 ${fullChains}/30`);
}

// ── S3 分支生成 ──────────────────────────────────────────────────
out("");
out("═".repeat(60));
out("S3 分支生成：共享起点 A，等强双通道 A→B / A→D + 抑制竞争（互斥概率选择）");
out("═".repeat(60));
{
  const net = buildNet();
  const clusters = makeClusters(20);
  trainWells(net, clusters);
  const wells = detectWells(net);
  const [clA, clB, clD] = [clusters[0]!, clusters[1]!, clusters[2]!];
  learnBranches(net, clA, [clB, clD], { repeats: 1, eta: 0.15, gamma: 0.3 });
  const wiOf = (cl: number[]) => wells.findIndex((w) => w.memberNeuronIds.includes(cl[0]!));
  const [wA, wB, wD] = [wiOf(clA), wiOf(clB), wiOf(clD)];
  if (wA < 0 || wB < 0 || wD < 0) {
    out(`  分支簇未全部检出（${wA},${wB},${wD}），跳过`);
  } else {
  const tally = { B: 0, D: 0, both: 0, neither: 0 };
  for (let seed = 1; seed <= 12; seed++) {
    net.settle(clA.slice(0, WELL_SIZE / 4));
    const r = timed(`branch-s${seed}`, () =>
      runTransitions(net, wells, { ...SEQ, seed }),
    );
    // 用累计激活步数判胜负（共存期前驱会掩盖后继的占优时间线）
    const bSteps = r.wellActiveSteps[wB] ?? 0;
    const dSteps = r.wellActiveSteps[wD] ?? 0;
    const bWon = bSteps >= 10;
    const dWon = dSteps >= 10;
    if (bWon && dWon) tally.both++;
    else if (bWon) tally.B++;
    else if (dWon) tally.D++;
    else tally.neither++;
  }
  out(`  12 种子分布：B 胜=${tally.B}，D 胜=${tally.D}，共存=${tally.both}，都无=${tally.neither}（互斥竞争：共存应≈0，胜负应均分）`);
  }
}

// ── S4 大规模下的对照 ────────────────────────────────────────────
out("");
out("═".repeat(60));
out("S4 对照（N=2000 规模抽查）：无噪声 / 无驱动");
out("═".repeat(60));
{
  const net = buildNet();
  const clusters = makeClusters(50);
  trainWells(net, clusters);
  const wells = detectWells(net);
  const chain = clusters.slice(0, 5);
  learnSequence(net, chain, 1, 0.15);
  for (const [label, opts] of [
    ["T=0", { ...SEQ, temperature: 0 }],
    ["β=0", { ...SEQ, driveBeta: 0 }],
  ] as const) {
    net.settle(chain[0]!.slice(0, WELL_SIZE / 4));
    const r = runTransitions(net, wells, { ...opts, seed: 1 });
    out(`  ${label}：转移事件 ${r.transitions.length} 个（期望 0）`);
  }
}

mkdirSync("runs", { recursive: true });
writeFileSync("runs/scale-v1.log", lines.join("\n") + "\n");
out(`\n日志已写入 runs/scale-v1.log`);
