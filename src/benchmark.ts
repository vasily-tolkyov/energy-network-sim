import os from "node:os";
import { EnergyNetwork, hebbianLearn, detectWells } from "./index.js";

/**
 * 规模基准：评估本机可用的最大网络规模，以及该规模下单次输入的时间成本。
 * 稠密权重矩阵占内存 8·N² 字节，这是规模的硬上限。
 * 运行：npm run build && node dist/src/benchmark.js
 */

const WELL_COUNT = 10; // 学习的势阱数
const WELL_SIZE = 32; // 每个势阱的神经元数
const SAMPLES = 3; // 每档规模的计时采样数（取最小值）
const SAMPLE_TIME_CAP_MS = 20_000; // 单档累计计时超过此值就停止采样
const SETTLE_STOP_MS = 60_000; // settle 单次输入超过此值即认为到顶，停止加码

interface Row {
  readonly n: number;
  readonly allocMs: number;
  readonly detectMs: number;
  readonly settleMs: number;
  readonly settleFlips: number;
  readonly annealMs: number;
  readonly rssGb: number;
}

function buildScaledNet(n: number): EnergyNetwork {
  const net = new EnergyNetwork({
    neuronCount: n,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 1.0,
  });
  // 10 个互不重叠的 32 神经元簇：簇内 0.8 + 峰边 1.0（与测试场景同构）
  for (let w = 0; w < WELL_COUNT; w++) {
    const base = w * WELL_SIZE;
    const cluster = Array.from({ length: WELL_SIZE }, (_, k) => base + k);
    hebbianLearn(net, cluster, 8);
    hebbianLearn(net, [base, base + 1], 2);
  }
  return net;
}

function timeIt(fn: () => void, samples: number): number {
  let best = Infinity;
  let spent = 0;
  for (let k = 0; k < samples; k++) {
    const t0 = performance.now();
    fn();
    const dt = performance.now() - t0;
    best = Math.min(best, dt);
    spent += dt;
    if (spent > SAMPLE_TIME_CAP_MS) break;
  }
  return best;
}

const rows: Row[] = [];
const freeBytes = os.freemem();
console.log(`本机：总内存 ${(os.totalmem() / 2 ** 30).toFixed(1)} GB，当前空闲 ${(freeBytes / 2 ** 30).toFixed(1)} GB`);
console.log(`权重矩阵为稠密 Float64：内存 = 8·N² 字节；每档学习 ${WELL_COUNT} 个 ${WELL_SIZE} 神经元势阱\n`);
console.log(
  "N".padStart(7) +
    "矩阵MB".padStart(9) +
    "分配ms".padStart(9) +
    "检阱ms".padStart(10) +
    "贪心ms/次".padStart(12) +
    "翻转数".padStart(8) +
    "退火ms/次".padStart(12) +
    "RSS GB".padStart(9),
);

for (let n = 1000; ; n = Math.round(n * 1.6)) {
  if (8 * n * n > freeBytes * 0.75) {
    console.log(
      `${String(n).padStart(7)}  矩阵 ${(8 * n * n / 2 ** 30).toFixed(1)} GB 超过空闲内存的 75%，跳过以防系统换页抖动`,
    );
    break;
  }
  let net: EnergyNetwork;
  const t0 = performance.now();
  try {
    net = buildScaledNet(n);
  } catch {
    console.log(`${String(n).padStart(7)}  —— 分配失败（内存上限），停止加码`);
    break;
  }
  const allocMs = performance.now() - t0;

  const t1 = performance.now();
  const wells = detectWells(net);
  const detectMs = performance.now() - t1;

  const input = [0, 2]; // 触及势阱 0
  let settleFlips = 0;
  const settleMs = timeIt(() => {
    settleFlips = net.settle(input).trace.flipCount;
  }, SAMPLES);
  const annealMs = timeIt(() => {
    net.settleAnnealed(input, wells, { seed: 7 });
  }, SAMPLES);

  const rssGb = process.memoryUsage().rss / 2 ** 30;
  rows.push({ n, allocMs, detectMs, settleMs, settleFlips, annealMs, rssGb });
  console.log(
    String(n).padStart(7) +
      (8 * n * n / 2 ** 20).toFixed(0).padStart(9) +
      allocMs.toFixed(0).padStart(9) +
      detectMs.toFixed(0).padStart(10) +
      settleMs.toFixed(1).padStart(12) +
      String(settleFlips).padStart(8) +
      annealMs.toFixed(1).padStart(12) +
      rssGb.toFixed(2).padStart(9),
  );

  if (settleMs > SETTLE_STOP_MS) {
    console.log(`\nsettle 单次输入超过 ${SETTLE_STOP_MS / 1000}s，判定为可交互使用的计算上限`);
    break;
  }
}
