import {
  EnergyNetwork,
  hebbianLearn,
  detectWells,
  learnSequence,
  runTransitions,
} from "./index.js";

/**
 * 有向通道与噪声转移的实机演示。
 * 运行：npm run build && node dist/src/demo-sequence.js
 */

const SEP = "─".repeat(64);
const show = (t: string): void => console.log(`\n${SEP}\n${t}\n${SEP}`);

const A = [0, 1, 2, 3];
const B = [8, 9, 10, 11];
const C = [16, 17, 18, 19];

function build(): EnergyNetwork {
  const net = new EnergyNetwork({
    neuronCount: 24,
    activationEnergy: 1.0,
    maintenanceEnergy: 0.5,
    learningRate: 0.1,
    maxWeight: 1.0,
    maxDirectedWeight: 1.0,
  });
  for (const c of [A, B, C]) {
    hebbianLearn(net, c, 8);
    hebbianLearn(net, [c[0]!, c[1]!], 2);
  }
  return net;
}

const names = ["A", "B", "C"];

show("① 对称关联（W）与有向通道（D）是两种机制");
{
  const net = build();
  console.log(`学习三个势阱 A/B/C 后：W(0,1)=${net.getWeight(0, 1).toFixed(2)}，D(0→8)=${net.getDirectedWeight(0, 8)}（尚未时序学习）`);
  learnSequence(net, [A, B, C], 7);
  console.log(`时序学习 [A,B,C] ×7 后：D(0→8)=${net.getDirectedWeight(0, 8).toFixed(2)}，D(8→16)=${net.getDirectedWeight(8, 16).toFixed(2)}`);
  console.log(`反方向：D(8→0)=${net.getDirectedWeight(8, 0)}，D(16→8)=${net.getDirectedWeight(16, 8)}（单向通道）`);
}

show("② 捕获 → 噪声推动的链式转移 A→B→C（T=0.15，β=0.5，κ=0.05）");
{
  const net = build();
  learnSequence(net, [A, B, C], 7);
  const wells = detectWells(net);
  const settled = net.settle([0, 2]);
  console.log(`输入线索 [0,2] → settle 捕获势阱 A：激活 [${settled.activeNeurons}]`);
  const r = runTransitions(net, wells, { steps: 120, temperature: 0.15, driveBeta: 0.5, fatigueRate: 0.05, seed: 7 });
  const tl = r.wellTimeline.map((w) => (w === null ? "·" : names[w]!));
  console.log("主导势阱时间线（每格 1 步，· = 无占优）：");
  for (let row = 0; row < tl.length; row += 40) {
    console.log(`  ${String(row).padStart(3)} |${tl.slice(row, row + 40).join("")}|`);
  }
  console.log(`转移事件：${r.transitions.map((t) => `${names[t.from]}→${names[t.to]}@${t.step}`).join("，")}`);
  console.log(`有向驱动做功 = ${r.driveWork.toFixed(2)}，接受翻转 ${r.acceptedFlips} 次（其中上坡 ${r.acceptedUphill} 次）`);
  console.log(`结束时激活集合 = [${r.finalActive}]（疲劳交接完毕，归于静息）`);
  net.reset(); // 回合结束恢复静息，连接保留
}

show("③ 统计：10 个种子的有序链命中率");
{
  let hits = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const net = build();
    learnSequence(net, [A, B, C], 7);
    const wells = detectWells(net);
    net.settle([0, 2]);
    const r = runTransitions(net, wells, { steps: 120, temperature: 0.15, driveBeta: 0.5, fatigueRate: 0.05, seed });
    const seq = r.transitions.map((t) => `${names[t.from]}→${names[t.to]}`).join(",");
    const ok = /A→B.*B→C/.test(seq);
    if (ok) hits++;
    console.log(`  seed=${String(seed).padStart(2)}  ${ok ? "✓" : "✗"}  ${seq || "(无转移)"}`);
  }
  console.log(`有序 A→B→C：${hits}/10`);
}

show("④ 对照：无噪声 / 无驱动 / 无疲劳");
{
  const mk = () => {
    const net = build();
    learnSequence(net, [A, B, C], 7);
    const wells = detectWells(net);
    net.settle([0, 2]);
    return { net, wells };
  };
  {
    const { net, wells } = mk();
    const r = runTransitions(net, wells, { steps: 120, temperature: 0, driveBeta: 0.5, fatigueRate: 0.05, seed: 7 });
    console.log(`无噪声（T=0）：转移事件 ${r.transitions.length} 个，结束激活 [${r.finalActive}] —— 驱动只压势垒不归零，没有噪声无人点火，A 疲劳后归于静息`);
  }
  {
    const { net, wells } = mk();
    const r = runTransitions(net, wells, { steps: 120, temperature: 0.15, driveBeta: 0, fatigueRate: 0.05, seed: 7 });
    console.log(`无驱动（β=0）：转移事件 ${r.transitions.length} 个 —— 单个噪声点火会立即熄灭，无法级联成簇`);
  }
  {
    const { net, wells } = mk();
    const r = runTransitions(net, wells, { steps: 120, temperature: 0.15, driveBeta: 0.5, fatigueRate: 0, seed: 7 });
    const finals = r.finalActive.map((id) => (A.includes(id) ? "A" : B.includes(id) ? "B" : "C")).join("");
    console.log(`无疲劳（κ=0）：结束激活含势阱成员 [${finals}] —— 旧势阱不崩塌，三者共存而非转移`);
  }
}
console.log();
