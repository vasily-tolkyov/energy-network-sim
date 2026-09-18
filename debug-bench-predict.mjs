// field-memory predict 速度基准（修复后）
import { pretrainContinuous } from "./dist/src/pop/concept/pretrain.js";
const { mem } = pretrainContinuous(40);
console.log(`规则 ${mem.ruleCount} 条，网络 ${mem.net.neuronCount} 神经元`);
const q = { color: 0.5, speed: 1.2, direction: 0.5, wallStiffness: 1.0, damping: 0.5, hitWall: 1.0 };
let worst = 0;
for (let i = 0; i < 5; i++) {
  const t0 = performance.now();
  mem.predict(q, i + 1);
  const ms = performance.now() - t0;
  worst = Math.max(worst, ms);
  console.log(`predict ${ms.toFixed(0)}ms`);
}
console.log(`峰值 ${worst.toFixed(0)}ms`);
