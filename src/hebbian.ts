import { learning, neuronId } from "./validate.js";
import type { EnergyNetwork } from "./network.js";

/**
 * 赫布学习（要求 6）：被同一输入共同激活的神经元之间，连接强度增加 η。
 * 重复呈现同一模式 → 连接逐步增强直至 maxWeight；
 * 连接越强，模式能量 E(s) 中的负耦合项越大，总能耗越低。
 */
export function hebbianLearn(
  network: EnergyNetwork,
  activePattern: Iterable<number>,
  repeats = 1,
  eta?: number,
  cap?: number,
): void {
  const neurons = [...new Set(activePattern)].sort((a, b) => a - b);
  const step = eta ?? network.config.learningRate;
  // Whole-batch atomicity: validate every endpoint, even singleton/zero-repeat batches.
  learning(repeats, step, cap ?? network.config.maxWeight);
  for (const id of neurons) neuronId(id, network.neuronCount);
  for (let r = 0; r < repeats; r++) {
    for (let a = 0; a < neurons.length; a++) {
      for (let b = a + 1; b < neurons.length; b++) {
        network.strengthen(neurons[a]!, neurons[b]!, step, cap);
      }
    }
  }
}
