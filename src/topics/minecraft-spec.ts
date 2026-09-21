import type { PerceptionSpec } from "../perception/scene-parser.js";
import type { ExecutionSpec } from "../execution/action-executor.js";

/**
 * Minecraft 对接的感知/执行规格（设计层）。
 * 链路：截图/状态流 → 视觉前端（图像→文字描述）→ Jev 语义解析 → 条件帧
 * → 能量网络学习与规划 → 执行映射 → mineflayer 命令。
 * 本文件只定义规格（声明式数据），不含运行时依赖；真实运行需要
 * Minecraft 服务器 + mineflayer + 视觉模型端点（后续接入）。
 *
 * 设计原则：全部维度粗粒度分档（Minecraft 的精确坐标不进网络——
 * 只保留对决策有意义的语义档位，与"感受野+概念"的感知哲学一致）。
 */

/** 感知规格：结构化状态文本（如 "生命值 18，饥饿值 15，脚下是草方块，前方 2 格是树，天黑中，附近有僵尸"） → 条件帧 */
export const MC_PERCEPTION: PerceptionSpec = {
  dims: [
    { name: "health", values: [
      { value: 0, label: "濒危", desc: "生命值极低，快死了" },
      { value: 1, label: "受伤", desc: "有损伤但还能行动" },
      { value: 2, label: "健康", desc: "生命值满|满血|接近满" },
    ], ask: "当前生命值处于哪一档？" },
    { name: "hunger", values: [
      { value: 0, label: "饥荒", desc: "饥饿值很低|快饿死|饥荒" },
      { value: 1, label: "饱足", desc: "饥饿正常|不饿|饱腹" },
    ], ask: "当前饥饿状态如何？" },
    { name: "time", values: [
      { value: 0, label: "白天", desc: "天亮，视野好" },
      { value: 1, label: "黑夜", desc: "天黑，怪物出没" },
    ], ask: "当前是白天还是黑夜？" },
    { name: "threat", values: [
      { value: 0, label: "无威胁", desc: "附近没有敌对生物" },
      { value: 1, label: "有威胁", desc: "附近有僵尸|骷髅|苦力怕|蜘蛛" },
    ], ask: "附近是否存在敌对生物威胁？" },
    { name: "ground", values: [
      { value: 0, label: "坚实地面", desc: "脚下是可站立的实体方块" },
      { value: 1, label: "危险地形", desc: "脚下是岩浆|水|悬崖边缘|虚空" },
    ], ask: "脚下地形是否安全？" },
  ],
  confidenceThreshold: 0.6,
  clarityAsk: (d) => `状态描述对${d}的描述是否明确无歧义？`,
};

/** 执行规格：抽象动作档 → mineflayer 风格命令文本（真实接入时替换为 bot API 调用） */
export const MC_EXECUTION: ExecutionSpec = {
  environment: "minecraft",
  actions: [
    { dimension: "move", commands: {
      "0": "前进一格", "1": "后退一格", "2": "向左移动", "3": "向右移动", "4": "原地不动",
    } },
    { dimension: "act", commands: {
      "0": "挖掘前方方块", "1": "放置方块", "2": "攻击最近目标", "3": "进食", "4": "无操作",
    }, feasibilityAsk: "这个动作在当前状态下是否安全可执行？" },
  ],
  confidenceThreshold: 0.5,
};
