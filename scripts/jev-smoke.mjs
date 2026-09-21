/** Jev 实网冒烟：读 TYPESAFE_API_KEY 环境变量，发一个 Noul 与一个 Choice，
 * 打印原始响应，验证端点形状与连通性。不产生任何费用量级（几分钱以下）。 */
import { JevApiBackend } from "../dist/src/perception/decision-backend.js";

const key = process.env.TYPESAFE_API_KEY;
if (!key) throw new Error("set TYPESAFE_API_KEY");
const api = new JevApiBackend("https://api.typesafe.ai/v1/systemone", key);

const state = "实验室桌上有一杯热水和几块冰，墙上的开关处于断开状态，电压表读数正常。";
const noul = await api.ask(state, { kind: "noul", text: "开关当前是否闭合？" });
console.log("noul:", JSON.stringify(noul));
const choice = await api.ask(state, { kind: "choice", text: "开关当前是哪种状态？", options: ["断开", "半开", "闭合"] });
console.log("choice:", JSON.stringify(choice));
