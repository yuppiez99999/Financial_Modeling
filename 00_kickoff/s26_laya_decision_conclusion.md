# J1 / S26 · Laya 类型化决策只读接入评估结论（只留本地 Laya，无 Jev 在线臂）

> 本文件是**证据与决策材料**，不是决策。
> 「是否接受 Laya 重型依赖代价」= **T26.1**，「整条链路是否保留」= **T26.5**，
> 两者均属人工检查点，NPC **不代签**，现状 `pending`。
> 全局纪律：`affects_gate` / `affects_signal` **恒为 `false`**（结构性保证，非约定）。

---

## 一、从哪来（Issue #66 选型 → 用户决策）

Issue #66 起因：Jev 与 Laya 哪个适合接入本项目。
上一轮评估给出方向：**两者分工不同、非二选一**，并建议 **Laya 打底 + Jev 可选补位（级联）**。
用户 2026-09-20 决策（会话指令原文「1同意 2只留Laya 3开工」）：

1. **同意** Laya 打底方向；
2. **只留 Laya** —— 移除 Jev 在线升级臂；
3. **开工** —— 直接落 T26.2~T26.4（纯离线、只读、零门禁改动）。

## 二、为什么只留 Laya（这不是妥协，是纪律）

Jev 是**闭源托管 API** = 网络依赖 + 成本 + **不可复现** ——
直接撞上 H5（LLM 投研辅助）被否的三条死因。纯 Jev 等于把 H5 的坑再踩一遍。
Laya 是 Jev 的**开放权重同构替代**（同一 request/response 形状：`choice` / `score` / `noul`），
**可本地跑**（MLX/CoreML/ONNX）⇒ **CI 离线可复现**。

本项目现在缺的恰好是这一层：
- 已交付只读契约 `decision-feed/1`，但 `advisory.recommended_threshold`
  是「已配置现行值」而非择优结果（文档明写）；
- 现有 LightGBM 概率**未校准**（`model_degeneration_conclusion`：输出常数化、spread≈1pp）；
- Laya 的**校准置信度 + 类型化决策**正好可做**只读的第二决策源 / 交叉验证臂**。

**级联简化**：原设计 `Laya 本地 → (置信度<阈值) 升级 Jev`；移除 Jev 臂后，
不可用 / 低置信一律**本地降级为 `noul`（弃权）** —— 零网络调用、零信号产出。

## 三、方法（离线优先 / 不可量化即止损）

- **适配器**（T26.2）：`LayaLocalAdapter` 本地只读适配，权重缺失即降级
  （`available=False`，不联网 / 不下载 / 不报错）；
- **离线对照**（T26.3）：Laya 置信度 vs 现有 LightGBM 概率
  （高置信子集命中率 + 置信度分档×收益单调性 + 两源 Spearman）；
  真实权重未接入时如实记 `unverifiable` + **止损**（延续 H5，不编造效果）；
- **级联冒烟**（T26.4）：只留 Laya 口径下的本地降级路径验证（`network_calls=0`、`raises=False`）；
- **保留决策**（T26.5）：决策单，无 `decided_by` 时最多 `defer` / `cancel`。

复现命令：

```bash
python main.py laya-decision                      # 适配器 + 离线对照 + 级联冒烟 + 决策单
python main.py laya-decision --laya-weights /path/to/laya   # 指定本地权重（仍只读）
```

落盘：`reports/laya_evaluation.json` / `laya_contrast.json` / `laya_cascade_smoke.json` / `laya_decision.json`。

## 四、准入评估结果（T26.1 材料）

| 候选 | 离线可复现 | 许可证 | 依赖重量 | 判定 |
|---|:---:|---|---|---|
| Laya（Convai 开放权重） | ✅ | 需逐项核 | **重型 ~1.7GB / ~2GB RAM** | `conditional`：**须 T26.1 人工确认依赖代价** |
| Jev（TypeSafe 闭源 API） | ❌ | 闭源托管 | 不可本地化 | `reject`：按用户决策只留 Laya |

`n_worth_introducing=1`、`n_needs_manual_admission=1`。
**Laya 权重属重型依赖**，与「零重型依赖」纪律冲突 ⇒ 必须先过 T26.1，不过则整阶段取消。

## 五、交付物

| 任务 | 交付 | 状态 |
|:---:|:---|:---:|
| T26.1 | 准入评估（许可证 / 依赖重量 / py3.8 CI 离线） | 🔶 人工（评估器已交付，采纳待定） |
| T26.2 | `src/eval/laya_typed_decision.py`：只读适配器（挂报告层） | ✅ |
| T26.3 | 离线对照（Laya vs LightGBM 概率） | ✅ |
| T26.4 | 级联冒烟（只留 Laya 的本地降级路径） | ✅ |
| T26.5 | 是否接入 / 是否仅评估 | 🔶 人工 |

CLI：`python main.py laya-decision [--laya-weights ... --decided-by ... --reason ...]`
测试：`tests/test_laya_typed_decision.py`（全离线合成数据，不触网、不下载权重）

## 六、边界（写在最前，也写在最后）

- **不进信号路径**：全部产出 `readonly=True`，无 `probability` / `direction` / `signal` 字段；
  `affects_gate` / `affects_signal` **恒为 False**（结构性保证）；
- **CI 离线**：只用本地权重；权重缺失如实降级 `unavailable`，**不联网、不下载**；
- **不编造效果**：真实权重未接入 ⇒ 对照记 `unverifiable` + 止损，
  **不得**以「Laya 有提升」为由放行；
- **strategy_gate 零改动**。
