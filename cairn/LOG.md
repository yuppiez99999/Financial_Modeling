# Project Cairn 项目日志（LOG）

本文件按倒序记录项目的实质性进展——最新条目紧贴本行下方。每条保持简短——只写摘要与指针；结论沉淀到 `cairn/<topic>.md` 知识专题文档。

## 2026-09-16 · 波动分层与置信度语义修正（Issue #55）

- **修正前一轮口径混淆**：`cairn/decision-source-contract` 的「高置信 ⇒ 收益为负」
  只在**低波动**状态成立。把高波/低波混在一个阈值分档里会得到自相矛盾结论。
- **置信度不是 edge 信号**：`IC(置信度, 未来收益) ≈ 0 ~ −0.09`（三周期一致），
  它度量的是「模型确定性」而非「这笔交易赚钱概率」。
- **波动分层是唯一正向读数**：高置信 × 高波动 → IC +0.23~+0.36、平均收益
  +1.5%~+3.8%、命中 0.65~0.67（三周期一致）；高置信 × 低波动 → 收益为负。
- **稳健性只到 suggestive**：随机 18 只子池 × 12 次，完整分歧仅 7/12 成立
  （`verdict=suggestive`）⇒ 够格作下一步实验方向，**不够格**改下游采纳口径。
- **周期权重重排被证伪**：单周期 IC 排序 ≠ 组合收益排序
  （纯 20d IC 最高而组合收益最差）⇒ 不按单周期 IC 重排权重。
- 新增 `src/eval/regime_conditioned_signal.py` + `python main.py regime-signal`；
  新增 `regime_stability_check`（池层面抽样，不重采样样本）。
- 顺带修一个复现性缺陷：同池打乱输入标的顺序会改变读数（date 同日 tie-break），
  改为 `sort_values(["date","_symbol"], kind="mergesort")`。
- 测试 `tests/test_regime_conditioned_signal.py` 27 条；只读 `affects_gate=false`。
- 详情见 `cairn/regime-conditioned-signal.md`。

## 2026-09-16 · 模型优化排查（Issue #55 步骤①②③）

- 按既定点定的优先级推进：**① 池共线性 → ② 标签口径 → ③ 周期选择**，全部落地为可复算命令
  `python main.py pool-collinearity` / `python main.py model-improve`（`affects_gate=false`）。
- **① 诊断成立但假设被证伪**：全池 38 只 → 有效独立维度 **7.91**（0.208），ETF 池 14→2.80（PC1 57%）；
  但**缩池无效** —— 去冗余后 IC 反降（0.039 → 0.012），ETF-only 反而最好（IC 0.054）。
  ⇒ 共线性不是区分度不足的原因；全池 0.039 部分来自共享市场 beta。
- **② 三重障碍法证伪**：同折同模型同样本，三周期全部 `improved=false`；
  5d 反而更差（IC 0.037 → 0.014）。判定纪律与 `decision_analytics.verdict` 同源
  （命中率与平均收益**两条腿都过**才算改善）。⇒ 不纳入主线。
- **③ 现行权重与证据方向相反**：5d 是唯一强 IC 且校正后显著的周期（IC 0.039, p<0.001），
  而现行 `0.30/0.35/0.35` 把最高权重给了**唯一不显著**的 10d；证据权重为
  `5d 0.576 / 10d 0.176 / 20d 0.248`。不自动重排（须人工签字），且全周期平均收益仍为负。
- **修掉一个真缺陷（比结论更重要）**：`get_feature_columns` 排除规则只认 `target_`，
  三重障碍法标签 `label_tb_*` 会被**当特征返回** → AUC=1.0 / IC=0.44 的假读数，
  **不报错不告警**。改为**前缀族排除** + fail-loud 自检；守卫 `tests/test_label_leak_guard.py`。
- 数据：`data/raw/` 38 标的真实日K（腾讯 26 + 新浪期货 10 + 新浪外汇 2，2020-01~2026-09）。
- 测试 `pytest tests` → **1311 passed**（新增 45 条守卫）；6 项失败**先于本次改动即失败**
  （optuna/hmmlearn 未装 + `reports/` 运行产物未落盘）。
- 详情见 `cairn/model-optimization-findings.md`。

## 2026-09-16 · 决策源契约层 + 信号有效性边界（Issue #55）

- 新增**决策源契约** `decision-feed/1`：`GET /api/v1/decision/feed` 与
  `python main.py decision-feed`（服务态 / 离线管道态逐字段一致）；原
  `/api/v1/portfolio/summary` 自动追加决策字段（向后兼容，旧消费方零改动）。
- 核心是 `net_up_probability`：每周期净看涨概率，下游**不必再判断 direction 字符串**
  ——根治与 tradingview `_aggregate` 之间的口径分裂（那边 0.2/0.5/0.3 + 0.6/0.4 阈值是自算的）。
- 结构性纪律：`position_role=observer` / `affects_gate=false` / `advisory_only=true`；
  不产出仓位，不改门禁；缺失周期不补 0.5，非有限值一律 None（不拿 0.5 冒充中性读数）。
- 新增 `src/eval/decision_analytics.py`：置信度分档 × **已实现收益**联合分布 + 门槛扫描 +
  保守判定（命中率与收益**两条腿都过**才算 effective）。
- **实测结论（如实入库）**：真实日K + 本地真实训练 LightGBM，15584 锚点 →
  置信度↑ ⇒ 命中率↑（54% → 98%）但**平均已实现收益↓（+1.92% → −0.78%）**，三周期均 `ineffective`。
  ⇒ 高置信 ≠ 可采信；「命中率高」不得单独作为采信依据。
- 顺带修两个真缺陷：推理期特征**按名对齐**（位置截断在列序不同时静默错位）、
  审计共享记账接口 `record_prediction_v2`（fail-loud + 来源入库）。
- 测试 1277 passed（新增 37 条守卫）；详情见 `cairn/decision-source-contract.md`。

## 2026-09-16 · TradingView 交付轮（图片信号卡 + Pine 数据层）

- 交付 `python main.py tv-export`：一次产出**图片信号卡**（真 PNG，tEXt 内嵌
  机器可读契约 + 全精度锚点）与 **Pine 外挂数据层**（`tv-pine/1`，`request.seed` 可读）。
  两者都是 `decision-feed/1` 的只读投影，不做二次换算。
- 新增 `src/export/tv/`（零依赖 PNG 编码器 / 信号卡 / Pine 契约 / 编排）、
  `src/eval/anchor_backfill.py`（无前视锚点回填）、`tests/test_tv_export.py`（17 条守卫）。
- 真实数据读数（38 标的池，2023-01~2026-09）：3420 锚点，命中 52.9%，
  平均已实现收益 +0.32%，`advisory_consumable` 0/38 —— 信号仍属弱区分度，
  定位为**只读观测 / 风险预警**，不按"高置信"放大仓位。
- **修掉一个静默真缺陷**：按名对齐的调用方把值矩阵当列名传入 → 全列判缺失 →
  整体 0 填充 → 产出与标的/日期无关的**常数概率**（全池锚点置信度恒 0.0303756）。
  现改为全列缺失时放弃按名对齐回落位置对齐；训练侧把 `feature_cols` 写进产物
  并同步为模型原生特征名（只读 property + booster 缓存两条实测约束）。
- 详情：`cairn/tradingview-handoff.md`；契约层见 `cairn/decision-source-contract.md`。

## 2026-09-13 · Project Cairn 初始化

- 初始化 Project Cairn 结构（AGENTS.md / CLAUDE.md / .cairn/config.yaml / cairn/LOG.md / cairn/ROADMAP.md）。
- 历史迁移模式：`start_fresh`（既有 00_kickoff 结论文档、SALES_PLAN、schedule 排期保持原样，不迁移）。
- git 策略：`track`（cairn/ 与 AGENTS.md 入库）；毕业 provider：暂缓对接。
- 详情：见 `AGENTS.md` 与 `.cairn/config.yaml`。
