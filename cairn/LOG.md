# Project Cairn 项目日志（LOG）

本文件按倒序记录项目的实质性进展——最新条目紧贴本行下方。每条保持简短——只写摘要与指针；结论沉淀到 `cairn/<topic>.md` 知识专题文档。

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

## 2026-09-13 · Project Cairn 初始化

- 初始化 Project Cairn 结构（AGENTS.md / CLAUDE.md / .cairn/config.yaml / cairn/LOG.md / cairn/ROADMAP.md）。
- 历史迁移模式：`start_fresh`（既有 00_kickoff 结论文档、SALES_PLAN、schedule 排期保持原样，不迁移）。
- git 策略：`track`（cairn/ 与 AGENTS.md 入库）；毕业 provider：暂缓对接。
- 详情：见 `AGENTS.md` 与 `.cairn/config.yaml`。
