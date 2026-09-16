# Project Cairn 项目日志（LOG）

本文件按倒序记录项目的实质性进展——最新条目紧贴本行下方。每条保持简短——只写摘要与指针；结论沉淀到 `cairn/<topic>.md` 知识专题文档。

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
