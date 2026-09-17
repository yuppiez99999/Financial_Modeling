# Project Cairn 项目日志（LOG）

本文件按倒序记录项目的实质性进展——最新条目紧贴本行下方。每条保持简短——只写摘要与指针；结论沉淀到 `cairn/<topic>.md` 知识专题文档。

## 2026-09-17 · 全池 38 标的状态分层 + 两条静默缺陷（Issue #55 第八轮）

- 第七轮留的"全池 38 标的分状态读数待跑"本轮跑完，并在过程中揪出**两条早就存在**
  的静默缺陷 —— 它们让第七轮的读数**其实从未真正跑在 HMM 口径上**。
- **缺陷①（本轮真事故）**：`regime_labels` 从第 1 个有效样本起就 fit，而 `fit_hmm`
  要求 ≥ `MIN_FIT_SAMPLES`(60) ⇒ 起步必然连续 `insufficient_fit_samples` 被静默吞掉。
  加 stalled 护栏后，**60 次起步失败正好触发护栏**，一个**完全可 fit** 的全池序列
  被误标 `stalled=True / refits=0`，静默降级到规则口径 —— 与文档写的 `hmm_expanding`
  不一致且无告警。修：跳过必然不足的前缀 + 护栏阈值 ≥ `MIN_FIT_SAMPLES`
  + `stalled/refits/fit_failures` 如实入 meta。修后 `refits=404`、`stalled=False`。
- **缺陷②**：`build_report` 调 `_date_regime_labels` **无异常保护**，hmmlearn 缺失抛
  `ModuleNotFoundError`（非 ValueError）⇒ 整个状态分层被判"计算失败"，
  **连规则口径降级都没走到**，而主读数明明可用。修：单独包一层，异常降级规则口径
  并写进 `hmm_meta.reason`、`mode=rules_fallback`（不冒充、不静默）。
- **主读数（全池 38 标的，真实日K 2020-01~2026-09，walk-forward，base 成本档）**：
  全局净超额 **−0.224%（t −3.62）** `no_edge`；基准年化 +10.5%，信号 +6.1%。
  三分 **range −0.250%（t −3.23）/ bear −0.134%（t −1.95）/ bull 期数 0 不外推**
  ⇒ 样本量上来后"相对等权显著为负"从待观察变**统计确认**，三态无一为正。
- **新增趋势/盘整二分**（`regime_breakdown.binary`，bull∪bear→trending、range→choppy，
  同一批逐期净超额，**展示归并不换口径**）：trending −0.134%（t −1.95）/
  choppy −0.250%（t −3.23），`conditional_edge_hint=False`
  ⇒ "信号只在趋势里有用"同样**不被支持**（三分因 bull 被吸收回答不了这一问）。
- 新增 `tests/test_regime_fit_guard.py` 11 条守卫；`pytest tests` → **1445 passed**
  （4 skipped 为环境可选依赖）。只读 `affects_gate=false`。
- 详情见 `cairn/benchmark-relative-edge.md` 第十节。

## 2026-09-17 · 状态分层的净超额：增量是不是只在某个市场状态下（Issue #55 第七轮）

- 前六轮排查全是**全样本平均**读数：信号"整体"相对等权显著为负。本轮补上
  **从未独立检验**的残留假设 —— 信号大部分时间没用、**只在某个市场状态才有增量**，
  被平均掉了。判据与 `edge-check` 全局口径**逐式同源**（净超额 = (sig−ben)−2×成本）。
- 新增 `edge-check` 的 `regime_breakdown` 段（默认开启，`--no-regime-breakdown` 关闭）：
  状态由**全池等权市场层**拟合，首选 HMM `expanding`（无前视），hmmlearn 缺失时
  **如实降级**规则口径并标 `mode=rules_fallback`（不冒充）；调仓日取**当日**标签，
  截断回归守卫证明无前视；单状态期数 < 8 → 该状态 `available=False` 不外推。
- **主读数（8 标的真实日K，walk-forward）**：全局净超额 −0.214%（`no_edge`）；
  分状态 **range −0.299%（t −1.91）/ bear −0.013%（t −0.07）** —— **没有一个状态为正**；
  bull 当期无有效样本，如实标不可用。`conditional_edge_hint=False`，
  `conclusion=no_conditional_edge`。⇒ "增量只藏在某状态"**不被支持**。
- 边界如实：8 标的冒烟、单窗、分状态每格期数偏少（28~66），`spread` 0.286% 极差
  只记为待观察量，**不构成口径变更依据**；全池 38 标的分状态读数待跑。
- 只读 `affects_gate=false`；新增 `tests/test_benchmark_relative.py` 9 条守卫；
  `pytest tests` → **1430 passed**（4 项失败先于本次改动即失败：optuna 未装）。
- 详情见 `cairn/benchmark-relative-edge.md` 第八节。

## 2026-09-16 · 特征集 × 模型族联合消融（Issue #55 第五轮，最后一条未量化嫌疑）

- 前四轮排查完池/标签/周期/置信度语义后，本轮把**特征集**与**模型族**两条
  从未量化的嫌疑一起消融，并**换掉记分板**：唯一判据 = `benchmark_relative` 的
  「相对全池等权的净超额」（扣 2× 成本、非重叠调仓、优于随机子集）；
  IC / 命中率 / AUC **不参与判定**。
- **主读数（26 标的真实日K，walk-forward 样本外）**：现行口径（LightGBM + 107 列）
  在三周期上**全部相对等权显著为负**：5d −0.292%/期（t −2.12）、
  10d −0.512%（t −2.79）、20d −0.397%（t −2.27）⇒ 独立判据**第二次**确认
  这套信号是"负增量"，TrendCast 只能作只读观测 / 风险预警。
- **特征消融：八个子集无一把净超额推正**（最好的 `returns_only`/`no_factor`
  也只是"没那么负"，相对基线臂配对 t ≤ 1.69）⇒ 107 列冗余**不是**当前瓶颈，
  **不做**加/减特征。
- **模型族消融：唯一正向读数在 10d** —— `random_forest`（配对 t 2.58,
  Holm p 0.039）、`logistic_ridge`（t 3.92, Holm p 0.0004）相对基线族
  **统计上确认"更不差"**；但**净超额仍为负**、且 5d/20d 不成立。
  ⇒ 够格作**下一步实验方向**（降复杂度 + 限定 10d），**不够格**改生产配置。
  线性族也能做到 ⇒ 瓶颈不是"非线性不够"，而是 10d 上树模型切分空间过多。
- 新增 `src/eval/feature_model_ablation.py` + `python main.py ablation`；
  判定引入 **Holm-Bonferroni 多重比较校正**，并把 `inconclusive` 与
  `worse_than_baseline` 区分开（不把"没证明更好"写成"更差"）。
- **修一个静默缺陷**：项目自研 `LightGBMModel` 入口是 `train()` 不是 `fit()`，
  适配层按 sklearn 惯例调 `fit` → 所有 lightgbm 臂被静默标成"样本不足"，
  报告看起来"跑完了但没数据"。已修并加守卫。
- 另修：特征分组用 `in` 会让 `ma_` 吞掉 `ema_/dema_/wma_/trima_`（整族错位）；
  分组漏列不披露（漏测）；每臂各自切折（臂间不同批样本）；消融臂写线上模型目录。
- 测试 `tests/test_feature_model_ablation.py` 32 条；`pytest tests` → **1392 passed**，
  6 项失败**先于本次改动即失败**（optuna/hmmlearn 未装 + `reports/` 运行产物未落盘），
  已 stash 逐条对照确认。
- 详情见 `cairn/feature-model-ablation.md`。

## 2026-09-16 · 基准相对决策增量：信号没有跑赢"什么都不做"（Issue #55）

- 新增 `benchmark_relative`（`python main.py edge-check` → `reports/benchmark_relative.json`）：
  **判据换成「相对全池等权的净超额」**（扣 `2×` 单边成本、非重叠调仓、随机子集对照）。
  前三轮判据（IC / 命中率 / 绝对收益）**都没有对照**，这是它们共同的盲点。
- **主读数（26 标的，真实日K + 真实训练 LightGBM，walk-forward）**：
  全池等权 buy&hold 年化 **+19.1%**；信号年化 +23.9%(5d) / +24.0%(10d) / +24.8%(20d)，
  但净超额 **−0.070%(5d) / +0.029%(10d) / +0.225%(20d)**，t = −0.84 / 0.18 / 0.66。
  ⇒ 5d 判 `no_edge`（不如等权）；10d/20d 判 `inconclusive`（远不显著）。
- **随机子集对照**：随机抽同样数量（16 只）标的，5d 有 **7.5%** 的随机次不亚于信号
  ⇒ "选中这些标的"没有信息含量，信号只是**满仓参与**了市场 beta。
- **据此否掉遗留候选**：① 周期权重重排 —— 五个候选（现行/等权/纯 5d/10d/20d）
  **无一带净超额**，差异全在噪声内（|t|≤1.18），**不做**；
  ② 「置信度×波动联读纳入采纳口径」—— 且在一个无 edge 的信号上调口径不改结论，**暂不**。
- 建议下一步：**换问题而非换参数** —— 做特征集 × 模型族联合消融，
  并以**本判据**（相对等权的净超额）作唯一记分板，不再看 IC / 命中率。
- 测试 1325 passed（新增 22 条守卫）；15 项失败**先于本次改动即失败**
  （fastapi/optuna/hmmlearn 未装 + `reports/` 运行产物未落盘），已 stash 对照确认。
- 详情见 `cairn/benchmark-relative-edge.md`。

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
