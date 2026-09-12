# 高质量项目集成 3 · 候选评估矩阵（Issue #54 · I1~I5）

> 检索范围：GitHub API / 网页核验（关键词：portfolio backtest、portfolio optimization、
> deflated sharpe ratio、data drift monitoring、machine learning attribution 等）。
> 本文件是**选型依据**，不是成果汇报：结论以「导入本项目的增量」为准，
> 参考 star 数只作成熟度信号，不作价值证据。
> 前置事实：G 轮（Issue #29 · S11~S15）与 H 轮（Issue #40 · S16~S20）已收官；
> S11 否掉周期切换、S12 否掉特征扩充、S13 判定瓶颈不在特征侧、S15 判定
> 「调参不是瓶颈、置信度门槛是唯一有效机制」、H5 判定 LLM 投研 0/3 候选满足准入。

## 一、本轮主题：组合闭环验证（为什么是它）

G 轮与 H 轮把「单标的读数的可信度」做透了：CPCV 收缩后 IC 仍为正（S17）、
保形区间覆盖率达标的（S16）、概率校准 ECE 0.0904（S19）、状态分层 differentiated
（S18）。但至今**没有任何组合层的收益证据**：全部评估停留在 IC / 命中率 /
覆盖率，从未回答「这套信号作为一个 A 股组合，扣完成本到底赚不赚钱」。
`backtest/run_minimal.py` 只是单标的收益路径演示；Q4 的 ATR 建议也只到
「建议」层。I 轮把已建立的可信评估**延伸到组合层**：

1. 信号 × 置信度门槛 × 成本三档（T11.2 已定稿）→ 组合净值/回撤/换手/夏普；
2. 回答 S15 遗留问题的最后一环：IC 与命中率脱节，**组合 PnL 是否也脱节**；
3. 组合参数（加权方式/门槛/成本档）本身构成新的选择自由度 → 必须接上 S17
   的过拟合审计口径（PSR/DSR/PBO）。

## 二、直接淘汰（不进入本项目）

| 候选 | star | 许可证 | 淘汰原因 |
|------|------|--------|----------|
| `skfolio/skfolio` | 2.4k | BSD-3 | 要求 **Python 3.10+** 且强依赖 **cvxpy**；本项目运行时 Python 3.8，违背零重型依赖纪律 |
| `robertmartin8/PyPortfolioOpt`（运行时） | 6k | MIT | 优化器需 cvxpy；等权/波动率倒数/置信度加权三臂对照**无需求解器**。仅保留为 I2「可选评估臂」做准入评估，准入不过则取消 |
| `esvhd/pypbo` | 140 | **AGPL-3.0** | 传染性 copyleft 许可证 + 停更迹象（statsmodels 钉 0.8.0）；PBO/PSR/DSR 公式来自 Bailey & López de Prado 论文，S17 已有自研统计底座，**仅作方法论引用不引入代码** |
| `pmorissette/bt` / `mementum/backtrader` / `ricequant/rqalpha` | 1.5k~9k | 各异 | S1 已定 vectorbt/自研回测口径，换回测框架属重复建设且撼动评价量尺（H 轮已否，维持） |
| `quantstats` / `empyrical` / `alphalens-reloaded` | 3k~7.6k | 各异 | 与 `eval_criteria.md` 双维量尺口径分叉（H 轮已否，维持） |
| `localhost:easytrader` 类实盘下单框架 | 2k+ | 各异 | 合规红线：本项目输出止于 `suggested_*` 建议，绝不接执行通道（延续 Q4 边界） |
| `MLflow` / `Weights & Biases` | 20k+ | Apache-2.0 | S17 自研 append-only 试验登记已闭环且零依赖；引入重型追踪平台属重复建设 |
| `great-expectations` / `pandera` | 3k~10k | Apache-2.0 | `src/data/quality_gate.py` 已覆盖数据质量门；引入全量数据契约框架边际低 |
| `microsoft/qlib`（运行时） | 48.5k | MIT | S13 已决：表达式级对齐复现，不引入运行时（维持） |
| `riskfolio-lib` | 1.5k+ | **GPL-3.0** | 传染性许可证 + cvxpy；同 skfolio 淘汰 |
| 新增 A 股数据源（tushare/baostock/adata 等） | — | 各异 | S14 已把 akshare 升 P1 并闭环，H 轮已判「再加同类源边际为零」（维持） |

## 三、入选（生成排期 I1~I5）

| 阶段 | 候选项目 / 方法 | star / 许可证 | 解决本项目什么问题 | 现有卡点证据 |
|:---:|------|------|------|------|
| **I1** (S21) | 自研组合回测闭环（零新依赖，vectorbt 仅作可选对照臂；`polakowo/vectorbt` 4.8k / BSD-3） | — | 把「信号→仓位→组合净值」从单标的演示升级为**池级组合回测**：38 标的 × 三周期概率 × 置信度门槛 × 成本三档（T11.2）→ 净值/回撤/换手/夏普；与命中率口径做一致性对照 | S15：「IC 与命中率脱节在收敛搜索后依旧存在」——该结论从未在组合 PnL 层复验；`backtest/run_minimal.py` 无组合层 |
| **I2** (S22) | 组合构建三臂对照（等权 / ATR 波动率倒数 / 校准置信度加权，零新依赖）＋ `PyPortfolioOpt` 可选优化臂 | 6k / MIT | 回答「权重怎么分」：三臂同口径 A/B，延续 S17 多重比较校正；优化器臂准入评估（cvxpy 依赖代价）不过则取消 | H 轮否掉纯特征扩充后，组合层是唯一未检验的自由度；Q4 已产出 ATR（波动结构）与校准置信分（S19）两个现成加权依据 |
| **I3** (S23) | `evidentlyai/evidently`（准入评估：依赖重量、py3.8 兼容；不过则按方法论自研 PSI/KS） | 7.9k / Apache-2.0 | 数据/预测分布漂移监控进报表层：特征分布漂移 × HMM 状态 × 命中率漂移交叉，回答「读数失效前有没有可观测前兆」 | H3 状态分层 differentiated 说明读数依赖市场状态；现有 `health_report` 只有命中率下降告警，无特征分布视角 |
| **I4** (S24) | LightGBM **TreeSHAP** 归因（`pred_contrib=True`，零新依赖；方法论 ref：`shap` 23.9k / MIT） | — | 状态×特征归因：解释 S18「状态分层 differentiated / 状态特征 A/B degraded」的特征成因；归因是**解释**不是 S12 被否的「扩充」 | S12 判定特征扩充无增量，但「模型到底在用什么」从未归因；lightgbm 已是登记过的可选依赖 |
| **I5** (S25) | 组合级过拟合审计：PSR / DSR / PBO 方法论（Bailey & López de Prado），自研实现，`pypbo` 仅公式参考 | — | 把 S17 的过拟合口径从 IC 层升到**组合净值层**：组合参数扫描（加权臂×门槛×成本档）计入统一试验预算，产出 DSR/MinTRL 类指标 | S17 已建统一试验预算与 CPCV；组合回测（I1/I2）引入新选择自由度，不审计则重蹈「事后补记」覆辙 |

## 四、选型原则（为什么是这五个）

1. **组合层是唯一未验证的层级**：标的维度（S9 分池）、时间维度（S18 状态）、
   评估可信度（S16/S17/S19）都已覆盖，组合 PnL 层是 G/H 轮结论的自然下一环；
2. **不再加模型族、不再单纯加特征**（延续 H 轮原则）：I4 是归因解释，不是扩充；
3. **零重型依赖、CI 离线可跑**：I1/I2/I4/I5 全部零新依赖（py3.8 可跑）；
   I3 为唯一外部库候选，须过准入评估，不过则方法论自研；
4. **一律只读、绝不进决策路径**：组合回测是**评估工具**不是策略生成器；
   门禁 `strategy_gate` 本轮内零改动，所有 `affects_gate` 恒为 false；
5. **允许整阶段取消**：I3 若准入评估不过且自研成本超预算则取消（延续 H5 先例）。

## 五、许可证与登记

- 本轮**规划期不新增任何运行时依赖**：I1/I2/I4/I5 零新增；I3（evidently，
  Apache-2.0）与 I2 可选臂（PyPortfolioOpt，MIT）仅在准入评估通过后才按
  `docs/THIRD_PARTY.md`「append 风格」登记并解注释 requirements.txt；
- 未实际 `pip install` 前不得声称已引入运行时（延续 H 轮纪律）；
- pypbo（AGPL-3.0）**只读论文公式不读代码**，避免传染性许可证污染。

## 六、对齐既有纪律

- 结论不管好坏如实入库（S12/S13/S15/S17 既有做法延续）；
- 每阶段须有 `tests/test_roadmap_s21.py` ~ `test_roadmap_s25.py` 对应守卫测试
  （离线合成数据、不触网）；
- 人工检查点（`manual_checkpoint`）不可自动通过：T21.4 / T22.4 / T23.4 / T24.4 / T25.4；
- 排期不是必须做满的清单：评估不过 → 如实记录 → 取消或止损（H5 先例）。
