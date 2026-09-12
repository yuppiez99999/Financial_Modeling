# 第三方组件登记（THIRD_PARTY）

> 依据 README §18A「统一原则」：MIT/Apache-2.0 代码集成进本仓库（「禁止商业用途」
> 自有许可）不冲突，但**必须登记来源与版本**。本文件是唯一登记入口，
> 引入新第三方组件时同步追加（append 风格，不删除历史条目）。

| 组件 | 来源 | 许可证 | 引入阶段 | 用途 | 使用方式 |
|------|------|--------|---------|------|---------|
| microsoft/qlib（Alpha158） | https://github.com/microsoft/qlib | MIT | S13 / G3（2026-09-11） | Alpha158 因子集（98 列） | **表达式级对齐复现**：`integrations/qlib/alpha158.py` 用纯 pandas 复现 qlib `Alpha158DataLoader` 的因子表达式，**未引入 qlib 运行时依赖、未复制 qlib 源码** |
| akfamily/akshare | https://github.com/akfamily/akshare | MIT | **S14 / G4（2026-09-11）由「可选依赖」升为 P1 数据源** | A股 / ETF / 国内期货 / 外汇 日K；宏观指标 | pip 依赖（`akshare>=1.12.0`，已在 requirements.txt）；**未安装时链路静默跳过** |
| 新浪财经公开接口（InnerFuturesNewService / NewForexService / stock_zh_a_daily 通道） | https://finance.sina.com.cn | 公开免费接口（无独立许可证，仅取数不复制代码） | S14 / G4（2026-09-11） | 国内期货日K、外汇日K | **HTTP 直连**：`src/data/akshare_client.py` 中 `_sina_futures_fetch` / `_sina_fx_fetch`。原因：akshare 的 `futures_zh_daily_sina` 内部请求**不带 UA/Referer**，在新浪限流下返回 HTTP 456 导致批量拉取 9/10 失败，故用复用 Session + 浏览器头直连**同一公开接口**（字段口径与 akshare 一致，见 `00_kickoff/akshare_data_source_conclusion.md` 第四节） |
| vectorbt | https://github.com/polakowo/vectorbt | 自定义（允许商用） | 既有（S1 回测层） | 向量化回测 | pip 依赖 |
| Kronos（快照） | https://github.com/shiyu-coder/Kronos | MIT | 既有（对照参考） | K 线基础模型 | 仓库内源码快照（`Kronos/THIRD_PARTY_NOTICE.md`） |
| LightGBM | https://github.com/microsoft/LightGBM | MIT | 既有 | 主力模型 | pip 依赖 |

## Alpha158 对齐表（qlib → 本项目）

qlib 官方 Alpha158（`qlib/contrib/data/loader.py`）共 158 个表达式 =
98 个特征列（部分表达式共享输出列）。本项目 `integrations/qlib/alpha158.py`
按**窗口 (5, 10, 20, 30, 60)** 复现其中 98 列：

- K 线形态项（8 个，当日，无窗口）：`KMID / KLEN / KUP / KLOW / KSFT / KMID2 / KUP2 / KLOW2`
- 滚动项 × 5 窗口（18 × 5 = 90 个）：
  `ROC / MA / STD / BETA / RSQR / RESI / QTLU / QTLD / RANK / RSV / CORR / CORD / CNTP / SUMP / VSUMP / VMA / VSTD / WVMA`

算子语义对齐说明（与 qlib 表达式引擎的差异点，如实登记）：

| qlib 算子 | 本项目实现 | 差异 |
|-----------|-----------|------|
| `Ref($close, i)` | `Series.shift(i)` | 无 |
| `Mean/Std/Max/Min/Quantile/Rank` | `rolling(w)` 同名 | 无 |
| `Slope / Rsquare / Resi` | 滚动 OLS（`rolling.apply`） | 无（数值精度 1e-9 级） |
| `Corr($close,$volume,w)` | `rolling(w).corr` | 无 |
| `Corr(Delta($close,i),$volume,w)` | `diff(w).rolling(w).corr(v)` | qlib 的 Delta 窗口语义按逐因子对齐 |
| `Count($close>Ref($close,i),w)` | `(c>c.shift(w)).rolling(w).mean()` | 无 |
| `Greater / Lesser` | `pd.concat.max/min(axis=1)` | 无 |
| `WVMA` | `Std(Abs(ret1)*volume, w)` | 无 |

**明确不复现的部分**（不影响验证口径，如实声明）：
- qlib Alpha360（1 年价格切片）未实现 —— 本轮验证只覆盖 Alpha158；
- qlib 表达式引擎本身（`DataHandlerLP` 惰性计算）未引入 —— 本项目按项目纪律
  （零重型依赖、CI 离线可跑）用 pandas 即时计算。

## 许可证合规结论

- qlib（MIT）：MIT 允许商用与再分发，本项目「禁止商业用途」是对**外授权的收紧**，
  内部使用与集成无冲突；
- 复现方式为**算法/表达式对齐**（因子定义属公开方法论），未复制 qlib 源码文本；
  若后续直接引入 qlib 包（`pip install pyqlib`），需在本表追加运行时版本号。

---

## optuna（S15 / G5 · T15.1 超参搜索）

| 项 | 内容 |
|----|------|
| 来源 | optuna/optuna（https://github.com/optuna/optuna） |
| 许可证 | MIT |
| 引入方式 | **可选运行时依赖**（`pip install optuna`）；未安装时 `tune` 命令明确报错指引，不降级、不静默 |
| 用途 | `src/eval/hyperopt_tuner.py`：LightGBM 超参搜索（walk-forward 测试折 IC 均值为目标），study 以 SQLite 持久化到 `reports/optuna_studies/` |
| 边界 | 仅离线研究用途；搜索结果属选择自由度，不直接落地配置（T15.3 人工检查点） |

## neuralforecast（S15 / G5 · T15.2 概率区间，间接引入）

| 项 | 内容 |
|----|------|
| 来源 | Nixtla neuralforecast（https://github.com/Nixtla/neuralforecast） |
| 许可证 | Apache-2.0 |
| 引入方式 | **本轮未安装运行时依赖**：`src/eval/confidence_curve.py` 的 `confidence_from_interval` 把「概率区间宽度 → 置信度」的换算机制先行抽象落地，与 LightGBM 概率口径（`|p−0.5|×2`）共用同一条阈值曲线链路；后续接入 neuralforecast 时只需喂区间数据，无需改链路 |
| 边界 | 若后续 `pip install neuralforecast`，需在本表追加运行时版本号 |

## 许可证合规结论（S15）

- optuna（MIT）与 neuralforecast（Apache-2.0）均允许内部使用与集成，
  与本项目「禁止商业用途」的外授权收紧不冲突；
- 本轮均为**可选依赖 + 缺省不安装**，CI 离线可跑（合成数据测试不触网）。

---

## 高质量项目集成 2 轮（H1~H5，Issue #40）

> 状态说明：以下条目为**规划期登记**（`schedule/plan.json` S16~S20）。
> 只有在本文件对应条目追加「运行时版本号」后，才代表已实际引入依赖。
> 选型依据见 `00_kickoff/high_value_projects_round2_candidates.md`。

| 组件 | 来源 | 许可证 | 引入阶段 | 用途 | 引入方式（规划） |
|------|------|--------|---------|------|-----------------|
| MAPIE | https://github.com/scikit-learn-contrib/MAPIE | BSD-3-Clause | S16 / H1（计划） | 保形预测 · 带覆盖率保证的预测区间 | 可选依赖 `pip install mapie`；未安装时区间口径明确报错，不降级、不静默 |
| hmmlearn | https://github.com/hmmlearn/hmmlearn | BSD-3-Clause | S18 / H3（计划） | 市场状态识别（牛/熊/震荡） | 可选依赖 `pip install hmmlearn`；状态标签严格无前视 |
| scikit-learn（校准器） | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | S19 / H4（计划） | `CalibratedClassifierCV`（isotonic / Platt）概率校准 | 既有依赖，无需新增；校准器仅在独立保留期验证，不自动落地 |
| TradingAgents 系（评估中） | https://github.com/TauricResearch/TradingAgents · https://github.com/hsliuping/TradingAgents-CN · https://github.com/qusong0627/QuantMind | Apache-2.0 / 部分仓库 NOASSERTION | S20 / H5（待 T20.1 评估） | LLM 投研辅助结论（**只读报告附注**） | **未确定引入**：T20.1 评估不通过则整阶段取消。若引入，严禁进入信号路径与门禁 |

### 规划期合规说明

- H1~H4 全部为 sklearn 风格轻量库或既有依赖，符合「零重型依赖、CI 离线可跑」纪律；
- H5 涉及的 LLM Agent 框架依赖较重且输出不可量化，**登记为待评估而非待引入**，
  评估结论（含许可证逐项核查）落 `00_kickoff/` 后另行追加运行时版本号；
- 配套候选评估与淘汰理由见 `00_kickoff/high_value_projects_round2_candidates.md`。
