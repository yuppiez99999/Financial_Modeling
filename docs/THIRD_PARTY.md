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
| MAPIE | https://github.com/scikit-learn-contrib/MAPIE | BSD-3-Clause | **S16 / H1（已实装）** | 保形预测 · 带覆盖率保证的预测区间 | 可选运行时依赖 `pip install mapie`（**已实装，MAPIE 1.5.0**）；未安装时 `conformal-interval` 明确报错，不降级、不静默 |
| hmmlearn | https://github.com/hmmlearn/hmmlearn | BSD-3-Clause | **S18 / H3（已实装）** | 市场状态识别（牛/熊/震荡） | 可选运行时依赖 `pip install hmmlearn`（**已实装，hmmlearn 0.3.3**）；`regime` 命令未安装时明确报错，不静默降级为规则口径 |
| scikit-learn（校准器） | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | S19 / H4（计划） | `CalibratedClassifierCV`（isotonic / Platt）概率校准 | 既有依赖，无需新增；校准器仅在独立保留期验证，不自动落地 |
| TradingAgents 系（评估中） | https://github.com/TauricResearch/TradingAgents · https://github.com/hsliuping/TradingAgents-CN · https://github.com/qusong0627/QuantMind | Apache-2.0 / 部分仓库 NOASSERTION | S20 / H5（待 T20.1 评估） | LLM 投研辅助结论（**只读报告附注**） | **未确定引入**：T20.1 评估不通过则整阶段取消。若引入，严禁进入信号路径与门禁 |

### 落地状态更新（2026-09-12，S16~S20 自动任务交付后）

> 以下为**实际落地事实**（不是计划）：只登记"真的用了什么"。

| 组件 | 最终状态 | 说明 |
|------|---------|------|
| MAPIE | **可选依赖，两条口径并存** | 主线为 `conformal-interval`（MAPIE `SplitConformalClassifier`，LAC）；另保留 `conformal` 覆盖率校准命令按 Vovk/分割保形口径**自实现**（`src/eval/conformal.py`），报告 `backend=native` 如实标注，不冒充 MAPIE 结果 |
| hmmlearn | **可选依赖（已实装 0.3.3）** | `src/eval/regime.py` 用 `GaussianHMM`（BSD-3-Clause）拟合市场状态；`regime` 命令缺省**明确报错不静默降级**，同时保留 `detect_regimes(..., backend="rules")` 旧口径供显式调用（如实标注 `backend=rules`）。已做状态占比 / 协方差退化检查，避免吸收成离群态 |
| scikit-learn（isotonic / Platt） | **既有依赖，已实际使用** | S19 校准层用 `IsotonicRegression` 与 `LogisticRegression`（logit 变换）；校准参数固化到 `models/probability_calibration_<horizon>.json`，推理期只读 |
| TradingAgents 系 | **不引入（整链路默认 cancel）** | S20 评估结论：0 个候选同时满足「A 股适配 + 离线可复现 + 许可证明确」。离线对照结构性不可量化 → 记 `unverifiable` 并止损。见 `00_kickoff/research_assist_conclusion.md` |

### 合规说明

- S16~S20 全部实现均为 **sklearn 风格轻量库 / 自实现 / 既有依赖**，无新增重型依赖，
  符合「零重型依赖、CI 离线可跑」纪律（全部测试用合成数据，不触网）；
- CPCV / DSR 属于**方法论引入**（López de Prado《Advances in Financial ML》口径），
  未复制源码；收缩指标自实现且如实标注为**一阶近似**（`method=cpcv_shrinkage`），
  不冒充 Sharpe DSR 精确解；
- 落地结论与实测读数见 `00_kickoff/round2_integration_conclusion.md`；

### MAPIE（S16 / H1 · T16.1 + T16.2 实装）

| 项 | 内容 |
|----|------|
| 来源 | scikit-learn-contrib/MAPIE（https://github.com/scikit-learn-contrib/MAPIE） |
| 许可证 | BSD-3-Clause |
| 运行时版本 | **mapie 1.5.0**（本轮实际 `pip install`，装于 CI 镜像） |
| 引入方式 | **可选运行时依赖**：`src/eval/conformal_probability.py` 用 `SplitConformalClassifier`（`prefit=True` + 保形集 `conformalize`）实现 split conformal；未安装时 `python main.py conformal-interval` 明确报错指引，不降级、不静默 |
| 用途 | ① T16.1：现行 LightGBM 上产出带覆盖率保证的预测集合 → [0,1] 区间 → 置信分（`confidence_from_interval`）；覆盖率审计 / 可靠性曲线（Brier/ECE）/ 区间宽度校准；② T16.2：区间置信分与现行 `\|p−0.5\|×2` 同数据同折对照 |
| 保形分数选择 | **`lac`（唯一）**：单周期二分类（两个标签）下 MAPIE 明确拒绝 `aps/raps`（`ValueError: Invalid conformity score for binary target`），LAC 在给定覆盖率下集合期望规模最小 —— 既是唯一可行也是最优选择 |
| 边界 | 仅离线研究用途；`model.factors.conformal.enabled` 缺省 **false**，区间口径**不进**特征/信号链路；`affects_gate` 恒 false；口径取舍属 T16.4 人工检查点 |

### hmmlearn（S18 / H3 · T18.1 + T18.2 + T18.3 实装）

| 项 | 内容 |
|----|------|
| 来源 | hmmlearn/hmmlearn（https://github.com/hmmlearn/hmmlearn） |
| 许可证 | BSD-3-Clause |
| 运行时版本 | **hmmlearn 0.3.3**（本轮实际 `pip install`） |
| 引入方式 | **可选运行时依赖**：`src/eval/regime.py` 用 `hmmlearn.hmm.GaussianHMM`（`covariance_type="diag"`、固定 `random_state=42`）拟合市场状态；未安装时 `python main.py regime` 明确报错指引，**不降级为规则口径**（规则口径会引入「换一套规则凑达标」的选择自由度） |
| 用途 | ① T18.1：市场层（多标的等权）日收益 + 20 日滚动波动 → 3 态隐状态 → 可读化为 `bull/range/bear`；② T18.2：状态内分层评估（逐状态 IC / 命中率 / 样本数）；③ T18.3：状态 one-hot 作为特征的**单变量**增量 A/B |
| 无前视口径 | 主线为 **expanding 重训**（第 t 天只用 `[0, t]` 观测 fit，`refit_every` 控制步长）；`full_sample` 全样本 decode 口径**明确标注 `lookahead_prefixed=true`**，仅作差异对照、不得作达标证据 |
| 边界 | 仅离线研究用途；`model.factors.regime.enabled` 缺省 **false**，状态**不进**特征/信号链路；`affects_gate` 恒 false；是否进门禁 / 风控属 T18.4 人工检查点 |

### 规划期合规说明

- H1~H4 全部为 sklearn 风格轻量库或既有依赖，符合「零重型依赖、CI 离线可跑」纪律；
- **H1（MAPIE）/ H3（hmmlearn）/ H4（scikit-learn 校准器）已由「规划期登记」转为「已实装」**：
  运行时版本号见上节；H2（CPCV / DSR）为**方法论引入**（不引入运行库）；
- **H5 评估结论：不引入**（`cancel`）—— 0 个候选满足「A 股适配 + 离线可复现 +
  许可证明确」三项准入，见 `00_kickoff/research_assist_conclusion.md`；
  若未来出现合格候选，按同一决策单重跑 `python main.py research-assist`；
- 配套候选评估与淘汰理由见 `00_kickoff/high_value_projects_round2_candidates.md`；
- **H 轮收口（2026-09-12）**：S16~S20 的 `auto_acceptable` 任务全部交付，
  运行时版本以本轮实际 `pip install` 为准；人工检查点
  （T16.4 / T17.4 / T18.4 / T19.4 / T20.4）保持 `pending`，采纳与否待人工签字。
