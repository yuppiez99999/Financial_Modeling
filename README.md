# TrendCast Pro · 金融市场预测模型

> 基于 LightGBM 梯度提升树的智能金融市场预测引擎 — 多周期方向预测 + 真实行情数据链路 + 命中率审计
>
> 已作为**只读信号源**集成至 [28-终极量化交易系统8.4](../28-终极量化交易系统8.4/)：`daily_runner.py` 步骤 2.5 每日自动拉取预测，信号卡片进入每日报告，命中率由 28 本地真实行情回溯审计（见《为28终极量化交易系统提供策略决策依据_设计方案_20260909.md》）。

---

## 一、它做什么

对 **A 股股票 / 商品期货 / 外汇** 标的，输出未来 **5 日（短期）/ 10 日（中期）/ 20 日（长期）** 的涨跌方向概率（二分类），并提供：

- 端到端流水线：数据采集 → 特征工程 → 训练 → 评估 → 导出 → 服务 → 报告
- 双维评估：机器学习指标（Accuracy / F1 / AUC）+ 金融指标（胜率 / 夏普 / 盈亏比 / 最大回撤）
- **推理数据每日自动刷新**：缓存过期经真实源刷新一次，绝不落 simulation 假数据（防随机游走数据伪装"今天"骗过新鲜度检查）
- FastAPI 预测服务（默认端口 8800），支持单只/批量/组合级预测（`/api/v1/portfolio/summary` 对齐 28 持仓池）
- 与 28 系统双向闭环：28 侧审计用本地真实行情回溯命中，命中率与漂移告警进入每日报告

### 最新训练评估（2026-09-09，腾讯财经真实行情 · 目标泄漏已修复）

数据口径：28 持仓池 26 只标的（12 个股 + 14 ETF），腾讯财经前复权日K，2020-01-01 ~ 2026-09-09，按时间三段切分。

> ⚠️ 早期评估（2026-06-27，10d/20d AUC 0.84）存在**目标泄漏**（target 列曾混入特征集）与 simulation 模拟数据口径问题，属虚高值，不可比。下表为修复后（`get_feature_columns` 强制排除 `target_*`）的真实可信基线：

| 模型周期 | 准确率 | AUC | 胜率 | 盈亏比 | 夏普(近似) | 评级 |
|----------|--------|-----|------|--------|-----------|------|
| 短期 5d | 52.7% | 0.5688 | 52.7% | 1.11 | 0.39 | 弱（略优于随机） |
| 中期 10d | 53.2% | 0.5689 | 53.2% | 1.14 | 0.32 | 弱（略优于随机） |
| 长期 20d | 51.3% | 0.5420 | 51.3% | 1.05 | 0.09 | 接近随机 |

**结论**：真实行情下三周期区分度均有限（AUC 0.54~0.57），信号仅可作**只读观测参考**（一期口径），不可单独作为交易依据。提升方向：宏观/情感/横截面特征扩充、按标的分层训练、命中率与 IC 达标后再进入决策路径（见《为28终极量化交易系统提供策略决策依据_设计方案_20260909.md》§5/§6 二期门禁）。

### 核心能力

| 维度 | 说明 |
|------|------|
| 市场覆盖 | 28 持仓池 26 只标的（12 个股 + 14 ETF，见 `configs/config_pro.yaml`）；期货/外汇暂关（腾讯源不支持，二期接 Wind 后开启） |
| 预测周期 | short_term(5d) / mid_term(10d) / long_term(20d) |
| 模型 | LightGBM（主力）；TimesFM、Kronos 为可选/实验性路径 |
| 特征 | MA/RSI/MACD/布林带/量价/波动率/时间特征；宏观与新闻情感为配置开关（当前关闭） |
| 数据源 | Wind MCP (P0，需 Key) → **腾讯财经 (P1，免费真实日K，前复权，已真实可用)** → 模拟数据 (P6 兜底，仅链路验证，不落盘) |
| 数据纪律 | simulation 兜底数据**绝不写入缓存**；推理缓存过期自动经真实源刷新 |
| 防泄漏 | `get_feature_columns` 强制排除 `target_*` 目标列（早期 AUC 0.84 虚高值的根因已修复） |
| 风控建议 | 止损止盈建议（ATR + 结构位 + 盈亏比约束）；**门禁未放行时不给可用价位**（fail-close），见 §13 |
| 出口 | CLI / REST API / ONNX 导出 / 每日与周度报告 |

---

## 二、快速开始

### 1. 安装

```bash
git clone https://github.com/yuppiez99999/Financial_Modeling.git
cd Financial_Modeling

python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

要求 Python ≥ 3.8（实际运行环境 3.8.9，代码全链路兼容）。使用 Wind 数据源时需配置 `WIND_API_KEY`；不用 Wind 时自动使用**腾讯财经免费日K（无需 Key）**，拉取前自动把腾讯域名并入 `NO_PROXY`（绕过系统代理拦截，Windows 代理环境必读）。

### 2. 训练

```bash
python main.py train
```

自动完成 采集 → 特征 → 训练 → 评估，产出 `models/lightgbm_{short,mid,long}_term_*.pkl` 与 `logs/evaluation_report.txt`。

### 3. 预测（缓存过期自动刷新）

```bash
python main.py predict 600519.SH --horizon all      # 单只，全部周期
python main.py batch 600519.SH 000858.SZ --horizon mid_term
```

> 预测时若本地缓存最后日期早于今天，`PredictionEngine._ensure_fresh` 会自动经腾讯源刷新**一次**（每标的每进程至多一次，节假日不空刷），刷新失败回退旧缓存（fail-open）。刷新只走真实数据源，绝不采用 simulation 兜底数据。

### 4. 报告与调度

```bash
python main.py daily-report     # 每日预测报告
python main.py weekly-report    # 周度预测报告
python main.py adaptive         # 自适应学习（性能监控 + 漂移检测）
python main.py schedule         # 自动重训练调度（常驻）
```

### 5. 启动 API 服务

```bash
python main.py serve            # http://localhost:8800
```

```bash
curl http://localhost:8800/health
curl "http://localhost:8800/api/v1/predict/300308.SZ?horizon=long_term"

curl -X POST http://localhost:8800/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"symbols":["300308.SZ","601088.SH","600276.SH"],"horizon":"long_term"}'

# 组合级摘要（28 系统每日消费的契约端点；缺省用 config 启用标的）
curl "http://localhost:8800/api/v1/portfolio/summary?symbols=600519.SH,300308.SZ"
```

---

## 三、命令行接口


| 命令 | 说明 |
|------|------|
| `train` | 完整训练流水线（采集 → 特征 → 训练 → 评估） |
| `evaluate` | 评估已训练模型 |
| `predict <symbol>` | 单只预测，`--horizon` 指定周期 |
| `batch <symbols...>` | 批量预测 |
| `export` | 导出 ONNX |
| `serve` | 启动 FastAPI 服务 |
| `schedule` | 启动自动重训练调度器 |
| `audit` | 预测审计（回溯验证历史预测） |
| `notify <symbol>` | 预测并推送信号（Webhook / 邮件） |
| `daily-report` / `weekly-report` | 生成日/周报 |
| `adaptive` | 运行自适应学习引擎 |
| `all` | 训练 → 评估 → 导出 全流程 |
| `macro` | 查看宏观指标（CPI/PMI/GDP/M2/LPR）数据源状态 |
| `monitor` | 生成模型监控报表（审计命中率 / 漂移 / 数据源 / 模型产物 / 实时流 / IC 趋势） |
| `ic` | 前视 IC / ICIR / 命中率评估（walk-forward 口径，门禁数据源） |
| `gate` | 策略门禁判定（IC + 可选审计命中率），输出 `reports/strategy_gate.json` |
| `factors <symbol>` | 多因子加权组合预测（模型因子 + 技术特征因子） |
| `factor-model [symbol]` | 多因子模型权重 / 族权重 / IC 诊断（可训练模型） |
| `stream [--once] [--symbols A,B]` | 盘中实时流：快照轮询 + 分钟级观点失真预警（Q3） |
| `intraday <symbol>` | 单只标的盘中信号更新（baseline vs live） |
| `consistency <symbol>` | 信号一致性校验（跨周期 / 跨模型 / 跨口径，Q3） |
| `ic-trend` | IC 时序 / 信号衰减监控：斜率 + 预计跌破门禁步数（Q5） |

常用参数：`--horizon {short_term,mid_term,long_term,all}`、`--config <path>`、
`--model-type {lightgbm,pytorch_lstm,timesfm,ensemble,factor_model,multifactor}`、
`--symbols <A,B>`（stream / ic）、`--once`（stream）、`--host` / `--port`。

---

## 四、项目结构

```
.
├── configs/
│   ├── config.yaml            # 默认配置（simulation 单源，测试用）
│   └── config_pro.yaml        # 生产配置（wind→tencent→simulation，标的对齐 28 持仓池 26 只）
├── data/
│   ├── raw/                   # 原始行情缓存 <symbol>.csv（simulation 兜底不落盘）
│   ├── realtime/              # 盘中快照 JSONL（Q3；与 raw 严格分离，绝不混入日K缓存）
│   ├── processed/             # 特征工程后的数据集
│   └── news/                  # 新闻缓存
├── models/                    # 训练产物（pkl）与 exported/（ONNX）
├── logs/                      # 运行日志与评估报告
├── reports/                   # 报告输出
├── src/
│   ├── data/                  # 采集 / 预处理 / 技术指标 / 情感分析 / 质量门控
│   │   ├── collector.py       # DataCollector：多源回退 + collect_all + simulation 不落盘
│   │   ├── tencent_client.py  # 腾讯财经免费日K客户端（前复权/分页/NO_PROXY/fail-open）
│   │   ├── macro_client.py    # 宏观指标客户端（wind→akshare→local 回退，asof 无前视对齐）
│   │   ├── streaming.py       # 实时快照 / 交易时段 / 滚动特征 / 快照存储（Q3）
│   │   └── preprocessor.py    # FeatureEngineer（防目标泄漏 + 非有限值清洗）+ DataPreprocessor
│   ├── train/                 # LightGBM 训练器、模型定义、自适应学习
│   ├── eval/                  # 双维评估器
│   ├── inference/             # 推理引擎（每日缓存刷新 _ensure_fresh + 特征对齐 _align_features）
│   │   ├── ic.py              # 前视 IC / ICIR / 命中率计算（Q2 门禁口径）
│   │   ├── factor_combiner.py # 推理期因子加权组合（Q2 多因子）
│   │   └── intraday.py        # 分钟级预测更新（Q3：日频 baseline + 盘中快照 → drift）
│   ├── api/                   # FastAPI 服务（含 /api/v1/portfolio/summary 组合契约端点）
│   ├── export/                # ONNX 导出
│   ├── report/                # 日报 / 周报生成
│   ├── scheduler/             # 自动重训练调度
│   ├── audit/                 # 预测审计
│   ├── notification/          # 信号推送
│   ├── trading/               # 量化交易适配层（信号/风控/订单/回测/门禁）
│   │   └── gate.py            # 策略门禁：IC/命中率判定信号能否进入决策路径
│   ├── monitor/               # 模型监控报表（审计命中率/漂移/数据源/模型产物/实时流）
│   │   └── signal_consistency.py  # 信号一致性校验（跨周期/跨模型/跨口径，Q3）
│   ├── utils/                 # 通用工具（dummy_models）
│   └── timesfm_predictor.py   # TimesFM 适配（可选）
├── Kronos/                    # 第三方基础模型源码快照（见下）
├── scripts/                   # 工具脚本（占位模型生成 / evaluate_models.py 预测评估）
├── tests/                     # pytest 测试（tencent_client/data_freshness/pipeline/api 等）
├── 为28终极量化交易系统提供策略决策依据_设计方案_20260909.md   # 对接设计与验证记录
├── main.py                    # CLI 入口
└── requirements.txt
```

---

## 五、配置

默认加载 `configs/config_pro.yaml`（存在时优先），可用 `--config` 覆盖。

```yaml
data:
  source: ["wind", "tencent", "simulation"]   # 按顺序回退；simulation 兜底不写缓存
  raw_dir: "data/raw"
  start_date: "2020-01-01"
  end_date: ""                                # 留空 = 采集至最新交易日
  min_train_rows: 60                          # 标的行数不足则训练时跳过
  markets:
    stock:   {enabled: true,  symbols: [...]} # 28 持仓池 12 个股
    etf:     {enabled: true,  symbols: [...]} # 28 持仓池 14 ETF（并入 stock 组）
    futures: {enabled: false}                 # 腾讯源不支持期货代码，二期接 Wind 后开启

features:
  technical:
    ma_windows: [5, 10, 20, 60]
    rsi_window: 14
    macd_fast: 12
    macd_slow: 26

training:
  adaptive_learning:
    enabled: true
    drift_threshold: 0.05                     # 准确率下降超阈值触发重训练
    auto_retrain: true
    monitor_interval: "daily"

report:
  output_dir: "reports"
  daily_report: true
  weekly_report: true
```

宏观指标配置（`features.macro_enabled: true` 时启用特征注入）：

```yaml
data:
  macro:
    source: ["wind", "akshare", "local"]   # 按顺序回退；全失败则注入显式零值
    dir: "data/macro"                      # 本地缓存目录（CSV 列: date,value）
    indicators: ["cpi", "pmi", "gdp", "m2", "lpr"]

features:
  macro_enabled: false                     # 开启后每个指标注入 latest/mom/yoy3 三列
```

> 宏观数据按**发布日期 asof 对齐**注入（每行行情只使用发布日期 ≤ 该日的宏观值），杜绝未来函数。
> 离线环境可把历史数据手工放入 `data/macro/macro_<indicator>.csv`（列 `date,value`）。

策略门禁与多因子配置（Q2）：

```yaml
model_factors:
  weighter: "static"          # static / ic（按 |IC| 自适应权重）
  weights:                    # 因子权重（自动归一化；缺失因子移出后重归一化）
    lightgbm: 0.45
    timesfm: 0.25
    momentum: 0.10
    trend: 0.10
    volume: 0.05
    volatility: 0.05

strategy_gate:
  enabled: true
  force_readonly: false       # 人工锁只读（即使达标也不放行）
  scope: "all"                # all / any
  min_ic: 0.03
  min_hit_rate: 0.52
  min_samples: 30
  min_windows: 3
  use_audit: false            # 叠加审计命中率交叉验证
  min_audit_verified: 10
  min_audit_hit_rate: 0.5
  report_dir: "reports"
```

数据源优先级：`Wind MCP (P0) → 腾讯财经 (P1) → 模拟数据 (P6 兜底)`。腾讯客户端按 6 字段参数格式拉取前复权日K（单页 800 条，向历史翻页覆盖 start_date），列序自动重排为标准 OHLCV；Wind 与腾讯均不可用时才用模拟数据，且模拟数据仅作链路验证、绝不写缓存污染真实历史。

---

## 六、可选模型后端

### TimesFM（可选依赖）

TimesFM 作为可插拔预测器接入，可替代或与 LightGBM 组成 `ensemble`。**默认关闭**，未安装 PyTorch 时自动使用 `models/timesfm_*.pkl` 占位模型，保证推理与测试可用。

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install timesfm[torch]
```

启用：`PredictionEngine.load_models(model_type)` 支持 `timesfm` 与 `ensemble`（LightGBM + TimesFM 融合），配置段为 `model.timesfm.{enabled,context_days,verbose}`（见 `configs/config.yaml`）；加载逻辑见 `src/inference/predictor.py`：优先读本地占位 pickle，不存在再懒加载真实 `TimesFMFinancePredictor`（懒加载 `timesfm`/`torch`）。

占位模型细节：
- 占位文件：`models/timesfm_short_term_5d.pkl` / `models/timesfm_mid_term_10d.pkl` / `models/timesfm_long_term_20d.pkl`，pickle 格式为 `{'model': DummyModel(), 'scaler': DummyScaler()}`（`DummyScaler.transform` 为恒等映射），确保无 PyTorch 环境下 `PredictionEngine.predict()` 正常工作并通过测试。
- 占位模型生成/修复脚本：`scripts/create_timesfm_placeholders.py`（创建占位模型）与 `scripts/fix_timesfm_placeholders.py`（将已有 pickles 包装为含 `scaler` 的 dict）。
- 最小化测试：`tests/test_predictor_minimal.py` 覆盖 LightGBM-only、TimesFM 占位路径与 ensemble，运行 `python -m pytest tests/test_predictor_minimal.py -q`。

CLI 用法（白名单已放开，见 `main.py` 的 `--model-type` choices 与 `valid_types`）：

```bash
python main.py predict 600519.SH --horizon mid_term --model-type timesfm
python main.py predict 600519.SH --horizon mid_term --model-type ensemble   # LightGBM + TimesFM 融合
```

> 注意：`timesfm` / `ensemble` 未安装 PyTorch 时会回退到 `models/timesfm_*.pkl` 占位模型，结果仅供链路验证，不代表真实预测精度。

Windows 上若出现 `WinError 126`（本机库加载失败），通常是 PyTorch 与 CUDA/CPU 版本不匹配，改用 CPU 版 PyTorch 即可。

### Kronos（第三方源码快照）

`Kronos/` 是开源金融 K 线基础模型 [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) 的源码快照（commit `67b630e`，2026-04-13），**非本项目原创**，仅作为后续接入基础模型的实验底座，目前尚未与 `src/` 主链路打通。来源与许可见 `Kronos/THIRD_PARTY_NOTICE.md` 与上游 LICENSE。

---

## 七、与主量化系统的对接（一期已交付并验证）

本仓库作为 **16_ 侧只读信号源**，与 `28-终极量化交易系统8.4` 通过 REST 对接。设计原则：**16_ 只出方向与概率，28 独享决策权**（一期只读注入，不改变 28 任何下单/调仓行为）。

```python
import requests

# 组合级契约端点（28 daily_runner 步骤 2.5 每日自动调用）
resp = requests.get(
    "http://localhost:8800/api/v1/portfolio/summary",
    params={"symbols": "600519.SH,300308.SZ"},
    timeout=30,
).json()
# 契约: {"generated_at", "model_type", "predictions":[{symbol, sector,
#        horizons:{h:{direction, probability, model}}}], "meta":{models_loaded, error_symbols}}
```

### 对接闭环（28 侧组件，位于 28 仓根目录）

| 组件 | 文件（28 仓） | 职责 |
|------|--------------|------|
| 客户端 | `trendcast_client.py` | health / portfolio summary / 读 28 持仓（fail-open，服务不可达→跳过） |
| 审计 | `trendcast_audit.py` | JSONL 落盘 + `price_source` 本地真实行情回溯 + 命中率/漂移告警 |
| 接线 | `daily_runner.py` 步骤 2.5 | 拉取→审计→信号卡片→每日报告；`logs/trendcast/signals_*.json` 快照供 LLM 上下文 |

### 验证状态（2026-09-09 E2E 冒烟通过）

- 正路径：26 标的 × 3 周期 = 78 条真实预测（概率 43%~55% 脱离 0.5 兜底）→ 快照 + 审计落盘
- 命中率链路：78 条已到期记录 → 57 条被 28 本地真实行情回溯 → **真实命中率 56.1%**（21 条本地无行情保持未验证，绝不误判）
- fail-open：16_ server 关闭 → 28 优雅跳过，主流程零中断
- 数据卫生：simulation 时代旧审计记录已清除；price_source 命中覆盖 6/10（4 只 ETF 本地缓存缺失，二期补齐）

> 早期配套的旧版 `trendcast_client.py`（指向 11_量化策略 v5.2.1）已废弃；现对接以 28 仓 `9d953128`/`7ad9b8e2` 与设计方案 v1.1 验证记录为准。

---

## 八、测试

```bash
python -m pytest tests/ -q     # 334 passed, 2 failed, 2 skipped
```

---

## 九、已知限制

- 真实行情下三周期模型区分度均有限（AUC 0.54~0.57，2026-09-09 防泄漏口径），信号仅作观测参考，不应单独作为交易依据
- 模型评估为历史回测口径，未扣除真实滑点与冲击成本，实盘前需做纸面跟踪
- Wind 数据源需终端/Key，缺失时降级到腾讯财经（免费）；期货/外汇代码腾讯不支持（futures 已暂关，二期接 Wind 后开启）
- TimesFM / Kronos 路径尚未接入主推理链路
- 宏观指标（CPI/PMI/GDP/M2/LPR）依赖 Wind Key 或可选依赖 `akshare`；两者均缺失时回退 `data/macro/*.csv`，仍无数据则注入**显式零值**（不编造数据）
- 新闻情感特征默认关闭（`features.sentiment_enabled: false`），开启后受新闻源可用性影响

- **策略门禁当前状态为 `readonly`**（2026-09-10 实测）：三周期 IC 均为正但 short/mid 命中率未过 52%，
  按 fail-close 口径信号**不进入决策路径**；详见「十一、Q2 路线进展」
- **多因子组合的特征因子尚未做过拟合校准**（权重为经验值），`weighter: ic` 需先有足量 IC 样本
- **实时流（Q3）定位为盘中观测预警，不是盘中交易决策**：公开免费源无分钟级行情，
  且模型训练口径是日频 —— 用盘中未收盘价重算特征会构成未来函数，因此只做「观点是否失真」提醒
- **实时流的节假日日历不完整**：`MarketClock` 仅按工作日 + 时段粗判，法定休市日会被判为交易时段，
  故快照一律带 `is_trading_hours` 标记，由下游自行采信
- **腾讯前复权数据在向历史分页时会退化**：接口对早期页的复权基准不一致，会返回非正价格，
  已由 `TencentClient._parse_rows` 剔除（中国神华实测剔除 945/3201 行）；需更长历史时建议接 Wind
- **信号一致性校验不参与策略门禁**：它只回答"各口径是否自相矛盾"，不替代 IC / 命中率门禁

> **Q1 排期已完成的修复**（2026-09-10）：原「已知限制」中「宏观指标 API 连接失败」「新闻采集性能待优化」两项已解决；
> 7 例遗留失败测试（`test_cli_api`/`test_modules`）已全部转绿。详见「十、Q1 排期进展」。

---

## 十、Q1 排期进展（2026-09-10 完成）

对照 `SALES_PLAN.md` §8.2 技术改进路线图 Q1 与 §10 落地执行计划，本仓库 Q1 项已全部落地。

### 10.1 路线图 Q1

| 路线图项 | 状态 | 实现 |
|---------|------|------|
| 修复宏观指标 API | ✅ | 新增 `src/data/macro_client.py`，四级回退链 wind → akshare → local CSV → 显式零值；`python main.py macro` 查看状态 |
| 优化新闻采集性能 | ✅ | `NewsCollector` 进程级缓存 + TTL；`SentimentFeatureGenerator` 整表一次分组聚合。26 标的 × 10 日由 260 次触网降至 **1 次** |
| 缺陷清零 | ✅ | 补齐 `main.build_parser` / `server._HAS_FASTAPI` + `create_app` / `DailyReportGenerator.generate` / `PredictionAudit.load_records` / `SignalNotifier.send_webhook` / `RetrainScheduler._should_run`；修复 pandas 3.0 `fillna(method=)` |

### 10.2 新增运维与评估能力

| 能力 | 入口 | 说明 |
|------|------|------|
| 预测评估 | `python scripts/evaluate_models.py` | **walk-forward 时序回测**（无泄漏），输出 Accuracy / AUC / **IC** / 夏普 / 最大回撤（可选手续费口径），生成 Markdown + JSON 报告 |
| 模型监控 | `python main.py monitor` | 汇总审计命中率（整体/分周期/近 30 天）、漂移信号、自适应性能历史、宏观指标可用性、行情缓存覆盖、模型产物缺失检测 |
| 监控 API | `GET /api/v1/monitor/report` | 同上，JSON 契约，供外部系统消费 |
| 宏观指标 | `python main.py macro` | CPI/PMI/GDP/M2/LPR 数据源可用性与当前特征值 |

### 10.3 评估脚本口径（防误读）

```bash
python scripts/evaluate_models.py                                   # 配置全标的池 × 三周期
python scripts/evaluate_models.py --symbols 600519.SH --horizon mid_term --folds 3
python scripts/evaluate_models.py --fee 0.0005                      # 计入双边手续费
python scripts/evaluate_models.py --offline --output logs/eval.json # 离线 + JSON 落盘
```

- **walk-forward 时序切分**：滚动扩窗，训练严格早于测试（打乱 = 未来数据泄漏，指标虚高）
- **特征口径与训练一致**：强制排除 `target_*` 与评估辅助列
- **逐标的构造目标**：避免多标的 concat 后跨标的 shift 污染
- **IC**：预测概率与未来收益的 Spearman 秩相关，衡量信号单调区分度（二期门禁关注项）

### 10.4 测试

```bash
python -m pytest tests/ -q     # 159 passed, 1 skipped
```

新增测试：`test_roadmap_q1.py`(15) · `test_macro_client.py`(17) · `test_sentiment_perf.py`(14) · `test_evaluate_models.py`(20) · `test_monitor_report.py`(18)

### 10.5 下一步（Q2 路线）

- ✅ 多因子模型集成，支持因子加权组合预测 → 见「十一、Q2 路线进展」
- ✅ 按评估脚本的 **IC / 命中率门禁**判定是否升级为「调仓打分因子」 → 见「十一」
- ✅ 用 28 真实持仓池扩标的与重训，`price_source` 接真实行情源 → 见「11.6」
- ✅ LSTM 上线（路线图 Q1 第 3 月项）→ 见「11.7」

---

## 十一、Q2 路线进展（多因子集成 · IC/命中率门禁）

对照 `SALES_PLAN.md` §8.2 路线图 Q2「多因子模型集成，支持因子加权组合预测」，以及
设计方案 §5「命中率与 IC 达标后再进入决策路径（决策路径 fail-close）」，本轮落地三件事：

### 11.1 前视 IC 计算（`src/inference/ic.py`）

- **IC**：预测分数与未来真实收益的 **Spearman 秩相关**（衡量信号单调区分度）；
- **ICIR**：滚动窗口 IC 的 均值/标准差（衡量信号稳定性），窗口 IC 无波动时记 0，不做除零放大；
- **命中率**：sign(score) 与 sign(return) 一致占比，支持观望带（`neutral_band`）；
- **未到期样本一律返回 `None` 并跳过**，绝不猜测未来价格（防未来函数）。

### 11.2 策略门禁（`src/trading/gate.py`）

把「只读观测信号」升级为「调仓打分因子」的准入判定，**fail-close**：缺数据 / 未达标一律不放行。

| 状态 | 含义 |
|------|------|
| `readonly` | 未过门禁（**默认态**），信号仅作参考，不进决策路径 |
| `gated` | 已过门禁，可作为打分因子进入决策路径（仍不产出仓位/下单建议） |
| `disabled` | 显式关闭门禁（`strategy_gate.enabled: false`） |

判定口径（`configs/config_pro.yaml` 的 `strategy_gate:` 段）：`min_ic` · `min_hit_rate` · `min_samples` ·
`min_windows` · `scope`（all/any）· 可选 `use_audit` 审计命中率**与门**交叉验证 · `force_readonly` 人工锁只读。

```bash
python main.py gate                       # 全标的池评估并落盘 reports/strategy_gate.json
python main.py gate reports/ic.json       # 复用既有 IC 评估结果，不重复训练
curl http://localhost:8800/api/v1/strategy/gate      # 供 28 系统读取门禁状态
```

### 11.3 多因子加权组合预测（`src/inference/factor_combiner.py`）

统一「模型因子」与「特征因子」的加权框架：`score = Σ weightᵢ × factor_scoreᵢ`（∈ [-1, 1]）。

| 因子 | 类型 | 说明 |
|------|------|------|
| `lightgbm` / `timesfm` | 模型因子 | 各分量模型概率 → 方向分（0.5 中性映射为 0） |
| `momentum` | 特征因子 | 20 日收益率的滚动分位 |
| `trend` | 特征因子 | 收盘价对 MA20 的偏离分位 |
| `volume` | 特征因子 | 近 5 日 / 近 20 日均量比（tanh 平滑） |
| `volatility` | 特征因子 | 20 日波动率分位（取负，高波动降权） |

- **缺失因子fail-soft**：缺失因子被移出后**重归一化**，而不是当作 0 分稀释得分（后者会引入方向性偏差）；
- **权重模式**：`static`（配置权重） / `ic`（按 |IC| 自适应缩放）；
- `python main.py factors <symbol>`、`GET /api/v1/factors/{symbol}`。

### 11.4 实测门禁结果（2026-09-10，26 只标的池 · walk-forward 口径）

| 周期 | IC | ICIR | 命中率 | 样本 | 是否达标 |
|------|-----|------|--------|------|----------|
| short_term 5d | 0.0400 | 0.185 | 51.37% | 12418 | ❌ 命中率未过 52% |
| mid_term 10d | 0.0513 | 0.274 | 50.49% | 12340 | ❌ 命中率未过 52% |
| long_term 20d | 0.0604 | 0.277 | **54.53%** | 12184 | ✅ |

**门禁结论：`readonly`（fail-close）**——三周期 IC 均为正（信号方向性存在，非噪声），
但 short/mid 命中率未过 52% 门槛，按 `scope: all` 判定整体**不放行**，
信号继续保持只读观测，**不进入调仓打分路径**。

> 该结果与 README §一「真实行情下三周期区分度均有限」的结论一致，是**如实呈现而非掩饰**：
> 门禁的作用正是把「看着还行」与「统计上可用」区分开。提升方向见 §11.5。

### 11.5 下一步（Q2 剩余 + Q3）

- 特征扩充（横截面 / 宏观 / 情感）提升 short/mid 周期 IC 与命中率至门禁线以上；
- 通过门禁后，将 `gated` 状态接入 28 的调仓打分因子（须先扩样本、过审）；
- ✅ LSTM 上线（路线图 Q1 遗留项）→ 见 §11.7；
- Q3：实时数据流接入，支持分钟级预测更新。

### 11.6 可训练多因子模型与真实持仓池（2026-09-10）

§11.3 的 `factor_combiner` 解决「**推理期**如何把已有因子组合成得分」；
本节解决「**训练期**如何学出因子权重并持久化」。

#### 11.6.1 因子库（`src/factors/factor_library.py`）

7 大因子族 · 15 个因子，每个因子都有明确**正向预期**并压到 `[-1, 1]`（可跨因子直接加权）：

| 族 | 因子 | 正向含义 |
|----|------|---------|
| trend | `ma_bias` · `ema_slope` · `adx_strength` | 价格在均线上方 / 均线上行 / 多头趋势强度 |
| momentum | `rsi_score` · `roc` · `macd_hist` | RSI 高于 50 / 20 日动量为正 / MACD 柱为正 |
| volatility | `volatility` · `atr_ratio` | **低波动**（取负，低波动溢价） |
| volume | `volume_ratio` · `obv_slope` · `cmf` | 放量 / OBV 上行 / 资金净流入 |
| reversal | `boll_revert` · `kdj_score` | 超买看跌（取负，均值回归） |
| sentiment | `sentiment` | 新闻情感（无数据 → 中性 0，不编造） |
| macro | `macro` | 宏观指标 z-score 合成（无数据 → 中性 0） |

- **无前视**：全部基于 rolling/ewm/shift，测试 `test_no_lookahead_factor_is_causal` 用"截断末尾数据后前缀不变"验证；
- **失败隔离**：单个因子族异常只置中性 0，不中断主链路。

#### 11.6.2 因子模型（`src/factors/factor_model.py`）

族级 → 因子级**两级权重**，得分经 logit **单调校准**为概率（测试保证「得分越高概率越高」）：

- 权重来源：显式配置优先，否则由**训练集 IC**（Spearman）估计，`|IC| < min_abs_ic` 的因子不参与；
- **族权重用族内最强 |IC|** 而非权重和：避免"1 个强因子"与"5 个弱因子"同权，也让**无数据族（IC=0）权重归零**而不稀释有效因子；
- `level2_shrink` 向等权收缩，防单一因子过度集中；
- 接口与 LightGBM/LSTM 同构（`train/predict/predict_proba/save/load`），可直接接入 `ModelTrainer`；
- `explain()` 输出因子/族权重与 IC 排名，报告、监控、API 直接消费。

```bash
python main.py train --model-type factor_model    # 训练并落盘 factor_model_*.pkl
python main.py factor-model 600519.SH             # 因子/族权重与 IC 诊断
curl http://localhost:8800/api/v1/factor-model    # JSON 契约
```

#### 11.6.3 类级加权组合预测（`src/factors/factor_predictor.py`）

`最终得分 = Σ w_class × 类得分`，各类先映射为方向分再融合：`factor`(0.40) + `tree`(0.35) + `sequence`(0.25)。

- **缺失类自动剔除并重归一化**：没装 torch 时 `sequence` 类自动缺席，**不报错、不稀释**；
- **列缺失不中断**：推理时缺失因子列按中性 0 补齐（`align_features`），也可直接吃原始 OHLCV 现算因子；
- `python main.py predict 600519.SH --model-type multifactor` → 结果含 `components`（各类得分）与 `explain`（因子解释）。

#### 11.6.4 真实持仓池与数据质量门控

- 持仓池 **26 只**（12 个股 + 14 ETF），对齐 28 终极量化交易系统；`source: [wind, tencent, simulation]`，仿真**不落盘**；
- `data.quality_gate: true` 接入质量体检（质量分 / A 级占比），**只告警不删数据**；
- 监控报表新增**持仓池覆盖度**（已缓存 / 已配置）与质量门控状态。

### 11.7 LSTM 上线与模型注册表（2026-09-10）

补齐 Q1 遗留的 LSTM 上线阻塞点，LSTM 已可**训练 → 落盘 → 加载 → 推理**全链路运行。

| 问题 | 修复 |
|------|------|
| 加载逻辑分散 | `src/train/registry/model_registry.py`：joblib/torch/factor/timesfm 四格式统一加载契约，新增格式只需注册 loader |
| 推理特征列靠猜 | LSTM checkpoint 写入 `meta.feature_cols`，推理按 meta 对齐列序（原先靠"非 factor_ 前缀"推断，极易错位） |
| 评估长度不一致 | 评估器按**实际预测长度**反向对齐标签，修复 `accuracy_score` 抛错（Q1 遗留阻塞点） |
| 滑窗丢最新一行 | 修正 `_create_sequences`：样本数 = `n - seq_len + 1`，**覆盖最新一行**（原先实盘等于永远少预测一天） |
| YAML 数值变字符串 | `weight_decay: 1e-5` 会被 YAML 解析为字符串导致 optimizer 报类型错，改为 `1.0e-5` 并在训练器内做二次兜底 |
| 无 torch 环境 | 多因子组合自动剔除 `sequence` 类并重归一化，推理不中断 |

> 注：`pytorch_lstm` 未安装 torch 时**自动降级**，其余模型类型不受影响。

### 11.8 测试

```bash
python -m pytest tests/ -q     # 230 passed（Q1 基线 159 → Q2 230）
```

| 测试文件 | 数量 | 覆盖 |
|---------|------|------|
| `test_q2_roadmap.py` | 25 | IC 计算 / 策略门禁 / 因子组合（同批 Q2 交付） |
| `test_roadmap_q2.py` | 35 | 因子库 / 因子模型 / 类级融合 / 模型注册表 / LSTM 上线 / 质量门控 / 适配层门禁强制 |
| `test_roadmap_q3.py` | 34 | 实时快照 / 盘中观点失真 / 信号一致性 / 非法行情防护 |
| `test_roadmap_q4.py` | 34 | ATR 止损 / 结构位 / 盈亏比约束 / 门禁 withhold 分支 |
| `test_roadmap_q5.py` | 33 | IC 时序 / 斜率与趋势判定 / 跌破外推 / 四个消费口分支 |

另修复 3 处**跨用例模块全局污染**（`test_predictor_minimal`/`test_ensemble_integration`/`test_predictor_integration`
直接给模块属性赋值 → 改为 `monkeypatch.setattr`）：原先会污染 `src.inference.predictor` 的
`DataCollector`/`FeatureEngineer`，导致后续用例拿到假数据管道。

> **Q2 剩余项**：short/mid 周期命中率提升至门禁线以上（特征扩充）、门禁 `gated` 后接入 28 调仓打分。

---

## 十二、Q3 路线进展（实时数据流 · 信号一致性 · 口径纠偏）

对照 `SALES_PLAN.md` §8.2 路线图 Q3「实时数据流接入，支持分钟级预测更新」，本轮落地三件事。

### 12.1 实时数据流接入（`src/data/streaming.py`）

| 组件 | 职责 |
|------|------|
| `RealtimeQuoteClient` | 腾讯实时快照（最新价 / 昨收 / 今开），批量请求，**逐只容错**，失败 fail-open |
| `MarketClock` | 交易时段粗判（工作日 + 上午/下午时段），不含节假日日历 → 快照带 `is_trading_hours` 标记由下游采信 |
| `IntradayStore` | 快照滚动存储（JSONL 按交易日分文件），保留期自动清理，单行损坏不影响其余 |
| `RollingFeatureBuilder` | 分钟级滚动特征（MA / 波动率 / 动量 / RSI），**严格只用已收盘日K** |

**数据纪律**：快照目录 `data/realtime/` 与日K缓存 `data/raw/` **严格分离** ——
分钟级数据混入日K缓存会污染训练集，是本项目明令禁止的事故类型（有测试守护）。

### 12.2 分钟级预测更新（`src/inference/intraday.py`）

先说清楚**"分钟级更新"到底更新了什么**：不是每分钟重训/重推模型（既无分钟级数据源支持，
也会引入未来函数），而是回答盘中才有意义的问题：

> 昨收时的模型观点，在盘中价格相对昨收变化 X% 后，是否仍成立？

输出三段可审计信息：

| 字段 | 含义 |
|------|------|
| `baseline` | T-1 收盘口径的日频预测（真实模型优先，失败降级**透明规则口径**并标注 `rules_fallback`） |
| `live` | 盘中快照（最新价 / 昨收 / 涨跌幅） |
| `drift` | `intact`（观点仍成立） / `stale`（盘中反向波动超阈值，观点可能失真） / `unknown`（无有效涨跌信息，不误报） |

**防未来函数**：baseline 只读**已落盘的日K**（T-1 及更早），盘中价只作"新观测"比对，
**绝不回写日K、绝不用于重算日K特征**（有测试用"截断尾部后前缀不变"验证）。

**陈旧保护**：日K缓存落后超过 `max_stale_days`（默认 5 天）时**拒绝给盘中结论**——
拿一周前的日K解释今天的盘中波动，得出的"观点失真"预警毫无意义。

```bash
python main.py stream --once --symbols 600519.SH,300308.SZ   # 一次性盘中更新
python main.py stream                                        # 常驻轮询（Ctrl+C 退出）
python main.py intraday 600519.SH                            # 单只标的
```

**实测（2026-09-10 盘中，真实快照）**：

```
600519.SH  最新价 1284.01  相对昨收 -0.53%  → intact（未与 baseline「看涨」相悖）
300308.SZ  无日K缓存 → available=false（如实标注，不臆测）
RB.SHF     实时源不支持该市场 → 跳过（不编造）
```

### 12.3 信号一致性校验（`src/monitor/signal_consistency.py`）

Q2 之后仓库里同时存在多条"给出方向"的路径（周期模型 / `FactorCombiner` / `FactorModel` / 盘中更新）。
它们各自都能跑通，但没人回答更基础的问题：**它们互相矛盾吗？**

| 校验维度 | 判定 |
|---------|------|
| 跨周期 | 中/长周期方向是否同向；`ignore_short_term: true` 时**允许短期噪声相悖**（仅有短期时以短期为准，不沉默） |
| 跨模型 | 可用分量（tree / factor / sequence）方向是否一致；**不足 2 个分量即 unknown**，不臆测"一致" |
| 跨口径 | 模型概率 vs 因子得分：方向相悖，或同向但**强度差距超容忍带**，均判 divergent |

- 中性带（`neutral_band`）内的概率视为**"无观点"**，不参与一致性判定；
- **一致性校验为观测项，不参与策略门禁**（门禁仍只看 IC / 命中率），输出中显式免责，
  避免下游误当成"第二道门禁"而产生隐式行为变更；
- `python main.py consistency 600519.SH`、`GET /api/v1/consistency/{symbol}`。

### 12.4 评估口径纠偏（防"看起来很强"的指标误读）

本轮把评估输出里**长期存在的口径歧义**显式化，避免内部/对外误读：

| 问题 | 修正 |
|------|------|
| 盈亏比在无亏损笔时取哨兵值 999，易被读作"盈亏比 999 倍" | 新增 `profit_factor_is_capped` 标记，报告用 `*` 标注并写明"已达解析上限" |
| 夏普是"每 N 日一次对赌"的近似年化，量级远高于真实资金夏普 | 字段 `sharpe_is_notional` + 报告注明"横向比较用，非资金夏普" |
| 最大回撤单位是"笔"而非百分比，易与资金回撤率混用 | 新增 `max_drawdown_unit`，报告显式标注 |
| 未计交易成本，胜率是否"够本"无参照 | 评估器 `evaluate(..., fee=)` 输出扣费口径 `*_net`，并给出**保本胜率** |
| walk-forward 报告缺少口径说明区 | `scripts/evaluate_models.py` 报告头部新增"指标口径（防误读）"区块 |

```bash
python scripts/evaluate_models.py --fee 0.0005     # 含双边费用口径 + 保本胜率
```

### 12.5 修复：真实数据源的非法行情污染（重要）

**症状**：部分标的（如 `601088.SH` 中国神华）**三周期预测全部失败**，报
`Input X contains infinity or a value too large for dtype('float64')`。

**根因**：腾讯 `qfq`（前复权）接口在**向历史分页**时复权基准不一致，返回"前复权价"退化
甚至为负 —— 中国神华 2013 年段实测 **912 行 `close` 为负**（如 `-0.32`）。这些行本身无经济含义，
但会连锁引发：

1. `pct_change()` 除零产生 `-inf` → 污染 `ret_*` / `roc_*` / `pvt` 等特征 →
   LightGBM 直接拒绝预测，**整只标的全挂**；
2. 训练集混入分布外极端值，**静默拉低模型质量**（且不会报错，最难发现）。

**修复（三层防御）**：

| 层 | 位置 | 做法 |
|----|------|------|
| 数据入口 | `TencentClient._parse_rows` | 剔除 `close/high/low <= 0` 与 `high < low` 的非法行，并记录剔除行数 |
| 特征出口 | `FeatureEngineer.transform` | `replace([±inf], nan)` 后统一 ffill/fillna，保证特征矩阵**全有限** |
| 指标内部 | `TechnicalIndicators._add_pvt_nvi` | NVI 单步乘法加 `np.isfinite` 守卫，溢出段保持上一值（不编造） |

**验证**：修复前 `601088.SH` 三周期全 error；修复后正常输出概率，且特征集
**非有限值计数为 0**（有 3 个回归测试守护）。

### 12.6 接口与配置

```bash
python main.py stream [--once] [--symbols A,B]   # 盘中实时流
python main.py intraday <symbol>                 # 单只盘中信号
python main.py consistency <symbol>              # 信号一致性校验
```

| API | 说明 |
|-----|------|
| `GET /api/v1/stream/status` | 实时流状态（快照覆盖 / 新鲜度 / 是否交易时段） |
| `GET /api/v1/stream/{symbol}` | 单标的盘中信号（baseline vs live → intact/stale） |
| `GET /api/v1/consistency/{symbol}` | 信号一致性校验 |

配置新增 `streaming:`（默认 **`enabled: false`**，需显式开启）与 `consistency:` 段；
监控报表新增「实时数据流（Q3）」章节。

### 12.7 测试

```bash
python -m pytest tests/ -q     # 260 passed（Q2 基线 230 → Q3 260）
```

新增 `tests/test_roadmap_q3.py`（33 项）：快照解析/落盘/清理、滚动特征无前视、
盘中 drift 判定（stale/intact/unknown）、陈旧缓存拒绝结论、引擎失败降级、
一致性三维校验、扣费口径与保本胜率、以及上述数据卫生的三项回归测试。

> **Q3 定位说明**：实时流是**盘中观测与预警**手段，不是盘中下单决策系统。
> 分钟级数据源的缺失与"训练口径是日频"这一事实决定了：盘中只做观点漂移提醒，
> 决策仍走日频 + 门禁路径（`strategy_gate` 当前 `readonly`）。

> **Q3 剩余 / Q4 展望**：分钟级行情源（需付费数据商）、盘中信号需积累样本后再评估门禁口径；
> Q4 智能风控模块（自动生成止损止盈建议）→ 已落地，见「十三、Q4 路线进展」。
>
> **Q4 遗留 / Q5 承接**：风控建议建立在门禁信号之上，而门禁长期为 `readonly`；
> 本轮 Q5 补齐「信号是否在衰减」的运维视角（见「十四、Q5 路线进展」），
> 使「要不要提前重训练、什么时候可能达标」变成可回答的问题。

---

## 十三、Q4 路线进展（智能风控模块 · 自动止损止盈建议）

对照 `SALES_PLAN.md` §8.2 路线图 Q4「智能风控模块，自动生成止损止盈建议」。

### 13.1 先说清楚它**不是**什么

`src/trading/risk.py`（既有）已经把信号变成仓位与止损止盈了 —— 但那是
`price * (1 - 3%)` 的**固定比例**，代价是两个：全池共用一个常数（3% 对 ETF 偏松、
对高波动个股偏紧），以及整数百分比止损**几乎必然落在噪音里**（要么被日内正常波动打掉，
要么宽到失去意义）。

Q4 模块（`src/trading/risk_advice.py`）只补这两件事，**不重算仓位、不下单、不改动
`RiskManager` / `OrderGenerator` / `TradingAdapter` 的任何行为**（有测试守护）。

### 13.2 止损：取「结构位」与「ATR」中更保守的一侧

| 输入 | 计算 | 说明 |
|------|------|------|
| 结构位 | 近 20 日最低价 × (1 − 0.5%) | **不含最后一根K线**（避免用当日极值自我参照） |
| 波动位 | 入场价 − ATR×2 | ATR 为真实波幅的 Wilder 平滑 |
| 最终 | 取更远的一侧 | 更保守 = 更不容易被正常波动扫掉 |

再压进 `[min_stop_pct, max_stop_pct]`（默认 1.5%~12%）区间；发生收敛时写入
`notes`，不静默改数。空头方向完全镜像（阻力位上方 / 入场价 + ATR×2）。

### 13.3 止盈：盈亏比下界与近端阻力取更近者

先按 `min_risk_reward`（默认 1.5）算出止盈下界，再与近端阻力/支撑比较，**取更近的一侧**
—— 不假设价格能穿越阻力。两边都不足以覆盖下界时，输出 `notes` **建议放弃该机会**，
而不是把止损放宽去凑盈亏比（后者是典型的"为了好看而破坏风控"）。

```bash
python main.py risk-advice 600519.SH          # 单只（Markdown）
python main.py risk-advice --all --json       # 全池（JSON，落盘 reports/risk_advice.json）
```

实测（2026-09-10 真实腾讯行情，门禁放行口径下）：

| 标的 | 参考价 | 建议止损 | 建议止盈 | 止损% | 止盈% | 盈亏比 | ATR% |
|------|--------|---------|---------|-------|-------|--------|------|
| `600519.SH` | 1285.13 | 1240.25 | 1352.45 | 3.49% | 5.24% | 1.50 | 1.75% |
| `000858.SZ` | 70.48 | 68.18 | 73.94 | 3.27% | 4.90% | 1.50 | 1.63% |
| `510300.SH` | 4.62 | 4.495 | 4.807 | 2.70% | 4.06% | 1.50 | 1.35% |

三只标的的止损比例互不相同（2.70% / 3.27% / 3.49%），这正是"比例由标的自身波动决定"
的直接体现 —— 若沿用固定 3%，`510300.SH` 会被打得偏紧、`600519.SH` 偏松。

### 13.4 合规与安全边界（本模块最重要的一节）

Q4 是路线图里**唯一直接产出"价位"**的模块。价位对下游有交易暗示性，因此加了四道闸：

| 边界 | 做法 |
|------|------|
| **门禁 fail-close** | `strategy_gate` 非 `gated` 时返回 `status=withheld`，`suggested_stop/take` 为 `null`，只保留波动结构与仓位侧信息。**"没评估过"（`unknown`）同样不放行** —— 未验证 ≠ 可用 |
| **信息不足拒绝给数** | 无行情 / 少于 30 根K线 / 缺 `high·low·close` 列 / 参考价非法 / ATR 不可用 → `status=unavailable` + 原因，**绝不用默认比例兜底算一个"看起来能用"的价** |
| **不冒充交易指令** | 字段统一 `suggested_*` 前缀，`not_trade_instruction: true`，附 `disclaimer`；`RiskManager` 输出保持不变（测试逐字段比对） |
| **无未来函数** | 只读已落盘日K，不触网、不重算当日盘中数据 |

> 当前 `strategy_gate` 为 `readonly`，因此**真实运行下本模块对本仓库全部标的均返回
> `withheld`** —— 这是预期行为，不是故障。门禁达标后自动恢复价位建议，无需改配置。

### 13.5 接口与配置

```bash
python main.py risk-advice 600519.SH     # 单只建议（Markdown）
python main.py risk-advice --all         # 全池建议（Markdown）
python main.py risk-advice --all --json  # JSON，同时落盘 reports/risk_advice.json
```

| API | 说明 |
|-----|------|
| `GET /api/v1/risk/advice/{symbol}` | 单标的止损止盈建议（契约见 §13.4） |
| `GET /api/v1/risk/advice?symbols=A,B` | 标的池批量建议 |

监控报表新增「智能风控建议（Q4）」章节（读 `reports/risk_advice.json`，只读、零副作用）：
门禁状态 / 建议产出条数 / 暂缓条数 / 平均盈亏比。

配置段 `trading.risk_advice`：`atr_window` · `atr_stop_mult` · `atr_take_mult` ·
`support_lookback` · `support_buffer` · `min_stop_pct` · `max_stop_pct` ·
`min_risk_reward` · `vol_target_pct` · `min_bars`。

### 13.6 波动缩放仓位建议

在 `RiskManager` 给出的仓位之上**只做缩放**（不出新仓位）：ATR 占比高于
`vol_target_pct`（默认 1.5%）时，按 `vol_target_pct / atr_pct` 等比缩小并写入
`vol_scaled_position_pct` 与 `notes`。缩放系数封顶 1.0 —— 低波动标的不会被放大仓位。

### 13.7 测试

```bash
python -m pytest tests/test_roadmap_q4.py -q   # 34 passed, 1 skipped
```

新增 `tests/test_roadmap_q4.py`（34 项）：ATR/结构位计算与样本不足拒答、止损取保守侧、
比例区间收敛与备注、止盈受阻力约束、盈亏比不足提示放弃而非放宽、门禁
`readonly`/`unknown`/无报告三种 withhold 分支、五类数据不足拒答、HOLD 不给价位、
波动缩放封顶、**建议不改变 `RiskManager` 输出**、`advise_payload` 与 `SignalEngine` 同源、
契约可序列化、配置段、CLI 命令、API 状态码、监控报表两种分支。

## 十四、S7 门禁解锁攻坚（门禁诊断 · 分层评估 · 训练折中性带）

> 背景：路线图 Q1~Q4 走完后，**门禁仍为 `readonly`** —— short/mid 命中率没过 52%，
> 导致 Q4 的止损止盈建议在真实运行下**一条价位都不会输出**（全部 `withheld`）。
> 功能是对的、边界是干净的，但它等于「装好了但锁着」。S7 就是去解锁它。
>
> ⚠️ S7 **只提升「可解释性 / 可评估性」，不改变门禁结论、不放宽风控标准**。
> 是否放行仍由 `strategy_gate` 按既定阈值判定。

### 14.1 门禁阻塞诊断（`gate-diagnose`）

既有 `strategy_gate.json` 只说一句「命中率 51.37% < 52%」。人看到的是一串数字，
不知道**还差多少、哪条腿在卡、值不值得动**。诊断把阻塞结构化：

| 字段 | 含义 |
|------|------|
| `shortfall` | 每个未达标指标距门槛的绝对差距（可排序、可追踪） |
| `headroom` | 已达标指标超门槛的余量（判断"要不要动它"） |
| `gaps` / `total_gap` | 归一化差距（0 = 刚好达标），用于跨指标比较 |
| `binding_horizon` / `binding_metric` | `scope=all` 下最难的腿；`scope=any` 下最易过的腿 |

口径：**纯读**既有产物，不训练、不触网、不改门禁结论；数据缺失 / 样本不足一律标注
`available=false`，**不给假门槛差**。

```bash
python main.py gate-diagnose            # 复用 reports/ 产物；无产物则先跑 ic
```

实测输出（README §11.4 的真实数字）：

```
scope=all 要求所有周期达标，当前最难补的是 mid_term（hit_rate 还差 1.51 个百分点）；
优先补齐该周期即可解锁
```

监控报表「策略门禁（Q2）」章节新增「🔧 阻塞诊断」行，`issues` 也会带上具体差距
（不再只有一句"门禁未放行"）。

### 14.2 按标的分层评估（`--stratify`）

`scope=all` 下整体指标会被"木桶短板"绑架，但不知道是哪只标的拖的。`--stratify`
额外**逐标的独立**跑一遍 walk-forward，产出精简指标（不塞原始数组）：

```bash
python main.py ic --stratify
```

结果进 `result["per_symbol"]`，每标的重带 `ic / hit_rate / hit_rate_raw / neutral_band`，
用于定位拖后腿标的。

### 14.3 训练折中性带（`--derive-band`，默认关闭）

**问题**：既有门禁把 0.5 概率**硬编码**为决策边界 —— 概率恰好 ≈0.5 的样本被强行
算作一个方向，噪声直接稀释命中率。

**诱惑与陷阱**：直接在测试折上挑一个阈值去抬命中率，是典型的数据泄漏 —— 指标会
虚高且不可复现。这正是本仓库反复强调的"看起来很强"。

**做法**：`derive_neutral_band` **只在训练折**上按网格搜索"保留比例不低于
`min_keep_ratio` 时命中率最高"的中性带，返回的是**训练折口径的常数**，由调用方
原样施加到测试折。训练折里学到的常数用在测试折上**不构成前视**。

- 退化保护：样本 < 10 直接返回 0.0（= 不设门槛，与既有口径一致），**绝不硬凑**；
- 保留约束：`min_keep_ratio`（默认 0.5）防止"只留 1 个样本 → 命中率 100%"的退化解；
- 多折取中位数：单折异常不污染整体口径，且全部来自训练折；
- `hit_rate_raw` 始终记录原始（0.5 边界）命中率，用于对照与追溯。

```bash
python main.py ic --derive-band         # 显式开启
# 或配置 strategy_gate.neutral_band.enabled: true（缺省 false）
```

> ⚠️ **默认关闭**：不开启时，评分口径与门禁判定**逐字段不变**（`neutral_band=0`）。
> 该机制是否真能推过门禁线，**必须用真实行情重跑验证**；本仓库 CI 环境无网络，
> 只做了合成序列上的正确性验证（见 §14.5），不做任何"已解锁"的结论。

### 14.4 真实链路实测（本机缓存行情，3 标的池）

> ⚠️ 以下是**真实数据**跑出来的结果，**不是**合成数据。结论对下一步决策有直接影响。

**池化口径（现状，门禁据此判定）**：short_term 未过（IC 0.0296 < 0.03 且命中率 51.36% < 52%），
mid/long 通过 → `scope=all` 下整体**仍为 readonly**。

**按标的分解（`--stratify`）** —— 这是关键：

| 标的 | short_term | mid_term | long_term |
|------|-----------|----------|-----------|
| `600519.SH` | ✅ IC 0.1324 / 54.39% | ✅ IC 0.1096 / 57.38% | ✅ IC 0.2813 / 61.97% |
| `000858.SZ` | ❌ IC 0.0257 / 55.35% | ✅ IC 0.0526 / 56.03% | ✅ IC 0.2901 / 63.68% |
| `510300.SH` | ❌ IC 0.0088 / 49.90% | ❌ IC 0.0587 / 48.84% | ✅ IC 0.3543 / 52.56% |

**结论（诚实版）**：**个股（600519 / 000858）分层后门禁轻松通过，拖后腿的是宽基 ETF `510300.SH`**——
把波动结构完全不同的 ETF 和个股混在一起池化训练，是 short_term 卡在门槛线上的直接原因。
这正是此前 README §11.5 提出的「按标的分层训练」假设的**实测证据**。

**训练折中性带（`--derive-band`）实测**：short_term 命中率 49.30% → 50.63%、mid_term
47.93% → 49.07%，**有提升但未跨过 52% 线**，且 long_term 略降。中性带较大
（0.15~0.29，丢弃样本较多）说明该池化数据集本身信噪比不足 —— **中性带不是银弹**。

**下一步（建议，非本轮结论）**：按标的分层建模 / 按资产类别分别设门禁，
而非继续在池化口径上调参。这需要人工确认后另开阶段。

### 14.5 合规边界

- 诊断与中性带**不改变门禁结论**，`strategy_gate` 仍按既定阈值判定；
- 中性带是**评估口径**的改进，不进入下单路径，不改变 `RiskManager` / `OrderGenerator` 行为；
- 对外材料**不得**据诊断/中性带宣称"信号已可用"；
- 门禁状态披露要求不变（见 `SALES_PLAN.md` §9.1）。

### 14.6 测试

```bash
python -m pytest tests/test_roadmap_s7.py -q   # 29 passed
python -m pytest tests/ -q                     # 334 passed, 2 failed, 2 skipped
```

新增 `tests/test_roadmap_s7.py`（29 项）：短板识别 / 归一化差距 / `scope=all` vs `any` /
不可用周期不误判 / 不诊断非门禁指标（icir）/ 可复算、中性带样本不足返回 0、
纯噪声不乱设门槛、`min_keep_ratio` 约束、**训练折学常数推升测试折命中率**、
`hit_rate_raw` 对照、默认关闭时口径不变、CLI 契约与签名向后兼容、
`per_symbol` 精简形状、监控报表两分支、诊断文案可读性。

2 例失败是**既有问题**（`test_predictor_utils.py` 依赖被 `.gitignore` 忽略的 `models/`
占位 pickle），改动前后各跑一次做对照，结果完全一致，与 S7 无关。

---

## 十五、Q5 路线进展（信号衰减监控 · 定期报告升级）

对照 `SALES_PLAN.md` §8.2 路线图 Q5「信号衰减监控，IC 时序与重训练预警」。

### 15.1 先说清楚它解决什么问题

Q2 的策略门禁（`python main.py gate`）回答的是**静态**问题：*此刻*的 IC / 命中率是否达标？
结论只有 `gated` / `readonly` 两种。

但运维上真正要命的是**动态**问题。看两个例子：

| 门禁看到的 | 实际含义 A | 实际含义 B |
|-----------|-----------|-----------|
| `|IC| = 0.031`（刚过线） | 稳定在 0.031，可用 | 从 0.09 一路衰减到 0.031，**下周就掉出放行线** |
| `|IC| = 0.028`（差一点） | 一直在 0.01 附近，信号本就弱 | 正从 0.01 回升，**即将达标** |

这两种情况在门禁看来**完全一样**，对使用者的含义却相反。
Q5 就是把「IC 快照」变成「IC 时序」：

```bash
python main.py ic-trend          # 全标的池，落盘 reports/ic_trend.json
```

### 15.2 怎么算的

1. 复用 `ic` 命令的**同一套** walk-forward 序列（`build_walkforward_sequences`），
   保证门禁与趋势**同源** —— 不会出现「监控说衰减、门禁说达标」的口径打架；
2. 在序列上按 `window`（默认 250 样本）切片、`step`（默认 40 样本）滑动，
   每个锚点算一段 IC，得到 IC 时序；
3. 对 IC 时序做最小二乘拟合，得到斜率（`slope_per_step`），并折算为
   `decay_per_step_pct`（以当前 |IC| 为基准的每步百分比变化）；
4. 按当前斜率外推，预估还有几个步长跌破 `strategy_gate.min_ic`（`steps_to_breach`）。

**关键设计**：为什么走 `evaluate_aligned` 而不是自己按收盘价算收益 ——
walk-forward 的**折与折之间时间不连续**，用相邻两行收盘价算 forward return 会跨折错位，
产出「看着像衰减、其实是折缝」的伪趋势。因此趋势监控直接消费门禁算好的
（分数，未来收益）对齐序列。

### 15.3 三种状态与「不外推永远安全」

| 状态 | 判定 | 含义 |
|------|------|------|
| `decaying` | 斜率 ≤ −`min_decay`（默认 0.001/步） | 显著衰减，建议提前安排重训练 |
| `improving` | 斜率 ≥ +`min_decay` | 信号在变强 |
| `stable` | 落在阈值带内 | 无明显趋势（**噪声抖动不算趋势**） |
| `unknown` | 锚点不足 / 样本不足 | 数据不足以判定，**如实沉默，绝不猜** |

`steps_to_breach` 有明确的语义边界：

- 已跌破 → `0`（就是现在）
- 斜率为负且尚未跌破 → 正步数（按线性外推，仅供参考）
- **斜率非负 → `None`**（不外推「永远安全」这种结论）
- 锚点不足 → `None`，且状态为 `unknown`

### 15.4 四个消费口

| 消费口 | 内容 |
|--------|------|
| CLI | `python main.py ic-trend`（JSON 输出 + 落盘 `reports/ic_trend.json`） |
| 监控报表 | `python main.py monitor` 新增「IC 趋势 / 信号衰减（Q5）」章节 + 待处理事项 |
| 日报 / 周报 | `daily-report` / `weekly-report` 新增「📉 信号衰减趋势（Q5）」明细表 |
| API | `GET /api/v1/monitor/ic-trend`（只读，报告缺失时返回 `available=false` 而非臆测） |

监控/日报中的预警分两档：`decaying`（衰减中）与 `near_breach`（预计 N 步内跌破，
N 由 `ic_trend.near_breach_steps` 配置）。两者都是 `warning` 级**趋势预警**，
不是当前故障。

### 15.5 与门禁的职责边界（重要）

> **趋势不改变门禁判定。** 放行与否仍然只看当前 IC / 命中率（`strategy_gate`）；
> 趋势章节只用于**提前安排重训练与特征迭代**，不参与 fail-close 判定。

这样分工的理由：门禁是**准入闸**，必须严进严出、只看可验证的当前指标；
趋势是**运维雷达**，用于提前几周发现退化。把趋势塞进门禁会让放行条件变得
不可解释（「斜率」不是风险指标，而「当前 IC」是）。

### 15.6 配置

```yaml
ic_trend:
  window: 250            # 每个锚点的 IC 观察窗（样本数）
  step: 40               # 锚点之间的步长（样本数）
  min_anchors: 5         # 判定趋势所需的最少锚点数（不足则 unknown，不臆测）
  min_obs: 50            # 单锚点最少有效样本
  min_decay: 0.001       # 显著衰减判定：每步 |IC| 下滑超过该值才算 decaying
  neutral_band: 0.0      # 命中率观望带（与门禁口径保持一致）
  report_dir: "reports"  # 趋势报告输出目录（ic_trend.json）
  near_breach_steps: 3   # 「预计 N 步内跌破门禁线」告警阈值
```

### 15.7 测试

```bash
python -m pytest tests/test_roadmap_q5.py -q   # 33 passed
```

新增 `tests/test_roadmap_q5.py`（33 项）：斜率工具、样本不足/未到期样本沉默、
锚点不足保留 IC 快照但不给斜率、衰减/改善/噪声三分支、`steps_to_breach`
三种语义（0 / 正数 / None）、最近锚点 IC 与直接 `spearman_ic` 可复算、
`evaluate_aligned` 不跨折错位、`min_ic` 取自门禁配置、汇总与契约可序列化
（无 NaN/Infinity）、配置段与取值合理性、CLI 命令与无数据时不落假报告、
监控报表三种分支（正常 / 衰减告警 / 报告缺失）、日报与周报章节、
API 两种状态码分支。

### 15.8 顺带修复的两个遗留卫生问题

这两个是上一轮如实标注、**本轮才处理**的既有问题（与 Q5 功能无关）：

1. `tests/test_predictor_utils.py` 有 2 例依赖 `models/` 下的占位 pickle，
   而该目录被 `.gitignore` 忽略、从未入库 → **干净克隆必然失败**。
   改为用例内调 `scripts.create_timesfm_placeholders.create_placeholders(tmp_path)`
   生成到临时目录（同时把该脚本的生成逻辑抽成可复用函数）。
2. `schedule/daily_run.ps1` 调用了**未定义**的 `Write-LogEntry`（PowerShell 路径下
   「按排期计划自动开发」必抛 `CommandNotFoundException`，日志文件从不落盘）。
   已补上函数定义 —— 与 `run_daily.py` 的日志契约（`run_<date>.json`）保持一致。

## 十六、技术栈

Python 3.10+ · LightGBM · scikit-learn · pandas / numpy · FastAPI + uvicorn · ONNX / onnxruntime · Wind MCP · 可选 TimesFM(PyTorch)

---

## 十七、免责声明

> **本项目仅供学习、交流、研究使用，不构成任何投资建议。**

1. **不构成投资建议**：本软件输出的所有预测结果、信号、报告、可视化图表均仅供学习与研究参考，**不构成任何形式的投资建议、理财建议或交易指令**，不应作为任何投资决策的依据。

2. **历史表现不代表未来收益**：金融市场具有高度不确定性，任何模型的历史回测表现均**不代表未来收益**。过去的胜率、夏普比率、盈亏比等指标不构成对未来业绩的承诺或保证。

3. **风险自负**：使用者在本软件基础上进行的一切投资行为，其风险与后果均由使用者**自行承担**。版权方不对因使用或无法使用本软件所导致的任何直接、间接、偶然、特殊、惩罚性或后果性损害承担任何责任，包括但不限于数据丢失、利润损失、投资亏损等。

4. **专业建议**：实际投资决策请咨询持牌专业金融顾问，并充分了解相关风险。

5. **合规责任**：使用者应确保其对本软件的使用符合其所在地区、机构的法律法规与监管要求，使用过程中的合规责任由使用者自行承担。

---

## 十八、许可证与版权

> **著作权归作者所有，禁止商用。**

本项目采用 **禁止商业用途许可协议（Non-Commercial License）**，详见仓库根目录 [LICENSE](./LICENSE)。

核心约定：

- **允许**：学习、交流、研究、教学、学术、非商业内部评估用途；
- **禁止**：一切商业用途，包括但不限于商业产品/服务、金融机构对外产品、以营利为目的的量化交易或信号售卖；
- **禁止**：转售、出租、分发牟利、再许可、去除版权标识；
- **禁止**：将本软件修改、改编后形成的衍生作品用于商业用途；
- 商用授权须事先取得版权方的**书面许可**。

本软件按"现状"（AS IS）提供，不附带任何明示或默示的担保。

### 联系方式

- **版权方**：yuppiez328（安然）
- **GitHub 仓库**：<https://github.com/yuppiez99999/Financial_Modeling>
- **联系邮箱**：liuzhuoleo@foxmail.com

如需商用授权、合作洽谈或对许可证有任何疑问，请通过上述邮箱联系版权方。

Copyright (c) 2026 yuppiez328（安然）。保留所有权利。