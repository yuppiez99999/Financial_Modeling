# TrendCast Pro · 金融市场预测模型

> 基于 LightGBM 梯度提升树的智能金融市场预测引擎 — 多周期方向预测 + 情感分析 + 自适应学习
>
> 已集成至 [量化策略系统 v5.2.1](../11_量化策略/)，支持 `daily_runner.py --trendcast` 每日信号注入

---

## 一、产品介绍

**TrendCast Pro** 是一款面向金融市场的智能预测引擎，通过 LightGBM 梯度提升树对 **A股股票、期货、外汇** 的未来走势进行多周期方向性二分类预测（看涨/看跌），现已完整集成至主量化策略系统的每日工作流。

### 最新训练评估（2026-06-27，Wind MCP 优先）

| 模型周期 | 准确率 | AUC | 夏普比率 | 盈亏比 | 最大回撤 | 评级 |
|----------|--------|-----|----------|--------|----------|------|
| 短期 5d | 50.7% | 0.51 | 0.22 | 1.03 | -179% | 差（等同随机） |
| 中期 10d | 74.9% | 0.84 | 9.09 | 2.98 | -7% | 优秀 |
| 长期 20d | 74.6% | 0.83 | 8.99 | 2.94 | -7% | 优秀 |

## TimesFM 集成说明（可选依赖）

本项目支持将 TimesFM（金融时序模型）作为可插拔预测器接入，用于替代或与 LightGBM 混合组成 `ensemble`。为了兼容测试与未安装 PyTorch 的环境，项目实现了占位模型（placeholder）与加载回退逻辑，默认不开启 TimesFM 依赖。

要点：
- TimesFM 为可选依赖：真实启用需安装 `timesfm[torch]` 与系统兼容的 PyTorch。Windows 上可能需选择合适的 CUDA/CPU 版本以避免 DLL 加载错误。
- 在测试/CI 环境中建议使用占位模型（已包含于 `models/`），以免强制导入 `torch`。

安装（可选，启用真实 TimesFM）：

```powershell
# 为 CPU 版本 PyTorch（示例，按需改为CUDA版本）
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install timesfm[torch]
```

启用方式：
- 在 `configs/config.yaml` 中，设置 `model.timesfm.enabled: true`（或在运行时通过环境/参数覆盖）。
- PredictionEngine 在加载 `model_type == "timesfm"` 时会优先尝试读取本地占位 pickle（`models/timesfm_<horizon>.pkl`），若不存在则尝试实例化 `TimesFMFinancePredictor`（懒加载 `timesfm`/`torch`）。

占位模型与回退（已实现）：
- 占位文件位置：`models/timesfm_short_term_5d.pkl`、`models/timesfm_mid_term_10d.pkl`、`models/timesfm_long_term_20d.pkl`。
- 占位 pickle 格式：`{'model': DummyModel(), 'scaler': DummyScaler()}`，其中 `DummyScaler.transform` 为恒等映射，`DummyModel.predict_classification` 提供可序列化的替代实现，确保在无 PyTorch 环境下 `PredictionEngine.predict()` 正常工作并通过测试。

测试
- 最小化单元测试文件：`tests/test_predictor_minimal.py`（覆盖 LightGBM-only、TimesFM-path（占位）以及 ensemble）。
- 运行：

```powershell
cd "e:\各种PY程序\16_金融市场预测模型"
python -m pytest tests/test_predictor_minimal.py -q -s
```

如果在本地启用了真实 TimesFM，请在运行前确认 PyTorch 成功安装并能导入（避免 WinError 126 的本机库加载失败）。

其它说明
- Predictor 回退逻辑位于 `src/inference/predictor.py`。如需改为在缺少 scaler 时区分占位类型，可修改该文件中的模型判定分支。
- 用于生成/修复占位 pickles 的脚本：`scripts/create_timesfm_placeholders.py`（创建占位模型）与 `scripts/fix_timesfm_placeholders.py`（将已有 pickles 包装为包含 `scaler` 的 dict）。

如需我把 `predictor.py` 中回退逻辑提取为 helper 或创建更多文档/集成测试，请回复指示。
中长周期模型表现优秀，建议优先使用中期（10日）和长期（20日）信号辅助交易决策。短期模型因5日噪声过大基本等同随机。

### 核心能力

| 能力维度 | 说明 |
|---------|------|
| 市场覆盖 | A股股票（14只核心持仓 + 32只扩展池）、期货、外汇 |
| 多周期预测 | 短期（5日）、中期（10日）、长期（20日）三档 |
| 模型架构 | LightGBM 梯度提升树（生产部署主力模型） |
| 特征工程 | 技术指标（MA/RSI/MACD/布林带）、收益率、波动率、量价特征、宏观指标（CPI/PMI/GDP） |
| 情感分析 | 多源新闻采集（东方财富、新浪财经）+ 情感词典分析 |
| 自适应学习 | 模型性能监控、漂移检测、增量学习、自动重训练 |
| 定期报告 | 每日预测报告、周度预测报告自动生成 |
| 双维评估 | 机器学习指标（Accuracy/F1/AUC）+ 金融指标（胜率/夏普/最大回撤） |
| API 服务 | FastAPI 端口 8800，RESTful 预测接口 |
| 数据源 | Wind MCP（第一优先）→ 腾讯数据 → 模拟兜底 |
| 量化集成 | `trendcast_client.py` 客户端，一键信号注入每日报告 |

### 工作流程

```
数据采集(Wind MCP优先) → 特征工程(技术指标+宏观+情感) → 数据集切分 → 模型训练 → 模型评估 → API服务 → 每日预测信号 → 自适应学习循环
```

### 数据源优先级

```
Wind MCP (P0) → 腾讯数据 (P1) → 模拟数据 (兜底)
```

---

## 二、产品定位

### 目标用户

| 用户群体 | 核心诉求 | 产品价值 |
|---------|---------|---------|
| 量化研究员 | 快速验证策略假设、回测模型效果 | 端到端训练评估流水线，降低工程门槛 |
| 私募/资管投研团队 | 批量预测多标的方向，辅助决策 | 多周期覆盖，一键训练并直接集成策略系统 |
| 金融科技开发商 | 将预测能力嵌入自有产品 | API 服务 + ONNX 导出，无缝对接生产环境 |

### 竞争优势

1. **Wind 数据直连**：对接专业金融数据终端，数据质量远高于免费源，100% 覆盖核心持仓
2. **金融场景原生**：内置夏普比率、最大回撤、盈亏比等金融评估指标，区别于通用 ML 框架
3. **多周期协同**：单一系统输出短/中/长三档预测，适配不同交易节奏
4. **多维度特征**：技术指标 + 宏观经济 + 新闻情感，综合判断市场走势
5. **自适应学习**：模型性能监控与漂移检测，自动触发增量学习或重训练
6. **量化系统紧集成**：`trendcast_client.py` + `daily_runner.py --trendcast` 零摩擦启用
7. **生产就绪**：FastAPI 服务 + 完整日志 + 优雅降级，可稳定运行于每日工作流

### 使用场景

- 每日量化流程：`daily_runner.py --trendcast` 在盘后报告自动注入全持仓 AI 预测信号
- 量化选股：批量预测股票池未来 5/10/20 日涨跌方向，构建多因子信号
- 仓位调整：根据中长周期模型信号辅助再平衡决策
- 定期报告：每日/周度自动生成市场预测报告，辅助投资决策
- 模型研究：快速对比不同特征工程方案在不同周期上的预测表现

---

## 三、项目结构

```
16_金融市场预测模型/
├── configs/
│   ├── config.yaml            # 全局配置
│   └── config_pro.yaml        # 生产配置（Wind优先, 情感分析/宏观指标启用）
├── data/
│   ├── raw/                   # 原始数据缓存
│   ├── processed/             # 预处理后的特征数据
│   └── news/                  # 新闻数据缓存
├── models/                    # 训练好的模型
│   ├── lightgbm_short_term_5d.pkl
│   ├── lightgbm_mid_term_10d.pkl
│   ├── lightgbm_long_term_20d.pkl
│   └── exported/              # ONNX 导出模型
├── reports/                   # 预测报告输出
├── logs/                      # 训练和服务日志
├── src/
│   ├── data/
│   │   ├── collector.py       # 数据采集（Wind MCP / 腾讯 / 模拟）
│   │   ├── preprocessor.py    # 特征工程与数据预处理（含宏观指标）
│   │   ├── sentiment_analyzer.py  # 新闻情感分析器
│   │   ├── indicators.py      # 技术指标计算
│   │   └── quality_gate.py    # 数据质量门控
│   ├── train/
│   │   ├── trainer.py         # LightGBM 训练器
│   │   ├── adaptive_learner.py  # 自适应学习引擎
│   │   └── models/            # 模型定义
│   ├── eval/
│   │   └── evaluator.py       # 模型评估器
│   ├── inference/
│   │   └── predictor.py       # 推理引擎
│   ├── api/
│   │   └── server.py          # FastAPI 服务（端口 8800）
│   ├── export/
│   │   └── exporter.py        # 模型导出
│   ├── report/
│   │   └── daily_report.py    # 每日/周度报告生成器
│   ├── scheduler/
│   │   └── retrain_scheduler.py  # 自动重训练调度
│   ├── audit/
│   │   └── prediction_audit.py   # 预测审计
│   └── notification/
│       └── notifier.py        # 信号推送
├── tests/                     # 测试
├── main.py                    # 主入口（train/serve/predict/daily-report等）
├── requirements.txt           # 依赖
└── README.md
```

---

## 四、快速开始

### 1. 环境准备

- Python ≥ 3.10
- Wind MCP 账号（已配置环境变量 `WIND_API_KEY`）

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python main.py train
```

系统自动执行：数据采集（Wind MCP 优先）→ 特征工程（含宏观指标和情感分析）→ 模型训练 → 模型评估。输出三组 pkl 模型文件和评估报告。

### 3. CLI 预测

```bash
# 单只预测（指定周期）
python main.py predict 300308.SZ --horizon long_term

# 所有周期预测
python main.py predict 600519.SH --horizon all

# 批量预测
python main.py batch 600519.SH 000858.SZ --horizon mid_term
```

### 4. 生成预测报告

```bash
# 生成每日预测报告
python main.py daily-report

# 生成周度预测报告
python main.py weekly-report
```

报告自动保存至 `reports/` 目录，包含市场概览、详细预测、新闻情感分析和投资建议。

### 5. 自适应学习

```bash
# 运行自适应学习引擎（性能监控 + 漂移检测）
python main.py adaptive
```

### 6. API 服务

```bash
# 启动服务（端口 8800）
python main.py serve

# 健康检查
curl http://localhost:8800/api/v1/health

# 单只预测
curl "http://localhost:8800/api/v1/predict/300308.SZ?horizon=long_term"

# 批量预测
curl -X POST http://localhost:8800/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"symbols":["300308.SZ","601088.SH","600276.SH"],"horizon":"long_term"}'
```

### 7. 集成到每日量化流程

```bash
# 终端1: 启动 API 服务
cd "16_金融市场预测模型"
python main.py serve

# 终端2: 运行每日流程 + AI 预测信号
cd "11_量化策略"
python daily_runner.py --trendcast
```

---

## 五、命令行接口

| 命令 | 说明 | 示例 |
|------|------|------|
| `train` | 训练模型 | `python main.py train` |
| `evaluate` | 评估模型 | `python main.py evaluate` |
| `predict <symbol>` | 单只标的预测 | `python main.py predict 600519.SH` |
| `batch <symbols>` | 批量预测 | `python main.py batch 600519.SH 000858.SZ` |
| `export` | 导出模型为 ONNX | `python main.py export` |
| `serve` | 启动 API 服务 | `python main.py serve` |
| `schedule` | 启动自动重训练调度 | `python main.py schedule` |
| `audit` | 生成预测审计报告 | `python main.py audit` |
| `notify <symbol>` | 推送预测信号 | `python main.py notify 600519.SH` |
| `daily-report` | 生成每日预测报告 | `python main.py daily-report` |
| `weekly-report` | 生成周度预测报告 | `python main.py weekly-report` |
| `adaptive` | 运行自适应学习引擎 | `python main.py adaptive` |

---

## 六、配置说明

编辑 `configs/config_pro.yaml` 自定义生产环境配置：

### 数据源配置

```yaml
data:
  source: ["wind", "tencent", "simulation"]  # 数据源优先级
  sentiment_enabled: true                     # 启用新闻情感分析
  macro_enabled: true                         # 启用宏观指标（CPI/PMI/GDP）
```

### 特征工程配置

```yaml
features:
  extended_indicators: true    # 启用扩展技术指标（50+）
  sentiment_enabled: true      # 新闻情感特征
  macro_enabled: true          # 宏观经济指标特征
  technical:
    ma_windows: [5, 10, 20, 60]
    rsi_window: 14
    macd_fast: 12
    macd_slow: 26
```

### 自适应学习配置

```yaml
training:
  adaptive_learning:
    enabled: true              # 启用自适应学习
    drift_threshold: 0.05      # 漂移检测阈值（准确率下降超过此值触发重训练）
    auto_retrain: true         # 自动重训练
    monitor_interval: "daily"  # 监控间隔
```

### 报告配置

```yaml
report:
  output_dir: "reports"        # 报告输出目录
  daily_report: true           # 启用每日报告
  weekly_report: true          # 启用周度报告
```

---

## 七、量化系统集成

主量化策略系统通过 `11_量化策略/trendcast_client.py` 调用 TrendCast Pro API：

```python
from trendcast_client import TrendCastClient

client = TrendCastClient()

# 单只预测
result = client.predict("300308.SZ", horizon="long_term")
# => {"direction": "看涨", "probability": 0.705, "confidence": 0.705}

# 全持仓批量预测
summary = client.get_portfolio_summary(horizon="mid_term")
```

`daily_runner.py --trendcast` 在工作流中自动调用客户端，将预测信号注入每日报告。

---

## 八、技术栈

- Python 3.10+
- LightGBM（梯度提升树）
- FastAPI（API 服务）
- Pandas / NumPy / Scikit-learn
- Wind MCP（金融数据终端）
- uvicorn（ASGI 服务器）
- ONNX（模型导出）

---

## 九、模块详解

### 1. 新闻情感分析 (`src/data/sentiment_analyzer.py`)

从东方财富、新浪财经等来源采集新闻，使用情感词典分析每条新闻的情感倾向，生成 `sentiment_score`、`positive_ratio`、`negative_ratio` 等特征。

### 2. 宏观指标获取 (`src/data/preprocessor.py`)

集成 `MacroIndicatorFetcher` 类，获取 CPI、PMI、GDP 等宏观经济指标，并与历史数据自动关联。

### 3. 自适应学习引擎 (`src/train/adaptive_learner.py`)

- **ModelPerformanceMonitor**：监控模型性能，记录预测准确率
- **IncrementalLearner**：增量学习，基于新数据更新模型
- **AdaptiveLearningEngine**：自适应学习引擎，检测模型漂移并触发重训练

### 4. 定期报告生成器 (`src/report/daily_report.py`)

- **DailyReportGenerator**：每日预测报告（市场概览、详细预测、情感分析、投资建议）
- **WeeklyReportGenerator**：周度预测报告（含周度总结和下周展望）

---

## 十、免责声明

本模型仅供学习和研究使用，不构成任何投资建议。金融市场预测存在不确定性，实际投资决策请咨询专业金融顾问。模型历史表现不代表未来收益，使用者需自行承担投资风险。