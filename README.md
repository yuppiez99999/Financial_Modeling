# TrendCast Pro · 金融市场预测模型

> 基于 LightGBM 梯度提升树的多周期市场方向预测引擎 —— 技术指标 + 宏观数据 + 新闻情感，配套自适应学习与 REST 预测服务。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io/)
[![API](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com/)

---

## 一、它做什么

对 **A 股股票 / 商品期货 / 外汇** 标的，输出未来 **5 日（短期）/ 10 日（中期）/ 20 日（长期）** 的涨跌方向概率（二分类），并提供：

- 端到端流水线：数据采集 → 特征工程 → 训练 → 评估 → 导出 → 服务 → 报告
- 双维评估：机器学习指标（Accuracy / F1 / AUC）+ 金融指标（胜率 / 夏普 / 盈亏比 / 最大回撤）
- 自适应学习：性能监控 + 漂移检测 + 增量学习 + 自动重训练
- FastAPI 预测服务（默认端口 8800），支持单只与批量预测
- 每日 / 周度预测报告自动生成，预测结果可审计回溯

### 最新训练评估（2026-06-27，Wind MCP 优先）

| 周期 | 准确率 | AUC | 夏普 | 盈亏比 | 最大回撤 | 结论 |
|------|--------|-----|------|--------|----------|------|
| 短期 5d | 50.7% | 0.51 | 0.22 | 1.03 | -179% | 等同随机，**不建议单独使用** |
| 中期 10d | 74.9% | 0.84 | 9.09 | 2.98 | -7% | 可用 |
| 长期 20d | 74.6% | 0.83 | 8.99 | 2.94 | -7% | 可用 |

> 说明：短期模型受 5 日噪声主导，样本外基本无区分度；中长周期表现显著更好，实盘应优先参考 10d / 20d 信号。上述为历史样本内回测口径，不等于未来收益。

### 核心能力

| 维度 | 说明 |
|------|------|
| 市场覆盖 | A 股（14 只核心持仓 + 5 只基准）、8 个商品期货、主要外汇对 |
| 预测周期 | short_term(5d) / mid_term(10d) / long_term(20d) |
| 模型 | LightGBM（主力）；TimesFM、Kronos 为可选/实验性路径 |
| 特征 | MA/RSI/MACD/布林带等 50+ 技术指标、收益率、波动率、量价、宏观（CPI/PMI/GDP）、新闻情感 |
| 数据源 | Wind MCP (P0) → 腾讯财经 (P1) → 模拟数据 (兜底) |
| 出口 | CLI / REST API / ONNX 导出 / 每日与周度报告 |

---

## 二、快速开始

### 1. 安装

```bash
git clone https://cnb.cool/yuppiez328/Financial_Modeling.git
cd Financial_Modeling

python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

要求 Python ≥ 3.10。使用 Wind 数据源时需配置 `WIND_API_KEY`；不用 Wind 时会自动降级到腾讯财经，无需 Key。

### 2. 训练

```bash
python main.py train
```

自动完成 采集 → 特征 → 训练 → 评估，产出 `models/lightgbm_{short,mid,long}_term_*.pkl` 与 `logs/evaluation_report.txt`。

### 3. 预测

```bash
python main.py predict 600519.SH --horizon all      # 单只，全部周期
python main.py predict RB.SHF --horizon long_term   # 期货
python main.py batch 600519.SH 000858.SZ --horizon mid_term
```

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
curl http://localhost:8800/api/v1/health
curl "http://localhost:8800/api/v1/predict/300308.SZ?horizon=long_term"

curl -X POST http://localhost:8800/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"symbols":["300308.SZ","601088.SH","600276.SH"],"horizon":"long_term"}'
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

常用参数：`--horizon {short_term,mid_term,long_term,all}`、`--config <path>`、`--model-type {lightgbm,pytorch_lstm}`、`--host` / `--port`。

---

## 四、项目结构

```
.
├── configs/
│   ├── config.yaml            # 默认配置
│   └── config_pro.yaml        # 生产配置（Wind 优先，启用情感与宏观特征）
├── data/
│   ├── raw/                   # 原始行情缓存
│   ├── processed/             # 特征工程后的数据集
│   └── news/                  # 新闻缓存
├── models/                    # 训练产物（pkl）与 exported/（ONNX）
├── reports/                   # 报告输出
├── src/
│   ├── data/                  # 采集 / 预处理 / 技术指标 / 情感分析 / 质量门控
│   ├── train/                 # LightGBM 训练器、模型定义、自适应学习
│   ├── eval/                  # 双维评估器
│   ├── inference/             # 推理引擎（含 TimesFM 回退）
│   ├── api/                   # FastAPI 服务
│   ├── export/                # ONNX 导出
│   ├── report/                # 日报 / 周报生成
│   ├── scheduler/             # 自动重训练调度
│   ├── audit/                 # 预测审计
│   ├── notification/          # 信号推送
│   └── timesfm_predictor.py   # TimesFM 适配（可选）
├── Kronos/                    # 第三方基础模型源码快照（见下）
├── scripts/                   # 占位模型生成等工具脚本
├── tests/                     # pytest 测试
├── main.py                    # CLI 入口
└── requirements.txt
```

---

## 五、配置

默认加载 `configs/config_pro.yaml`（存在时优先），可用 `--config` 覆盖。

```yaml
data:
  source: ["wind", "tencent", "simulation"]   # 按顺序回退
  sentiment_enabled: true                     # 新闻情感特征
  macro_enabled: true                         # 宏观指标特征

features:
  extended_indicators: true                   # 50+ 扩展技术指标
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

数据源优先级：`Wind MCP (P0) → 腾讯财经 (P1) → 模拟数据 (P6 兜底)`。Wind 不可用时自动降级，不会中断流程。

---

## 六、可选模型后端

### TimesFM（可选依赖）

TimesFM 作为可插拔预测器接入，可替代或与 LightGBM 组成 `ensemble`。**默认关闭**，未安装 PyTorch 时自动使用 `models/timesfm_*.pkl` 占位模型，保证推理与测试可用。

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install timesfm[torch]
```

启用：`PredictionEngine.load_models(model_type)` 支持 `timesfm` 与 `ensemble`（LightGBM + TimesFM 融合），配置段为 `model.timesfm.{context_days,verbose}`；加载逻辑见 `src/inference/predictor.py`：优先读本地占位 pickle，不存在再懒加载真实 `TimesFMFinancePredictor`（懒加载 `timesfm`/`torch`）。

> 注意：`main.py` CLI 的模型类型白名单当前仅 `lightgbm` / `pytorch_lstm`，**`timesfm` 与 `ensemble` 只能通过直接调用 `PredictionEngine` 使用**，经 CLI 会因类型校验而退出。

Windows 上若出现 `WinError 126`（本机库加载失败），通常是 PyTorch 与 CUDA/CPU 版本不匹配，改用 CPU 版 PyTorch 即可。

### Kronos（第三方源码快照）

`Kronos/` 是开源金融 K 线基础模型 [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) 的源码快照（commit `67b630e`，2026-04-13），**非本项目原创**，仅作为后续接入基础模型的实验底座，目前尚未与 `src/` 主链路打通。来源与许可见 `Kronos/THIRD_PARTY_NOTICE.md` 与上游 LICENSE。

---

## 七、与主量化系统的对接

本仓库可独立运行。若需把预测信号注入外部量化工作流，通过 HTTP 调用本服务的 REST 接口即可：

```python
import requests

resp = requests.post(
    "http://localhost:8800/api/v1/predict/batch",
    json={"symbols": ["300308.SZ", "601088.SH"], "horizon": "mid_term"},
    timeout=30,
).json()
```

> 注：早期版本配套的 `trendcast_client.py` 位于外部量化系统侧，不在本仓库；跨系统调用统一走 REST 接口。

---

## 八、测试

```bash
python -m pytest tests/ -q
```

---

## 九、已知限制

- 短期（5d）模型区分度不足，样本外近似随机，不应单独作为交易依据
- 模型评估为历史回测口径，未扣除真实滑点与冲击成本，实盘前需做纸面跟踪
- Wind 数据源需终端/Key，缺失时降级到免费源，特征完整度会下降
- TimesFM / Kronos 路径尚未接入主推理链路

---

## 十、技术栈

Python 3.10+ · LightGBM · scikit-learn · pandas / numpy · FastAPI + uvicorn · ONNX / onnxruntime · Wind MCP · 可选 TimesFM(PyTorch)

---

## 十一、免责声明

本项目仅供学习与研究使用，**不构成任何投资建议**。金融市场预测存在高度不确定性，历史回测表现不代表未来收益，据此操作的投资风险由使用者自行承担。
