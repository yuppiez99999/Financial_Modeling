# 决策源契约与有效性边界（16_ → tradingview / 28）

**当前真相**：16_ 对下游的交付物是 `contract_version = decision-feed/1` 的**只读决策源契约**
（`GET /api/v1/decision/feed`、`python main.py decision-feed`）。

## 一、为什么要有契约层（而不是让下游自己换算）

16_ 的定位是「只出方向与概率，决策权归下游」。但下游落地时每个消费方都自己重写一遍同样的换算，
于是**同一份预测在不同链路上被翻译成不同口径**：

| 环节 | 下游自算的口径 | 16_ 侧同义实现 |
|---|---|---|
| 多周期聚合权重 | tradingview `trendcast_signal_source._aggregate`：`short 0.2 / mid 0.5 / long 0.3` | `SignalEngine.DEFAULT_HORIZON_WEIGHTS`：`0.30 / 0.35 / 0.35` |
| 动作阈值 | `up_prob > 0.6 → BUY`、`< 0.4 → SELL`（就地硬编码） | 门禁 `min_hit_rate 0.52` / 置信度子集 thr ∈ [0.2,0.3] |
| 置信度口径 | `|up_prob − 0.5| × 2`（无命名） | `confidence_from_proba` / `uncertainty_from_probability` |
| 单周期方向+概率 | 自行判断 `direction == "看涨"`，否则取 `1 − p` | 无（此前未出口） |

**结论**：口径必须由生产方显式出口。契约层的核心字段 `net_up_probability` 就是
「每周期净看涨概率」——下游**不再需要判断 direction 字符串**，
`P(up)` 与「方向 + 该方向概率」不再可能被读成相反结论。

## 二、契约字段（增量，原始字段逐字段保留）

- `horizons.<h>.net_up_probability`：净看涨概率；`direction` 看跌时 = `1 − p`
- `horizons.<h>.calibrated_probability` / `calibration_applied` / `uncertainty`：S19 校准层
- `aggregate`：多周期加权综合分（`composite_score` ∈ [0,1]、`composite_signed` ∈ [-1,1]）、
  `coverage`、`missing_horizons`、`weights_used` —— **缺失周期不补 0.5**
- `advisory`：`advisory_consumable` / `recommended_threshold` / `calibrated_consumable`
- `audit`：已回溯命中率摘要（含 24h 窗口）
- `analytics`：置信度分档 × 已实现收益的联合分布 + 门槛扫描 + 保守判定
- 结构性纪律字段：`position_role = observer`、`affects_gate = false`、`advisory_only = true`

## 三、有效性边界（本轮最重要结论，如实入库）

用真实日K + 本地真实训练的 LightGBM，按每 5 交易日一个锚点回填已实现收益
（15584 锚点，26 标的，3 周期；`reports/decision_feed/analytics_backfill.json`）：

| 置信度档 | 覆盖率 | 命中率 | 平均已实现收益 |
|---|---|---|---|
| [0.0,0.1) | 22.7% | 54.0% | **+1.92%** |
| [0.1,0.2) | 21.0% | 58.9% | **+1.69%** |
| [0.2,0.3) | 19.1% | 61.7% | **+1.08%** |
| [0.3,0.5) | 25.4% | 71.1% | **+0.51%** |
| [0.5,0.7) | 10.3% | 82.9% | **−0.49%** |
| [0.7,1.0] | 1.4% | **98.2%** | **−0.78%** |

**置信度越高 → 方向命中率越高，但平均已实现收益越低（单调反向）。**
杠杆口径下同样反向（`mean_return / mean_abs_return`：低档 +0.355、高档 −0.080）。
三周期判定均为 `ineffective`。

机制解释：模型在趋势加速段最自信，而该段恰是**短期已过热、后续均值回复**的位置
（极端档 mean_abs_return 0.097 vs 低档 0.054，波动近乎翻倍而方向仍对 —— 典型的高位高波动）。

**对下游的含义**：
- 高置信 ≠ 可采信；**「命中率高」不能单独作为采信依据**，必须与收益联合判定
  （判定函数强制两条腿都过，见 `decision_analytics.verdict`）；
- 建议门槛因此**不能**按「命中率最高」来挑；本契约的 `recommended_threshold=0.2`
  是「已配置现行值」（S15/G5 口径，T15.3=defer），不是本层择优结果；
- 结论为 `ineffective` 时，下游应继续把信号当**只读观测 / 反向参考 / 风险预警**，
  不得据此放大仓位。

**边界声明**：锚点间有重叠（每 5 交易日取值、5/10/20 日视界），水平读数不可当独立样本；
但方向性结论在重叠下依然成立且更强。该结论**不改变** 16_ 任何门禁判定
（`affects_gate=false`），是否据此调整门槛属人工检查点。

## 四、踩坑与解法（contains）

- **contains 推理特征位置截断会静默错位**：`_align_features` 在「特征列数相同但列序/列集合不同」时
  **不报警不报错**，把 A 列值喂给期望 B 列的模型，产出看似正常的错概率。现实触发路径：
  `evaluate_models.build_supervised` 的列集合（含 `_symbol` / `_fwd_ret`）与推理期
  `FeatureEngineer.get_feature_columns` 不逐字相同。
  解法：`_align_by_feature_names` —— 模型有 `feature_name_` 时按名取列并按模型顺序排列，
  缺列 0 填充 + WARNING 留痕；无特征名回落位置逻辑。守卫：`tests/test_predictor_feature_alignment.py`。
- **contains 共享记账接口字段口径不同会静默写坏记录**：下游按逐字段 dict 送预测，
  与 `record_prediction(prediction_dict)` 口径不同 → 被写成 `symbol=None` 的坏记录，
  不报错也无法分辨来源。解法：`PredictionAudit.record_prediction_v2` 关键字段缺失 fail-loud，
  来源显式入库。守卫：`tests/test_audit_record_v2.py`。
- **contains 0.5 不能既当"中性读数"又当"数据坏了"**：契约层所有换算在缺失/非有限值上一律 `None`，
  只有真实算出来的 0.5 才是中性读数（`net_up_probability` / `aggregate` 同此纪律）。

## 五、下游接入方式

```bash
# 服务态（推荐：与 28 现有的 :8800 链路同源）
curl "http://127.0.0.1:8800/api/v1/decision/feed?symbols=300308.SZ,510300.SH"

# 离线管道态（tradingview scripts/ 逐行读标的清单的用法）——离线/服务态**逐字段一致**
python main.py decision-feed --symbols-file ~/positions.txt --stdout
```

原 `/api/v1/portfolio/summary` 亦已自动追加决策字段（向后兼容，旧消费方零改动）。
