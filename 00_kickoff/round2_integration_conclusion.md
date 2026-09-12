# 高质量项目集成 2 · 落地结论（Issue #40 · H1~H5 / S16~S20）

> 执行方式：按 `schedule/plan.json` 的 S16~S20 排期，**只做自动可执行任务**
> （`auto_acceptable`），人工检查点（T16.4 / T17.4 / T18.4 / T19.4 / T20.4）
> 一律保持 `pending`，不代签。
> 全局纪律：`strategy_gate` **零改动**，所有阶段 `affects_gate` 恒为 false。

---

## 一、总览

| 阶段 | 主题 | 交付物 | 测试 | 自动任务 |
|:---:|:---|:---|:---:|:---:|
| S16 / H1 | 概率区间校准 | `src/eval/conformal.py`、`src/eval/calibration_ab.py`、`main.py conformal` | 18 passed | T16.1~T16.3 |
| S17 / H2 | 评估防过拟合加固 | `src/eval/cpcv.py`、`src/eval/overfit_audit.py`、`main.py overfit-audit` | 27 passed | T17.1~T17.3 |
| S18 / H3 | 市场状态分层 | `src/eval/regime.py`、`main.py regime` | 16 passed | T18.1~T18.3 |
| S19 / H4 | 推理链路概率校准 | `src/eval/probability_calibration.py`、`src/inference/probability_calibrator.py`、`main.py calibration`、API 字段 | 22 passed | T19.1~T19.3 |
| S20 / H5 | 投研辅助链路 | `src/eval/research_assist.py`、`main.py research-assist` | 22 passed | T20.2 / T20.3（T20.1 调研亦已交付） |

全量回归：**860 passed, 2 skipped**（零回归）。

---

## 二、真实数据实测读数（本机 `data/raw` 缓存 5 只标的池）

> 读数来自本地运行，**只作呈现**：未经多重比较校正不作达标证据；
> 是否落地全部属于对应人工检查点。

### H1 保形预测覆盖率（`reports/calibration/conformal_5d.json`）

| α | 名义覆盖 | 实测覆盖 | 平均区间宽度 |
|:---:|:---:|:---:|:---:|
| 0.05 | 95% | 91.7% | 0.1500 |
| 0.10 | 90% | 90.0% | 0.1270 |
| 0.20 | 80% | 86.7% | 0.0957 |
| 0.30 | 70% | 85.0% | 0.0848 |

结论：**实测覆盖率整体不低于名义值**（保守），α 增大区间变窄、覆盖下降，
口径自洽。`backend=native`（未安装 MAPIE 时如实降级，不冒充）。

口径对照（`ab_5d.json`）：`proba_distance` 与 `interval_width` 两口径在本数据上
**无实质差异**（命中率均差 0.0000）→ T16.4 人工取舍。

多时段复验（`rolling_5d.json`，2 段口径）：**partially_stable** ——
第 0 段命中率随阈值单调不减，第 1 段不成立。即 **T15.3 的单时段结论未能跨时段复现**，
这是本轮最重要的负面结论，如实入库。

### H2 CPCV 与过拟合审计（`reports/overfit_audit.json`、`cpcv_evaluation.json`）

- 组合式路径 **15 条**（N=6, k=2），purge + embargo(5) 审计 **全部通过**（overlaps=0）；
- 路径得分均值 0.0542；收缩指标（`cpcv_shrinkage`，n_trials=8）：
  best_raw 0.1391 → penalty 0.0854 → **best_shrunk 0.0538（仍为正）**；
- 统一试验预算：跨命令累计 **8~10 次**（`ic` / `horizon-scan` / `conformal` / `label-ab` 等），
  报告里自动标注"本次读数已扫描 N 次"，解决 S13 逐命令登记的盲区。

历史读数回算：`label_ab` 的 `hit_rate_z_new=4.398` → p≈6.3e-05，在 1 次试验下仍显著；
其余 S11/S12/S13/S15 报告当前不在 `reports/` 中（缺失即如实标注，不猜）。

### H3 市场状态分层（`reports/regime_stratified.json`）

状态识别：`backend=rules`（未安装 hmmlearn 时降级；安装后 HMM 亦可用，已做状态占比与协方差退化检查）。

| 状态 | 样本 | 命中率 | IC | 命中率 vs 整体 |
|:---:|:---:|:---:|:---:|:---:|
| 牛 | 6520 | 53.42% | -0.0400 | -0.56pp |
| 震荡 | 3230 | 48.72% | +0.0698 | -5.26pp |
| **熊** | 2412 | **62.55%** | **+0.1104** | **+8.56pp** |

**这一轮最有价值的发现**：命中率在熊市与震荡市间差 **13.83pp**。
它解释了 S15「IC 为正但命中率卡线」的时间维度成因 ——
**整池平均把"熊市很准、震荡市不准"平均成了 ~50%**。
状态增量 A/B（T18.3）：`delta_ic = -0.0090`、`delta_hit_rate = 0.0` → **不纳入主线**。

### H4 概率校准（`reports/probability_calibration_5d.json`）

| 口径 | Brier | ECE |
|:---:|:---:|:---:|
| 校准前（基准） | 0.2650 | 0.1042 |
| isotonic | 0.2563 | 0.0386 |
| **platt** | **0.2498** | **0.0025** |

校准后 ECE 从 **0.104 降到 0.0025**（降幅 97.6%），Brier 同步下降 ——
说明现行概率输出**确实存在系统性高估/低估**，校准层有实质价值。
参数已固化到 `models/probability_calibration_<horizon>.json`（按需生成，不入 git）；
API 追加 `calibrated_probability` / `uncertainty` 字段，**既有字段逐字段不变**。

### H5 投研辅助（`reports/research_assist_evaluation.json`）

| 候选 | 许可证 | A 股适配 | 离线可复现 | 判定 |
|:---|:---:|:---:|:---:|:---:|
| TauricResearch/TradingAgents | Apache-2.0 | 无 | 否 | reject |
| hsliuping/TradingAgents-CN | Apache-2.0 系 | 有 | 否 | defer |
| qusong0627/QuantMind | NOASSERTION | 部分 | 否 | reject |

**结论：0 个候选同时满足「A 股适配 + 离线可复现 + 许可证明确」**。
离线对照因"附注不参与打分"而**结构性不可量化** → 记 `unverifiable` 并**止损**，
不编造"LLM 提升命中率"的故事。决策单默认 `cancel` / `pending`（无人工签字不放行）。

---

## 三、边界与纪律（可审计）

1. **门禁零改动**：5 个阶段 `affects_gate` 恒为 false；`strategy_gate` 判据逐字段未变；
2. **人工检查点不代签**：T16.4 / T17.4 / T18.4 / T19.4 / T20.4 全部保持 `pending`；
3. **不做无依据声明**：MAPIE 未装 → `backend=native`；hmmlearn 未装 → `backend=rules`；
   报告缺失 → `available=false` + 原因；一律不冒充、不外推；
4. **无前视**：保形覆盖率只在更晚的复验段读数；CPCV 只做剔除；状态用前向滤波；
   校准器只在校准段拟合、指标只在复验段读数；
5. **结论好坏如实入库**：多时段复验**未复现** T15.3 的单时段结论、状态特征增量**为负**、
   投研链路**整阶段默认取消** —— 均如实记录，未做修饰。

---

## 四、待人工决策（T16.4 / T17.4 / T18.4 / T19.4 / T20.4）

- **T16.4**：区间口径 vs 概率距离口径取舍（含覆盖率下限与信号量权衡）；
- **T17.4**：是否把过拟合概率下限引入门禁；
- **T18.4**：状态分层是否进入信号门禁 / 风控 `withheld` 语义
  （本轮已给出"熊市 62.55% vs 震荡 48.72%"的强证据，值得优先评估）；
- **T19.4**：校准层是否进主推理链路（ECE 0.104→0.0025，证据较强）；
- **T20.4**：投研辅助链路是否保留（当前默认 cancel）。
