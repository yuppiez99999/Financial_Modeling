# S23 + S24 + S25（I3/I4/I5）自动任务交付结论 · 2026-09-13

> Issue #54 · 高质量项目集成 3 轮 ｜ 三个阶段的 auto_acceptable 任务一次性交付，
> T23.4 / T24.4 / T25.4 待人工检查点。全部 `affects_gate=false`，只读报表。

## 一、S23 漂移监控（T23.1 + T23.2 + T23.3）

- **T23.1 准入评估 = 不引入 evidently**：现版（0.7.x）要求 Python ≥ 3.10，
  本项目运行时 3.8；老版本停更线不明 + 重依赖。按候选矩阵预授权走
  **PSI/KS 方法论自研**（`src/eval/drift_monitor.py`，零新依赖；
  PSI 分位 10 桶 + ε 平滑，KS 手写 ECDF，阈值 0.1/0.25/0.2 为业界惯例只读引用）。
- **T23.2 真实数据读数**（3 标的，参考窗 2020-01~2023-03 vs 当前窗 2023-03~2026-06）：
  收益分布（ret_1d/ret_5d）在后半窗**显著漂移**（PSI 0.26~0.73）；成交量基本
  稳定（000408.SZ 例外，PSI 0.517）。含义：用 2020~2023 训练的模型面对
  2023~2026 的市场时，**输入分布已变** —— 这正是「读数会失效」的可观测前兆之一。
- **T23.3 交叉分析**：rolling PSI × 滚动波动率在 3 标的均为**负相关**
  （-0.84/-0.79/-0.53）—— 高波动期相对起点的分布漂移读数反而回落（参考窗
  本身被高波动段主导的口径效应，3 标的冒烟不作为统计证据，如实入库）；
  与 S18 状态转移、命中率漂移的真实交叉**待跑**（hmmlearn 未装、预测审计
  无工件）。
- 命令：`python main.py drift-monitor`；报告 `reports/drift/drift_monitor.json`。

## 二、S24 TreeSHAP 归因（T24.1 + T24.2 + T24.3）

- **T24.1 归因管线**：`src/eval/feature_attribution.py` —— lightgbm 原生
  `pred_contrib=True`（TreeSHAP，零新依赖，无需 shap 包）；全局重要性
  （mean|SHAP| 占比）、时间窗归因漂移（前/后半窗排名变化）、状态×特征交叉
  接口。fail-close：非 LightGBM 模型（含 DummyModel 占位）**明确报错，
  拒绝产出假归因**。
- **重要发现（如实登记）**：`models/*.pkl` 当前为 **DummyModel 占位符**
  —— 真实 LightGBM 训练待跑。因此本阶段真实归因待 `python main.py train`
  产出真模型后运行；机制正确性由 `tests/test_roadmap_s24.py` 合成 LGBM
  证明（9 例：强特征排第一、排名漂移可检、状态交叉可算、fail-close）。
- **T24.2 状态交叉**：接口就绪；真实状态标签依赖 hmmlearn（未装）→
  真实交叉待跑（与 S23 的状态交叉同源缺口）。
- **T24.3 命令**：`python main.py feature-attribution`（真实模型就绪后即用）。

## 三、S25 组合级过拟合审计（T25.1 + T25.2 + T25.3）

- **T25.1 统计自研**：`src/eval/overfit_stats.py`（零新依赖）——
  PSR / DSR（N 次试验期望最大 SR 校正）/ MinTRL / PBO（CSCV 按**时间块**
  对称切分，块级 Σx/Σx² 预聚合，C(16,8)=12870 组合秒级）。
  公式直译 Bailey & López de Prado；pypbo（AGPL）仅公式参考不引入代码。
- **T25.2 实测回算**（9 个配置 = 3 臂 × 3 成本档，S21/S22 的全部净值序列；
  N = 9 + 此前已登记 24 次试验 = 33）：
  - **PBO = 0.011**：IS 冠军在 OS 几乎不掉队 —— 这批配置间没有「选出假冠军」
    的过拟合迹象（因为三臂中置信度臂退化为等权、差异主要由成本档决定）；
  - **但 DSR 全线偏低（0.08~0.24，PSR 0.77~0.92）**：把 33 次试验的选择
    自由度校正后，这些 Sharpe > 0 的置信度大幅缩水；
  - **MinTRL 2173~8382 天 ≫ 现有 1569 天**：现有样本长度不足以把这些
    读数钉到 0.95 置信。
  - 一句话：**「读数为正」但按统一试验预算校正后证据不足** —— 这正是
    S17 纪律想在组合层阻止的「事后看都是信号」。
- **T25.3 报表接入**：`portfolio-backtest` 输出新增 `overfit_audit` 段
  （`--no-audit` 可关），审计读数与统一试验预算联动自动标注。

## 四、待人工检查点（不可代签）

| 检查点 | 问题 |
|------|------|
| T23.4 | 漂移监控是否纳入 health_report 常规输出 |
| T24.4 | TreeSHAP 归因是否纳入例行报告（依赖真实训练模型就绪） |
| T25.4 | 是否引入「DSR 校正后仍为正才采信」的组合读数采信标准 |

## 五、轮次遗留（如实登记）

1. **真实训练模型 + 38 标的池**：S24 真实归因、S21/S22 完整组合回测、
   S25 全池审计，全部等 `python main.py train` 真实训练（当前 models/ 为
   占位符）；
2. **hmmlearn 未装**：S18 状态标签、S23/S24 的状态交叉待跑；
3. 跨阶段证据链：`reports/portfolio/portfolio_backtest.json`（含
   overfit_audit）、`reports/drift/drift_monitor.json`、
   `reports/attribution/`（真实模型就绪后生成）。
