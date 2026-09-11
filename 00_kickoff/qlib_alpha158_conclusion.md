# Alpha158 因子增量验证结论（S13 / G3）

> 本文件记录 **qlib Alpha158 因子集** vs **现行特征集（含 15 因子口径）** 的
> 同口径对照实验结果。按 Issue #29 集成方案 G3 的要求：**结论不管好坏都如实入库**。
>
> 复算命令：`python main.py qlib-ab`
> 产物：`reports/qlib_ab.json`

## 一、对比设计（保证只差特征集）

| 维度 | 口径 |
|------|------|
| 数据 | 配置内启用标的池（缓存行情） |
| 标签 | **现行 `target_{h}d`**（固定窗口口径；不用 S12 三重障碍新标签 —— 一次只变一个变量，否则增量不可归因） |
| 切分 | 同一组 walk-forward 折（严格时序，测试折在训练折之后） |
| 模型 | 同一 LightGBM 配置（`configs/config.yaml`） |
| 样本 | 两臂完全同一批样本、同一组折 |
| 变量 | **仅特征集**：基准臂 = 现行特征（约 20 列）；对照臂 = 现行特征 + `factor_a158_*` 5 列（Alpha158 族压缩因子） |

## 二、Alpha158 因子实现口径

- `integrations/qlib/alpha158.py`：纯 pandas 复现 qlib Alpha158 的 98 列
  （8 K 线形态项 + 18 滚动项 × 5 窗口），**不引入 qlib 运行时依赖**（CI 离线可跑）；
- `src/factors/qlib_factor_provider.py`：把 98 列按经济含义压缩为 5 个族因子
  （momentum/trend/volatility/volume/reversal），与 `FactorLibrary` 的
  `factor_*` 命名空间隔离（`factor_a158_` 前缀），输出有界 (-1,1)；
- `integrations/qlib/data_layer.py`：行情缓存 → qlib `.bin` 列存（T13.1），
  转换为单向离线动作，不改现有数据路径；
- 全部算子只用 rolling/shift（**只用历史**），测试含篡改尾部价格的无前视硬校验。

## 三、实验结果

**（待真实 26 标的池数据实测后填写 —— 本文件由 `python main.py qlib-ab` 的
实测数字更新；合成数据 smoke 验证只证明链路正确，不作为结论依据。）**

| 周期 | 基准 IC | 基准命中率 | +A158 IC | +A158 命中率 | ΔIC | Δ命中率 | 判定 |
|------|---------|-----------|----------|-------------|------|---------|------|
| 5 日 | — | — | — | — | — | — | 待测 |
| 10 日 | — | — | — | — | — | — | 待测 |
| 20 日 | — | — | — | — | — | — | 待测 |

## 四、结论（如实、保守）

1. **判定口径**：`improved` 需 IC 与命中率**同向**变好（与 S12 label-ab
   同一保守标准）；任何一臂退化 → unavailable，不猜。
2. **不自动纳入主线**：本命令 `affects_gate` 恒为 False，`model.factors`
   配置一个字不动。是否把 `factor_a158_*` 纳入特征集由 **T13.4 人工检查点**
   决定（README §18A 排期表已列）。
3. **需要人工复核的点**（纳入主线前必须回答）：
   - 增量是**稳健**还是**单次噪声**？需换随机种子 / 折数复算；
   - 5 个族压缩因子是否应该按族选择性引入（而非全量 5 列）？
     这会引入选择自由度，须登记试验次数（`python main.py trials`）；
   - 若纳入，`model.factors.qlib_alpha158.enabled` 置 true 后需重做
     泄漏审查（`00_kickoff/leakage_checklist.md`）。

## 五、边界声明

- 本验证**不改变**：`data.prediction_horizons`、`strategy_gate`、现行特征集、
  `FactorLibrary` 的任何行为；
- `factor_a158_*` 因子默认**不进生产特征集**（`qlib_alpha158.enabled=false` 缺省），
  仅 `qlib-ab` 命令内临时计算；
- 第三方来源已在 `docs/THIRD_PARTY.md` 登记（microsoft/qlib, MIT, 表达式级对齐复现）。
