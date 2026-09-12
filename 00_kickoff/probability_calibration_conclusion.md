# H4 / S19 · 推理链路概率校准结论（isotonic / Platt）

> 本文件是**证据与决策材料**，不是决策。
> 「校准层是否进主推理链路」属 **T19.4 人工检查点**，NPC **不代签**。
> 全局纪律：`strategy_gate` 零改动，本阶段 `affects_gate` 恒为 `false`。

---

## 一、要解决的问题

S15 的置信度门槛（`thr ∈ [0.2, 0.3]`）吃的是**原始概率** `p`。
若 `p` 本身系统性高估/低估，**阈值语义就会漂移** ——
「0.3 的置信」在不同时间/标的上可能意味着完全不同的真实胜率。
本阶段补的是这个前提：把「模型说的概率」变成「可以当概率用的概率」。

## 二、方法（无前视口径）

- 校准器接在 LightGBM 输出之后：`isotonic`（等渗回归，非参、单调）与 `platt`（Sigmoid，两参）；
- **严格分段**：校准器**只在校准段拟合**，Brier / ECE **只在复验段读数** ——
  同一段既拟合又读数即为前视，本实现结构上禁止；
- 指标：`Brier`（越小越好）+ `ECE`（期望校准误差，可靠性曲线与对角线的加权距离）；
- 落盘：`reports/probability_calibration_<h>d.json`；
  参数固化到 `models/probability_calibration_<horizon>.json`（按需生成，不入 git）。

复现命令：

```bash
python main.py calibration                       # 缺省 both：isotonic + platt 都跑
python main.py calibration --calibration-method platt
```

## 三、真实读数（本地 `data/raw` 缓存池，三周期）

| 周期 | 口径 | Brier | ECE | 校准样本 |
|:---:|:---:|:---:|:---:|:---:|
| **5d** | 校准前 | 0.2650 | 0.1042 | —— |
| | isotonic | 0.2563 | 0.0386 | 3040 |
| | **platt** | **0.2498** | **0.0025** | 3040 |
| **10d** | 校准前 | 0.2713 | 0.1258 | —— |
| | isotonic | 0.2619 | 0.0737 | 3021 |
| | **platt** | **0.2523** | **0.0463** | 3021 |
| **20d** | 校准前 | 0.2402 | 0.0710 | —— |
| | isotonic | 0.2420 | 0.0577 | 2982 |
| | **platt** | **0.2394** | **0.0530** | 2982 |

**读数（如实呈现，好坏都记）**

1. **三周期 ECE 全部下降**，platt 一律优于 isotonic（按 Brier 选优 → `best_by_brier=platt`）：
   - 5d：ECE 0.1042 → **0.0025**（−97.6%）；
   - 10d：0.1258 → 0.0463（−63.2%）；
   - 20d：0.0710 → 0.0530（−25.4%）。
2. **改善幅度随周期递减**（5d 最强、20d 最弱）；
3. 20d 的 isotonic **Brier 反而略升**（0.2402 → 0.2420）—— 非参校准在样本量最少的周期上
   出现过拟合式抖动。**这正是 T19.4 不能只看 5d 的原因**；
4. 可靠性曲线显示**高概率端系统性高估**：5d 的 `[0.6,0.7)` 桶预测 0.644 / 实际 0.485、
   `[0.8,0.9)` 桶预测 0.830 / 实际 0.627（gap 约 −0.16 ~ −0.20）——
   即**模型「很自信」时反而没那么准**，与「IC 为正但命中率卡线」现象方向一致。

**结论：现行概率确实存在系统性高估/低估，校准层有实质价值** ——
但证据强度**随周期与口径而变**，不能只看最好的那一格。

> ⚠️ 只作呈现：读数来自单池单次，未做多重比较校正；
> 它是**决策材料**，不是「已达标」的结论。
> 复现：`python main.py calibration`（落盘 `reports/probability_calibration_{5,10,20}d.json`）。

## 四、§B 决策读数（`calibration-ablation`）

**为什么 §A 与 §B 必须分开看**：T19.4 要定的是"要不要进**主推理链路**"，
其判据是**决策读数**（命中率 / IC / 阈值子集），不是概率质量。
二者可以完全脱节 —— 校准是**单调映射**，因此：

> 全样本口径下方向命中率与 AUC **恒等**；
> 差异只可能出现在**阈值子集**口径上 —— 而那正是 T15.3 想让门禁依赖的东西。

`calibration-ablation` 的对照纪律：

- 三条腿共享**同一个**分类器与同一段校准样本，只变"概率是否被校准"；
- 逐指标并排：门禁点命中率（clf=no_call 口径）/ IC / AUC / Brier / ECE /
  阈值子集命中率；
- **一升一降不择优、不合成总分**；负面读数如实入库；
- 报告内含 **AUC 不变性校验**（校准是单调映射，AUC 变了说明装配有问题）。

> 复现：`python main.py calibration-ablation`
> 落盘：`reports/calibration/calibration_ablation.json`
> ⚠️ CI 离线环境无真实行情，`reports/` 不入 git —— 该文件按需生成，
> 缺失时视为证据未生成（不冒充）。

---

## 五、待人工决策（T19.4）

**要你定什么**：校准后的概率是否成为**主推理链路**的默认语义 ——
即 ① 门禁阈值吃校准概率 ② `/api/v1/portfolio/summary` 与 `/signal`
的 `probability` 字段直接给校准值（`calibrated_probability` 从「附加字段」升为「主字段」）。

**现状默认**：`model.factors.probability_calibration.enabled=false`；
API 的 `calibrated_probability` / `uncertainty` 是**追加字段**，
`probability` / `direction` / `signal` **逐字段不变**（向后兼容已由 `tests/test_roadmap_s19.py` 钉死）。

**签 / 不签的后果**

- **不签**：概率语义维持现状，改造成本为 0；代价是「校准证据已出但没用上」；
- **签**：证据只是证据 —— 真改链路仍需**另开 PR 逐字段 diff + 人工签字**，
  且必须同步处理下方耦合项。
  ⚠️ 若签，**建议 platt + 三周期统一**：5d 证据最强并不代表该口径在 20d 也成立
  （20d isotonic 的 Brier 反而升），逐周期挑口径就是新的选择自由度。

**⚠️ 耦合提醒（不要单签这一项）**

T19.4（概率语义）与 **T15.3**（阈值语义）、**T18.4**（状态与概率的关系）
操作的是同一条「概率 → 阈值 → 信号」链：

- T15.3 已把门禁**结构与阈值**绑在一起（双指标，`thr ∈ [0.2, 0.3]`）；
- 校准层一旦成为默认语义，**同一组 thr 的物理含义就变了** ——
  原先 `thr=0.3` 对应的子集 ≠ 校准后 `thr=0.3` 对应的子集（覆盖率和命中率都会变）；
- 因此三项**建议一起看、一起签**，否则阈值语义会漂移。

**落法（供参考，不代做）**

```bash
python main.py calibration --calibration-method platt   # 生成报告 + 固化参数
python main.py confidence-holdout                       # 用新概率重算保留期曲线
python main.py confidence-gate                          # 再看候选阈值（覆盖/命中权衡）
```

## 六、边界与纪律

1. `affects_gate` 恒为 `false`，`strategy_gate` 判据逐字段未变；
2. 校准器**只读**套用：`src/inference/probability_calibrator.py` 在参数文件缺失时
   如实降级为原始概率（不猜、不外推）；
3. T19.4 **不得标 `completed`**；2026-09-12 经用户确认（Issue #40）落定为
   `confirmed`（决策 `defer`：暂不进主推理链路），带签署字段（签字，非代签）；
4. 本阶段自动任务（T19.1/T19.2/T19.3）已交付，收口字段见
   `schedule/plan.json` → `stages[S19].closing`。
