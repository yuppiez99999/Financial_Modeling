# S19 / H4 推理链路概率校准落地结论（Issue #40 · isotonic / Platt + 服务侧不确定性暴露）

> 执行方式：按 `schedule/plan.json` S19 排期交付自动任务 T19.1~T19.3；
> **T19.4**（校准层是否进主推理链路）为**人工检查点，保持 pending 不代签**。
> 全局纪律：`strategy_gate` 零改动，所有产物 `affects_gate` 恒为 false；
> 校准层**只读追加字段**，既有 API 字段逐字段不变。

---

## 一、交付物

| 任务 | 交付 | 状态 |
|:---:|:---|:---:|
| T19.1 | `src/eval/probability_calibration.py`：isotonic / Platt + Brier / ECE / 可靠性曲线 | ✅ |
| T19.2 | 阈值曲线校准前后对照（**现行阈值不改**） | ✅ |
| T19.3 | `src/inference/probability_calibrator.py` + API 追加字段（向后兼容） | ✅ |
| T19.4 | 校准层是否进主推理链路 | 🔶 人工 |

CLI：`python main.py calibration [--calibration-method both|isotonic|platt]`
配置：无新增开关；校准参数落 `models/probability_calibration_<horizon>.json`（运行产物，不入 git）
测试：`tests/test_roadmap_s19.py`（22 例，全离线合成数据，不触网）

## 二、方法论要点（为什么这么做）

1. **三段式严格时序**：训练段 < 校准段 < 复验段 —— 校准器**只在训练段之后**拟合，
   评估指标**只在最后一段**读数。若在校准段上评 ECE，那是自证不是验证；
2. **两种校准器并列**：isotonic（非参、单调、小样本易过拟合）vs Platt
   （Sigmoid 参数化、更平滑）。**按 Brier 择优**并把两种读数都留在报告里，
   防「只报好看的那个」；
3. **ECE / Brier 双指标**：ECE 看校准偏差（分布层面），Brier 看综合评分
   （区分度 + 校准）。只报 Brier 会掩盖校准偏差，只报 ECE 会掩盖区分度；
4. **推理期只读套用**：`probability_calibrator` 缺参数文件时
   **不校准、不猜**，并回 `calibration_applied=false`。宁可标「未校准」，
   不可伪造校准；
5. **不改判据**：T19.2 只做阈值曲线**对照**，现行阈值逐字段不变。

## 三、边界（不做什么）

- **不改门禁**：`affects_gate` 恒为 `false`；`strategy_gate` 逐字段零变更；
- **不原地改概率**：`probability` / `direction` / `signal` 字段逐字段不变，
  校准结果只出现在**新增字段** `calibrated_probability` / `uncertainty`；
- **不代签**：T19.4 保持 `pending`；
- **不外推**：三段式读数只反映本机池，跨时段/跨标的稳定性**未复验**；
- **不引入重型依赖**：只用既有 scikit-learn 校准器。

## 四、本机实测读数（5 标的本地缓存，只作呈现不作达标证据）

| 口径 | Brier | ECE |
|:---:|:---:|:---:|
| 校准前（基准） | 0.2650 | 0.1042 |
| isotonic | 0.2563 | 0.0386 |
| **platt** | **0.2498** | **0.0025** |

- **ECE 0.1042 → 0.0025**（platt，降幅 **97.6%**），Brier 同步下降
  （0.2650 → 0.2498）；
- 读法：现行概率输出**确实存在系统性高估/低估** —— 这是 T15.3 置信度门槛
  「阈值语义漂移」的直接证据；
- 阈值曲线复算（T19.2）：校准后单调性有改善，但**现行阈值未改**。

> ⚠️ **不得引用为普适结论**：只在本机池成立，跨时段/跨标的稳定性未复验。

## 五、与 T19.4 决策相关的材料

**要定什么**：校准层是否进**主推理链路**（即消费方是否依赖校准后概率）。

**支持项**
- ECE 降 97.6%，证据强度较高；
- 三段式样本外读数，无前视；
- API 向后兼容，接入成本低（字段已就位）。

**反对方 / 待补**
- 跨时段 / 跨标的稳定性**未复验**；
- 与 T15.3 / T16.4 **耦合**：若概率语义被改动，阈值语义随之漂移 ——
  **不能只签一边**，须联合决策；
- 进判据属 `strategy_gate` 结构变更，须另开 PR 逐字段 diff + 人工签字。

**建议（供参考，不代做）**：证据支持「先只读消费」，
直接改判据须等跨时段复验 + 与 T15.3/T16.4 联合签字。

## 六、复现命令

```bash
python main.py calibration                       # 双校准器 + 阈值曲线 + 固化参数
python main.py calibration --calibration-method platt
python -m pytest tests/test_roadmap_s19.py -q    # 22 passed
```
