# S20 / H5 投研辅助链路落地结论（Issue #40 · LLM 投研只读接入 + 离线对照）

> 执行方式：按 `schedule/plan.json` S20 排期交付自动任务 T20.2 / T20.3，
> 并完成 T20.1 的候选调研；
> **T20.1 / T20.4**（是否引入 / 是否保留）为**人工检查点，保持 pending 不代签**。
> 全局纪律：`affects_gate` / `affects_signal` 恒为 false —— **结构性不进信号路径**。

---

## 一、交付物

| 任务 | 交付 | 状态 |
|:---:|:---|:---:|
| T20.1 | 候选调研评估（A 股适配 / 依赖代价 / 许可证） | 🔶 人工（结论已出，采纳待定） |
| T20.2 | `src/eval/research_assist.py`：只读附注（挂报告层） | ✅ |
| T20.3 | 离线对照实验（有/无附注的命中率差异） | ✅ |
| T20.4 | 是否保留该链路 | 🔶 人工 |

CLI：`python main.py research-assist [--decided-by ... --reason ...]`
测试：`tests/test_roadmap_s20.py`（22 例，全离线合成数据，不触网）

## 二、方法论要点（为什么这么做）

1. **只读挂载**：`attach_research_note` 只写 `research_note` /
   `research_source` / `research_note_readonly` 三个字段，
   **不触碰** `probability` / `direction` / `signal`；且**不原地修改**输入 dict；
2. **三项准入**：候选必须**同时**满足「A 股适配 + 离线可复现 + 许可证明确」
   才可引入 —— 缺一即不通过。这条规则写进 `build_research_evaluation`，
   不靠人工印象；
3. **离线对照如实止损**：附注**不参与打分** → 差异**结构性为 0** →
   记 `unverifiable`。**不编造「LLM 提升命中率」的故事**；
4. **允许整阶段取消**：排期不是必须做满的清单 —— 若评估不通过，
   整阶段取消是**正确结果**，不是失败。

## 三、边界（不做什么）

- **不进信号路径**：`affects_signal` / `affects_gate` 恒为 `false`；
- **不代签**：T20.4 保持 `pending`；`build_decision` 无人工签字恒为
  `defer` / `cancel`，且 **0 个合格候选时即便签字也不得 `proceed`**；
- **不编效果**：对照不可量化即标 `unverifiable`，并作为 blocker 上报；
- **不引入重型依赖**：不 `pip install` 任何 LLM 框架（评估阶段零引入）。

## 四、候选评估结论（0/3 满足准入）

| 候选 | 许可证 | A 股适配 | 离线可复现 | 判定 |
|:---|:---:|:---:|:---:|:---:|
| TauricResearch/TradingAgents | Apache-2.0 | 无 | 否 | reject |
| hsliuping/TradingAgents-CN | Apache-2.0 系 | 有 | 否 | defer |
| qusong0627/QuantMind | NOASSERTION | 部分 | 否 | reject |

**结论：0 个候选同时满足「A 股适配 + 离线可复现 + 许可证明确」。**

- `TradingAgents-CN` 有 A 股适配，但**离线不可复现**（需在线 LLM 调用），
  与「离线 CI 可跑」纪律冲突 → `defer`；
- 另外两个许可证或适配不达标 → `reject`；
- 离线对照（T20.3）：附注不参与打分 → 差异结构性 0 → `unverifiable`。

**默认动作**：决策单 `verdict=cancel` / `status=pending`。

## 五、与 T20.4 决策相关的材料

**要定什么**：LLM 投研附注链路是否保留。

- **支持保留**：工程接口已就位（只读、零侵入），未来若有合格候选可直接复用；
- **支持取消（默认）**：0/3 候选满足准入；对照不可量化；
  链路本身对现有读数**零贡献**，保留即维护成本；
- **风险**：一旦有人把 `research_note` 接到信号路径上，
  就违反了本轮纪律 —— 结构性防呆（字段命名 + `_readonly` 标记）已做，
  但**纪律最终靠人**。

**建议（供参考，不代做）**：**维持 cancel**。若未来出现满足三项准入的候选，
可按同一决策单重跑 `python main.py research-assist`。

## 六、复现命令

```bash
python main.py research-assist                    # 候选评估 + 离线对照 + 决策单
python -m pytest tests/test_roadmap_s20.py -q     # 22 passed
```
