# S18 / H3 市场状态分层落地结论（Issue #29 · HMM 状态识别 + 状态内分层评估）

> 执行方式：按 `schedule/plan.json` S18 排期交付自动任务 T18.1~T18.3；
> **T18.4**（状态是否进门禁 / 风控 `withheld` 语义）为**人工检查点，保持 pending 不代签**。
> 全局纪律：`strategy_gate` 零改动，所有产物 `affects_gate` 恒为 false。

---

## 一、交付物

| 任务 | 交付 | 状态 |
|:---:|:---|:---:|
| T18.1 | `src/eval/regime.py`：HMM 市场状态识别（`bull/range/bear`，无前视） | ✅ |
| T18.2 | 状态内分层评估（逐状态 IC / 命中率 / 样本数 + 分得开判定） | ✅ |
| T18.3 | 状态 one-hot 作为特征的增量 A/B（只加一个变量） | ✅ |
| T18.4 | 状态是否进门禁 / 风控 `withheld` 语义 | 🔶 人工 |

CLI：`python main.py regime [--refit-every 20 --no-ab --no-full-sample]`
配置：`model.factors.regime`（缺省 `enabled: false`，状态不进生产链路）
测试：`tests/test_roadmap_s18_h3.py`（34 例，全离线合成数据，不触网）
依赖：`hmmlearn 0.3.3`（BSD-3-Clause，已 `pip install`，登记进 `docs/THIRD_PARTY.md`）

## 二、方法论要点（为什么这么做）

1. **固定 3 态，写死不配置**：`N_STATES = 3` 是常量而非配置项。若做成可扫参数，
   「扫几个 n_states 挑好看的那个」就是 S11/S17 反复要防的**选择自由度回流**；
2. **牛熊命名只是可读化映射**：`name_states` 按**拟合状态均值收益升序**映射
   `bear/range/bull`，报告如实标注 `mapping="state_mean_return_ascending"` ——
   它是统计归纳，不是客观牛熊判定；
3. **无前视是双口径，不是一句话**：HMM 一次 `fit` 会看到整段观测，直接对全样本
   decode 就是前视。因此：
   - `expanding`（**主线，缺省**）：第 t 天只用 `[0, t]` 观测重训
     （`--refit-every` 控制步长，步长内沿用最近模型），`lookahead_prefixed=false`；
   - `full_sample`（**仅对照**）：全样本一次 fit 再 decode，
     `lookahead_prefixed=true`，**不得作达标证据**——保留它只是让口径差异**可测量**；
4. **状态只由市场层收益 / 波动决定**：不消费标签、不消费未来收益；观测两列 =
   日对数收益 + 20 日滚动波动率（rolling std，只回看）；
5. **判定语言与 S12/S13/S17 同构**：A/B 用保守四态（`improved/degraded/mixed/
   insufficient_samples`），**一升一降一律 mixed，不择优**。

## 三、边界（不做什么）

- **不改门禁**：`affects_gate` 恒 false，`strategy_gate` 逐字段零变更；
  状态是否进门禁 / 风控 `withheld` 属 T18.4；
- **不自动落地**：`model.factors.regime.enabled` 缺省 false；
- **不猜**：hmmlearn 缺失 / 观测不足 / 单状态样本 < 30 / 可用状态 < 2 时如实
  `available=false` + 原因，不降级成假标签、不硬凑三段；
- **不虚报因果**：本模块只回答「状态间命中率是否分得开」，**不判定**
  「模型只在某状态有效」的因果结论——那属 T18.4。

## 四、本机冒烟实测（单标的 600519.SH 腾讯真实源 800 行，只作呈现不外推）

- 市场层观测 **781/800** 有效；expanding 重训 **14 次**；状态可用 = True；
- **10d 分层：`differentiated`**（可用状态 2，命中率极差 **0.3816**）：

| 状态 | 样本 | IC | 命中率 |
|:---:|:---:|:---:|:---:|
| range | 177 | +0.2540 | **56.50%** |
| bear | 60 | −0.1792 | 18.33% |
| （全样本基线） | 237 | +0.1345 | 46.84% |

- **10d 状态特征 A/B：`degraded`**（ΔIC = −0.0065，Δhit = 0.0000）——
  单标的下加状态 one-hot 没有改善，如实记录；
- 单标的样本量小、三态里只两态可评估（`bull` 段落在训练区），
  **38 标的池完整结论**待 PR 合并后跑 `python main.py regime`。

**一条值得注意的读数**：`bear` 段命中率 18.33% 远低于随机——在这个单标的上，
模型在低波动/下跌段**方向反着做**。这正是 H3 想解释的「IC 为正但命中率卡线」的
时间维度候选成因，但**单标的不足以支撑结论**，必须全池复跑后再看。

## 五、38 标的池完整结论（待跑）

```bash
python main.py regime                      # expanding 无前视 + 分层 + A/B（默认全池）
python main.py regime --no-full-sample     # 不出有前视对照口径（更快）
```

预期：若「模型只在特定状态有效」成立，`summarize_regime_spread` 应给
`differentiated` 且最优状态样本占比 ≥ 10%；若给 `uniform`，则说明
**时间维度解释力弱**——那同样是有价值的否定结论，如实入库。

## 六、与 T18.4 决策相关的材料

1. 状态分层**分得开 ≠ 该进门禁**：门禁引入状态维度意味着「某些状态不出信号」，
   直接影响信号量与业务可用性，需要你综合 S15 置信度门禁一起判；
2. **A/B 目前是 `degraded`**：把状态 one-hot 当普通特征喂给 LightGBM 没有增量，
   H3 的价值更可能在**评估与风控**（分层读数 / withheld），而不是**特征**；
3. 与 S9 分池正交：S9 回答「谁的池」（资产维度），H3 回答「什么时候」（时间维度），
   两者**不可互相替代**，T18.4 决策时请一并考虑是否需要一个统一的
   「分层放行」结构（那将是 `strategy_gate` 的又一次结构变更，须另开 PR 逐字段 diff）。
