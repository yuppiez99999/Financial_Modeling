# 高质量项目集成 2 · 候选评估矩阵（Issue #40 · H1~H5）

> 检索范围：CNB（`cnb search list-public-repos`，关键词 量化 / 金融 / 因子 / 回测 / 多智能体 等）
> + GitHub API（关键词 20+ 组，按 star 排序取前列）。
> 本文件是**选型依据**，不是成果汇报：结论以「导入本项目的增量」为准，
> 参考 star 数只作成熟度信号，不作价值证据。

## 一、检索结论（平台侧现状）

| 平台 | 观察 | 对本项目的影响 |
|------|------|----------------|
| CNB | 金融/量化类公开仓库**极少**，命中的多为模型量化（GGUF 权重）、课程仓库、ComfyUI 整合包；相关度最高的是 `quantskills-cn/*` 只读镜像（源在 GitHub）与 `8888/github.com/*` 的 GitHub 镜像 | CNB 侧无「原生可参考」的高质量量化项目；镜像仓库建议**回源到 GitHub 评估**，不直接依赖镜像 |
| GitHub | 量化/金融 ML 生态成熟，候选充足 | 作为主要候选来源 |

> 注：`quantskills-cn/skill-factor-alpha191-alpha101`、`agent-alpha-portfolio-guardian` 等
> 为「只读镜像，源仓库在 GitHub」，按仓库自述应前往源仓库提交与反馈 —— 故本轮统一按 GitHub 侧项目评估。

## 二、候选清单与筛选（GitHub）

### 2.1 直接淘汰（不进入本项目）

| 候选 | star | 淘汰原因 |
|------|------|----------|
| `google-research/timesfm` | 32.2k | 本项目已有 `src/timesfm_predictor.py` 实验路径；S5~S15 结论指向**模型族不是瓶颈**，重复引入无增量 |
| `microsoft/qlib`（运行时） | 48.5k | S13 / G3 已做 **Alpha158 表达式级对齐复现**，明确「不引入 qlib 运行时依赖、不复制源码」；引入完整运行时违背零重型依赖纪律 |
| `kernc/backtesting.py` / `ricequant/rqalpha` / `hikyuu` / `zvt` | 3.5k~9k | S1 已落地 vectorbt 回测层，`eval_criteria.md` 已定口径；换回测框架属重复建设且会撼动既有评价量尺 |
| `AI4Finance-Foundation/FinGPT` / `FinRobot` | 8k~21k | 大模型微调/Agent 平台，依赖重型（torch + 权重 + 算力），且与「只读信号源、离线 CI 可跑」纪律冲突 |
| `TauricResearch/TradingAgents` | 104k | 已在下游以 A 股适配版形态存在（见 H5）；原版不含 A 股数据适配 |
| `achillesrasquinha/bulbea` / `tsfresh` 类特征库 | 0.2k~2.3k | S12 已用**正交对照实验**否掉纯特征扩充（无显著增量），重复做属已知无效方向 |
| `mpquant/Ashare` / `1nchaos/adata` / `hello245m/free-stockdb` | 2.5k~5.2k | 数据源类；S14 已把 akshare 升 P1 并打开期货/外汇，数据通道已闭环，再加同类源边际为零 |
| `rxiv`/`Empyrical` 类绩效库（`quantstats`） | 7.6k | 已有双维评估（ML + 金融指标）与 `eval_criteria.md` 量尺；引入会引入口径分叉 |

### 2.2 入选（生成排期 H1~H5）

| 阶段 | 候选项目 | star / 许可证 | 解决本项目什么问题 | 现有卡点证据 |
|:---:|------|------|------|------|
| **H1** ✅ | `scikit-learn-contrib/MAPIE`（+ 已登记的 `Nixtla/neuralforecast`） | 1.6k / BSD-3-Clause | 给概率**覆盖率保证**；把 S15 就绪但未实装的 `confidence_from_interval` 真正喂上数据 | T15.3 遗留：「保留期只有一个，多时段滚动复验是下一轮值得做的验证」「neuralforecast 概率区间机制已就绪但未实装」。**已实装（2026-09-12，S16/T16.1+T16.2）**：`src/eval/conformal_probability.py` + `python main.py conformal-interval`，MAPIE 1.5.0 split conformal（LAC）；结论 `00_kickoff/conformal_interval_conclusion.md`，对照判定 mixed（置信分粒度限制），口径取舍待 T16.4 人工定稿 |
| **H2** | 方法论引入（Advances in Financial ML · CPCV/DSR），不引入重型包 | 方法论 / MIT 级 | 把「选择自由度」从事后补记变为流水线内建；检验 S11~S15 结论在校正后是否仍成立 | S11 只做了「多重比较校正」一次性检查；S15 明确「阈值读数未经多重比较校正，不得直接引用为达标证据」 |
| **H3** | `hmmlearn`（HMM 市场状态识别） | 成熟 sklearn 风格库 | 解释「IC 为正但命中率卡线」的**时间维度**成因；与 S9 资产维度分池正交 | S15 结论二：「IC 与命中率脱节在收敛搜索后依旧存在」；README 已知限制提出「按标的分层」，缺时间状态维度 |
| **H4** | `scikit-learn` 校准器（isotonic/Platt） | 既有依赖 | 让「置信度」建立在**校准过的概率**上，而非未校准的原始输出 | S15 置信度机制建立在 `|p−0.5|×2` 上，p 本身未做校准验证（Brier/ECE 缺失） |
| **H5** | `TauricResearch/TradingAgents` / `hsliuping/TradingAgents-CN` / `qusong0627/QuantMind` | 104k / 31.7k / 1.4k | 投研辅助视角（基本面/事件/情绪）互补 | 优先级最低：H5 明确**只读、不进信号路径**，评估不通过则整阶段取消 |

## 三、选型原则（为什么是这五个）

1. **不再加模型族、不再单纯加特征**：S5/S12/S13/S15 四轮结论一致指向瓶颈不在模型与特征侧；
2. **只做「能让现有读数更可信」的事**：H1/H2 加固评估与概率语义，H3/H4 拆解「命中率卡线」的成因；
3. **零重型依赖、CI 离线可跑**：全部候选均为 sklearn 风格轻量库或方法论引入，合成数据测试不触网；
4. **一律只读、绝不进决策路径**：门禁 `strategy_gate` 在本轮内**零改动**，所有 `affects_gate` 恒为 false；
5. **允许整阶段取消**：H5 若 T20.1 评估不通过则取消 —— 排期不是必须做满的清单。

## 四、许可证与登记

- MAPIE：BSD-3-Clause；hmmlearn：BSD-3-Clause；scikit-learn：BSD-3-Clause；
  neuralforecast：Apache-2.0；TradingAgents 系：Apache-2.0 / 部分 NOASSERTION（需逐项核）。
- 引入时按 `docs/THIRD_PARTY.md`「append 风格、不删历史」同步登记来源与版本；
  未实际 `pip install` 前不得声称已引入运行时。
- **H1/MAPIE 已完成运行时登记**（mapie 1.5.0，2026-09-12）；H2~H4 仍为规划期登记。

## 五、对齐既有纪律

- 结论不管好坏如实入库（S12/S13/S15 既有做法延续）；
- 每阶段须有 `tests/test_roadmap_s16.py` ~ `test_roadmap_s20.py` 对应守卫测试（离线合成数据）；
- 人工检查点（`manual_checkpoint`）不可自动通过：T16.4 / T17.4 / T18.4 / T19.4 / T20.1 / T20.4。
