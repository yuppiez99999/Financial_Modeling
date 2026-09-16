# TradingView 交付（图片信号卡 + Pine 数据层）

**当前真相**：16_ 对 TradingView 的交付物是 `python main.py tv-export` 一次产出的
**图片信号卡（PNG）** 与 **Pine 外挂数据层（`tv-pine/1`）**，两者都只是
`decision-feed/1` 契约的**只读投影**，不做二次换算。

## 一、为什么是 PNG + JSON，而不是「再发一份接口文档」

下游要的是**能被读进去的东西**。TradingView 侧只有两条近原生入口：

| 入口 | TradingView 侧怎么用 | 本项目的交付物 |
|---|---|---|
| 图片导入 | 客户端读**一张静态 PNG** | `signals/<symbol>.png` —— 单图承载三周期净看涨概率 / 综合分 / 置信度 / 门禁结论 / 采纳权重 / 锚点收益条 |
| `request.seed` | Pine 读一个**变量 × 时序**的 JSON 表 | `pine/trendcast/<symbol>.json` —— `ret_*` 列 + 列字典 |

平台**不做二次渲染**，所以图片必须是真 PNG；Pine 也只能读表结构，所以 JSON 必须带列字典。
两件事都不是"风格选择"，是消费方的硬约束。

## 二、图片信号卡（`src/export/tv/signal_card.py`）

### 像素即契约

卡片上的每个数值都来自同一份 `decision_feed` 产出，**逐字段相等** ——
不允许图上显示 0.586 而 JSON 里是别的数。守卫：`tests/test_tv_export.py`
用契约值反查卡片元数据。

### 元数据不靠肉眼（tEXt 块）

同一张 PNG 的 `tEXt` 块里放机器可读内容，中文按 **UTF-8** 落盘
（TradingView / Chromium / ImageMagick / PIL 均按 UTF-8 解 `tEXt`）：

| 键 | 内容 |
|---|---|
| `signal_contract` | 契约摘要 JSON（读数 + 纪律字段 + 锚点统计） |
| `anchors_json` | **全精度**锚点序列（日期 / 分数 / 命中 / 已实现收益） |
| `Title` / `Description` | 平台预览里直接可见的一行说明 |
| `Source` | `TrendCast Pro decision-feed/1` |
| `Boundary` | `position_role=observer; affects_gate=false; advisory_only=true; ...` |
| `AnchorEvidence` | `validated=false; backfilled history; not an independent sample` |

### 零依赖 PNG 编码器（`src/export/tv/png_writer.py`）

本项目运行时没有图像库。为一张信号卡引入 Pillow 不划算，因此按 PNG 规范直译：
8bit truecolor / `filter=0` / zlib，含 5×7 点阵字库与 Bresenham 折线。
越界绘制**裁剪**而不抛错（报表产物不能因多画一个像素打断交付链路）。

## 三、Pine 数据层（`src/export/tv/pine_json.py`）

```pine
//@version=6
import TradingView/ta/8 as ta
var matrix<float> m = request.seed("TRENDCAST_510300_SH", "trendcast/510300.SH.json")
if m.size() > 0
    float comp = m.get(1, m.rows() - 1)     // ret_composite
    plot(comp * 100, "composite %", color.blue)
```

三条结构纪律：

1. `ret_*` 列**优先**（`ret_composite` / `ret_short_term` / `ret_mid_term` /
   `ret_long_term` / `ret_confidence` / `ret_coverage` / `ret_calibrated_composite`）；
2. 每列都有对应 `columns[]` 条目，`id` 与列名**逐字一致**（Pine 对齐全靠它）；
3. **不出口可交易字段**：`is_trade` / 仓位 / 权重一律不在列里 ——
   本层用途是「把 16_ 工厂搬到同一张图上做对照」，不是让 TradingView 下单。
   纪律写在 `meta` 里而不是写在文档里，下游图里就能看见 `affects_gate: false`。

合约缺失的周期**不写列值**（`None`），不补 0.5 —— 与契约层同一条纪律。

## 四、锚点回填（`src/eval/anchor_backfill.py`）

回答「这条信号值不值得看」需要**分数 × 已实现收益**的联合分布，而审计记录
依赖系统在线跑过一段时间（冷启动 / 换池后为空）。本模块补这一段：**离线回填**。

**无前视纪律（唯一红线）**：

- 特征行只取到 `t`（`FeatureEngineer.transform` 全部指标都是滚动/后视窗）；
- 结果只在 `t+h` 已收线时回填（`close[t+h]/close[t]-1`），未到期标 `pending` 且**不参与**统计；
- 模型是**当下这一个**（不逐锚点重训）→ 所有读数标 `validated=false`，卡片上显示 `NOT VALIDATED`。

`step < horizon_days` 时锚点视窗重叠，水平读数不可当独立样本 →
每锚点带 `overlap` 标记。

## 五、实测读数（2026-09-16，38 标的池）

真实日K（腾讯前复权 26 只 A股/ETF + akshare 期货主连 10 + 新浪外汇 2，
2023-01 ~ 2026-09）+ 本地真实训练 LightGBM：

| 项 | 读数 |
|---|---|
| 锚点数（5 日周期，step=5） | 3420 |
| 锚点命中率 | 52.9% |
| 锚点平均已实现收益 | +0.32% |
| 置信度分档 | `[0,0.02)` n=3185 hit 52.8% ret +0.31%；`[0.02,0.04)` n=234 hit 55.1% ret +0.41% |
| 契约 `advisory_consumable_count` | 0 / 38 |

**读数偏弱且置信度普遍贴地（`|composite-0.5|` 几乎全在 0.02 以内）**：
本轮模型 AUC ≈ 0.50~0.54，属于「尚未跑出区分度」的状态。
契约里 `advisory_consumable` 恒为 0 是**如实读数**，不是缺陷 ——
门槛就是用来挡住这种信号的。

⇒ 结论：**当前信号宜作只读观测 / 风险预警，不宜按"高置信"放大仓位。**
与 `cairn/decision-source-contract.md` 的既有结论（高置信 ≠ 可采信）一致且更强。

## 六、踩坑与解法（contains）

- **contains 按名对齐的调用契约被误用，静默产出常数概率**：
  `_align_by_feature_names` 的 `feature_names` 参数被传成了**值矩阵**，
  于是「用小数去匹配模型特征名」全部判成缺失，整体 0 填充，
  **不报错地**产出与任何标的、日期都无关的常数概率。
  证据：全池 3409 个锚点置信度**恒为 0.0303756**。
  解法：函数对 `feature_names` 做类型校验（非字符串序列直接返回 None），
  并且**全列缺失时返回 None**（放弃按名对齐）而不是 0 填充，交由调用方回落位置对齐。
  守卫：`tests/test_predictor_feature_alignment.py`。
- **contains 训练侧只留 Column_N 占位名，按名对齐等于没有**：
  LightGBM 从纯 ndarray 训练只记 `Column_0..N`，推理侧拿的是语义列名
  （`ma_5` / `rsi` ...），两边对不上。解法：训练侧把 `feature_cols` 写进产物
  **并同步为模型原生特征名**；加载侧对老产物同样补写一次（不必重训）。
  两条实测约束（漏任一条都等于没写）：`LGBMClassifier.feature_name_` 是**只读 property**，
  且 `fit` 会用 `Column_N` 覆盖构造期的名字 → 只能 `fit` 之后经 `booster_.feature_name` 写；
  sklearn 包装的 booster 会**缓存** `feature_name()` → 必须清掉 `__boosters` 缓存。
- **contains lambda 不可 pickle，会把模型落盘打断**：
  写 `booster_.feature_name = lambda: ...` 后 `joblib.dump` 直接抛
  `Can't pickle <lambda>`（`tests/test_trainer.py` 实测炸）。
  解法：用模块级类 `_BoosterFeatureNames` 代替 lambda。
- **contains 0.5 不能既当"中性读数"又当"数据坏了"**（延用既有纪律）：
  卡片与 Pine 列在缺失/非有限值上一律 `None`，只有真算出来的 0.5 才是中性读数。

## 七、下游接入方式

```bash
python main.py tv-export                          # 全池，一次产出三件套
python main.py tv-export --symbols 510300.SH      # 单标的
python main.py tv-export --no-anchors --card-limit 8   # 只出卡片，只出前 8 张
```

产出（`outputs/tv/exports/`，gitignored，与 `reports/` 同性质）：

```
signals/<symbol>.png        图片信号卡（tEXt 内嵌契约 + 锚点）
signals/index.json          卡片清单
pine/trendcast/<symbol>.json  request.seed 直接读
pine/index.json             Pine 数据层清单
handoff_report.json         全量报告（数值 / 路径 / 纪律声明）
anchor_backfill.json        锚点回填全量读数
decision_feed.json          底层契约（= `main.py decision-feed` 的产出）
```

配置：`configs/config_pro.yaml` 的 `decision_feed.tv` 段（输出目录 / banner /
锚点参数 / 是否给不可用标的出卡 / 卡片上限）。

## 八、边界（比功能重要）

- **只读**：`position_role=observer`、`affects_gate=false`、`advisory_only=true`，
  三处（契约 / 卡片元数据 / Pine 载荷）结构性存在；
- **不产出仓位**：不给权重（只有周期聚合权重这个**读数口径**，不是仓位权重）、
  不给手数、不给下单建议；
- **不做下单依据**：卡片页脚与 Pine `meta.boundary` 都写明「同图对照，不做下单依据」；
- **未验证**：锚点标 `validated=false`，卡片顶部 banner 显示 `NOT VALIDATED`；
- 以本交付为交易依据须经人工检查点批准。
