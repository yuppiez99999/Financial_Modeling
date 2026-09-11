# S14 / G4 结论：akshare 数据源升级与实际验收数字

- **状态**：`auto` 部分已完成，验收数字已入库；**T14.3 为人工检查点，pending**
- **日期**：2026-09-11
- **来源**：Issue #29 高质量项目集成 · G4（`akfamily/akshare`）
- **验收脚本**：`python scripts/verify_data_sources.py --source akshare`
- **原始报告**：`reports/s14_data_source_acceptance.json`

---

## 一、做了什么（一句话）

把 akshare 从"写在 `requirements.txt` 里的可选依赖"升为**回退链 P1**，
补上本项目**唯一缺失的免费多市场通道**：期货与外汇。链路变为

```
wind (P0 机构级) → akshare (P1 免费多市场) → tencent (P2 免费 A股/ETF) → simulation (P6 兜底)
```

## 二、为什么是它（本项目真实卡点，不是"再加一个库"）

| 卡点 | 升级前证据 | akshare 如何解决 |
|------|-----------|-----------------|
| **期货未开** | `config_pro.yaml` 注释：「腾讯源不支持期货代码（RB.SHF 等）；二期接 Wind 行情后再开启」，`futures.enabled: false` | 新增 `futures` 通道，10 个主连代码全部实测可取 |
| **外汇未开** | 注释：「腾讯/Wind MCP 暂不支持 FXCM 外汇代码」，`forex.enabled: false` 且**无 symbols** | 新增 `forex` 通道，USDCNH/USDCNY 实测可取 |
| **腾讯前复权历史退化** | 实测中国神华 2013 年段出现 `close=-0.32` 的 912 行，剔除 945/3201 行 | 提供**独立于腾讯**的 A股/ETF 第二通道，两条源可交叉识别单源退化 |
| **宏观 LPR 拿不到** | `macro` 5 项指标中 LPR 恒 unavailable | 修列名归一（`TRADE_DATE`/`LPR1Y`），5/5 全部可用 |

## 三、验收数字（T14.3 口径：全量拉取成功率 ≥ 95%）

**实测通过，且远超验收线：**

| 项目 | 成功 / 总数 | 成功率 | 验收线 | 判定 |
|------|------------|--------|--------|------|
| **标的池（全量）** | **38 / 38** | **100.0%** | 95% | ✅ |
| └ stock（A股 12 + ETF 14） | 26 / 26 | 100.0% | | ✅ |
| └ futures（国内期货主连） | 10 / 10 | 100.0% | | ✅ |
| └ forex（外汇） | 2 / 2 | 100.0% | | ✅ |
| **宏观指标（CPI/PMI/GDP/M2/LPR）** | **5 / 5** | **100.0%** | 95% | ✅ |

**期货（10/10，2020-01-02 → 2026-09-11，均为 1624 行）**
RB.SHF 3108 · CU.SHF 108990 · AU.SHF 944.94 · SC.INE 812.9 · I.DCE 718 ·
M.DCE 3402 · Y.DCE 9026 · P.DCE 10035 · C.DCE 2255 · SR.CZC 5441

**外汇（2/2）**
USDCNH.FXCM 1747 行（→ 2026-09-10，6.7145）· USDCNY.FXCM 1700 行（→ 2026-09-11，6.7137）

> 数字口径：`config_pro.yaml` `start_date: 2020-01-01`，`end_date` 留空（采至最新交易日）。
> 报告由脚本一次跑出，可复现；未落任何行情缓存（只读不写，避免验收数据混进训练缓存）。

## 四、实现要点与两个"踩坑"

### 1. 按代码形态分派，不建外部字典

`split_symbol()` 只看后缀：`RB.SHF`→futures、`USDCNH.FXCM`→forex、
`.SH/.SZ` 且 5xx/15x/16x/18x→etf、其余→stock。**未知代码如实返回 None**，不猜。

### 2. 列序是交叉验证结论，不是猜的

新浪外汇日K 返回 `date, open, low, high, close`（**注意 open 后是 low，不是 high**）。
验证方式：枚举 4 个价格列的全部角色排列，`open/low/high/close` 是**唯一**
在 800/800 行历史上满足 `low ≤ min(open,close) ≤ max(open,close) ≤ high` 的排列。
列序取错 = 静默污染全部外汇特征，故在测试里**钉死**（`test_sina_fx_column_order_is_fixed`）。

### 3. 坑一：新浪期货限流（HTTP 456）

`ak.futures_zh_daily_sina` 内部 `requests.get` **不带 UA/Referer**，
批量拉取时新浪返回 456 反爬页 → akshare 抛 `IndexError`。实测 10 个期货主连 **9 个失败**。
修复：改为**复用同一 `requests.Session` + 浏览器头**直连同一公开接口
（字段口径与 akshare 一致），成功率 76% → 100%。

### 4. 坑二：JSONP 切片起点差一个字符

响应为 `var _v=([{...},{...}]);`。切片从 `"=("` **+2** 起（落在 `[` 上）才能解出全部记录；
从 +3 起（落在 `{` 上）`raw_decode` **只解出第 1 条**且不报错 —— 这是最危险的一类 bug
（静默丢 99.98% 数据）。已在 `test_sina_futures_jsonp_parsing` 钉死为回归用例。

此外对"空返回/解析失败/网络异常"做了**有界退避重试**（缺省 2 次），
重试只针对真实网络调用；解析后确无数据不重试（不放大无意义请求）。

## 五、边界（不改门禁、不伪装）

- **只影响取数，不动门禁**：`strategy_gate` / `prediction_horizons` **一个字未改**；
  本次升级**不产生任何"信号可用/门禁解锁"含义**。
- **不编造数据**：外汇无成交量，如实填 `volume=1.0` 占位（防下游量类特征除零），
  **价格一律来自真实源**，绝不用模拟值顶替；退化行（非正价格 / `high<low`）直接剔除，
  与腾讯源同口径（宁可少数据，不可脏数据）。
- **未安装 akshare 不报错**：`ImportError` 静默跳过，链路行为与升级前**完全一致**；
  全量测试离线可跑（akshare 调用全部 mock，不触网）。
- **标准版 `config.yaml` 仍为 `["simulation"]`**（离线可跑），只有专业版启用真实链路。

## 六、待人工决策（T14.3）

1. **回退链顺序是否定稿**：`wind → akshare → tencent → simulation`。
   现方案让 akshare 排在腾讯前（覆盖更广、且提供第二通道做交叉验证）；
   若你更希望"腾讯优先（A股更快）"，一条配置即可对调。
2. **期货/外汇是否正式纳入训练主线**：本次已把 `futures.enabled`/`forex.enabled`
   置为 `true`，**这意味着 `collect_all()` 会把 38 个标的都纳入训练**。
   若你希望"先只取数、暂不参与训练"，需回退这两个开关（数据通道保留，配置关掉即可）。
3. **外汇品种池是否扩充**：当前仅 USDCNH/USDCNY 两个（已实测）；
   `_FX_SYMBOLS` 中另有 EURUSD/USDJPY/GBPUSD/AUDUSD/USDHKD/XAUUSD 已登记映射，
   未实测是否可取 —— 需要则我逐条实测后加入。

## 七、复现方式

```bash
pip install akshare                      # 未安装时链路自动跳过，不影响其它功能
python scripts/verify_data_sources.py --source akshare     # 真实网络验收（≈2 分钟）
python scripts/verify_data_sources.py --offline            # 离线自检（配置 + 代码映射）
python main.py macro                                       # 宏观 5 项状态
```
