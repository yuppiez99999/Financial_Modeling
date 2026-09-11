# 第三方组件登记（THIRD_PARTY）

> 依据 README §18A「统一原则」：MIT/Apache-2.0 代码集成进本仓库（「禁止商业用途」
> 自有许可）不冲突，但**必须登记来源与版本**。本文件是唯一登记入口，
> 引入新第三方组件时同步追加（append 风格，不删除历史条目）。

| 组件 | 来源 | 许可证 | 引入阶段 | 用途 | 使用方式 |
|------|------|--------|---------|------|---------|
| microsoft/qlib（Alpha158） | https://github.com/microsoft/qlib | MIT | S13 / G3（2026-09-11） | Alpha158 因子集（98 列） | **表达式级对齐复现**：`integrations/qlib/alpha158.py` 用纯 pandas 复现 qlib `Alpha158DataLoader` 的因子表达式，**未引入 qlib 运行时依赖、未复制 qlib 源码** |
| akfamily/akshare | https://github.com/akfamily/akshare | MIT | 既有（requirements.txt，可选） | 财经数据接口 | pip 依赖 |
| vectorbt | https://github.com/polakowo/vectorbt | 自定义（允许商用） | 既有（S1 回测层） | 向量化回测 | pip 依赖 |
| Kronos（快照） | https://github.com/shiyu-coder/Kronos | MIT | 既有（对照参考） | K 线基础模型 | 仓库内源码快照（`Kronos/THIRD_PARTY_NOTICE.md`） |
| LightGBM | https://github.com/microsoft/LightGBM | MIT | 既有 | 主力模型 | pip 依赖 |

## Alpha158 对齐表（qlib → 本项目）

qlib 官方 Alpha158（`qlib/contrib/data/loader.py`）共 158 个表达式 =
98 个特征列（部分表达式共享输出列）。本项目 `integrations/qlib/alpha158.py`
按**窗口 (5, 10, 20, 30, 60)** 复现其中 98 列：

- K 线形态项（8 个，当日，无窗口）：`KMID / KLEN / KUP / KLOW / KSFT / KMID2 / KUP2 / KLOW2`
- 滚动项 × 5 窗口（18 × 5 = 90 个）：
  `ROC / MA / STD / BETA / RSQR / RESI / QTLU / QTLD / RANK / RSV / CORR / CORD / CNTP / SUMP / VSUMP / VMA / VSTD / WVMA`

算子语义对齐说明（与 qlib 表达式引擎的差异点，如实登记）：

| qlib 算子 | 本项目实现 | 差异 |
|-----------|-----------|------|
| `Ref($close, i)` | `Series.shift(i)` | 无 |
| `Mean/Std/Max/Min/Quantile/Rank` | `rolling(w)` 同名 | 无 |
| `Slope / Rsquare / Resi` | 滚动 OLS（`rolling.apply`） | 无（数值精度 1e-9 级） |
| `Corr($close,$volume,w)` | `rolling(w).corr` | 无 |
| `Corr(Delta($close,i),$volume,w)` | `diff(w).rolling(w).corr(v)` | qlib 的 Delta 窗口语义按逐因子对齐 |
| `Count($close>Ref($close,i),w)` | `(c>c.shift(w)).rolling(w).mean()` | 无 |
| `Greater / Lesser` | `pd.concat.max/min(axis=1)` | 无 |
| `WVMA` | `Std(Abs(ret1)*volume, w)` | 无 |

**明确不复现的部分**（不影响验证口径，如实声明）：
- qlib Alpha360（1 年价格切片）未实现 —— 本轮验证只覆盖 Alpha158；
- qlib 表达式引擎本身（`DataHandlerLP` 惰性计算）未引入 —— 本项目按项目纪律
  （零重型依赖、CI 离线可跑）用 pandas 即时计算。

## 许可证合规结论

- qlib（MIT）：MIT 允许商用与再分发，本项目「禁止商业用途」是对**外授权的收紧**，
  内部使用与集成无冲突；
- 复现方式为**算法/表达式对齐**（因子定义属公开方法论），未复制 qlib 源码文本；
  若后续直接引入 qlib 包（`pip install pyqlib`），需在本表追加运行时版本号。
