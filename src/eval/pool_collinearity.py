"""池共线性诊断（Issue #55 步骤①）：把「38 只池的有效样本量有多大」变成可复算读数。

## 问题从哪来

Issue #55 的实测读数：38 标的池、真实日K、真实训练 LightGBM，
锚点命中率 52.9%、AUC ≈ 0.50~0.54、`advisory_consumable` 恒 0 / 38。
交付形态这条路已经通了，但**模型没跑出可用区分度**。

排查优先级第一条就是**池的共线性**：38 只里 24 只是宽基 / 行业 ETF
（沪深300、中证500、中证1000、科创50×2…），彼此高度相关。
如果池内标的日收益的**有效独立维度**远小于 38，那么：

- 名义样本量（≈ 38 × 交易日）严重高估了**有效样本量**；
- 模型学到的主要是**市场 beta**（共同因子），而不是区分度；
- 门禁在"全池混合样本"上评估，命中率会被beta主导，看起来接近 50% 也不奇怪。

## 本模块做什么

**只做诊断，不做修复** —— 先回答"共线性到底有多严重"，再谈要不要动池：

1. `correlation_matrix`：标的收益相关矩阵 + 高相关对清单；
2. `effective_dimension`：**有效独立维度**（相关矩阵特征值的参与比
   participation ratio），以及常见口径下的"等效标的数"；
3. `variance_decomposition`：第一主成分解释的方差占比（≈ 市场 beta 强度）；
4. `residual_effective_dimension`：**把等权市场因子剥掉之后还剩多少**——
   对每只标的做 `r_i = α + β·r_market + ε`，用残差 ε 重算标的间相关与有效维度。

   ⚠️ **读数解读有坑**（实测暴露）：等权平均因子在"独立标的占多数"时，
   残差维度可能**低于**原始维度，而**不是**直觉上的"剥掉共同成分后维度上升"。
   原因是等权平均本身由池内标的构成 —— OLS 残差会与"该标的在因子里的权重"
   产生结构性负相关（合成伪迹），把独立标的也拉成相关的。
   因此该字段只能**同池内横向对照**（如全池 vs 分池），**不可**当成
   "剥 beta 后还剩多少独立信息"的绝对口径。要真正的市场因子应换成外部指数
   （如沪深300指数本身），本模块刻意不引入外部数据源以保持离线可复算。
5. `compare_pools`：把全池、仅 ETF、仅个股、仅期货等分池放在一起对照，
   让"共线性主要来自哪一块"可读。

## 边界（比功能重要）

- **只读**：不重训模型、不改池、不改门禁（`affects_gate=false`）；
- **不猜**：样本不足 / 时间轴对不齐 → 如实 `available=false` + reason；
- **不擅动池**：结论是**证据**。要不要缩池 / 换池属产品口径变更，
  按仓库既有纪律须人工签字（与 `horizon_decision` / `confidence_gate` 同款）；
- **口径一致**：收益一律用**日对数收益**，按日期**取交集**对齐，
  缺一天的标的按可用日期算（并如实报告覆盖天数）。
"""
from __future__ import annotations

import itertools
import logging
import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 相关阈值：|ρ| ≥ 该值即算"高相关对"（按经验口径，非统计检验）
HIGH_CORR_THRESHOLD = 0.7
# 参与比下限：低于该值视为共线性显著（有效维度不足名义维度的一半）
LOW_DIM_RATIO = 0.5
# 计算相关矩阵最少需要的时间点数与标的数
MIN_DAYS = 60
MIN_SYMBOLS = 3


def _finite_ret(series: pd.Series) -> pd.Series:
    """日对数收益；非有限值一律置 NaN（不拿 0 冒充无变化）。"""
    s = pd.to_numeric(series, errors="coerce")
    out = np.log(s / s.shift(1))
    return out.replace([np.inf, -np.inf], np.nan)


def build_returns(price_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """多标的日对数收益对齐表（index=日期并集，columns=标的）。

    只做**对齐**，不做填充：某标的当日无数据即 NaN，
    由调用方按对可用（`dropna`）或成对删除（相关系数默认）处理。
    """
    cols: Dict[str, pd.Series] = {}
    for symbol, df in (price_frames or {}).items():
        if df is None or len(df) == 0 or "close" not in df.columns:
            continue
        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        frame = frame.drop_duplicates(subset="date", keep="last")
        ret = _finite_ret(frame.set_index("date")["close"])
        ret = ret[ret.index >= pd.Timestamp("2000-01-01")]
        cols[str(symbol)] = ret
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols)
    return out.sort_index()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """皮尔逊相关矩阵（成对完整观测，`min_periods` 自适应）。"""
    n = int(returns.shape[0])
    min_p = max(min(int(n * 0.5), n), 10)
    return returns.corr(min_periods=min_p)


def participation_ratio(eigenvalues: Sequence[float]) -> float:
    """参与比（participation ratio）：Σλ² 归一化下的**有效独立维度**。

    等价于 (Σλ)² / Σλ²。全部特征值相等时 = 维度数（完全独立）；
    只有一个非零特征值时 = 1（完全共线）。比"数几个特征值 > 1"稳健得多，
    不依赖阈值选择。
    """
    vals = np.asarray([float(v) for v in eigenvalues], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 0.0
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    return float((total ** 2) / float((vals ** 2).sum()))


def high_correlation_pairs(corr: pd.DataFrame,
                           threshold: float = HIGH_CORR_THRESHOLD
                           ) -> List[Dict[str, Any]]:
    """列出 |ρ| ≥ threshold 的标的对（按 |ρ| 降序）。"""
    names = list(corr.columns)
    pairs: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(names, 2):
        try:
            rho = float(corr.loc[a, b])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(rho):
            continue
        if abs(rho) >= float(threshold):
            pairs.append({"a": str(a), "b": str(b), "corr": round(rho, 6)})
    pairs.sort(key=lambda p: -abs(p["corr"]))
    return pairs


def demean_by_market(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """剥掉等权市场因子：`r_i = α + β·r_mkt + ε`，返回 (残差表, 市场因子)。

    市场因子 = 当日**可得标的**的等权平均收益（不是全样本固定截面的均值，
    避免利用未来信息构成因子）。β 用**全样本** OLS 估计并如实标注 ——
    这是诊断（刻画既有相关性结构），不是交易信号，因此不构成前视；
    用于交易决策前必须走 walk-forward（见边界声明）。
    """
    mkt = returns.mean(axis=1, skipna=True)
    resid = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    betas: Dict[str, float] = {}
    for col in returns.columns:
        pair = pd.concat([returns[col], mkt], axis=1).dropna()
        if len(pair) < MIN_DAYS:
            betas[str(col)] = float("nan")
            continue
        y = pair.iloc[:, 0].to_numpy(dtype=float)
        x = pair.iloc[:, 1].to_numpy(dtype=float)
        var = float(np.var(x))
        if var <= 0:
            betas[str(col)] = float("nan")
            continue
        beta = float(np.cov(x, y, ddof=1)[0, 1] / var) if len(x) > 1 else 0.0
        alpha = float(np.mean(y) - beta * np.mean(x))
        betas[str(col)] = beta
        resid.iloc[:, resid.columns.get_loc(col)] = returns[col].to_numpy(dtype=float) - (alpha + beta * mkt.to_numpy(dtype=float))
    return resid, mkt


def diagnose(returns: pd.DataFrame,
             high_corr_threshold: float = HIGH_CORR_THRESHOLD,
             label: str = "pool") -> Dict[str, Any]:
    """对一张多标的收益表做共线性诊断（纯函数，不落盘、不碰配置）。"""
    out: Dict[str, Any] = {
        "label": str(label),
        "available": False,
        "reason": "",
        "n_symbols": int(returns.shape[1]) if not returns.empty else 0,
        "n_days": int(returns.shape[0]) if not returns.empty else 0,
        "high_corr_threshold": float(high_corr_threshold),
    }
    if returns is None or returns.empty or returns.shape[1] < MIN_SYMBOLS:
        out["reason"] = f"标的数不足（{out['n_symbols']} < {MIN_SYMBOLS}）"
        return out
    # 每标的有效天数（如实报告，不掩盖缺失）
    per_symbol_days = {str(c): int(returns[c].notna().sum()) for c in returns.columns}
    usable = [c for c, d in per_symbol_days.items() if d >= MIN_DAYS]
    if len(usable) < MIN_SYMBOLS:
        out["reason"] = f"有效天数充足的标的不足（{len(usable)} < {MIN_SYMBOLS}）"
        out["per_symbol_days"] = per_symbol_days
        return out
    sub = returns[usable]
    corr = correlation_matrix(sub)
    out["per_symbol_days"] = per_symbol_days
    out["usable_symbols"] = usable

    # 特征值谱：用完整观测的子集算（相关矩阵特征值对缺失敏感）
    complete = sub.dropna()
    if len(complete) < MIN_DAYS:
        out["reason"] = f"完整观测天数不足（{len(complete)} < {MIN_DAYS}），无法做特征分解"
        return out
    mat = np.corrcoef(complete.to_numpy(dtype=float).T)
    eig = np.linalg.eigvalsh(mat)
    eig = np.sort(eig)[::-1]
    pr = participation_ratio(eig)
    total_var = float(eig.sum())
    top1 = float(eig[0] / total_var) if total_var > 0 else float("nan")
    top3 = float(eig[:3].sum() / total_var) if total_var > 0 else float("nan")

    pairs = high_correlation_pairs(corr, high_corr_threshold)
    n_pairs_possible = len(usable) * (len(usable) - 1) // 2

    # 剥掉市场因子后的残差维度
    resid, _mkt = demean_by_market(sub)
    resid_complete = resid.dropna()
    resid_pr = float("nan")
    resid_top1 = float("nan")
    if len(resid_complete) >= MIN_DAYS and resid_complete.shape[1] >= MIN_SYMBOLS:
        rmat = np.corrcoef(resid_complete.to_numpy(dtype=float).T)
        reig = np.sort(np.linalg.eigvalsh(rmat))[::-1]
        resid_pr = participation_ratio(reig)
        rtot = float(reig.sum())
        resid_top1 = float(reig[0] / rtot) if rtot > 0 else float("nan")

    out.update({
        "available": True,
        "n_complete_days": int(len(complete)),
        "eigenvalues_head": [round(float(v), 6) for v in eig[:10]],
        "effective_dimension": round(pr, 4),
        "effective_dimension_ratio": round(pr / max(len(usable), 1), 4),
        "pc1_variance_share": round(top1, 4) if math.isfinite(top1) else None,
        "pc1_3_variance_share": round(top3, 4) if math.isfinite(top3) else None,
        "market_beta_share": round(top1, 4) if math.isfinite(top1) else None,
        "high_corr_pairs": pairs[:40],
        "n_high_corr_pairs": int(len(pairs)),
        "n_pairs_possible": int(n_pairs_possible),
        "high_corr_pair_ratio": round(len(pairs) / n_pairs_possible, 4) if n_pairs_possible else None,
        "residual_effective_dimension": round(resid_pr, 4) if math.isfinite(resid_pr) else None,
        "residual_pc1_variance_share": round(resid_top1, 4) if math.isfinite(resid_top1) else None,
        "collinear": bool((pr / max(len(usable), 1)) < LOW_DIM_RATIO),
        "note": (
            "有效维度 = 相关矩阵特征值的参与比 (Σλ)²/Σλ²；"
            "pc1_variance_share 可读作「等权市场 beta 解释了多大比例的共同波动」；"
            "residual_* 为剥掉**等权平均**因子后的残差口径 —— 该因子由池内标的构成，"
            "残差维度可能因合成伪迹而低于原始维度，故只可同池内横向对照，"
            "不可当作「剥 beta 后还剩多少独立信息」的绝对读数。"
            "诊断只产出证据，不构成缩池/换池决策。"
        ),
    })
    return out


def pool_breakdown(symbols: Sequence[str]) -> Dict[str, List[str]]:
    """按标的类型粗分池：ETF / 个股 / 期货 / 外汇（用于分池对照）。

    规则只看代码后缀与命名习惯，**不做行业分类**（那需要外部数据，
    且答案不稳定）。A股 6 开头为沪市个股、3/0/688 段按代码段区分 ETF 与个股。
    """
    etf_prefixes = ("51", "52", "56", "58", "15", "16", "159", "588")
    out: Dict[str, List[str]] = {"etf": [], "stock": [], "futures": [], "forex": [], "other": []}
    for s in symbols:
        sym = str(s)
        if sym.endswith(".FXCM"):
            out["forex"].append(sym)
        elif any(sym.endswith(sfx) for sfx in (".SHF", ".INE", ".DCE", ".CZC", ".CFX")):
            out["futures"].append(sym)
        elif sym.startswith(etf_prefixes):
            out["etf"].append(sym)
        elif sym.endswith((".SH", ".SZ")):
            out["stock"].append(sym)
        else:
            out["other"].append(sym)
    return {k: v for k, v in out.items() if v}


def compare_pools(returns: pd.DataFrame,
                  high_corr_threshold: float = HIGH_CORR_THRESHOLD,
                  extra_pools: Optional[Dict[str, Sequence[str]]] = None
                  ) -> Dict[str, Any]:
    """全池 vs 分类分池 vs 自定义分池的共线性对照。"""
    if returns is None or returns.empty:
        return {"available": False, "reason": "无可用收益数据"}
    pools: Dict[str, List[str]] = {"all": [str(c) for c in returns.columns]}
    for name, syms in pool_breakdown(list(returns.columns)).items():
        pools[name] = list(syms)
    for name, syms in (extra_pools or {}).items():
        pools[str(name)] = [str(s) for s in syms if str(s) in returns.columns]
    result: Dict[str, Any] = {
        "available": True,
        "n_symbols": int(returns.shape[1]),
        "pools": {},
        "affects_gate": False,
        "note": "诊断只产出证据；缩池/换池属产品口径变更，须人工签字并重做偏差审查。",
    }
    for name, syms in pools.items():
        if len(syms) < MIN_SYMBOLS:
            result["pools"][name] = {"available": False,
                                     "reason": f"标的数不足（{len(syms)} < {MIN_SYMBOLS}）",
                                     "symbols": syms}
            continue
        result["pools"][name] = diagnose(returns[syms], high_corr_threshold, label=name)
        result["pools"][name]["symbols"] = syms
    return result


def build_report(price_frames: Dict[str, pd.DataFrame],
                 high_corr_threshold: float = HIGH_CORR_THRESHOLD,
                 extra_pools: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """从行情表产出完整诊断报告（落盘结构）。"""
    returns = build_returns(price_frames)
    report: Dict[str, Any] = {
        "kind": "pool_collinearity",
        "generated_at": datetime.now().astimezone().isoformat(),
        "affects_gate": False,
        "readonly": True,
        "n_symbols_input": int(len(price_frames or {})),
        "high_corr_threshold": float(high_corr_threshold),
    }
    if returns.empty:
        report.update({"available": False, "reason": "无可用行情数据（data/raw 为空或列不全）"})
        return report
    report["n_symbols_with_returns"] = int(returns.shape[1])
    report["date_range"] = [str(returns.index.min())[:10], str(returns.index.max())[:10]]
    report["comparison"] = compare_pools(returns, high_corr_threshold, extra_pools)
    report["overall"] = report["comparison"]["pools"].get("all", {"available": False,
                                                                  "reason": "all 池不可用"})
    report["available"] = bool(report.get("overall", {}).get("available"))
    if not report["available"]:
        report["reason"] = report.get("overall", {}).get("reason", "诊断不可用")
    return report
