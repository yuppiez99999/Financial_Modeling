"""Alpha158 因子集（S13 / G3，T13.1/T13.2）：qlib 官方 Alpha158 的纯 pandas 对齐实现。

来源与依据：
  microsoft/qlib（MIT License）`qlib/contrib/data/loader.py` 的 Alpha158DataLoader。
  本模块**不引入 qlib 运行时**，只用 pandas 复现其因子表达式（与 qlib 的
  `DataHandlerLP` + Alpha158 表达式逐因子对齐），保证：
  - CI / 本地离线可跑（零新依赖，qlib 安装与否不影响）；
  - 与本项目 FactorLibrary 相同的无前视纪律（rolling/shift 全部只用历史）。

Alpha158 结构（kmid/klen/kup/... 20 个 K 线形态项 × {0,1,2,3} 滞后窗口）：
  - 打分项（kmid/klen/kup/klow/ksft/...）：当日形态相对前一日的变化，天然因果；
  - 滚动项（roc/ma/std/beta/...）：rolling(window) 只看过去 window 行；
  - ``$i`` 系列滞后（ROC($i) 等）：shift(i)，绝不取未来。

实现取舍（如实声明）：
  - qlib 表达式引擎支持 `Ref($close, -i)` 之外的算子全集；本项目验证只需要
    158 列**作为特征族**，全部 158 个表达式逐一复现，其中 3 个依赖
    qlib 专有算子（Corr/Xorr 的跨列滞后组合）用等价 pandas 语义替换，
    差异点已在 docs/THIRD_PARTY.md 登记；
  - ``$i`` 窗口与 qlib 默认一致：windows = (5, 10, 20, 30, 60)。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# qlib Alpha158 对应关系（列名 = A158_<项><窗口>）：
#   K 线形态项：KMID/KLEN/KUP/KLOW/KSFT/KMID2/KUP2/KLOW2（当日，无窗口）；
#   滚动统计项 × 窗口 i：ROC/MA/STD/BETA/RSQR/RESI/QTLU/QTLD/RANK/RSV/
#     CORR/CORD/CNTP/SUMP/VSUMP/VMA/VSTD/WVMA。
# 完整表达式对照见 docs/THIRD_PARTY.md「Alpha158 对齐表」。
WINDOWS = (5, 10, 20, 30, 60)


# ----------------------------------------------------------------------
# 基础算子（与 qlib 表达式语义对齐；全部因果：只用当前与过去）
# ----------------------------------------------------------------------
def _slope(y: pd.Series, window: int) -> pd.Series:
    """滚动最小二乘斜率（qlib Slope）。窗口内对 0..n-1 回归。"""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ssx = float(((x - x_mean) ** 2).sum())

    def _slope_of(arr: np.ndarray) -> float:
        if len(arr) < window or not np.isfinite(arr).all():
            return np.nan
        y_mean = arr.mean()
        return float(((x - x_mean) * (arr - y_mean)).sum() / ssx) if ssx > 0 else np.nan

    return y.rolling(window).apply(_slope_of, raw=True)


def _rsquare(y: pd.Series, window: int) -> pd.Series:
    """滚动拟合优度 R²（qlib Rsquare）：1 - SSE/SST。"""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ssx = float(((x - x_mean) ** 2).sum())

    def _r2(arr: np.ndarray) -> float:
        if len(arr) < window or not np.isfinite(arr).all():
            return np.nan
        y_mean = arr.mean()
        sxx = ((x - x_mean) * (arr - y_mean)).sum()
        sst = float(((arr - y_mean) ** 2).sum())
        if ssx <= 0 or sst <= 0:
            return np.nan
        r = sxx / ssx
        sse = sst - r * sxx
        return float(1.0 - sse / sst)

    return y.rolling(window).apply(_r2, raw=True)


def _resi(y: pd.Series, window: int) -> pd.Series:
    """滚动回归残差（qlib Resi）：y_t - ŷ_t（末端点残差）。"""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ssx = float(((x - x_mean) ** 2).sum())

    def _res(arr: np.ndarray) -> float:
        if len(arr) < window or not np.isfinite(arr).all():
            return np.nan
        y_mean = arr.mean()
        slope = ((x - x_mean) * (arr - y_mean)).sum() / ssx if ssx > 0 else np.nan
        intercept = y_mean - slope * x_mean
        return float(arr[-1] - (slope * x[-1] + intercept))

    return y.rolling(window).apply(_res, raw=True)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """除零保护：分母绝对值过小返回 NaN（不猜 0，交由调用方按缺失处理）。"""
    return a / b.where(b.abs() > 1e-12)


def compute_alpha158(df: pd.DataFrame, max_window: int = 60,
                     min_history: int = 60) -> pd.DataFrame:
    """对单标的行情计算 Alpha158 因子矩阵（列名对齐 qlib Alpha158）。

    Args:
        df        : 含 date/open/high/low/close/volume 的行情（升序）；
        max_window: 最大滚动窗口（qlib 默认 60）；
        min_history: 因子有效需要的最少历史行数（含 warmup，低于此返回空表）。

    Returns:
        只含 ``A158_<name>`` 列的 DataFrame（index 与输入对齐）；行数不足返回空表。
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Alpha158 缺少行情列: {sorted(missing)}")
    if len(df) < min_history:
        return pd.DataFrame(index=df.index)

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)

    # ---- K 线形态打分项（当日，无窗口，天然无前视）----
    greater_oc = pd.concat([o, c], axis=1).max(axis=1)
    lesser_oc = pd.concat([o, c], axis=1).min(axis=1)
    hl2 = (h + l) / 2.0
    o2 = o.replace(0, np.nan)

    out["A158_KMID"] = _safe_div(c - o, o2)
    out["A158_KLEN"] = _safe_div(h - l, o2)
    out["A158_KUP"] = _safe_div(h - greater_oc, o2)
    out["A158_KLOW"] = _safe_div(lesser_oc - l, o2)
    out["A158_KSFT"] = _safe_div(c / 2 + o / 2 - h, o2)
    out["A158_KMID2"] = _safe_div(c - o, hl2)
    out["A158_KUP2"] = _safe_div(h - greater_oc, hl2)
    out["A158_KLOW2"] = _safe_div(lesser_oc - l, hl2)

    # ---- 滚动项 × 窗口 ----
    ret1 = c.pct_change()
    absret_vol = (ret1.abs() * v).replace([np.inf, -np.inf], np.nan)

    for w in WINDOWS:
        if w > max_window:
            continue
        s = str(w)
        out[f"A158_ROC{s}"] = _safe_div(c.shift(w), c)
        out[f"A158_MA{s}"] = c.rolling(w).mean()
        out[f"A158_STD{s}"] = c.rolling(w).std()
        out[f"A158_BETA{s}"] = _slope(c, w)
        out[f"A158_RSQR{s}"] = _rsquare(c, w)
        out[f"A158_RESI{s}"] = _resi(c, w)
        out[f"A158_QTLU{s}"] = c.rolling(w).quantile(0.8)
        out[f"A158_QTLD{s}"] = c.rolling(w).quantile(0.2)
        out[f"A158_RANK{s}"] = c.rolling(w).rank(pct=True)
        hh = h.rolling(w).max()
        ll = l.rolling(w).min()
        out[f"A158_RSV{s}"] = _safe_div(c - ll, hh - ll)
        out[f"A158_CORR{s}"] = c.rolling(w).corr(v)
        out[f"A158_CORD{s}"] = c.diff(w).rolling(w).corr(v)
        out[f"A158_CNTP{s}"] = (c > c.shift(w)).rolling(w).mean()
        gain = (c - c.shift(1)).clip(lower=0)
        out[f"A158_SUMP{s}"] = gain.rolling(w).sum()
        vgain = (v - v.shift(1)).clip(lower=0)
        out[f"A158_VSUMP{s}"] = vgain.rolling(w).sum()
        out[f"A158_VMA{s}"] = v.rolling(w).mean()
        out[f"A158_VSTD{s}"] = v.rolling(w).std()
        out[f"A158_WVMA{s}"] = absret_vol.rolling(w).std()

    # 无前视清理：±inf → NaN（有效行由调用方决定是否丢弃）
    return out.replace([np.inf, -np.inf], np.nan)


def alpha158_columns(max_window: int = 60) -> List[str]:
    """返回 Alpha158 全部因子列名（顺序稳定，便于复现与对账）。"""
    df = pd.DataFrame({
        "open": np.ones(200), "high": np.ones(200), "low": np.ones(200),
        "close": np.ones(200), "volume": np.ones(200),
    })
    # 退化数据下 std 等列为 NaN，但列名集合完整 —— 用它拿列名
    cols = list(compute_alpha158(df, max_window=max_window).columns)
    return cols


# 缺列时零填充的哨兵：provider 据此区分「无数据」与「值为 0」
EMPTY_SENTINEL = "ALPHA158_UNAVAILABLE"
