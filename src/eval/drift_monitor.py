"""漂移监控（S23 / I3：evidently 准入不过 → PSI/KS 方法论自研，T23.2 fallback）。

准入结论（T23.1，2026-09-13）：evidently 现版（0.7.x）要求 Python ≥ 3.10，
本项目运行时 Python 3.8；老版本停更线不明 + 重依赖（plotly/pydantic），
**不引入运行时**，按候选矩阵预授权的 fallback 走 PSI/KS 方法论自研。

## 口径（写死，防自由度回流）

- **PSI**（Population Stability Index）：以参考窗分位数为桶（缺省 10 桶），
  PSI = Σ (a% − e%)·ln(a%/e%)；零桶以 ε=1e-6 平滑。判定阈值沿用业界惯例：
  <0.1 稳定 / 0.1~0.25 中度漂移 / >0.25 显著漂移（阈值为**只读引用**，
  不构成任何决策）。
- **KS**（两样本 Kolmogorov–Smirnov 统计量）：手写 ECDF 最大绝对差，
  不新增 scipy 之外的依赖。
- **无未来函数**：参考窗/当前窗按 asof 显式切分；rolling PSI 只用窗内数据。
- **fail-close**：任一侧有效样本 < 30 → 该特征 ``available=False``，不猜。

## 不做什么

- 只读监控、只进报表：漂移读数**不进任何决策路径**，`affects_gate` 恒 false；
- 不自动触发重训练/告警动作（那属 T23.4 人工检查点的范围）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_SAMPLE = 30          # 任一侧有效样本下限（低于则拒绝给读数）
PSI_BINS = 10            # PSI 分位桶数
PSI_EPS = 1e-6           # 零桶平滑
PSI_STABLE = 0.10        # 业界惯例阈值（只读引用）
PSI_SIGNIFICANT = 0.25
KS_SIGNIFICANT = 0.20    # KS 统计量经验阈值（只读引用）


def psi(expected: Sequence[float], actual: Sequence[float],
        bins: int = PSI_BINS) -> Optional[float]:
    """ Population Stability Index（expected 为参考分布，actual 为当前分布）。"""
    e = pd.Series(expected, dtype="float64").dropna()
    a = pd.Series(actual, dtype="float64").dropna()
    if len(e) < MIN_SAMPLE or len(a) < MIN_SAMPLE:
        return None
    edges = np.quantile(e.to_numpy(), np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e.to_numpy(), bins=edges)[0] / len(e)
    a_pct = np.histogram(a.to_numpy(), bins=edges)[0] / len(a)
    e_pct = np.clip(e_pct, PSI_EPS, None)
    a_pct = np.clip(a_pct, PSI_EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_statistic(expected: Sequence[float], actual: Sequence[float]) -> Optional[float]:
    """两样本 KS 统计量 = sup|ECDF_e − ECDF_a|（手写实现，零新依赖）。"""
    e = np.sort(pd.Series(expected, dtype="float64").dropna().to_numpy())
    a = np.sort(pd.Series(actual, dtype="float64").dropna().to_numpy())
    if len(e) < MIN_SAMPLE or len(a) < MIN_SAMPLE:
        return None
    grid = np.sort(np.concatenate([e, a]))
    cdf_e = np.searchsorted(e, grid, side="right") / len(e)
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    return float(np.max(np.abs(cdf_e - cdf_a)))


def _verdict(psi_value: Optional[float], ks_value: Optional[float]) -> str:
    if psi_value is None or ks_value is None:
        return "unavailable"
    if psi_value > PSI_SIGNIFICANT or ks_value > KS_SIGNIFICANT:
        return "significant_drift"
    if psi_value > PSI_STABLE:
        return "moderate_drift"
    return "stable"


def drift_report(features: Dict[str, pd.Series],
                 reference_mask: pd.Series,
                 current_mask: pd.Series,
                 ) -> Dict[str, Any]:
    """对一组特征做 参考窗 vs 当前窗 漂移读数（T23.2 核心）。

    Args:
        features: ``{特征名: Series}``（index 对齐，含时间索引）。
        reference_mask / current_mask: 布尔掩码（按时间显式切分，无未来函数）。
    """
    rows: List[Dict[str, Any]] = []
    for name, series in features.items():
        s = pd.Series(series, dtype="float64")
        ref = s[reference_mask.reindex(s.index, fill_value=False)]
        cur = s[current_mask.reindex(s.index, fill_value=False)]
        p = psi(ref, cur)
        k = ks_statistic(ref, cur)
        rows.append({
            "feature": name,
            "available": p is not None and k is not None,
            "psi": round(p, 6) if p is not None else None,
            "ks": round(k, 6) if k is not None else None,
            "ref_n": int(len(ref)), "cur_n": int(len(cur)),
            "verdict": _verdict(p, k),
        })
    available_rows = [r for r in rows if r["available"]]
    drifted = [r["feature"] for r in available_rows
               if r["verdict"] == "significant_drift"]
    return {
        "available": bool(available_rows),
        "affects_gate": False,
        "report_only": True,
        "n_features": len(rows),
        "n_available": len(available_rows),
        "n_significant_drift": len(drifted),
        "drifted_features": drifted,
        "rows": rows,
        "thresholds": {"psi_stable": PSI_STABLE, "psi_significant": PSI_SIGNIFICANT,
                       "ks_significant": KS_SIGNIFICANT,
                       "note": "业界惯例只读引用，阈值为观察口径非决策口径"},
    }


def rolling_psi(series: pd.Series, window: int = 60,
                step: int = 20) -> pd.Series:
    """漂移时间线：以「首个 window 段」为参考，滚动窗 PSI（T23.3 输入）。

    每个时点的参考分布固定为**序列开头的第一段**（无未来函数：参考窗
    不随滚动窗后移），当前窗为该时点前 ``window`` 天 —— 回答「相对起点
    变了多少」。
    """
    s = pd.Series(series, dtype="float64").dropna()
    if len(s) < window * 2:
        return pd.Series(dtype="float64")
    ref = s.iloc[:window]
    out: Dict[pd.Timestamp, float] = {}
    for end in range(window * 2, len(s) + 1, step):
        cur = s.iloc[end - window:end]
        p = psi(ref, cur)
        if p is not None:
            out[s.index[end - 1]] = p
    return pd.Series(out, dtype="float64")


def cross_drift_with_timeline(drift_timeline: pd.Series,
                              other: pd.Series,
                              name: str = "other") -> Dict[str, Any]:
    """T23.3 交叉分析：漂移时间线 vs 外部时间线（状态转移密度/命中率变化）。

    两条时间线按共同日期对齐后做皮尔逊/秩相关。样本不足或常数序列 →
    ``available=False``（不猜）。**相关不构成因果**，读数只进报表。
    """
    d = pd.Series(drift_timeline, dtype="float64").dropna()
    o = pd.Series(other, dtype="float64").dropna()
    common = d.index.intersection(o.index)
    if len(common) < MIN_SAMPLE:
        return {"available": False, "name": name,
                "reason": f"共同日期不足（{len(common)} < {MIN_SAMPLE}）",
                "affects_gate": False}
    dv = d[common]
    ov = o[common]
    if dv.std() == 0 or ov.std() == 0:
        return {"available": False, "name": name,
                "reason": "存在常数序列，相关无定义", "affects_gate": False}
    return {
        "available": True,
        "name": name,
        "affects_gate": False,
        "n_days": int(len(common)),
        "pearson": round(float(dv.corr(ov)), 6),
        "spearman": round(float(dv.corr(ov, method="spearman")), 6),
        "note": "相关不构成因果；漂移与状态/命中率变化是否互为前兆属 T23.4 人工检查点",
    }
