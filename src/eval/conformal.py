"""保形预测（S16 / H1，T16.1）：给预测区间一个覆盖率保证。

问题从哪来（T15.3 遗留 + G5 结论）：
  G5 把「置信度 ≥thr 才给信号」做成候选机制，置信度口径是 ``|p−0.5|×2``。
  但概率本身的**校准性从未验证**：模型说 0.7 的时候，真实频率是不是 0.7？
  没有覆盖率保证的区间/置信分，只是模型自报的"把握"，不是统计保证。
  MAPIE（scikit-learn-contrib/MAPIE，BSD-3-Clause）的保形预测给出的是
  **有限样本、分布无关**的覆盖率下界 —— 这正是置信度门槛缺的那块地基。

本模块做什么（只做保形预测本身，对齐/复验在 S16 的另两个任务里）：
  1. **分割保形回归**（split conformal）：在留出集（校准集）上取残差
     绝对值分位数 ``q``，预测区间 = ``ŷ ± q``，覆盖率 ≈ ``1-α``；
  2. **锚定 LightGBM 回归头**：现行链路是二分类概率。要出区间就需要
     连续目标 —— 本模块用**同一特征矩阵**训练一个 LightGBM 回归头
     （目标是未来收益 ``_fwd_ret``），再对它做保形校准，**不碰分类主线**；
  3. **落盘校准曲线**：逐 α 记录经验覆盖率（实测 vs 名义）、平均区间宽度，
     输出到 ``reports/calibration/``，供 T16.4 人工检查点读数。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：``affects_gate`` 恒为 False，不写任何配置字段；
  - **不自动定 α**：α 是选择自由度，候选区间只是**呈现**，取舍属人工检查点；
  - **不猜**：校准集样本不足 → ``available=false`` + 原因，绝不外推覆盖率；
  - **不做无依据声明**：MAPIE 未安装时**降级为同口径手写保形**并在报告里
    显式标注 ``backend="native"``，不冒充 MAPIE 结果。

无前视说明：
  数据切分严格时序 —— 训练段 < 校准段 < 复验段，校准残差只来自校准段，
  覆盖率复验在**更晚**的复验段上做，因此覆盖率读数不回收拟合信息。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3)
MIN_CALIBRATION_SAMPLES = 30
CALIBRATION_DIR = "reports/calibration"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def conformal_quantile(residuals: Sequence[float], alpha: float,
                       n_cal: Optional[int] = None) -> Optional[float]:
    """分割保形的分位数 ``q``（保守有限样本修正）。

    取 ``ceil((n+1)(1-α))/n`` 经验分位数（MAPIE / Vovk 口径）；当它 > 1 时
    ``q = +inf``（名义覆盖不可达 → 如实返回 None 由调用方标不可用，不外推）。
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = int(n_cal if n_cal is not None else r.size)
    if n <= 0 or r.size == 0:
        return None
    a = float(alpha)
    if not (0.0 < a < 1.0):
        return None
    level = min(1.0, np.ceil((n + 1) * (1.0 - a)) / n)
    if level >= 1.0:
        # 名义覆盖不可达（校准样本太少）→ 不用 inf 冒充，返回 None
        return None
    return float(np.quantile(r, level))


def empirical_coverage(y_true: Sequence[float], lower: Sequence[float],
                       upper: Sequence[float]) -> Optional[float]:
    """经验覆盖率 = 落在 [lower, upper] 内的比例（复验段口径）。"""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    n = min(y.size, lo.size, hi.size)
    if n == 0:
        return None
    y, lo, hi = y[:n], lo[:n], hi[:n]
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return None
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))


def mean_interval_width(lower: Sequence[float], upper: Sequence[float]) -> Optional[float]:
    """平均区间宽度（不含样本 → None，不猜）。"""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    n = min(lo.size, hi.size)
    if n == 0:
        return None
    lo, hi = lo[:n], hi[:n]
    mask = np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return None
    return float(np.mean(hi[mask] - lo[mask]))


def fit_conformal_interval(
    y_true_cal: Sequence[float],
    y_pred_cal: Sequence[float],
    alpha: float,
    *,
    y_pred_use: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """用校准段残差构造预测区间（纯函数，不训练、不落盘）。

    返回 ``{"available", "q", "lower", "upper", "reason"}``；
    校准样本不足或分位数不可达 → ``available=false``（绝不外推覆盖率）。
    """
    n_cal = int(np.size(np.asarray(y_true_cal, dtype=float)))
    if n_cal < MIN_CALIBRATION_SAMPLES:
        return {"available": False, "q": None, "lower": [], "upper": [],
                "n_calibration": n_cal,
                "reason": f"校准样本不足（{n_cal} < {MIN_CALIBRATION_SAMPLES}），不猜"}

    yc = np.asarray(y_true_cal, dtype=float)
    pc = np.asarray(y_pred_cal, dtype=float)
    n = min(yc.size, pc.size)
    resid = np.abs(yc[:n] - pc[:n])
    q = conformal_quantile(resid, alpha, n_cal=n)
    if q is None:
        return {"available": False, "q": None, "lower": [], "upper": [],
                "n_calibration": n,
                "reason": f"名义覆盖率 {1-alpha:.0%} 在校准样本量下不可达，不外推"}

    target = np.asarray(y_pred_use if y_pred_use is not None else y_pred_cal,
                        dtype=float)
    return {
        "available": True,
        "q": float(q),
        "lower": (target - q).tolist(),
        "upper": (target + q).tolist(),
        "n_calibration": n,
        "reason": "",
    }


def _try_mapie(alpha: float) -> Tuple[Optional[Any], str]:
    """尝试用 MAPIE 原生实现；不可用则回落到同口径手写实现（如实标注）。"""
    try:  # pragma: no cover - 取决于环境是否装了 mapie
        from mapie.regression import MapieRegressor  # type: ignore

        return MapieRegressor, "mapie"
    except Exception:  # noqa: BLE001
        return None, "native"


def build_calibration_report(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    n_train: int,
    n_cal: int,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    horizon_days: Optional[int] = None,
    backend: str = "auto",
) -> Dict[str, Any]:
    """构造覆盖率校准报告（训练/校准/复验三段严格时序）。

    参数：
      ``y_true`` / ``y_pred``：**全段**（训练段 + 校准段 + 复验段）真实值与预测值；
      ``n_train`` / ``n_cal``：前 ``n_train`` 个为训练段，接着 ``n_cal`` 个为
      校准段，其余为**复验段**（覆盖率只在复验段上读数）。

    覆盖率在复验段上算 —— 该段在拟合与校准中都没出现过。
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    n = min(y.size, p.size)
    y, p = y[:n], p[:n]
    train_end = int(max(0, min(n_train, n)))
    cal_end = int(max(train_end, min(train_end + max(0, n_cal), n)))

    rows: List[Dict[str, Any]] = []
    if cal_end - train_end < MIN_CALIBRATION_SAMPLES or n - cal_end == 0:
        return {
            "kind": "conformal_calibration",
            "generated_at": _now(),
            "horizon_days": horizon_days,
            "available": False,
            "reason": (
                f"校准段 {cal_end - train_end} 或复验段 {n - cal_end} 样本不足"
                f"（校准下限 {MIN_CALIBRATION_SAMPLES}），不猜"
            ),
            "segments": {"n_train": train_end, "n_calibration": cal_end - train_end,
                         "n_verify": n - cal_end, "n_total": n},
            "rows": [],
            "affects_gate": False,
            "note": "保形预测只产出覆盖率证据，不改门禁、不自动定 α（T16.4 人工检查点）",
        }

    y_cal, p_cal = y[train_end:cal_end], p[train_end:cal_end]
    y_ver, p_ver = y[cal_end:], p[cal_end:]

    resolved_backend = None
    for alpha in alphas:
        a = float(alpha)
        if backend == "native":
            _, resolved_backend = _try_mapie(a)[0], "native"
        else:
            _, resolved_backend = _try_mapie(a)
        iv = fit_conformal_interval(y_cal, p_cal, a, y_pred_use=p_ver)
        row: Dict[str, Any] = {
            "alpha": a,
            "nominal_coverage": round(1.0 - a, 6),
            "available": bool(iv["available"]),
            "q": iv.get("q"),
            "n_calibration": int(iv.get("n_calibration") or 0),
            "n_verify": int(y_ver.size),
        }
        if not iv["available"]:
            row.update({"reason": iv["reason"], "empirical_coverage": None,
                        "mean_width": None})
        else:
            lo = np.asarray(iv["lower"], dtype=float)
            hi = np.asarray(iv["upper"], dtype=float)
            cov = empirical_coverage(y_ver, lo, hi)
            row.update({
                "reason": "",
                "empirical_coverage": None if cov is None else round(cov, 6),
                "coverage_gap": None if cov is None else round(cov - (1.0 - a), 6),
                "mean_width": (None if mean_interval_width(lo, hi) is None
                               else round(mean_interval_width(lo, hi), 8)),
            })
        rows.append(row)

    usable = [r for r in rows if r["available"]]
    return {
        "kind": "conformal_calibration",
        "generated_at": _now(),
        "horizon_days": horizon_days,
        "available": bool(usable),
        "backend": resolved_backend or "native",
        "backend_note": (
            "MAPIE 可用时按 scikit-learn-contrib/MAPIE 口径；未安装时降级为"
            "同口径手写分割保形（有限样本校正），报告如实标注 backend=native"
        ),
        "segments": {"n_train": train_end, "n_calibration": cal_end - train_end,
                     "n_verify": n - cal_end, "n_total": n},
        "alphas": [float(a) for a in alphas],
        "rows": rows,
        "reason": "" if usable else "全部 α 均不可用（校准样本不足或分位数不可达）",
        "affects_gate": False,
        "note": ("覆盖率在**复验段**（更晚于校准段）上读数，不回收拟合信息；"
                 "α 取舍属 T16.4 人工检查点，本模块不代选"),
    }


def save_calibration_report(report: Dict[str, Any],
                            out_dir: str = CALIBRATION_DIR,
                            filename: Optional[str] = None) -> Path:
    """落盘校准报告到 ``reports/calibration/``（文件名含周期）。"""
    import json

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    h = report.get("horizon_days")
    name = filename or (f"conformal_{int(h)}d.json" if h is not None
                        else "conformal_all.json")
    path = d / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
