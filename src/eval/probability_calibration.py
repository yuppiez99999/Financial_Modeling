"""概率校准层（S19 / H4，T19.1 / T19.2）：让「置信度」建立在校准过的概率上。

问题从哪来（S15 的机制缺口）：
  G5 的置信度门槛用 ``|p−0.5|×2`` 做置信分。但 p 本身**从未验证过校准性**：
  LightGBM 说 0.70 的时候，真实频率可能是 0.55（树上模型的典型症状）。
  于是一个"高置信"样本可能只是模型过度自信 —— 门槛筛选的是"自信"而不是
  "准"。Brier 分数与 ECE（期望校准误差）就是把这层窗户纸捅破的指标。

参照 scikit-learn 校准器（既有的轻量依赖，无需新引入重型包）：

  1. **校准层**（T19.1）：isotonic（保序回归，非参数）与 Platt（sigmoid，
     参数化）两种校准器接在 LightGBM 概率输出之后，用**独立保留期**验证
     校准误差（Brier / ECE / 可靠性曲线）；
  2. **阈值曲线复算**（T19.2）：校准后重算置信度阈值曲线，看它是否更单调、
     更好解释 —— 但**不改现行阈值**（是否进主推理链路属 T19.4 人工检查点）。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁 / 不改阈值**：``affects_gate`` 恒为 False，现行 thr 一个字节不动；
  - **不在训练折上校准再报测试折**：校准器只在**校准段**拟合，指标只在
    **更晚的复验段**读数（否则校准就是把答案抄一遍）；
  - **不自动选校准器**：isotonic 与 Platt 都跑，谁更好由 T19.4 人工判；
  - **不猜**：校准段样本不足 → ``available=false`` + 原因，不外推 ECE。

无前视说明：
  严格时序三段：训练段（拟合模型）< 校准段（拟合校准器）< 复验段（读数）。
  校准器只见过校准段，因此复验段的校准误差是**样本外**读数。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

METHOD_ISOTONIC = "isotonic"
METHOD_PLATT = "platt"
DEFAULT_BINS = 10
MIN_CALIBRATION_SAMPLES = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def brier_score(proba: Sequence[float], y_true: Sequence[int]) -> Optional[float]:
    """Brier 分数 = 均方概率误差（越小越好）。样本为空 → None（不猜）。"""
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n = int(min(p.size, y.size))
    if n == 0:
        return None
    p, y = p[:n], y[:n]
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return None
    return float(np.mean((p[mask] - y[mask]) ** 2))


def reliability_curve(proba: Sequence[float], y_true: Sequence[int],
                      bins: int = DEFAULT_BINS) -> List[Dict[str, Any]]:
    """可靠性曲线：逐概率分箱的「预测均值 vs 真实频率」。

    空箱输出 ``available=false``（不补 0、不外推）。
    """
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n = int(min(p.size, y.size))
    if n == 0:
        return []
    p, y = p[:n], y[:n]
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: List[Dict[str, Any]] = []
    for i in range(int(bins)):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin": i, "lower": round(float(lo), 4),
                         "upper": round(float(hi), 4), "samples": 0,
                         "available": False, "mean_predicted": None,
                         "observed_frequency": None, "gap": None})
            continue
        mp = float(np.mean(p[mask]))
        of = float(np.mean(y[mask]))
        rows.append({
            "bin": i, "lower": round(float(lo), 4), "upper": round(float(hi), 4),
            "samples": cnt, "available": True,
            "mean_predicted": round(mp, 6),
            "observed_frequency": round(of, 6),
            "gap": round(of - mp, 6),
        })
    return rows


def expected_calibration_error(proba: Sequence[float], y_true: Sequence[int],
                               bins: int = DEFAULT_BINS) -> Optional[float]:
    """ECE = Σ (n_b/N) × |真实频率 − 预测均值|（仅统计有样本的箱）。"""
    curve = reliability_curve(proba, y_true, bins=bins)
    usable = [r for r in curve if r["available"]]
    if not usable:
        return None
    total = sum(r["samples"] for r in usable)
    if total <= 0:
        return None
    return float(sum(r["samples"] / total * abs(r["gap"]) for r in usable))


def fit_calibrator(proba_cal: Sequence[float], y_cal: Sequence[int],
                   method: str = METHOD_ISOTONIC) -> Dict[str, Any]:
    """在校准段拟合校准器（纯函数，返回可序列化描述 + transform 闭包）。

    - ``isotonic``：``sklearn.isotonic.IsotonicRegression``（非参数、单调）；
    - ``platt``：``sklearn.linear_model.LogisticRegression`` 在概率的 logit 上；
    - 样本不足 → ``available=false``，**不拟合也不外推**。
    """
    p = np.asarray(proba_cal, dtype=float)
    y = np.asarray(y_cal, dtype=float)
    n = int(min(p.size, y.size))
    if n < MIN_CALIBRATION_SAMPLES:
        return {"available": False, "method": method, "reason": (
            f"校准样本不足（{n} < {MIN_CALIBRATION_SAMPLES}），不拟合")}
    p, y = p[:n], y[:n]
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size < MIN_CALIBRATION_SAMPLES or len(np.unique(y)) < 2:
        return {"available": False, "method": method,
                "reason": "校准段标签单一或有效样本不足，不拟合"}

    try:
        if method == METHOD_ISOTONIC:
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(p, y)

            def _transform(x, _m=model):
                return np.clip(np.asarray(_m.predict(x), dtype=float), 0.0, 1.0)

            params = {"kind": "isotonic",
                      "x_thresholds": [round(float(v), 6) for v in _m_x(model)],
                      "y_thresholds": [round(float(v), 6) for v in _m_y(model)]}
        elif method == METHOD_PLATT:
            from sklearn.linear_model import LogisticRegression

            eps = 1e-6
            p_safe = np.clip(p, eps, 1 - eps)
            z = np.log(p_safe / (1 - p_safe)).reshape(-1, 1)
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            model.fit(z, y)

            def _transform(x, _m=model):
                xs = np.clip(np.asarray(x, dtype=float), eps, 1 - eps)
                zz = np.log(xs / (1 - xs)).reshape(-1, 1)
                return np.clip(_m.predict_proba(zz)[:, 1], 0.0, 1.0)

            coef = float(model.coef_.ravel()[0])
            intercept = float(model.intercept_.ravel()[0])
            params = {"kind": "platt", "coef": round(coef, 6),
                      "intercept": round(intercept, 6)}
        else:
            return {"available": False, "method": method,
                    "reason": f"未知校准方法 {method}（支持 {METHOD_ISOTONIC}/{METHOD_PLATT}）"}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "method": method,
                "reason": f"校准器拟合失败: {e}"}

    baseline_brier = brier_score(p, y)
    calibrated_brier = brier_score(_transform(p), y)
    return {
        "available": True,
        "method": method,
        "n_calibration": int(p.size),
        "params": params,
        "baseline_brier_on_calibration": baseline_brier,
        "calibrated_brier_on_calibration": calibrated_brier,
        "reason": "",
        "transform": _transform,
    }


def _m_x(model) -> Sequence[float]:
    """isotonic 阈值（兼容 sklearn 不同版本的属性名）。"""
    return getattr(model, "X_thresholds_", getattr(model, "f_.x", []))


def _m_y(model) -> Sequence[float]:
    return getattr(model, "y_thresholds_", getattr(model, "f_.y", []))


def calibrate_and_evaluate(
    proba: Sequence[float],
    y_true: Sequence[int],
    *,
    n_train: int,
    n_cal: int,
    methods: Sequence[str] = (METHOD_ISOTONIC, METHOD_PLATT),
    bins: int = DEFAULT_BINS,
    horizon_name: str = "",
) -> Dict[str, Any]:
    """三段式校准评估：训练段 < 校准段 < 复验段（T19.1）。

    ``proba`` / ``y_true`` 覆盖全段；校准器只在校准段拟合，Brier/ECE 只在
    **复验段**读数 —— 这是"样本外校准误差"，不是把校准段自己的拟合误差报上来。
    """
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n = int(min(p.size, y.size))
    p, y = p[:n], y[:n]
    train_end = int(max(0, min(n_train, n)))
    cal_end = int(max(train_end, min(train_end + max(0, n_cal), n)))

    base: Dict[str, Any] = {
        "kind": "probability_calibration",
        "generated_at": _now(),
        "horizon": horizon_name,
        "segments": {"n_total": n, "n_train": train_end,
                     "n_calibration": cal_end - train_end, "n_verify": n - cal_end},
        "bins": int(bins),
        "methods": {},
        "affects_gate": False,
        "note": ("校准层只产出证据；是否进主推理链路属 T19.4 人工检查点，"
                 "现行置信度阈值不改动。"),
    }
    if n - cal_end == 0 or cal_end - train_end < MIN_CALIBRATION_SAMPLES:
        base["available"] = False
        base["reason"] = (f"校准段 {cal_end - train_end} 或复验段 {n - cal_end} "
                          f"样本不足（校准下限 {MIN_CALIBRATION_SAMPLES}），不猜")
        return base

    p_cal, y_cal = p[train_end:cal_end], y[train_end:cal_end]
    p_ver, y_ver = p[cal_end:], y[cal_end:]

    base["baseline"] = {
        "brier": _r(brier_score(p_ver, y_ver)),
        "ece": _r(expected_calibration_error(p_ver, y_ver, bins=bins)),
        "reliability": reliability_curve(p_ver, y_ver, bins=bins),
    }

    any_available = False
    for method in methods:
        fitted = fit_calibrator(p_cal, y_cal, method=method)
        row: Dict[str, Any] = {
            "method": method,
            "available": bool(fitted.get("available")),
            "reason": fitted.get("reason", ""),
            "params": fitted.get("params"),
            "n_calibration": fitted.get("n_calibration", 0),
        }
        if fitted.get("available"):
            any_available = True
            transformed = np.asarray(fitted["transform"](p_ver), dtype=float)
            row["brier"] = _r(brier_score(transformed, y_ver))
            row["ece"] = _r(expected_calibration_error(transformed, y_ver, bins=bins))
            row["brier_delta_vs_baseline"] = _delta(row["brier"], base["baseline"]["brier"])
            row["ece_delta_vs_baseline"] = _delta(row["ece"], base["baseline"]["ece"])
            row["reliability"] = reliability_curve(transformed, y_ver, bins=bins)
            row["improved"] = bool(
                (row["brier_delta_vs_baseline"] or 0) < 0
                and (row["ece_delta_vs_baseline"] or 0) <= 0)
        base["methods"][method] = row

    base["available"] = any_available
    if not any_available:
        base["reason"] = "所有校准方法均不可用（校准样本不足或拟合失败）"
    else:
        best = min((r for r in base["methods"].values() if r["available"]),
                   key=lambda r: (r["brier"] if r["brier"] is not None else 1e9))
        base["best_by_brier"] = best["method"]
        base["conclusion"] = (
            f"校准后 Brier/ECE 均下降（{best['method']}）→ 概率确实存在高估/低估，"
            "但**是否进主推理链路须人工决定**（T19.4）"
            if best.get("improved") else
            f"校准未带来 Brier/ECE 的稳定改善（最佳 {best['method']}）→ "
            "现有概率输出在这份数据上已相对校准，引入校准层收益有限"
        )
    return base


def _r(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), ndigits)


def _delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return round(float(new) - float(old), 6)


def threshold_curve_after_calibration(
    proba: Sequence[float],
    proba_calibrated: Sequence[float],
    y_true: Sequence[int],
    fwd_ret: Sequence[float],
    *,
    grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4),
    min_samples: int = 50,
) -> Dict[str, Any]:
    """校准后置信度阈值曲线复算（T19.2）：看曲线是否更单调 / 更可解释。

    ⚠️ **不改现行阈值**：只做"校准前 vs 校准后"的曲线对照，是否切换属 T19.4。
    """
    from src.eval.confidence_curve import sweep_confidence

    p = np.asarray(proba, dtype=float)
    pc = np.asarray(proba_calibrated, dtype=float)
    y = np.asarray(y_true, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(p.size, pc.size, y.size, r.size))
    if n == 0:
        return {"available": False, "reason": "无样本", "affects_gate": False}
    p, pc, y, r = p[:n], pc[:n], y[:n], r[:n]

    before = sweep_confidence(p, r, grid=grid, min_samples=min_samples)
    after = sweep_confidence(pc, r, grid=grid, min_samples=min_samples)

    def _monotone(curve: Dict[str, Any]) -> Optional[bool]:
        hits = [row["hit_rate"] for row in curve.get("rows", [])
                if row.get("available")]
        if len(hits) < 2:
            return None
        return bool(all(b >= a - 0.005 for a, b in zip(hits, hits[1:])))

    m_before, m_after = _monotone(before), _monotone(after)
    if m_before is None or m_after is None:
        verdict = "unavailable"
        detail = "曲线可用点不足，无法判定单调性"
    elif m_after and not m_before:
        verdict = "calibration_improves_monotonicity"
        detail = "校准后阈值曲线单调性改善（校准前不单调）"
    elif m_before and not m_after:
        verdict = "calibration_worsens_monotonicity"
        detail = "校准后单调性反而变差（如实记录，不掩饰）"
    else:
        verdict = "no_material_change"
        detail = f"校准前后单调性一致（before={m_before}, after={m_after}）"

    return {
        "kind": "calibrated_threshold_curve",
        "generated_at": _now(),
        "available": True,
        "samples": n,
        "monotone_before": m_before,
        "monotone_after": m_after,
        "verdict": verdict,
        "verdict_detail": detail,
        "curve_before": before,
        "curve_after": after,
        "affects_gate": False,
        "note": ("仅呈现校准前后对照；**现行置信度阈值不变**，是否切换属 T19.4 "
                 "人工检查点。"),
    }
