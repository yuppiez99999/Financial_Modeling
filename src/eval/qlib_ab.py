"""Alpha158 增量验证（S13 / G3，T13.3）：qlib 因子 vs 现有 15 因子的同口径 A/B。

为什么需要（Issue #29 集成方案 G3 验收口径）：
  「Alpha158 有没有增量」必须变成**同数据、同折、同模型配置**下可复算的数字，
  而不是"qlib 名气大、因子多，所以引入"。本模块与 S12 的 label_ab 同构：

    基准臂（base） : 现有特征（``FeatureEngineer.get_feature_columns``，
                    含现行 15 因子口径的技术特征集）；
    对照臂（a158） : 基准特征 + ``factor_a158_*`` 5 列（Alpha158 族压缩因子）。

  两臂**只差特征集**：同一批样本、同一组 walk-forward 折、同一个 LightGBM
  配置、同一个标签口径（现行 ``target_{h}d``，不用 S12 新标签 —— 一次只变
  一个变量，否则增量来源不可归因）。

判定（保守）：
  - ``improved`` 需 IC 与命中率**同向**变好，且命中率差给出二项 z 提示；
  - 任何一臂单折退化 / 样本不足 → 如实标 unavailable，不猜。

边界：``affects_gate`` 恒为 False；结论不管好坏写入
``00_kickoff/qlib_alpha158_conclusion.md``（T13.4 人工检查点的输入）。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.inference.ic import hit_rate, spearman_ic

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_Z = 30


def _z_hit_rate(acc: float, n: int) -> Optional[float]:
    """命中率相对 0.5 的二项近似 z 值；样本不足返回 None（与 label_ab 同源）。"""
    if n < MIN_SAMPLES_FOR_Z:
        return None
    se = math.sqrt(0.25 / n)
    if se <= 0:
        return None
    return (acc - 0.5) / se


def evaluate_arm(
    features: np.ndarray,
    labels: np.ndarray,
    forward_returns: np.ndarray,
    splits: List[tuple],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """单臂 walk-forward LightGBM 评估（与 label_ab.evaluate_label_variant 同源）。"""
    from src.train.models.lightgbm_model import LightGBMModel

    scores: List[float] = []
    returns: List[float] = []
    folds_used = 0
    for train_idx, test_idx in splits:
        y_tr = labels[train_idx]
        if len(np.unique(y_tr[~np.isnan(y_tr)])) < 2:
            continue
        try:
            model = LightGBMModel(config)
            model.train(features[train_idx], y_tr)
            proba = model.predict_proba(features[test_idx])[:, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning("[qlib-ab] 折训练失败: %s", e)
            continue
        scores.extend([float(p) - 0.5 for p in proba])
        returns.extend([float(r) for r in forward_returns[test_idx]])
        folds_used += 1

    if not scores:
        return {"available": False, "reason": "no_folds_trained", "ic": None,
                "hit_rate": None, "samples": 0, "folds": 0, "z": None}
    ic = spearman_ic(scores, returns)
    hr = hit_rate(scores, returns)
    z = _z_hit_rate(hr, len(scores)) if hr is not None else None
    return {
        "available": True,
        "ic": round(float(ic), 6) if ic is not None else None,
        "hit_rate": round(float(hr), 6) if hr is not None else None,
        "samples": len(scores),
        "folds": folds_used,
        "z": round(float(z), 3) if z is not None else None,
        "reason": "",
    }


class QlibABExperiment:
    """Alpha158 特征增量对照实验（特征 A/B，标签/折/模型全同）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def compare(
        self,
        combined,
        horizon_days: int,
        base_cols: List[str],
        extra_cols: List[str],
        splits: List[tuple],
    ) -> Dict[str, Any]:
        """在已拼接的监督数据集上做特征增量 A/B。

        Args:
          - ``combined``：``build_supervised`` 产出，且已由调用方追加
            ``factor_a158_*`` 列（Alpha158 族压缩因子）；
          - ``base_cols``：基准特征列（现行特征集）；
          - ``extra_cols``：增量特征列（factor_a158_*）；
          - ``splits``：walk-forward 折（两臂共用）。
        """
        h = int(horizon_days)
        target_col = f"target_{h}d"
        if target_col not in combined.columns or "_fwd_ret" not in combined.columns:
            return {"available": False, "reason": "missing_target_or_forward_return"}

        if not extra_cols or not all(c in combined.columns for c in extra_cols):
            return {"available": False, "reason": "missing_a158_columns"}

        y = combined[target_col].to_numpy(dtype=float)
        fwd = combined["_fwd_ret"].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < MIN_SAMPLES_FOR_Z:
            return {"available": False, "reason": "insufficient_samples",
                    "valid": int(valid.sum())}

        X_base = combined[base_cols].to_numpy(dtype=float)
        X_full = combined[base_cols + extra_cols].to_numpy(dtype=float)

        base_res = evaluate_arm(X_base, y, fwd, splits, self.config)
        full_res = evaluate_arm(X_full, y, fwd, splits, self.config)

        comparison: Dict[str, Any] = {
            "available": bool(base_res["available"] and full_res["available"]),
            "horizon_days": h,
            "base_features": len(base_cols),
            "a158_features": len(base_cols) + len(extra_cols),
            "extra_cols": list(extra_cols),
            "folds": len(splits),
            "base": base_res,
            "with_a158": full_res,
        }
        if comparison["available"]:
            d_ic = (full_res["ic"] or 0.0) - (base_res["ic"] or 0.0)
            d_hr = (full_res["hit_rate"] or 0.0) - (base_res["hit_rate"] or 0.0)
            comparison["delta"] = {
                "ic": round(d_ic, 6),
                "hit_rate": round(d_hr, 6),
                "hit_rate_z_base": base_res["z"],
                "hit_rate_z_a158": full_res["z"],
                # 与 label_ab 同一保守口径：IC 与命中率同向变好才算"有改善迹象"
                "improved": bool(d_ic > 0 and d_hr > 0),
            }
        else:
            comparison["reason"] = "one_or_both_arms_unavailable"
        comparison["affects_gate"] = False
        return comparison


def build_ab_report(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """把多周期特征 A/B 结果汇总为报告（report_only）。"""
    from datetime import datetime

    out: Dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "kind": "qlib_alpha158_ab",
        "affects_gate": False,
        "note": (
            "基准臂 = 现行特征集；对照臂 = 现行特征 + factor_a158_*（Alpha158 族压缩）。"
            "两臂同数据、同折、同模型配置、同标签（现行 target_{h}d）。"
        ),
        "horizons": {},
    }
    any_improved = False
    for hname, res in results.items():
        out["horizons"][hname] = res
        if isinstance(res, dict) and (res.get("delta") or {}).get("improved"):
            any_improved = True
    out["summary"] = {
        "any_improved": any_improved,
        "conclusion": (
            "Alpha158 因子在现行特征之上出现正向增量（需 T13.4 人工复核后再决定是否纳入主线）"
            if any_improved
            else "Alpha158 因子未跑出正向增量；按验收口径如实入库，不纳入主线"
        ),
    }
    return out
