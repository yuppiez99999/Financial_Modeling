"""标签口径 A/B 对比（S12 / G2）：旧固定窗口标签 vs 新三重障碍法标签。

为什么需要（Issue #29 集成方案 G2 验收口径）：
  标签重构是本轮唯一**可能直接改变门禁判定**的一步（README §18A 里程碑：
  09-19 拿到「标签重构是否有效」的硬结论）。因此必须有一个**同配置、同数据、
  同切分**的对照实验，把「换了标签之后 IC / 命中率是否变好」变成可复算的数字，
  而不是靠感觉宣布"有效"。

本模块做什么：
  - 用**同一个 LightGBM 配置**、同一份 walk-forward 折，分别对：
      · 旧标签：``target_{h}d``（未来 h 日收益 > 0，写死窗口）
      · 新标签：``label_tb_{h}d_bin``（三重障碍法折叠为二分类）
    训练并评估，输出每个口径的 IC / 命中率 / 样本 / 折数；
  - 给出**增量对照**（新 − 旧）、显著性提示（命中率差是否超出二项噪声）；
  - 结论**不管好坏都如实入库**（说明书要求：证明"没用"同样是有价值结论）。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：``affects_gate`` 恒为 False，只产出证据；
  - **不自动切换标签**：是否纳入主线由人工检查点（T12.3）决定；
  - **不挑口径**：两个口径用完全一致的切分与模型，禁止"调参调出好结果"；
  - **不猜**：样本不足 / 单折退化时如实标 unavailable。

统计口径备注：
  - IC 用 Spearman（与门禁同源 ``src.inference.ic.spearman_ic``）；
  - 命中率用 ``hit_rate``（概率去中性 0.5 后与方向一致的比例）；
  - 显著性用命中率的二项近似：``z = (acc - 0.5) / sqrt(0.25 / n)``，
    只做提示，**不据此宣布门禁解锁**。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.labeling import TripleBarrierLabeler, to_binary
from src.inference.ic import hit_rate, spearman_ic

logger = logging.getLogger(__name__)

# 命中率显著性的最小样本（低于此不给 z，避免小数样本的假显著）
MIN_SAMPLES_FOR_Z = 30


def _z_hit_rate(acc: float, n: int) -> Optional[float]:
    """命中率相对 0.5 的二项近似 z 值；样本不足返回 None。"""
    if n < MIN_SAMPLES_FOR_Z:
        return None
    se = math.sqrt(0.25 / n)
    if se <= 0:
        return None
    return (acc - 0.5) / se


def evaluate_label_variant(
    features: np.ndarray,
    labels: np.ndarray,
    forward_returns: np.ndarray,
    splits: List[tuple],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """在给定标签上跑 walk-forward LightGBM 评估（与门禁同折）。

    返回 {ic, hit_rate, samples, folds, z, available, reason}。
    """
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
            logger.warning(f"[label-ab] 折训练失败: {e}")
            continue
        scores.extend([float(p) - 0.5 for p in proba])
        returns.extend([float(r) for r in forward_returns[test_idx]])
        folds_used += 1

    if not scores:
        return {
            "available": False,
            "reason": "no_folds_trained",
            "ic": None,
            "hit_rate": None,
            "samples": 0,
            "folds": 0,
            "z": None,
        }
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


class LabelABExperiment:
    """旧标签 vs 三重障碍法标签的同折对照实验。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.labeler = TripleBarrierLabeler(self.config)

    def compare(
        self,
        combined,
        horizon_days: int,
        feature_cols: List[str],
        splits: List[tuple],
    ) -> Dict[str, Any]:
        """在已拼接的监督数据集上做 A/B。

        参数：
          - ``combined``：含 ``close`` / 特征列 / ``target_{h}d`` / ``_fwd_ret``
            的 DataFrame（由 ``scripts.evaluate_models.build_supervised`` 产出）；
            若含 ``high``/``low`` 则新标签用日内价判定触碰。
          - ``horizon_days``：预测周期。
          - ``feature_cols``：特征列名（两个口径共用，确保只变标签）。
          - ``splits``：walk-forward 折（两个口径共用，确保只变标签）。
        """
        h = int(horizon_days)
        old_col = f"target_{h}d"
        if old_col not in combined.columns or "_fwd_ret" not in combined.columns:
            return {
                "available": False,
                "reason": "missing_target_or_forward_return",
            }

        # 新标签逐行构造（combined 已是单序列/按时间排序的监督集）
        highs = combined["high"].tolist() if "high" in combined.columns else None
        lows = combined["low"].tolist() if "low" in combined.columns else None
        tb_labels = self.labeler.label_series(
            combined[["close"]].assign(
                **({"high": highs} if highs is not None else {}),
                **({"low": lows} if lows is not None else {}),
            ),
            h,
        )
        tb_bin = to_binary(tb_labels)

        X = combined[feature_cols].to_numpy(dtype=float)
        y_old = combined[old_col].to_numpy(dtype=float)
        y_new = np.array(
            [np.nan if v is None else float(v) for v in tb_bin], dtype=float
        )
        fwd = combined["_fwd_ret"].to_numpy(dtype=float)

        n = len(X)
        # 新标签的有效掩码：两端样本（波动率不足 / 未到期）为 NaN，
        # 对照实验必须在**同一批样本**上比，否则差异可能来自样本集不同。
        valid = ~np.isnan(y_new)
        if valid.sum() < MIN_SAMPLES_FOR_Z:
            return {
                "available": False,
                "reason": "insufficient_new_label_samples",
                "new_valid": int(valid.sum()),
            }
        idx = np.where(valid)[0]

        # 折内样本同样取交集，保证两口径样本完全一致。
        # 关键：splits 里的索引是 combined 的**原始位置**，而下面 Xc 已按 idx
        # 重排 —— 必须把折索引重映射为「在 idx 中的位置」，否则会错位甚至越界
        # （越界即索引到 Xc 之外）。用 pos_of[orig] = 在 idx 中的下标。
        pos_of = {int(orig): pos for pos, orig in enumerate(idx)}
        common_splits: List[tuple] = []
        for tr, te in splits:
            tr_c = np.array([pos_of[int(o)] for o in tr if int(o) in pos_of], dtype=int)
            te_c = np.array([pos_of[int(o)] for o in te if int(o) in pos_of], dtype=int)
            if len(tr_c) > 0 and len(te_c) > 0:
                common_splits.append((tr_c, te_c))
        if not common_splits:
            return {"available": False, "reason": "no_common_splits"}

        Xc = X[idx]
        fwd_c = fwd[idx]
        y_old_c = y_old[idx]
        y_new_c = y_new[idx]

        old_res = evaluate_label_variant(Xc, y_old_c, fwd_c, common_splits, self.config)
        new_res = evaluate_label_variant(Xc, y_new_c, fwd_c, common_splits, self.config)

        comparison: Dict[str, Any] = {
            "available": bool(old_res["available"] and new_res["available"]),
            "horizon_days": h,
            "common_samples": int(len(idx)),
            "folds": len(common_splits),
            "old_label": old_res,
            "new_label": new_res,
        }
        if comparison["available"]:
            d_ic = (new_res["ic"] or 0.0) - (old_res["ic"] or 0.0)
            d_hr = (new_res["hit_rate"] or 0.0) - (old_res["hit_rate"] or 0.0)
            comparison["delta"] = {
                "ic": round(d_ic, 6),
                "hit_rate": round(d_hr, 6),
                "hit_rate_z_old": old_res["z"],
                "hit_rate_z_new": new_res["z"],
                # 仅当新口径 z 更接近显著且增量 > 0 时给"有改善迹象"；
                # 措辞保守，不宣布结论（结论由 A/B 报告与人工检查点给出）
                "improved": bool(d_ic > 0 and d_hr > 0),
            }
        else:
            comparison["reason"] = "one_or_both_variants_unavailable"
        comparison["affects_gate"] = False
        return comparison


def build_ab_report(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """把多周期 A/B 结果汇总为报告（report_only）。

    ``results``：{horizon_str: compare(...) 的返回值}
    """
    from datetime import datetime

    out: Dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "kind": "label_ab",
        "affects_gate": False,
        "note": (
            "旧标签 = 未来 h 日收益 > 0（固定窗口）；"
            "新标签 = 三重障碍法折叠二分类。两口径同数据、同折、同模型配置。"
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
            "标签重构在同口径下出现正向增量（需人工复核后再决定是否纳入主线）"
            if any_improved
            else "标签重构未跑出正向增量；按说明书要求如实入库，不纳入主线"
        ),
    }
    return out
