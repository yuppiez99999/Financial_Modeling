"""模型优化对照实验（Issue #55 步骤②③）：标签口径对照 + 周期权重重排。

## 问题从哪来

Issue #55 的实测读数把下一步指得很清楚（38 标的池、真实日K、真实训练 LightGBM）：

- 锚点命中率 52.9%、AUC ≈ 0.50~0.54、`advisory_consumable` 恒 0 / 38；
- 三周期置信度几乎全部贴地，契约综合分只在 0.5079~0.5135 之间动。

排查优先级：① 池共线性（见 `pool_collinearity.py`）→ ② 标签口径 → ③ 周期选择。

本模块落地 **②** 与 **③**，两者共用同一套 walk-forward 折与同一个模型配置
（**只变被检验的那一个变量**，其余逐字节一致）：

### ② 标签口径对照（`label_variant_experiment`）

- 旧标签：`target_{h}d` —— 未来 h 日收盘收益 > 0（固定窗口，写死 5/10/20）；
- 新标签：三重障碍法折叠二分类 `label_tb_{h}d_bin` —— 止盈/止损阈值随波动率自适应，
  且**看路径不看终点**（先跌 15% 再涨 2% 与一路上涨 2% 不再同标）。

对照口径不止「IC / 命中率」，还带**平均已实现收益**——因为 Issue #55 已经证明
「命中率高」可以和高置信度一起反向，必须与收益联合判定（与
`decision_analytics.verdict` 同一条纪律：两条腿都过才算改善）。

### ③ 周期选择重排（`horizon_weight_experiment`）

现行聚合权重 `short 0.30 / mid 0.35 / long 0.35` 是**沿用下来的**；
历史上 5d ≈ 随机、10d/20d 较强（README 有记录）。本实验：

- 用同一套折分别评估 5/10/20 日的 IC / 命中率 / 平均收益 / 显著性；
- 给出**按证据重排**的权重建议（`evidence_weights`），与现行权重并列对照；
- 给出「若按建议权重聚合，综合分的区分度是否变好」的读数。

**权重重排也不自动生效** —— 属产品口径变更，须人工签字（与 S11/S12/S13/S15 同款纪律）。

## 边界（比功能重要）

- **只读**：`affects_gate=False`；不写配置、不改 `data.prediction_horizons`、
  不改 `SignalEngine.DEFAULT_HORIZON_WEIGHTS`；
- **不挑口径**：两个口径共用同一折、同一模型配置、同一批样本（标签 NaN 取交集）；
- **不择优**：多个候选同时给出并整体校正（Bonferroni/Holm 沿用 `horizon_decision`），
  不因为某个数字好看就放宽口径；
- **不猜**：样本不足 / 单折退化 → 如实 `available=false` + reason；
- **无前视**：标签与特征都逐标的构造；折按时间顺序；波动率只用 `[.., t]`。
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 与门禁同源的最小样本口径（低于此不下结论）
MIN_SAMPLES = 30
MIN_HIT_SAMPLES = 30


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        pa = pd.Series(a).rank().to_numpy(dtype=float)
        pb = pd.Series(b).rank().to_numpy(dtype=float)
        if pa.std() == 0 or pb.std() == 0:
            return None
        return float(np.corrcoef(pa, pb)[0, 1])
    except Exception:  # noqa: BLE001
        return None


def _hit_rate(proba: Sequence[float], fwd: Sequence[float]) -> Optional[float]:
    """命中率：概率去中性 0.5 后与实际收益方向一致的比例（与门禁同源口径）。"""
    if not len(proba):
        return None
    ok = 0
    n = 0
    for p, r in zip(proba, fwd):
        pf = _finite(p)
        rf = _finite(r)
        if pf is None or rf is None:
            continue
        n += 1
        if (pf - 0.5) * rf > 0:
            ok += 1
    return (ok / n) if n else None


def _mean_return(proba: Sequence[float], fwd: Sequence[float]) -> Optional[float]:
    """按信号方向建仓的**平均已实现收益**（去中性后 >0 做多、<0 做空）。"""
    vals = []
    for p, r in zip(proba, fwd):
        pf = _finite(p)
        rf = _finite(r)
        if pf is None or rf is None:
            continue
        vals.append((1.0 if pf - 0.5 > 0 else -1.0) * rf)
    return float(np.mean(vals)) if vals else None


def _auc(y: Sequence[float], proba: Sequence[float]) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        yy = np.asarray([_finite(v) for v in y], dtype=float)
        pp = np.asarray([_finite(v) for v in proba], dtype=float)
        m = np.isfinite(yy) & np.isfinite(pp)
        if m.sum() < MIN_SAMPLES or len(np.unique(yy[m])) < 2:
            return None
        return float(roc_auc_score(yy[m], pp[m]))
    except Exception:  # noqa: BLE001
        return None


def evaluate_variant(X: np.ndarray, y: np.ndarray, fwd: np.ndarray,
                     splits: Sequence[Tuple[np.ndarray, np.ndarray]],
                     config: Dict[str, Any],
                     label: str = "") -> Dict[str, Any]:
    """在给定标签上跑 walk-forward 训练并评估（两个口径共用，确保只变标签）。"""
    from src.train.models.lightgbm_model import LightGBMModel

    probas: List[float] = []
    ys: List[float] = []
    rets: List[float] = []
    folds_used = 0
    for train_idx, test_idx in splits:
        y_tr = y[train_idx]
        y_tr = y_tr[np.isfinite(y_tr)]
        if len(np.unique(y_tr)) < 2:
            continue
        try:
            model = LightGBMModel(dict(config))
            model.train(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[model-improve] {label} 折训练失败: {e}")
            continue
        probas.extend([float(p) for p in proba])
        ys.extend([float(v) for v in y[test_idx]])
        rets.extend([float(r) for r in fwd[test_idx]])
        folds_used += 1

    out: Dict[str, Any] = {
        "label": str(label),
        "available": False,
        "reason": "",
        "n_samples": len(probas),
        "folds": folds_used,
        "ic": None,
        "hit_rate": None,
        "mean_realized_return": None,
        "auc": None,
    }
    if len(probas) < MIN_SAMPLES:
        out["reason"] = f"样本不足（{len(probas)} < {MIN_SAMPLES}）或无可训练折"
        return out
    ic = _spearman(probas, rets)
    hr = _hit_rate(probas, rets)
    mr = _mean_return(probas, rets)
    auc = _auc(ys, probas)  # 只算一次（此前重复调用两遍，纯浪费）
    out.update({
        "available": True,
        "ic": round(ic, 6) if ic is not None else None,
        "hit_rate": round(hr, 6) if hr is not None else None,
        "mean_realized_return": round(mr, 8) if mr is not None else None,
        "auc": round(auc, 6) if auc is not None else None,
    })
    return out


def _build_supervised_with_tb(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                              horizon_days: int) -> pd.DataFrame:
    """逐标的构造监督集：**同时**带旧标签 `target_{h}d` 与三重障碍法标签。

    逐标的构造（不是 concat 后统一 shift）—— 多标的 concat 后 shift 会跨标的取价。
    """
    from src.data.labeling import TripleBarrierLabeler, to_binary
    from src.data.preprocessor import FeatureEngineer

    fe = FeatureEngineer(config)
    labeler = TripleBarrierLabeler(config)
    h = int(horizon_days)
    parts: List[pd.DataFrame] = []
    for symbol, df in (data or {}).items():
        try:
            feats = fe.transform(df.copy(), h)
            feats = fe.create_target(feats, h)
            feats = feats.copy()
            feats["_symbol"] = symbol
            feats["_fwd_ret"] = feats["close"].shift(-h) / feats["close"] - 1
            tb = labeler.label_series(
                feats[["close"]].assign(
                    **({"high": feats["high"]} if "high" in feats.columns else {}),
                    **({"low": feats["low"]} if "low" in feats.columns else {}),
                ), h)
            feats[f"label_tb_{h}d_bin"] = [
                np.nan if v is None else float(v) for v in to_binary(tb)
            ]
            feats = feats.dropna(subset=[f"target_{h}d"])
            if feats.empty:
                continue
            parts.append(feats)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[model-improve] {symbol} 监督集构造失败: {e}")
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return combined


def label_variant_experiment(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                             horizon_days: int,
                             folds: int = 3) -> Dict[str, Any]:
    """② 标签口径对照：固定 h 二分类 vs 三重障碍法（同折、同模型、同样本）。"""
    from scripts.evaluate_models import walk_forward_splits

    h = int(horizon_days)
    combined = _build_supervised_with_tb(data, config, h)
    out: Dict[str, Any] = {
        "kind": "label_variant",
        "horizon_days": h,
        "available": False,
        "reason": "",
        "affects_gate": False,
    }
    if combined.empty:
        out["reason"] = "无可用监督数据（data/raw 为空或特征构造失败）"
        return out

    from src.data.preprocessor import FeatureEngineer
    fe = FeatureEngineer(config)
    cols = [c for c in fe.get_feature_columns(combined, h) if not str(c).startswith("_")]
    new_col = f"label_tb_{h}d_bin"
    if new_col not in combined.columns:
        out["reason"] = "三重障碍法标签列缺失"
        return out

    y_new = combined[new_col].to_numpy(dtype=float)
    valid = np.isfinite(y_new)
    if valid.sum() < MIN_SAMPLES:
        out["reason"] = f"新标签有效样本不足（{int(valid.sum())} < {MIN_SAMPLES}）"
        return out
    idx = np.where(valid)[0]
    X = combined.iloc[idx][cols].to_numpy(dtype=float)
    y_old = combined.iloc[idx][f"target_{h}d"].to_numpy(dtype=float)
    y_new_v = y_new[idx]
    fwd = combined.iloc[idx]["_fwd_ret"].to_numpy(dtype=float)

    # 折在**取交集后的样本空间**上生成一次，两口径共用（只变标签）
    splits = walk_forward_splits(len(X), int(folds))
    if not splits:
        out["reason"] = "无法生成 walk-forward 折"
        return out

    old_res = evaluate_variant(X, y_old, fwd, splits, config, label="fixed_h")
    new_res = evaluate_variant(X, y_new_v, fwd, splits, config, label="triple_barrier")
    out["available"] = bool(old_res["available"] and new_res["available"])
    out["common_samples"] = int(len(X))
    out["n_folds"] = len(splits)
    out["old_label"] = old_res
    out["new_label"] = new_res
    if not out["available"]:
        out["reason"] = "一个或两个口径不可用（样本不足或无可训练折）"
        return out

    d_ic = (new_res["ic"] or 0.0) - (old_res["ic"] or 0.0)
    d_hr = (new_res["hit_rate"] or 0.0) - (old_res["hit_rate"] or 0.0)
    d_mr = (new_res["mean_realized_return"] or 0.0) - (old_res["mean_realized_return"] or 0.0)
    # 与 decision_analytics.verdict 同一条纪律：**两条腿都过**才算改善
    # （命中率与平均已实现收益同时不劣，且 IC 不劣）
    improved = bool(d_hr > 0 and d_mr > 0 and d_ic >= 0)
    out["delta"] = {
        "ic": round(d_ic, 6),
        "hit_rate": round(d_hr, 6),
        "mean_realized_return": round(d_mr, 8),
        "improved": improved,
        "improved_basis": ("命中率与平均已实现收益需同时改善，且 IC 不劣 —— "
                           "与 decision_analytics.verdict 同一条纪律"),
    }
    out["conclusion"] = (
        "三重障碍法标签在同折同模型下跑出正向增量，可作为口径变更候选（须人工签字）"
        if improved else
        "三重障碍法标签未跑出正向增量：如实入库，不纳入主线"
    )
    return out


def horizon_weight_experiment(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                              horizons: Sequence[int], folds: int = 3,
                              current_weights: Optional[Dict[str, float]] = None
                              ) -> Dict[str, Any]:
    """③ 周期选择：逐周期独立评估 + 按证据重排聚合权重（仅建议，不生效）。"""
    from src.eval.horizon_decision import holm_adjust, bonferroni, ic_p_value
    from scripts.evaluate_models import walk_forward_splits

    out: Dict[str, Any] = {
        "kind": "horizon_weight",
        "available": False,
        "reason": "",
        "affects_gate": False,
        "current_weights": dict(current_weights or {}),
        "horizons": {},
    }
    per: Dict[str, Dict[str, Any]] = {}
    for h in horizons:
        h = int(h)
        combined = _build_supervised_with_tb(data, config, h)
        if combined.empty:
            per[str(h)] = {"available": False, "reason": "无可用监督数据", "horizon_days": h}
            continue
        from src.data.preprocessor import FeatureEngineer
        fe = FeatureEngineer(config)
        cols = [c for c in fe.get_feature_columns(combined, h) if not str(c).startswith("_")]
        X = combined[cols].to_numpy(dtype=float)
        y = combined[f"target_{h}d"].to_numpy(dtype=float)
        fwd = combined["_fwd_ret"].to_numpy(dtype=float)
        splits = walk_forward_splits(len(X), int(folds))
        res = evaluate_variant(X, y, fwd, splits, config, label=f"{h}d")
        res["horizon_days"] = h
        if res.get("available"):
            # 显著性：IC 的近似 t 检验（与 horizon_decision 同源）
            p = ic_p_value(res["ic"] or 0.0, res["n_samples"])
            res["p_value"] = round(float(p), 6)
        per[str(h)] = res
    out["horizons"] = per

    usable = [r for r in per.values() if r.get("available")]
    if not usable:
        out["reason"] = "全部周期均不可用（样本不足或无可训练折）"
        return out
    out["available"] = True

    # 多重比较：候选数是 5/10/20 三个，整体校正后不显著就不该当证据
    pvals = [float(r.get("p_value", 1.0)) for r in usable]
    holm = holm_adjust(pvals)
    n_trials = len(usable)
    for r, ph in zip(usable, holm):
        r["p_bonferroni"] = round(bonferroni(float(r.get("p_value", 1.0)), n_trials), 6)
        r["p_holm"] = round(ph, 6)
        r["significant"] = bool(max(r["p_bonferroni"], r["p_holm"]) <= 0.05)

    # 证据权重：**只用非负、单调的信噪比**，且不因为某周期"好看"而无限放大
    # 口径：权重 ∝ max(0, |IC|) × max(0, 命中率-0.5)，全部不达标则退化为等权
    raw: Dict[str, float] = {}
    for r in usable:
        ic = abs(float(r.get("ic") or 0.0))
        edge = max((float(r.get("hit_rate") or 0.5)) - 0.5, 0.0)
        raw[str(r["horizon_days"])] = ic * edge
    total = sum(raw.values())
    if total <= 0:
        evidence_weights = {k: round(1.0 / len(raw), 6) for k in raw}
        out["weights_reason"] = ("所有周期的 IC×超额命中率均为 0：无证据支持倾斜，"
                                 "退化为等权（不猜、不择优）")
    else:
        evidence_weights = {k: round(v / total, 6) for k, v in raw.items()}
        out["weights_reason"] = ("证据权重 ∝ |IC| × (命中率-0.5) 归一化；"
                                 "全部不显著时该建议只作参考，不得据此切换口径")
    out["evidence_weights"] = evidence_weights
    out["raw_scores"] = {k: round(v, 8) for k, v in raw.items()}
    out["any_significant"] = any(r.get("significant") for r in usable)
    out["conclusion"] = (
        "存在经多重比较校正后仍显著的周期：" + str([r["horizon_days"] for r in usable
                                                      if r.get("significant")])
        if out["any_significant"] else
        "无周期通过多重比较校正：按仓库纪律，不据此重排聚合权重（如实入库）"
    )
    return out


def build_report(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                 horizons: Sequence[int] = (5, 10, 20),
                 folds: int = 3) -> Dict[str, Any]:
    """② + ③ 合并报告（落盘结构）。"""
    current_weights = (config.get("signal", {}) or {}).get("horizon_weights") or {
        "5": 0.30, "10": 0.35, "20": 0.35,
    }
    report: Dict[str, Any] = {
        "kind": "model_improvement",
        "generated_at": datetime.now().astimezone().isoformat(),
        "affects_gate": False,
        "readonly": True,
        "n_symbols": int(len(data or {})),
        "horizons": [int(h) for h in horizons],
        "folds": int(folds),
    }
    if not data:
        report.update({"available": False, "reason": "无可用行情数据（data/raw 为空）"})
        return report
    report["label_variant"] = {
        str(int(h)): label_variant_experiment(data, config, int(h), folds=folds)
        for h in horizons
    }
    report["horizon_weight"] = horizon_weight_experiment(
        data, config, horizons, folds=folds, current_weights=current_weights)
    # 只做「5/10/20 三周期」的标签对照 + 周期重排；其他周期（若配置里有）不动
    report["available"] = bool(
        any(v.get("available") for v in report["label_variant"].values())
        or report["horizon_weight"].get("available"))
    if not report["available"]:
        report["reason"] = "标签对照与周期重排均不可用"
    report["conclusion_note"] = (
        "本报告只产出证据：标签口径切换与周期权重重排均属**产品口径变更**，"
        "`affects_gate=False`，须人工签字并重做泄漏/偏差审查后才可生效。"
    )
    return report
