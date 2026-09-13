"""TreeSHAP 状态×特征归因（S24 / I4：lightgbm pred_contrib，零新依赖）。

回答的问题（Issue #54）：S12 判定特征扩充无增量，但「模型到底在用什么」
从未归因；S18 状态分层 differentiated / 状态特征 A/B degraded 的**特征成因**
需要归因证据。归因是**解释**，不是扩充 —— 不新增任何特征进训练。

## 实现口径

- 用 lightgbm 原生 ``predict(X, pred_contrib=True)``（TreeSHAP，零新依赖，
  不需要 shap 包）；贡献矩阵形状 (n_samples, n_features + 1)，末列为
  bias（期望值），聚合时剔除。
- **无未来函数**：归因只解释「模型对既有特征的读数」，特征本身由上游
  管线保证 T 日可得；本模块不做任何特征计算。
- **fail-close**：非 LightGBM 模型（如 DummyModel 占位符）→ 明确报错，
  绝不静默降级为「假归因」。

## 已知边界（2026-09-13 如实登记）

`models/*.pkl` 当前为 DummyModel 占位符（真实训练待跑）→ 真实归因待
`python main.py train` 产出真 LightGBM 后运行；本模块 + 测试（合成数据
训练微型 LGBM）证明机制正确。状态交叉需要 S18 状态标签（hmmlearn 未装、
reports/regime 无工件）→ 状态交叉接口就绪、真实交叉**待跑**。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BIAS_COL = "__bias__"


def tree_shap_contrib(model: Any, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP 贡献矩阵（(n, n_features+1)，末列 bias）。

    仅接受 LightGBM（Booster 或 sklearn API）；其他模型 fail-close 报错。
    """
    is_lgbm = (
        model.__class__.__module__.startswith("lightgbm")
        or type(model).__name__ in ("Booster", "LGBMClassifier", "LGBMRegressor")
    )
    if not is_lgbm:
        raise ValueError(
            f"TreeSHAP 归因需要 LightGBM 模型，得到 {type(model).__name__}"
            "（DummyModel/占位模型不能产出真归因，拒绝伪装）")
    try:
        contrib = model.predict(X, pred_contrib=True)
    except TypeError:
        contrib = model.predict(X.values, pred_contrib=True)
    contrib = np.asarray(contrib, dtype="float64")
    if contrib.ndim != 2 or contrib.shape[1] != X.shape[1] + 1:
        raise ValueError(f"贡献矩阵形状异常: {contrib.shape}")
    return contrib


def contribution_frame(contrib: np.ndarray, feature_names: List[str],
                       index: Optional[pd.Index] = None) -> pd.DataFrame:
    """贡献矩阵 → DataFrame（列 = 特征名 + bias），index 可挂时间。"""
    names = list(feature_names) + [BIAS_COL]
    if contrib.shape[1] != len(names):
        raise ValueError(f"特征名数 {len(feature_names)} 与贡献矩阵不匹配 {contrib.shape}")
    return pd.DataFrame(contrib, columns=names, index=index)


def aggregate_importance(contrib_df: pd.DataFrame) -> pd.DataFrame:
    """全局重要性：mean|SHAP| 与占比（bias 剔除；占比按 mean|SHAP| 归一）。"""
    imp = contrib_df.drop(columns=[BIAS_COL]).abs().mean()
    share = imp / imp.sum() if imp.sum() > 0 else imp * 0.0
    out = pd.DataFrame({
        "mean_abs_shap": imp.round(8),
        "share": share.round(6),
    }).sort_values("mean_abs_shap", ascending=False)
    return out.reset_index().rename(columns={"index": "feature"})


def window_drift(contrib_df: pd.DataFrame, split: float = 0.5) -> Dict[str, Any]:
    """时间维度归因漂移：前半窗 vs 后半窗的 mean|SHAP| 排名变化。

    回答「模型在用什么」是否随时间漂移（与 S23 分布漂移互补：分布漂移看
    输入，归因漂移看模型读数）。
    """
    n = len(contrib_df)
    if n < 20:
        return {"available": False, "reason": f"样本不足（{n} < 20）",
                "affects_gate": False}
    cut = int(n * float(split))
    early = aggregate_importance(contrib_df.iloc[:cut]).set_index("feature")
    late = aggregate_importance(contrib_df.iloc[cut:]).set_index("feature")
    joined = early.join(late, lsuffix="_early", rsuffix="_late", how="outer").fillna(0.0)
    joined["rank_early"] = joined["mean_abs_shap_early"].rank(ascending=False)
    joined["rank_late"] = joined["mean_abs_shap_late"].rank(ascending=False)
    joined["rank_shift"] = (joined["rank_late"] - joined["rank_early"]).abs()
    top_shift = joined.sort_values("rank_shift", ascending=False).head(5)
    return {
        "available": True,
        "affects_gate": False,
        "split": [int(cut), int(n - cut)],
        "max_rank_shift": float(joined["rank_shift"].max()),
        "top_moved_features": [
            {"feature": f,
             "rank_early": int(r["rank_early"]), "rank_late": int(r["rank_late"])}
            for f, r in top_shift.iterrows()],
        "table": joined.round(8).reset_index().to_dict(orient="records"),
    }


def state_cross(contrib_df: pd.DataFrame,
                state_labels: pd.Series) -> Dict[str, Any]:
    """状态×特征交叉（T24.2）：各状态下特征的 mean|SHAP| 差异。

    state_labels index 需与 contrib_df 对齐（S18 的 bull/range/bear）。
    真实状态标签依赖 hmmlearn（未装）→ 接口就绪、真实交叉待跑。
    """
    if len(state_labels) != len(contrib_df):
        return {"available": False,
                "reason": "状态标签与贡献矩阵长度不一致", "affects_gate": False}
    states = pd.Series(state_labels).astype(str).to_numpy()
    feats = contrib_df.drop(columns=[BIAS_COL]).abs()
    rows: List[Dict[str, Any]] = []
    for st in sorted(set(states)):
        mask = states == st
        if mask.sum() == 0:
            continue
        imp = feats[mask].mean()
        top = imp.sort_values(ascending=False).head(3)
        rows.append({
            "state": st, "n": int(mask.sum()),
            "top_features": [{"feature": f, "mean_abs_shap": round(float(v), 8)}
                             for f, v in top.items()],
        })
    return {"available": len(rows) >= 2, "affects_gate": False,
            "states": rows,
            "note": "状态交叉只解释「不同状态下模型读什么」，不改变训练特征集"}
