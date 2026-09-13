"""S24 TreeSHAP 归因守卫：机制正确性（合成 LGBM）+ fail-close 边界。

全部离线：合成数据训练微型 LightGBM（<1s）证明 pred_contrib 归因链路；
不依赖 models/ 下的占位模型。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

lightgbm = pytest.importorskip("lightgbm")

from src.eval import feature_attribution as fa  # noqa: E402


def _train_tiny_lgbm(n=400, seed=11):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "strong": rng.normal(size=n),
        "weak": rng.normal(size=n),
        "noise": rng.normal(size=n),
    })
    # y 主要由 strong 驱动（非线性），weak 微弱，noise 无信息
    y = (0.9 * X["strong"] + 0.05 * X["weak"]
         + 0.3 * np.sin(X["strong"] * 2.0)
         + rng.normal(0, 0.1, n))
    model = lightgbm.LGBMRegressor(
        n_estimators=40, max_depth=3, verbose=-1, random_state=seed)
    model.fit(X, y)
    return model, X


# ---------------------------------------------------------------- fail-close
def test_rejects_non_lightgbm_model():
    class FakeModel:
        def predict(self, X, pred_contrib=True):
            return np.zeros((len(X), X.shape[1] + 1))
    with pytest.raises(ValueError, match="LightGBM"):
        fa.tree_shap_contrib(FakeModel(), pd.DataFrame({"a": [1.0]}))


def test_rejects_malformed_contrib_matrix():
    model = _train_tiny_lgbm()[0]
    good = fa.tree_shap_contrib(model, pd.DataFrame({"strong": [0.1], "weak": [0.2], "noise": [0.3]}))
    with pytest.raises(ValueError):
        fa.contribution_frame(good[:, :2], ["strong", "weak", "noise"])


# ---------------------------------------------------------------- 归因方向
def test_attribution_ranks_strong_feature_first():
    model, X = _train_tiny_lgbm()
    contrib = fa.tree_shap_contrib(model, X)
    cdf = fa.contribution_frame(contrib, list(X.columns))
    imp = fa.aggregate_importance(cdf)
    assert imp.iloc[0]["feature"] == "strong"
    rank = {r["feature"]: i for i, r in imp.iterrows()}
    assert rank["strong"] < rank["weak"]
    # share 归一
    assert imp["share"].sum() == pytest.approx(1.0, abs=1e-9)


def test_contrib_sums_to_prediction():
    """SHAP 可加性：行内贡献（含 bias）之和 == 模型预测。"""
    model, X = _train_tiny_lgbm(n=300)
    contrib = fa.tree_shap_contrib(model, X)
    pred = np.asarray(model.predict(X))
    assert np.allclose(contrib.sum(axis=1), pred, atol=1e-6)


# ---------------------------------------------------------------- 窗口漂移
def test_window_drift_detects_regime_change_in_usage():
    """前半 strong 驱动、后半改 weak 驱动 → 排名明显移动。"""
    rng = np.random.default_rng(3)
    n = 240
    X = pd.DataFrame({"strong": rng.normal(size=n),
                      "weak": rng.normal(size=n),
                      "noise": rng.normal(size=n)})
    y = np.concatenate([
        0.9 * X["strong"].to_numpy()[:n // 2],
        0.9 * X["weak"].to_numpy()[n // 2:],
    ]) + rng.normal(0, 0.05, n)
    model = lightgbm.LGBMRegressor(n_estimators=60, max_depth=3,
                                   verbose=-1, random_state=3)
    model.fit(X, y)
    contrib = fa.tree_shap_contrib(model, X)
    cdf = fa.contribution_frame(contrib, list(X.columns))
    drift = fa.window_drift(cdf)
    assert drift["available"] is True
    moved = {d["feature"] for d in drift["top_moved_features"]}
    assert moved & {"strong", "weak"}


def test_window_drift_fails_closed_below_min_sample():
    cdf = pd.DataFrame({"a": np.random.default_rng(0).normal(size=10),
                        fa.BIAS_COL: np.zeros(10)})
    out = fa.window_drift(cdf)
    assert out["available"] is False


# ---------------------------------------------------------------- 状态交叉
def test_state_cross_profiles_differ_by_state():
    rng = np.random.default_rng(4)
    n = 300
    X = pd.DataFrame({"f": rng.normal(size=n), "g": rng.normal(size=n)})
    contrib = np.column_stack([
        np.where(np.arange(n) < n // 2, 0.8, 0.05),   # bull 期 f 重要
        np.where(np.arange(n) < n // 2, 0.05, 0.8),   # bear 期 g 重要
        np.zeros(n),
    ])
    cdf = pd.DataFrame(contrib, columns=["f", "g", fa.BIAS_COL])
    states = pd.Series(["bull"] * (n // 2) + ["bear"] * (n - n // 2))
    out = fa.state_cross(cdf, states)
    assert out["available"] is True
    by_state = {s["state"]: s for s in out["states"]}
    assert by_state["bull"]["top_features"][0]["feature"] == "f"
    assert by_state["bear"]["top_features"][0]["feature"] == "g"


def test_state_cross_rejects_length_mismatch():
    cdf = pd.DataFrame({"a": [1.0, 2.0], fa.BIAS_COL: [0.0, 0.0]})
    out = fa.state_cross(cdf, pd.Series(["bull"]))
    assert out["available"] is False
