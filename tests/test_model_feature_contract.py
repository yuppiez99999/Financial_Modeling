"""模型特征契约守卫：feature_cols 持久化 + 推理按名对齐 fail-close。

背景（2026-09-15 模型退化诊断 · 遗留隐患）：旧格式 pkl 不含 feature_cols，
推理侧按数量对齐（多则截断、少则填 0）——特征管线一旦变更就会静默错位且
完全不可见。本文件把修复固化为断言：
  1. 训练时 feature_cols 持久化进 pkl，load 后往返一致；
  2. 推理按名对齐：缺列 / 数量不符 → 显式报错（拒绝静默错位）；
  3. 旧格式无契约 → 退回数量对齐但显式告警（不冒充安全）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import _align_features_by_contract  # noqa: E402
from src.train.models.lightgbm_model import LightGBMModel  # noqa: E402

FEATURE_COLS = [f"f{i:02d}" for i in range(6)]


def _tiny_config(tmp_path: Path) -> dict:
    return {
        "model": {"lightgbm": {
            "objective": "binary", "n_estimators": 10, "learning_rate": 0.1,
            "max_depth": 2, "num_leaves": 4, "subsample": 1.0,
            "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0,
            "verbose": -1, "early_stopping_rounds": 5,
        }},
        "training": {"save_dir": str(tmp_path)},
    }


def _blob(rng: np.random.RandomState, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, len(FEATURE_COLS)))
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _artifact(model, cols, n_features):
    from types import SimpleNamespace
    return SimpleNamespace(feature_cols=cols), model, n_features


def _train_tiny(tmp_path: Path) -> LightGBMModel:
    m = LightGBMModel(_tiny_config(tmp_path))
    rng = np.random.RandomState(42)
    X, y = _blob(rng)
    m.train(X, y, feature_cols=FEATURE_COLS)
    return m


def test_feature_cols_roundtrip_through_pkl(tmp_path):
    """训练契约必须原样持久化进 pkl 并能加载回来（名称 + 顺序）。"""
    m = _train_tiny(tmp_path)
    path = m.save(str(tmp_path / "m.pkl"))
    m2 = LightGBMModel(_tiny_config(tmp_path))
    m2.load(str(path))
    assert m2.feature_cols_ == FEATURE_COLS
    import joblib
    raw = joblib.load(path)
    assert raw["feature_cols"] == FEATURE_COLS


def test_train_without_contract_warns_and_saves_none(tmp_path):
    """不提供契约时允许训练（兼容旧路径），但 pkl 中为 None。"""
    m = LightGBMModel(_tiny_config(tmp_path))
    rng = np.random.RandomState(0)
    X, y = _blob(rng)
    m.train(X, y)
    assert m.feature_cols_ is None
    m.save(str(tmp_path / "m.pkl"))
    import joblib
    assert joblib.load(tmp_path / "m.pkl")["feature_cols"] is None


def _model_data_with_contract(trained: LightGBMModel):
    artifact, model, _ = _artifact(trained, FEATURE_COLS, len(FEATURE_COLS))
    return {"model": trained.model, "scaler": trained.scaler, "_artifact": artifact}


def test_contract_alignment_selects_named_columns_in_order(tmp_path):
    """按名取列、按训练顺序排列：打乱列序的推理 frame 必须被纠正。"""
    trained = _train_tiny(tmp_path)
    model_data = _model_data_with_contract(trained)
    rng = np.random.RandomState(7)
    df = pd.DataFrame(rng.normal(size=(10, len(FEATURE_COLS))),
                      columns=list(reversed(FEATURE_COLS)))
    out = _align_features_by_contract(df, model_data, fallback_cols=[])
    assert out.shape == (1, len(FEATURE_COLS))
    # 第 0 列必须是 f00（训练顺序），而不是打乱后的第一列 f05
    assert np.allclose(out[0], df[FEATURE_COLS].iloc[-1].to_numpy())


def test_contract_alignment_missing_column_fails_closed(tmp_path):
    trained = _train_tiny(tmp_path)
    model_data = _model_data_with_contract(trained)
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.normal(size=(10, 5)),
                      columns=FEATURE_COLS[:5])   # 缺 f05
    with pytest.raises(ValueError, match="管线漂移"):
        _align_features_by_contract(df, model_data, fallback_cols=[])


def test_contract_alignment_count_mismatch_fails_closed(tmp_path):
    """契约列数与模型 n_features_in_ 不符 → 显式报错。"""
    trained = _train_tiny(tmp_path)
    artifact = _artifact(trained, FEATURE_COLS + ["extra"], 7)[0]
    model_data = {"model": trained.model, "scaler": trained.scaler,
                  "_artifact": artifact}
    rng = np.random.RandomState(2)
    df = pd.DataFrame(rng.normal(size=(10, 7)),
                      columns=FEATURE_COLS + ["extra"])
    with pytest.raises(ValueError, match="特征数与模型不符"):
        _align_features_by_contract(df, model_data, fallback_cols=[])


def test_legacy_bundle_falls_back_with_warning(tmp_path, caplog):
    """旧格式（无 _artifact/契约）退回数量对齐，但必须显式告警。"""
    trained = _train_tiny(tmp_path)
    model_data = {"model": trained.model, "scaler": trained.scaler}  # 无 _artifact
    rng = np.random.RandomState(3)
    df = pd.DataFrame(rng.normal(size=(10, len(FEATURE_COLS))),
                      columns=FEATURE_COLS)
    import logging
    with caplog.at_level(logging.WARNING):
        out = _align_features_by_contract(df, model_data,
                                          fallback_cols=FEATURE_COLS)
    assert out.shape == (1, len(FEATURE_COLS))
    assert any("数量对齐" in r.message for r in caplog.records)
