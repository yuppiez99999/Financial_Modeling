"""测试 DataCollector 与 FeatureEngineer。"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collector import DataCollector  # noqa: E402
from src.data.preprocessor import FeatureEngineer  # noqa: E402


def _sample_df():
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, 100))
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.default_rng(1).integers(1000, 5000, 100),
    })


def test_collector_simulation_fallback(tmp_path):
    cfg = {"training": {"save_dir": "models"}, "data": {"source": ["simulation"], "raw_dir": str(tmp_path), "start_date": "2026-01-01", "end_date": "2026-04-10"}}
    col = DataCollector(cfg)
    df = col._fetch_with_fallback("TEST.SZ")
    assert df is not None and len(df) > 0
    assert {"date", "open", "high", "low", "close", "volume"} <= set(df.columns)


def test_collector_cache_roundtrip(tmp_path):
    cfg = {"training": {"save_dir": "models"}, "data": {"source": ["simulation"], "raw_dir": str(tmp_path), "start_date": "2026-01-01", "end_date": "2026-04-10"}}
    col = DataCollector(cfg)
    df = col._fetch_simulation("TEST.SZ", n=50)
    col._save_cache("TEST.SZ", df)
    loaded = col.load_cached("TEST.SZ")
    assert loaded is not None
    assert len(loaded) == 50


def test_feature_engineer_transform():
    cfg = {"features": {"technical": {"ma_windows": [5, 10]}}}
    fe = FeatureEngineer(cfg)
    df = _sample_df()
    out = fe.transform(df, horizon_days=5)
    cols = fe.get_feature_columns(out, 5)
    assert "ma_5" in out.columns
    assert "rsi" in out.columns
    assert "ret_1" in out.columns
    assert "date" not in cols
    assert "close" not in cols
    assert len(cols) > 0


def test_split_data_target_per_symbol(tmp_path):
    """多标的训练时 target 必须逐标的构造，禁止 concat 后跨标的 shift。

    两只标的价格量级差异巨大（10 元 vs 3000 元）。若在 concat 后统一
    shift(-horizon)，目标会退化成"其他标的的价格比"，本测试断言逐标的
    口径的 ground truth 与训练切分结果完全一致（回归防护）。
    """
    import numpy as np

    from src.train.trainer import ModelTrainer

    def _make_df(base_price: float) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        rng = np.random.default_rng(7)
        close = base_price * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        return pd.DataFrame({
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 100_000, len(dates)).astype(float),
        })

    cfg = {
        "model": {"type": "lightgbm"},
        "data": {
            "prediction_horizons": {"short_term": 5},
            "forecast_horizon": 5,
            "source": ["simulation"],
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "start_date": "2024-01-01",
            "end_date": "2024-04-29",
            "split_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "training": {"save_dir": str(tmp_path / "models")},
    }
    trainer = ModelTrainer(cfg)
    processed = {"AAA": _make_df(10.0), "BBB": _make_df(3000.0)}

    datasets, _feat_cols = trainer._split_data(processed, 10)

    # ground truth：逐标的 shift(-10) 后与训练集全量标签逐行比对
    truth_frames = []
    for sym, dfp in processed.items():
        fr = dfp["close"].shift(-10) / dfp["close"] - 1
        # 无未来数据的尾部行不参与比对（trainer._split_data 会 dropna 掉这些行）
        truth_frames.append(
            dfp.assign(
                target_10d=((fr > 0) & fr.notna()).astype(float),
                _ok=fr.notna(),
            )
        )
    combined = (
        pd.concat(truth_frames, ignore_index=True)
        .sort_values("date")
    )
    combined = combined[combined["_ok"]].drop(columns="_ok").dropna()

    y_all = np.concatenate([datasets["y_train"], datasets["y_val"], datasets["y_test"]])
    np.testing.assert_array_equal(y_all, combined["target_10d"].values)
