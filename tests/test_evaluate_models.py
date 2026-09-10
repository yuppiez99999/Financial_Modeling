"""Q1 排期：预测评估脚本（scripts/evaluate_models.py）测试。

重点验证「评估口径正确性」：时序无泄漏、目标逐标的构造、IC 计算、
手续费计入、walk-forward 切分合法性。全部离线运行。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_models", PROJECT_ROOT / "scripts" / "evaluate_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load_module()


def _cfg(tmp_path):
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["raw_dir"] = str(tmp_path)
    cfg["data"]["source"] = ["simulation"]
    cfg["features"]["macro_enabled"] = False
    cfg["training"]["save_dir"] = str(tmp_path / "models")
    return cfg


def _ohlcv(n=400, seed=1, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="B"),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1_000, 10_000, n).astype(float),
    })


# ---------- walk-forward 切分 ----------

def test_walk_forward_splits_are_temporal():
    splits = ev.walk_forward_splits(1000, folds=3)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()      # 训练严格早于测试
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_walk_forward_test_windows_cover_tail():
    splits = ev.walk_forward_splits(1000, folds=3)
    last_test = splits[-1][1]
    assert last_test.max() == 999                     # 最后一折覆盖到数据末尾
    # 训练集单调增长（滚动扩窗）
    sizes = [len(t) for t, _ in splits]
    assert sizes == sorted(sizes)


def test_walk_forward_fallback_single_fold():
    splits = ev.walk_forward_splits(60, folds=3)
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 42 and len(test_idx) == 18


def test_walk_forward_empty():
    assert ev.walk_forward_splits(0, folds=3) == []


# ---------- IC ----------

def test_ic_perfect_monotonic_is_one():
    proba = np.linspace(0, 1, 50)
    fwd = np.linspace(-0.1, 0.1, 50)
    assert ev.information_coefficient(proba, fwd) == pytest.approx(1.0)


def test_ic_inverse_is_minus_one():
    proba = np.linspace(0, 1, 50)
    fwd = np.linspace(0.1, -0.1, 50)
    assert ev.information_coefficient(proba, fwd) == pytest.approx(-1.0)


def test_ic_too_few_samples():
    assert ev.information_coefficient(np.array([0.1, 0.9]), np.array([1.0, 2.0])) == 0.0


# ---------- 指标计算 ----------

def test_compute_metrics_perfect_prediction():
    y = np.array([1, 1, 0, 0] * 10)
    proba = np.where(y == 1, 0.9, 0.1)
    fwd = np.where(y == 1, 0.05, -0.05)
    m = ev.compute_metrics(y, proba, fwd, horizon_days=5, fee=0.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)
    assert m["hit_rate"] == pytest.approx(1.0)
    assert m["total_return"] > 0
    assert m["win_rate"] == pytest.approx(1.0)


def test_compute_metrics_inverted_prediction_loses():
    y = np.array([1, 0] * 20)
    proba = np.where(y == 1, 0.1, 0.9)     # 完全反着预测
    fwd = np.where(y == 1, 0.05, -0.05)
    m = ev.compute_metrics(y, proba, fwd, horizon_days=5, fee=0.0)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["total_return"] < 0
    assert m["max_drawdown_pct"] <= 0


def test_fee_reduces_total_return():
    y = np.array([1, 0] * 30)
    proba = np.where(y == 1, 0.8, 0.2)
    fwd = np.where(y == 1, 0.02, -0.02)
    free = ev.compute_metrics(y, proba, fwd, 5, fee=0.0)
    paid = ev.compute_metrics(y, proba, fwd, 5, fee=0.005)
    assert paid["total_return"] < free["total_return"]
    assert paid["fee_per_side"] == 0.005


def test_single_class_labels_do_not_crash():
    y = np.ones(20)
    proba = np.full(20, 0.7)
    fwd = np.full(20, 0.01)
    m = ev.compute_metrics(y, proba, fwd, 5, fee=0.0)
    assert m["auc"] == 0.5                       # 单类别时 AUC 无定义 → 中性值


# ---------- 数据集构建：无泄漏 ----------

def test_build_supervised_excludes_target_and_helpers(tmp_path):
    cfg = _cfg(tmp_path)
    data = {"AAA.SH": _ohlcv(300, seed=1), "BBB.SH": _ohlcv(300, seed=2)}
    combined = ev.build_supervised(data, cfg, horizon_days=5)

    assert not combined.empty
    assert combined["date"].is_monotonic_increasing
    # 目标列不含 NaN
    assert not combined["target_5d"].isna().any()

    from src.data.preprocessor import FeatureEngineer
    fe = FeatureEngineer(cfg)
    feature_cols = [c for c in fe.get_feature_columns(combined, 5) if not str(c).startswith("_")]
    assert all(not c.startswith("target_") for c in feature_cols)
    assert "_symbol" not in feature_cols and "_fwd_ret" not in feature_cols


def test_build_supervised_target_is_per_symbol(tmp_path):
    """逐标的构造目标：不同标的首尾行不应互相污染。"""
    cfg = _cfg(tmp_path)
    a, b = _ohlcv(120, seed=1), _ohlcv(120, seed=2)
    combined = ev.build_supervised({"A.SH": a, "B.SH": b}, cfg, 5)

    for symbol in ("A.SH", "B.SH"):
        sub = combined[combined["_symbol"] == symbol].sort_values("date")
        # 尾部 horizon 行应被丢弃（无未来数据），每标的行数 = 120 - 5
        assert len(sub) == 115


def test_build_supervised_empty_when_no_data(tmp_path):
    cfg = _cfg(tmp_path)
    assert ev.build_supervised({}, cfg, 5).empty


# ---------- 端到端 ----------

def test_evaluate_horizon_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    data = {f"S{i}.SH": _ohlcv(400, seed=i) for i in range(3)}
    result = ev.evaluate_horizon(cfg, data, "short_term", 5, "lightgbm", folds=3, fee=0.0)
    assert result["status"] == "ok"
    assert result["horizon"] == "short_term"
    assert len(result["folds"]) == 3
    for key in ("accuracy", "auc", "ic", "sharpe_ratio", "max_drawdown_pct", "win_rate"):
        assert key in result["weighted"]
    assert result["feature_count"] > 0
    assert result["total_samples"] > 0


def test_evaluate_horizon_no_data(tmp_path):
    result = ev.evaluate_horizon(_cfg(tmp_path), {}, "short_term", 5, "lightgbm", 3, 0.0)
    assert result["status"] == "no_data"


def test_render_markdown_contains_key_sections(tmp_path):
    cfg = _cfg(tmp_path)
    data = {"S0.SH": _ohlcv(400, seed=0)}
    results = [ev.evaluate_horizon(cfg, data, "short_term", 5, "lightgbm", 2, 0.001)]
    meta = {"generated_at": "2026-09-10T00:00:00", "model_type": "lightgbm",
            "symbol_count": 1, "folds": 2, "fee": 0.001, "data_source": "local_cache_only"}
    md = ev.render_markdown(results, meta)
    assert "# TrendCast Pro · 预测评估报告" in md
    assert "## 汇总" in md and "## 分折明细" in md
    assert "IC" in md and "不构成投资建议" in md


def test_main_cli_runs_offline(tmp_path, monkeypatch, capsys):
    """完整 CLI 冒烟：离线模式读本地缓存并产出报告。"""
    cfg = _cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    from src.data.collector import DataCollector
    for i in range(2):
        symbol = f"CLI{i}.SH"
        DataCollector(cfg)._save_cache(symbol, _ohlcv(400, seed=i))

    report_path = tmp_path / "eval.md"
    json_path = tmp_path / "eval.json"
    code = ev.main([
        "--config", str(cfg_path),
        "--symbols", "CLI0.SH", "CLI1.SH",
        "--horizon", "short_term", "--folds", "2",
        "--offline", "--log-level", "ERROR",
        "--report", str(report_path), "--output", str(json_path),
    ])
    assert code == 0
    assert report_path.exists() and json_path.exists()
    import json
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["meta"]["symbol_count"] == 2
    assert payload["results"][0]["status"] == "ok"


def test_main_cli_errors_without_symbols(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["data"]["markets"] = {"stock": {"enabled": False, "symbols": []}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    assert ev.main(["--config", str(cfg_path), "--offline", "--log-level", "ERROR"]) == 2


def test_resolve_horizons(tmp_path):
    cfg = _cfg(tmp_path)
    assert set(ev.resolve_horizons(cfg, "all")) == {"short_term", "mid_term", "long_term"}
    assert ev.resolve_horizons(cfg, "mid_term") == {"mid_term": 10}
    with pytest.raises(SystemExit):
        ev.resolve_horizons(cfg, "nope")
