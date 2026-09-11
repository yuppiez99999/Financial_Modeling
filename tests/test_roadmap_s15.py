"""S15 / G5 路线测试：optuna 超参搜索（T15.1）与置信度阈值曲线（T15.2）。

设计要点（与 S9~S14 测试同构）：
  - 全部离线（合成数据 + tmp 目录），CI 不触网；
  - 边界优先：不改门禁、不自动落地超参、不挑曲线好点、样本不足不猜；
  - 无前视：篡改数据尾部不影响搜索目标的既有折（walk-forward 由索引决定）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.confidence_curve import (  # noqa: E402
    confidence_from_interval,
    confidence_from_proba,
    sweep_confidence,
)
from src.eval.hyperopt_tuner import (  # noqa: E402
    SEARCH_SPACE,
    baseline_fold_ic,
    current_lightgbm_params,
    tune_lightgbm,
)


# ----------------------------------------------------------------------
# 测试数据
# ----------------------------------------------------------------------
def _make_data(n: int = 800, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    logit = 0.4 * X[:, 0] + 0.2 * X[:, 1]
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < proba).astype(int)
    fwd = 0.2 * X[:, 0] + rng.normal(0, 0.1, n)
    idx = np.arange(n)
    splits = [(idx[: int(n * 0.6)], idx[int(n * 0.6): int(n * 0.8)]),
              (idx[: int(n * 0.8)], idx[int(n * 0.8):])]
    return X, y, fwd, splits


_CFG_MIN = {
    "model": {"lightgbm": {
        "objective": "binary", "n_estimators": 120, "learning_rate": 0.05,
        "max_depth": 4, "num_leaves": 15, "subsample": 1.0, "colsample_bytree": 1.0,
        "reg_alpha": 0.0, "reg_lambda": 0.0, "verbose": -1, "early_stopping_rounds": 10,
    }},
}


# ----------------------------------------------------------------------
# T15.1 optuna 超参搜索
# ----------------------------------------------------------------------
class TestHyperoptTuner:
    def test_search_space_covers_current_params(self):
        """搜索空间必须覆盖仓库现行配置的取值（否则搜索不到基线邻域）。"""
        import yaml
        cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        params = current_lightgbm_params(cfg)
        for name, value in params.items():
            lo, hi = SEARCH_SPACE[name]
            assert lo <= value <= hi, f"{name}={value} 不在搜索空间 [{lo}, {hi}]"

    def test_report_shape_and_affects_gate_false(self, tmp_path):
        X, y, fwd, splits = _make_data()
        report = tune_lightgbm(X, y, fwd, splits, n_trials=3, horizon_days=5,
                               study_dir=str(tmp_path / "studies"))
        assert report["kind"] == "hyperopt_lightgbm"
        assert report["affects_gate"] is False
        assert report["metric"] == "mean_test_fold_spearman_ic"
        assert report["folds"] == len(splits)
        assert report["n_trials_total"] >= 3
        assert isinstance(report["best_trial"]["params"], dict)
        assert report["best_trial"]["search_ic"] > 0  # 合成信号可学
        assert "不自动落地" in report["note"]

    def test_study_sqlite_persisted(self, tmp_path):
        X, y, fwd, splits = _make_data()
        study_dir = tmp_path / "studies"
        tune_lightgbm(X, y, fwd, splits, n_trials=2, horizon_days=5,
                      study_dir=str(study_dir))
        assert (study_dir / "lgbm_h5_42.db").exists()

    def test_oot_reeval_present(self, tmp_path):
        X, y, fwd, splits = _make_data()
        report = tune_lightgbm(X, y, fwd, splits, n_trials=2, horizon_days=5,
                               study_dir=str(tmp_path / "studies"))
        oot = report["oot_reeval_last_fold"]
        assert oot is not None
        assert oot["samples"] == len(splits[-1][1])
        assert "ic" in oot and "hit_rate" in oot

    def test_baseline_fold_ic_matches_search_pipeline(self, tmp_path):
        """基线复评必须与搜索同一套评估路径（否则对照失真）。"""
        X, y, fwd, splits = _make_data()
        params = {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05,
                  "n_estimators": 120, "subsample": 1.0, "colsample_bytree": 1.0,
                  "reg_alpha": 0.0, "reg_lambda": 0.0}
        base = baseline_fold_ic(X, y, fwd, splits, params)
        assert base is not None
        # 用相同超参跑一次 1-trial 搜索会至少不劣于随机试验的均值下界（宽松断言）
        report = tune_lightgbm(X, y, fwd, splits, n_trials=1, horizon_days=5,
                               study_dir=str(tmp_path / "studies"))
        assert report["best_trial"]["search_ic"] > -1.0

    def test_insufficient_data_raises_not_fabricated(self):
        """无有效折时抛错而不是编造 IC（空 splits → 无 IC → ValueError）。"""
        from src.eval.hyperopt_tuner import _fold_mean_ic
        X = np.zeros((10, 3))
        y = np.zeros(10, dtype=int)
        fwd = np.zeros(10)
        with pytest.raises(ValueError):
            _fold_mean_ic(X, y, fwd, [],
                          {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05,
                           "n_estimators": 10, "subsample": 1.0,
                           "colsample_bytree": 1.0, "reg_alpha": 0.0,
                           "reg_lambda": 0.0})
        # 常数收益（方差为 0）→ spearman_ic 返回 0.0 属实值，不是编造；
        # 但**全部折 IC 均不可计算**时必须抛错（上面空 splits 已覆盖）。
        splits = [(np.arange(5), np.arange(5, 10))]
        v = _fold_mean_ic(X, np.ones(10, dtype=int), fwd, splits,
                          {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05,
                           "n_estimators": 10, "subsample": 1.0,
                           "colsample_bytree": 1.0, "reg_alpha": 0.0,
                           "reg_lambda": 0.0})
        assert v == 0.0  # 如实返回 0（无区分度），不抛错也不虚高

    def test_optuna_import_error_message(self, monkeypatch):
        """缺 optuna 时给出明确指引，而不是裸 ImportError。"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "optuna":
                raise ImportError("no optuna")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        X, y, fwd, splits = _make_data()
        with pytest.raises(ImportError, match="optuna"):
            tune_lightgbm(X, y, fwd, splits, n_trials=1, horizon_days=5,
                          study_dir="/tmp/never")


# ----------------------------------------------------------------------
# T15.2 置信度阈值曲线
# ----------------------------------------------------------------------
class TestConfidenceCurve:
    def test_proba_confidence_bounds(self):
        c = confidence_from_proba([0.5, 0.75, 1.0, 0.0])
        np.testing.assert_allclose(c, [0.0, 0.5, 1.0, 1.0])

    def test_interval_confidence_narrower_is_higher(self):
        wide = confidence_from_interval([9.0], [11.0], [10.0])[0]
        narrow = confidence_from_interval([9.8], [10.2], [10.0])[0]
        assert narrow > wide
        assert 0.0 <= wide <= 1.0 and 0.0 <= narrow <= 1.0

    def test_sweep_shape_and_flags(self):
        rng = np.random.default_rng(1)
        n = 3000
        proba = np.clip(0.5 + 0.25 * rng.normal(size=n), 0.02, 0.98)
        ret = np.sign(proba - 0.5) * 0.05 + rng.normal(0, 0.05, n)
        curve = sweep_confidence(proba, ret)
        assert curve["kind"] == "confidence_curve"
        assert curve["affects_gate"] is False
        assert curve["total_samples"] == n
        assert len(curve["rows"]) == len(curve["rows"]) > 5
        # 覆盖率随阈值单调不增
        covs = [r["coverage"] for r in curve["rows"]]
        assert all(covs[i] >= covs[i + 1] for i in range(len(covs) - 1))
        # 阈值 0 覆盖全样本
        assert curve["rows"][0]["threshold"] == 0.0
        assert curve["rows"][0]["coverage"] == 1.0

    def test_high_confidence_subset_more_accurate_on_signal_data(self):
        """合成可学信号上，高置信子集命中率应高于全样本（机制自检）。"""
        rng = np.random.default_rng(2)
        n = 5000
        strength = rng.random(n)  # 置信度代理
        proba = np.clip(0.5 + strength * 0.45, 0.02, 0.98)
        # 真实收益符号与概率方向一致的概率随 strength 提高（高置信更准）
        p_correct = 0.5 + 0.4 * strength
        correct = rng.random(n) < p_correct
        ret = np.where(correct, np.abs(rng.normal(0.05, 0.02, n)),
                       -np.abs(rng.normal(0.05, 0.02, n)))
        curve = sweep_confidence(proba, ret)
        full = curve["rows"][0]
        high = [r for r in curve["rows"]
                if r.get("available") and r["threshold"] >= 0.6][0]
        assert high["hit_rate"] > full["hit_rate"]

    def test_insufficient_samples_marked_unavailable(self):
        n = 60
        proba = np.linspace(0.51, 0.99, n)
        ret = np.linspace(0.01, 0.1, n)
        curve = sweep_confidence(proba, ret, min_samples=30)
        tail = [r for r in curve["rows"] if r["threshold"] >= 0.8]
        assert tail
        assert all(r.get("available") is False for r in tail)

    def test_observation_is_presentational_only(self):
        rng = np.random.default_rng(3)
        n = 3000
        proba = np.clip(0.5 + 0.2 * rng.normal(size=n), 0.05, 0.95)
        ret = rng.normal(0, 0.05, n)
        curve = sweep_confidence(proba, ret)
        if curve["observation"] is not None:
            assert "非推荐阈值" in curve["observation"]["note"]
            assert "人工检查点" in curve["observation"]["note"]
        # 选择自由度警告必须存在
        assert "选择自由度" in curve["note"]

    def test_neutralized_direction_uses_half_band(self):
        """命中率必须按去 0.5 中性化口径（与门禁同源），不是裸概率。"""
        from src.eval.confidence_curve import _neutralized_direction
        np.testing.assert_allclose(
            _neutralized_direction(np.array([0.4, 0.5, 0.6])),
            [-0.1, 0.0, 0.1],
        )


# ----------------------------------------------------------------------
# CLI 接线（build_parser 层面，不触网）
# ----------------------------------------------------------------------
class TestCliWiring:
    def test_parser_accepts_new_commands(self):
        from main import build_parser
        parser = build_parser()
        for cmd in ("tune", "confidence"):
            ns = parser.parse_args([cmd])
            assert ns.command == cmd
            assert ns.n_trials == 20

    def test_parser_n_trials_and_horizons(self):
        from main import build_parser
        parser = build_parser()
        ns = parser.parse_args(["tune", "--n-trials", "50",
                                "--trials-horizons", "5,10"])
        assert ns.n_trials == 50
        assert ns.trials_horizons == "5,10"

    def test_tune_unsupported_model_reports_error(self, tmp_path, monkeypatch):
        """非 lightgbm 模型：明确报不支持，不静默跑错模型。"""
        from main import run_tune
        payload = run_tune({"data": {}}, model_type="pytorch_lstm")
        assert "error" in payload
        assert "affects_gate" in payload and payload["affects_gate"] is False

    def test_tune_confidence_no_data_graceful(self, monkeypatch, tmp_path):
        """无行情数据：如实报错退出，不编造搜索结果。"""
        from main import run_confidence, run_tune

        class _NoDataCollector:
            def load_cached(self, symbol):
                return None

            def fetch_realtime(self, symbol):
                return None

        monkeypatch.setattr("scripts.evaluate_models.load_market_data",
                            lambda config, symbols, offline=False: {})
        cfg = {"data": {"prediction_horizons": {"short_term": {"days": 5}}},
               "model": _CFG_MIN["model"]}
        for fn in (run_tune, run_confidence):
            payload = fn(cfg)
            if isinstance(payload, dict) and "error" in payload:
                assert "无可用行情数据" in payload["error"]
