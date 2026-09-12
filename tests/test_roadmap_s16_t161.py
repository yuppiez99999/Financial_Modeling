"""S16 / T16.1 + T16.2 守卫测试：保形预测区间 + 区间/概率口径对照。

覆盖（全部离线，合成数据，不触网）：
  - three_way_split：60/10/30 时间序、无重叠无遗漏、样本不足如实给空；
  - interval_from_prediction_sets / interval_confidence：
    双标签与空集 → 最宽区间（置信分 0）、单标签 → 半宽（置信分 0.5）；
  - 无前视硬校验：篡改 holdout 标签 → 区间与置信分**逐位不变**；
  - coverage_report：覆盖率三态（on_target / conservative / anti_conservative）
    与样本不足不猜；
  - reliability_curve：过度自信分布必须被 ECE/Brier 抓到（方向性断言）；
  - compare_interval_vs_proba：改善 / 退步 / 混合 / 样本不足四态判定口径；
  - 报告结构：affects_gate 恒 false、路径尊重配置、JSON 可序列化；
  - CLI 注册（conformal-interval 在 choices 里，--confidence-levels 等可解析）
    与无数据 fail-soft。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import conformal_probability as cp  # noqa: E402


# ----------------------------------------------------------------------
# 三段切分
# ----------------------------------------------------------------------
class TestThreeWaySplit:
    def test_default_60_10_30(self):
        """缺省三段 ≈ 60% / 10% / 30%（保留期必须恰好是后 30%）。"""
        sp = cp.three_way_split(1000)
        assert len(sp.holdout) == 300                 # 保留期 = 后 30%（硬要求）
        assert abs(len(sp.calibration) / 1000 - 0.10) < 0.02
        assert abs(len(sp.proper_train) / 1000 - 0.60) < 0.02
        assert sp.available is True

    def test_strict_time_order(self):
        sp = cp.three_way_split(500)
        assert sp.proper_train.max() < sp.calibration.min() < sp.holdout.min()

    def test_no_overlap_no_gap(self):
        sp = cp.three_way_split(997)
        merged = np.concatenate([sp.proper_train, sp.calibration, sp.holdout])
        assert sorted(merged.tolist()) == list(range(997))

    def test_holdout_aligns_with_t163_holdout(self):
        """保留期必须与 T16.3 holdout_split 完全同段（两条证据链可同源对照）。"""
        from src.eval import confidence_holdout as ch
        n = 1234
        _tr, ho = ch.holdout_split(n, 0.7)
        sp = cp.three_way_split(n, holdout_ratio=0.3)
        assert np.array_equal(ho, sp.holdout)

    def test_empty_and_tiny_input(self):
        sp = cp.three_way_split(0)
        assert sp.available is False and len(sp.holdout) == 0
        sp2 = cp.three_way_split(1)
        assert sp2.available is False       # 切不出三段，如实不可用

    def test_as_dict_roundtrip(self):
        d = cp.three_way_split(100).as_dict()
        assert d["holdout"] == 30 and d["available"] is True
        assert d["proper_train"] + d["calibration"] == 70
        assert abs(d["calibration"] - 10) <= 2       # 目标 10%，实际以整除切点为准


# ----------------------------------------------------------------------
# 区间换算
# ----------------------------------------------------------------------
class TestIntervalMapping:
    def test_both_labels_is_widest(self):
        sets = np.array([[True, True], [False, False]])
        lo, hi = cp.interval_from_prediction_sets(sets)
        assert np.allclose(lo, 0.0) and np.allclose(hi, 1.0)
        assert np.allclose(cp.interval_confidence(sets), 0.0)

    def test_single_label_is_half_width(self):
        sets = np.array([[False, True], [True, False]])
        conf = cp.interval_confidence(sets)
        assert np.allclose(conf, 0.5)

    def test_empty_set_maps_to_lowest_confidence(self):
        """空集 = 哪个方向都不敢说 → 与「两个标签都在」同等低置信（宁漏不假自信）。"""
        assert np.allclose(cp.interval_confidence(np.array([[False, False]])), 0.0)

    def test_confidence_bounded_and_monotone(self):
        sets = np.array([[False, True], [True, True], [True, False]])
        conf = cp.interval_confidence(sets)
        assert conf.min() >= 0.0 and conf.max() <= 1.0

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            cp.interval_from_prediction_sets(np.array([[True, False, True]]))


# ----------------------------------------------------------------------
# 覆盖率审计
# ----------------------------------------------------------------------
def _sets_from_flags(covered: np.ndarray) -> np.ndarray:
    """按「是否覆盖真实标签」造预测集合：覆盖→单标签，不覆盖→错标签。"""
    n = len(covered)
    out = np.zeros((n, 2), dtype=bool)
    out[np.arange(n), np.zeros(n, dtype=int)] = covered
    out[~covered] = out[~covered][:, ::-1]
    return out


class TestCoverageReport:
    def test_on_target(self):
        y = np.zeros(500, dtype=int)
        sets = _sets_from_flags(np.concatenate([np.ones(405, bool), np.zeros(95, bool)]))
        rep = cp.coverage_report(y, sets, 0.8)
        assert rep["available"] is True
        assert rep["verdict"] == "on_target"
        assert abs(rep["empirical_coverage"] - 0.81) < 1e-9

    def test_anti_conservative_flagged(self):
        """实测覆盖率低于目标覆盖率 = 覆盖承诺未兑现，必须显式标注。"""
        y = np.zeros(500, dtype=int)
        sets = _sets_from_flags(np.concatenate([np.ones(300, bool), np.zeros(200, bool)]))
        rep = cp.coverage_report(y, sets, 0.9)
        assert rep["verdict"] == "anti_conservative"
        assert rep["coverage_gap"] < 0

    def test_insufficient_samples_not_guessed(self):
        rep = cp.coverage_report(np.array([0, 1]), np.array([[True, False], [True, False]]), 0.8)
        assert rep["available"] is False and rep["reason"] == "insufficient_samples"

    def test_empty_set_ratio_reported(self):
        y = np.zeros(100, dtype=int)
        sets = np.zeros((100, 2), dtype=bool)     # 全空集
        rep = cp.coverage_report(y, sets, 0.8)
        assert rep["empty_set_ratio"] == 1.0
        assert rep["verdict"] == "anti_conservative"


# ----------------------------------------------------------------------
# 可靠性曲线
# ----------------------------------------------------------------------
class TestReliabilityCurve:
    def test_overconfident_distribution_detected(self):
        """过度自信的概率必须被 ECE/Brier 抓到（方向性断言，不设绝对阈值）。"""
        rng = np.random.RandomState(7)
        base = rng.rand(2000)
        y = (rng.rand(2000) < base).astype(int)
        honest = cp.reliability_curve(base, y)
        over = cp.reliability_curve(np.clip(0.5 + (base - 0.5) * 3.0, 0, 1), y)
        assert over["ece"] > honest["ece"]
        assert over["brier"] > honest["brier"]

    def test_isotonic_reference_is_labeled(self):
        rng = np.random.RandomState(11)
        p = rng.rand(500)
        y = (rng.rand(500) < p).astype(int)
        rep = cp.reliability_curve(p, y, isotonic=True)
        assert rep["source"] == "isotonic_in_sample_reference"
        assert "不得与 raw 做" in (rep["note"] or "")

    def test_insufficient_samples(self):
        assert cp.reliability_curve([0.5] * 5, [0] * 5)["available"] is False

    def test_bin_rows_have_gap(self):
        rng = np.random.RandomState(3)
        p = rng.rand(600)
        y = (rng.rand(600) < p).astype(int)
        rep = cp.reliability_curve(p, y)
        avail = [r for r in rep["bins"] if r["available"]]
        assert avail and all("gap" in r for r in avail)


class TestWidthCalibration:
    def test_narrow_width_flagged(self):
        rng = np.random.RandomState(5)
        errors = rng.rand(400)          # 误差大
        widths = np.full(400, 0.01)     # 声明宽度极小
        rep = cp.coverage_width_curve(errors, widths)
        assert rep["available"] is True
        assert rep["violation_ratio"] == 1.0

    def test_insufficient_samples(self):
        assert cp.coverage_width_curve([0.1] * 3, [0.1] * 3)["available"] is False


# ----------------------------------------------------------------------
# 对照判定（T16.2）
# ----------------------------------------------------------------------
def _mk(proba, returns, conf_interval, conf_proba):
    return cp.compare_interval_vs_proba(proba, returns, conf_interval,
                                        conf_proba=conf_proba,
                                        grid=(0.0, 0.1, 0.2, 0.3))


class TestCompareVerdicts:
    def _data(self):
        rng = np.random.RandomState(42)
        n = 600
        proba = np.clip(rng.rand(n), 0.01, 0.99)
        returns = rng.randn(n) * 0.01
        return proba, returns

    def test_identical_confidence_is_mixed(self):
        """两条口径完全相同时，逐阈值 delta = 0 → 不得判成 improved。"""
        p, r = self._data()
        c = np.linspace(0, 1, len(p))
        res = _mk(p, r, c, c.copy())
        assert res["verdict"] == "mixed"

    def test_verdict_enum_and_affects_gate(self):
        p, r = self._data()
        c = np.linspace(0, 1, len(p))
        res = _mk(p, r, c, 1 - c)
        assert res["verdict"] in {"improved", "degraded", "mixed", "insufficient_samples"}
        assert res["affects_gate"] is False

    def test_insufficient_samples_when_too_few(self):
        p, r = self._data()
        res = cp.compare_interval_vs_proba(p, r, np.zeros(len(p)),
                                           conf_proba=np.zeros(len(p)),
                                           grid=(0.9,))
        assert res["verdict"] == "insufficient_samples"
        assert res["usable_thresholds"] == 0

    def test_pairs_are_paired_by_threshold(self):
        p, r = self._data()
        c = np.linspace(0, 1, len(p))
        res = _mk(p, r, c, c.copy())
        got = [x["threshold"] for x in res["pairs"] if x.get("available")]
        assert got == [0.0, 0.1, 0.2, 0.3]

    def test_not_choosing_the_best_row(self):
        """判定必须覆盖全部可用阈值行，不得择优 —— 一升一降就要 mixed。"""
        n = 400
        proba = np.linspace(0.01, 0.99, n)
        returns = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
        ci = np.where(np.arange(n) % 2 == 0, 1.0, 0.0)
        cprob = np.where(np.arange(n) % 2 == 1, 1.0, 0.0)
        res = _mk(proba, returns, ci, cprob)
        assert res["verdict"] == "mixed"
        assert "不择优" in res["reason"]


# ----------------------------------------------------------------------
# 报告结构与路径
# ----------------------------------------------------------------------
class TestReports:
    def test_interval_report_shape(self):
        rep = cp.build_interval_report({"5d": {"available": True}},
                                       meta={"confidence_levels": [0.8]})
        assert rep["kind"] == "conformal_interval_report"
        assert rep["affects_gate"] is False
        assert rep["confidence_levels"] == [0.8]
        assert json.loads(json.dumps(rep))["horizons"]["5d"]["available"] is True

    def test_comparison_report_shape(self):
        rep = cp.build_comparison_report({})
        assert rep["kind"] == "conformal_vs_proba_report"
        assert rep["affects_gate"] is False

    def test_paths_respect_config(self):
        cfg = {"strategy_gate": {"confidence_gate": {"report_dir": "out"}}}
        assert cp.calibration_dir(cfg).as_posix() == "out/calibration"
        assert cp.interval_report_path(cfg).as_posix() == "out/calibration/conformal_interval.json"
        assert cp.comparison_report_path(None).as_posix() == \
            "reports/calibration/conformal_vs_proba.json"

    def test_generated_at_present(self):
        assert cp.build_interval_report({})["generated_at"]
        assert cp.build_comparison_report({})["generated_at"]


# ----------------------------------------------------------------------
# 端到端（离线合成数据，MAPIE 可用时）
# ----------------------------------------------------------------------
class TestEndToEndOffline:
    @pytest.mark.skipif(not cp.mapie_available(), reason="需要可选依赖 mapie")
    def test_full_pipeline_no_lookahead(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb

        rng = np.random.RandomState(7)
        n = 1200
        X = rng.randn(n, 4)
        y = (X[:, 0] + 0.5 * rng.randn(n) > 0).astype(int)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        df["target_5d"] = y
        df["_fwd_ret"] = np.where(y == 1, 0.01, -0.01) + rng.randn(n) * 0.01

        split = cp.three_way_split(n)
        out = cp.train_and_conformal(
            df, [f"f{i}" for i in range(4)], "target_5d", split, config={},
            lgb_module=lgb, scaler_cls=StandardScaler)
        sets = out["sets"][0.8]
        assert len(sets) == len(split.holdout)

        conf_a = cp.interval_confidence(sets)
        # 篡改 holdout 标签：预测/区间必须逐位不变（模型没见过 holdout 标签）
        df2 = df.copy()
        df2.loc[split.holdout, "target_5d"] = 1 - df2.loc[split.holdout, "target_5d"]
        out2 = cp.train_and_conformal(
            df2, [f"f{i}" for i in range(4)], "target_5d", split, config={},
            lgb_module=lgb, scaler_cls=StandardScaler)
        assert np.array_equal(conf_a, cp.interval_confidence(out2["sets"][0.8]))

        rep = cp.coverage_report(out["y_true"], sets, 0.8)
        assert rep["available"] is True
        assert 0.0 <= rep["empirical_coverage"] <= 1.0


# ----------------------------------------------------------------------
# 配置真的被消费（不是文档式假配置）
# ----------------------------------------------------------------------
class TestConfigConsumed:
    def test_config_keys_exist_and_default_off(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        for name in ("config.yaml", "config_pro.yaml"):
            cfg = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
            cc = cfg["model"]["factors"]["conformal"]
            assert cc["enabled"] is False, f"{name}: 区间口径必须缺省关闭"
            assert cc["confidence_levels"] == [0.8, 0.9]
            assert cc["conformity_score"] == "lac"
            assert cc["holdout_ratio"] == 0.3

    def test_strategy_gate_untouched_by_this_round(self):
        """本轮零门禁改动：config 段必须与主线逐字段一致（防顺手改门禁）。"""
        import yaml
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
        gate = cfg["strategy_gate"]
        assert gate["min_hit_rate"] == 0.52
        assert gate["confidence_gate"]["mode"] == "report_only"
        assert gate["confidence_gate"]["freeze_structure"] is True

    def test_cli_reads_confidence_levels_from_config(self, monkeypatch):
        """改配置必须改行为 —— 否则就是「文档式假配置」。"""
        import main as m
        captured = {}

        def fake_train(combined, cols, target, split, config=None, **kw):
            captured["levels"] = list(kw.get("confidence_levels") or [])
            captured["holdout_ratio"] = split.meta.get("holdout_ratio_of_all")
            n = len(split.holdout)
            return {"estimator": None,
                    "proba": np.full(n, 0.5),
                    "returns": np.zeros(n),
                    "y_true": np.zeros(n, dtype=int),
                    "sets": {0.7: np.tile([[False, True]], (n, 1)),
                             0.95: np.tile([[True, True]], (n, 1))}}

        monkeypatch.setattr("scripts.evaluate_models.load_market_data",
                            lambda cfg, syms, offline=False: {"X": _fake_df()})
        # 特征列与监督数据集走桩：本用例只验「配置是否真被消费」，不验建模
        monkeypatch.setattr("scripts.evaluate_models.build_supervised",
                            lambda data, cfg, days: _fake_df())
        monkeypatch.setattr("src.data.preprocessor.FeatureEngineer.get_feature_columns",
                            lambda self, df, days: ["close"])
        monkeypatch.setattr(cp, "mapie_available", lambda: True)
        monkeypatch.setattr(cp, "three_way_split",
                            lambda n, r=0.3, **kw: cp.ThreeWaySplit(
                                np.arange(3), np.array([3]), np.arange(4, 10),
                                {"holdout_ratio_of_all": r}))
        monkeypatch.setattr(cp, "train_and_conformal", fake_train)
        cfg = {"data": {"markets": {}, "prediction_horizons": {}},
               "model": {"factors": {"conformal": {
                   "confidence_levels": [0.7, 0.95], "holdout_ratio": 0.4}}}}
        m.run_conformal_interval(cfg, symbols=["X"], horizons=[5])
        assert captured["levels"] == [0.7, 0.95]          # 配置覆盖了缺省 0.8/0.9
        assert abs(captured["holdout_ratio"] - 0.4) < 1e-9


def _fake_df():
    import pandas as pd
    return pd.DataFrame({"date": ["2024-01-01"], "close": [1.0]})


# ----------------------------------------------------------------------
# 结论复算脚本（换机器/过期后可一键复现文档数字）
# ----------------------------------------------------------------------
class TestVerificationScript:
    def test_script_exists_and_is_importable(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "verify_conformal_interval.py"
        assert path.exists(), "结论复算脚本必须落盘（结论须可复现）"
        src = path.read_text(encoding="utf-8")
        assert "SEED = 7" in src, "复算种子必须与结论文档一致"

    def test_synthetic_dataset_is_deterministic(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import verify_conformal_interval as v
        a, cols = v.synthetic_dataset()
        b, _ = v.synthetic_dataset()
        assert cols == [f"f{i}" for i in range(6)]
        assert (a["target_5d"].to_numpy() == b["target_5d"].to_numpy()).all()

    def test_script_fails_loud_when_mapie_missing(self, monkeypatch):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import verify_conformal_interval as v
        monkeypatch.setattr(cp, "mapie_available", lambda: False)
        assert v.main([]) == 2      # 明确失败，不静默降级


# ----------------------------------------------------------------------
# CLI 注册
# ----------------------------------------------------------------------
class TestCli:
    def test_command_registered_and_flags_present(self):
        src = Path(__import__("main").__file__).read_text(encoding="utf-8")
        assert '"conformal-interval"' in src
        assert "--confidence-levels" in src
        assert "--no-compare" in src and "--holdout-ratio" in src
        assert "run_conformal_interval" in src

    def test_no_data_fails_soft(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import main as m
        monkeypatch.setattr("scripts.evaluate_models.load_market_data",
                            lambda cfg, syms, offline=False: {})
        payload = m.run_conformal_interval({"data": {"prediction_horizons": {}}})
        assert payload["affects_gate"] is False
        assert "error" in payload
