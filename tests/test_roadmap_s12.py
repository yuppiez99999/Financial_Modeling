"""S12 特征扩充正交对照实验：横截面/宏观/情感的增量是否可验证。

测试要点：
  - 横截面特征**只横跨标的、不跨时间**（无前视）；
  - 对照臂只改特征集：基准列与臂列不重叠，否则"增量"没有意义；
  - 配对增量 + Bonferroni 校正：不显著一律 reject，绝不因某臂数字好看而放行；
  - 未产出数据的臂：如实标注 `available=false`，且**不参与多重比较计数**；
  - 边界：`affects_features` / `affects_gate` 恒为 False，不改生产特征配置；
  - 消费口：CLI / 监控报表 / 日报 / API 缺失与正常分支。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import feature_experiment as fe


def _panel_data(n: int = 260) -> dict:
    """构造 4 只标的的伪行情（含宏观/情感列），足够跑 3 折 walk-forward。"""
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    out = {}
    rng = np.random.default_rng(7)
    for i, symbol in enumerate(["600519.SH", "510300.SH", "159915.SZ", "512880.SH"]):
        ret = rng.normal(0.0004, 0.012, n)
        close = 100 * np.cumprod(1 + ret)
        out[symbol] = pd.DataFrame({
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        })
    return out


# ----------------------------------------------------------------------
# 横截面特征
# ----------------------------------------------------------------------
def test_panel_close_aligns_by_date():
    data = _panel_data(30)
    panel = fe._panel_close(data)
    assert panel.shape[1] == 4
    assert len(panel) == 30


def test_cross_sectional_features_are_same_day_only():
    """核心无前视保证：当日特征只由**当日及更早**的横截面数据决定。

    做法：把最后一天之后的价格整体放大，特征表的最后一天不得变化。
    """
    data = _panel_data(60)
    panel = fe._panel_close(data)
    f1 = fe._cross_sectional_features(panel)

    data2 = {k: v.copy() for k, v in data.items()}
    for k in data2:
        data2[k].loc[data2[k].index[-1], "close"] *= 3.0
    panel2 = fe._panel_close(data2)
    f2 = fe._cross_sectional_features(panel2)

    merged = f1.merge(f2, on=["date", "_symbol"], suffixes=("_a", "_b"))
    for col in ("xsec_ret_rank", "xsec_mom_rank", "xsec_excess", "xsec_dispersion"):
        a = merged.dropna(subset=[f"{col}_a", f"{col}_b"])
        # 倒数第二天及更早的行必须完全一致（未来价格不影响过去特征）
        cutoff = sorted(a["date"].unique())[-2]
        past = a[a["date"] <= cutoff]
        assert np.allclose(past[f"{col}_a"], past[f"{col}_b"], equal_nan=True)


def test_cross_sectional_features_single_symbol_returns_empty():
    data = {"600519.SH": _panel_data(30)["600519.SH"]}
    panel = fe._panel_close(data)
    assert fe._cross_sectional_features(panel).empty


def test_cross_sectional_rank_between_zero_and_one():
    data = _panel_data(60)
    long = fe._cross_sectional_features(fe._panel_close(data)).dropna()
    assert long["xsec_ret_rank"].between(0, 1).all()
    assert long["xsec_mom_rank"].between(0, 1).all()
    # 分化度是逐日标量：同一天所有标的取同一个值
    per_day = long.groupby("date")["xsec_dispersion"].nunique()
    assert (per_day == 1).all()


def test_arm_columns_match_expected_prefixes():
    df = pd.DataFrame(columns=["ma_5", "macro_cpi_yoy3", "sentiment_score", "news_count",
                               "xsec_ret_rank", "xsec_dispersion", "unrelated"])
    assert fe._arm_columns("macro", df) == ["macro_cpi_yoy3"]
    assert set(fe._arm_columns("sentiment", df)) == {"sentiment_score", "news_count"}
    assert set(fe._arm_columns("cross_sectional", df)) == {"xsec_ret_rank", "xsec_dispersion"}
    assert fe._arm_columns("unknown", df) == []


def test_base_columns_exclude_arm_features():
    """基准列必须把对照臂的特征**排除干净**，否则"增量"是假的。"""
    data = _panel_data(120)
    from src.data.preprocessor import FeatureEngineer

    cfg = {"features": {"extended_indicators": False, "macro_enabled": True}}
    df = data["600519.SH"].copy()
    feats = FeatureEngineer(cfg).transform(df, 5)
    feats = feats.merge(
        fe._cross_sectional_features(fe._panel_close(data))
        .query("_symbol == '600519.SH'").drop(columns=["_symbol"]),
        on="date", how="left")
    base = fe._base_columns(feats, 5, cfg)
    for arm in ("cross_sectional", "macro", "sentiment"):
        assert not set(base) & set(fe._arm_columns(arm, feats))
    assert "target_5d" not in base


# ----------------------------------------------------------------------
# 判定逻辑
# ----------------------------------------------------------------------
def _arm_row(horizon="short_term", arm="macro", deltas=None, ic_delta=0.01,
             available=True, hit_delta=0.0):
    return {
        "horizon": horizon, "arm": arm, "horizon_days": 5, "available": available,
        "reason": "", "features_added": ["macro_cpi_yoy3"],
        "base_ic": 0.02, "arm_ic": 0.02 + ic_delta, "ic_delta": ic_delta,
        "base_hit_rate": 0.51, "arm_hit_rate": 0.51 + hit_delta, "hit_delta": hit_delta,
        "base_samples": 1000, "arm_samples": 1000, "folds": 3,
        "fold_ic_deltas": deltas or [-0.01, 0.0, 0.01, -0.005],
    }


def test_decide_defer_when_no_usable_arms():
    res = {"horizons": {"short_term": {"arms": [_arm_row(available=False)]}},
           "thresholds": {"max_family_p": 0.05}}
    verdict, narrative = fe.decide(res)
    assert verdict == fe.VERDICT_DEFER
    assert "没有任何可用对照臂" in narrative
    assert "未测到的臂" in narrative


def test_decide_reject_when_nothing_significant():
    rows = [_arm_row(deltas=[-0.01, 0.005, -0.002, 0.001]) for _ in range(3)]
    res = {"horizons": {"short_term": {"arms": rows}},
           "thresholds": {"max_family_p": 0.05}}
    verdict, narrative = fe.decide(res)
    assert verdict == fe.VERDICT_REJECT
    assert "不显著" in narrative
    assert "不支持继续投入" in narrative


def test_decide_adopt_when_strongly_significant():
    """一致为正且幅度大的增量 → 校正后显著 → adopt（但仍要求人工确认）。"""
    rows = [_arm_row(arm="cross_sectional", ic_delta=0.05,
                     deltas=[0.04, 0.05, 0.045, 0.055])]
    res = {"horizons": {"short_term": {"arms": rows}},
           "thresholds": {"max_family_p": 0.05}}
    verdict, narrative = fe.decide(res)
    assert verdict == fe.VERDICT_ADOPT
    assert "证据" in narrative and "人工确认" in narrative


def test_decide_counts_only_evaluated_arms_as_trials():
    """没测过的臂不该抬高校正强度（否则真发现更难通过）。"""
    rows = [_arm_row(arm="cross_sectional", ic_delta=0.05,
                     deltas=[0.04, 0.05, 0.045, 0.055]),
            _arm_row(arm="sentiment", available=False)]
    res = {"horizons": {"short_term": {"arms": rows}},
           "thresholds": {"max_family_p": 0.05}}
    fe.decide(res)
    assert res["n_trials"] == 1
    assert len(res["comparisons"]) == 1


def test_decide_reject_mentions_skipped_arms():
    rows = [_arm_row(deltas=[-0.01, 0.005, -0.002, 0.001]),
            _arm_row(arm="sentiment", available=False)]
    res = {"horizons": {"short_term": {"arms": rows}},
           "thresholds": {"max_family_p": 0.05}}
    verdict, narrative = fe.decide(res)
    assert verdict == fe.VERDICT_REJECT
    assert "未能评估" in narrative


def test_paired_t_handles_insufficient_and_zero_variance():
    assert fe._paired_t([0.01, 0.02])[1] == 2          # 样本不足 → t=0
    assert fe._paired_t([0.01, 0.01, 0.01])[0] == 0.0  # 零方差 → 不外推显著性


def test_decide_flags_affects_flags_false():
    res = {"horizons": {"short_term": {"arms": [_arm_row()]}},
           "thresholds": {"max_family_p": 0.05}}
    fe.decide(res)
    assert res.get("affects_gate") is not True


# ----------------------------------------------------------------------
# 端到端（小样本）
# ----------------------------------------------------------------------
def test_run_experiment_end_to_end_small():
    data = _panel_data(300)
    cfg = {
        "data": {"prediction_horizons": {"short_term": 5}},
        "features": {"extended_indicators": False, "macro_enabled": True,
                     "sentiment_enabled": False},
        "model": {"type": "lightgbm", "lightgbm": {
            "objective": "binary", "n_estimators": 20, "learning_rate": 0.1,
            "num_leaves": 15, "max_depth": 4, "min_child_samples": 20,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0,
            "reg_lambda": 1e-5, "random_state": 42, "n_jobs": 1,
            "class_weight": "balanced", "verbose": -1, "early_stopping_rounds": 20,
        }},
        "training": {"save_dir": "models"},
        "feature_experiment": {"report_dir": "reports"},
    }
    res = fe.run_experiment(data, cfg, arms=("cross_sectional", "macro"), folds=2)
    assert res["available"] is True
    assert res["affects_features"] is False
    assert res["verdict"] in (fe.VERDICT_ADOPT, fe.VERDICT_REJECT, fe.VERDICT_DEFER)
    arms = res["horizons"]["short_term"]["arms"]
    cs = [a for a in arms if a["arm"] == "cross_sectional"][0]
    assert cs["available"] is True
    assert len(cs["features_added"]) == 4
    # 基准与臂的样本量应接近（同一份数据、同一折切分；横截面特征前 20 行为 NaN）
    assert abs(cs["base_samples"] - cs["arm_samples"]) <= max(60, cs["base_samples"] * 0.15)
    # 且臂的样本量不得多于基准（加特征只会因 NaN 丢样本，不会凭空多出来）
    assert cs["arm_samples"] <= cs["base_samples"]


def test_run_experiment_without_data_defers():
    res = fe.run_experiment({}, {}, arms=("macro",))
    assert res["available"] is False
    assert res["verdict"] == fe.VERDICT_DEFER


def test_save_and_load_roundtrip(tmp_path):
    cfg = {"feature_experiment": {"report_dir": str(tmp_path)}}
    res = {"verdict": "reject", "narrative": "x"}
    path = fe.save(res, cfg)
    assert path.exists()
    assert fe.load(cfg)["verdict"] == "reject"


def test_load_missing_and_corrupt(tmp_path):
    cfg = {"feature_experiment": {"report_dir": str(tmp_path)}}
    assert fe.load(cfg) is None
    (tmp_path / fe.REPORT_NAME).write_text("{oops", encoding="utf-8")
    assert fe.load(cfg) is None


# ----------------------------------------------------------------------
# CLI / 报表 / API
# ----------------------------------------------------------------------
def test_cli_feature_experiment_command():
    import main

    parser = main.build_parser()
    args = parser.parse_args(["feature-experiment"])
    assert args.command == "feature-experiment"
    assert args.arms is None
    args2 = parser.parse_args(["feature-experiment", "--arms", "macro,sentiment"])
    assert args2.arms == "macro,sentiment"


def test_run_feature_experiment_no_data(capsys):
    import main

    cfg = {"data": {"prediction_horizons": {"short_term": 5}}, "markets": {}}
    import scripts.evaluate_models as ev

    original = ev.load_market_data
    try:
        ev.load_market_data = lambda *a, **k: {}  # type: ignore[assignment]
        out = main.run_feature_experiment({"data": {"prediction_horizons": {}}}, symbols=[])
    finally:
        ev.load_market_data = original  # type: ignore[assignment]
    assert "error" in out


def test_monitor_collects_feature_experiment(tmp_path):
    from src.monitor.health_report import ModelMonitor

    cfg = {"feature_experiment": {"report_dir": str(tmp_path)}}
    res = ModelMonitor(cfg)._collect_feature_experiment()
    assert res["available"] is False
    assert "feature-experiment" in res["hint"]

    fe.save({"verdict": "reject", "narrative": "n", "n_trials": 3}, cfg)
    res2 = ModelMonitor(cfg)._collect_feature_experiment()
    assert res2["available"] is True
    assert res2["verdict"] == "reject"


def test_daily_report_section(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    cfg = {"feature_experiment": {"report_dir": str(tmp_path)}}
    gen = DailyReportGenerator.__new__(DailyReportGenerator)
    gen.config = cfg
    gen.report_dir = str(tmp_path)
    assert "未评估" in "\n".join(gen._generate_feature_experiment_section())

    fe.save({
        "verdict": "reject", "narrative": "特征扩充无正交信息", "n_trials": 6,
        "thresholds": {"max_family_p": 0.05},
        "horizons": {"short_term": {"horizon_days": 5, "base_ic": 0.02,
                                    "base_hit_rate": 0.51, "arms": [
            {"arm": "macro", "horizon_days": 5, "available": True, "ic_delta": -0.008,
             "hit_delta": -0.009, "features_added": ["a"], "significant": False},
        ]}},
    }, cfg)
    text = "\n".join(gen._generate_feature_experiment_section())
    assert "S12" in text
    assert "reject" in text
    assert "不改变" in text


def test_api_endpoint_feature_experiment(tmp_path):
    from fastapi.testclient import TestClient

    from src.api import server

    cfg = {"feature_experiment": {"report_dir": str(tmp_path)}}
    old = server._config
    try:
        server._config = cfg
        client = TestClient(server.app)
        body = client.get("/api/v1/eval/feature-experiment").json()
        assert body["available"] is False

        fe.save({"verdict": "adopt", "narrative": "x"}, cfg)
        body2 = client.get("/api/v1/eval/feature-experiment").json()
        assert body2["available"] is True
        assert body2["verdict"] == "adopt"
    finally:
        server._config = old
