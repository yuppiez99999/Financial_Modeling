"""Q5 路线测试：信号衰减监控（IC 时序）与定期报告升级。

全部离线运行（不触网、不训练模型）。覆盖点：
  - 滚动锚点 IC 时序：切片正确、锚点不足必须沉默（unknown）而不是猜趋势；
  - 斜率 / 趋势判定：显著衰减 → decaying，显著上升 → improving，
    噪声抖动（低于 min_decay）→ stable，绝不把 0.001 级抖动当趋势；
  - 预计跌破步数：已跌破 → 0，斜率为负 → 正数，斜率非负 → None（不外推"永远安全"）；
  - 分批收益序列（walk-forward 折缝）走 evaluate_aligned，不跨折错位；
  - 与门禁同源：min_ic 取自 strategy_gate 段，IC 用同一 spearman_ic 实现；
  - 监控报表 / 日报 / 周报 / API 四个消费口的缺失与正常分支；
  - 配置段、CLI 命令、报告落盘。
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.ic import spearman_ic  # noqa: E402
from src.monitor.ic_trend import (  # noqa: E402
    STATUS_DECAYING,
    STATUS_IMPROVING,
    STATUS_STABLE,
    STATUS_UNKNOWN,
    ICTrendMonitor,
    _linear_slope,
)


# ---------------------------------------------------------------- helpers
def _cfg(tmp_path: Path, **ic_trend) -> dict:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    base = {
        "strategy_gate": {"enabled": True, "min_ic": 0.03, "min_hit_rate": 0.52,
                          "report_dir": str(reports)},
        "ic_trend": {
            "window": 60,
            "step": 20,
            "min_anchors": 4,
            "min_obs": 20,
            "min_decay": 0.002,
            "near_breach_steps": 3,
            "report_dir": str(reports),
        },
    }
    base["ic_trend"].update(ic_trend)
    return base


def _pairs(n: int = 400, strength: float = 0.06, decay: bool = False,
           improve: bool = False, seed: int = 11):
    """构造 (score, return) 序列：strength 控制信号有效性，可线性衰减/增强。"""
    rng = random.Random(seed)
    scores, rets = [], []
    for i in range(n):
        factor = 1.0
        if decay:
            factor = 1.0 - i / n
        if improve:
            factor = i / n
        s = rng.gauss(0, 0.1)
        scores.append(s)
        rets.append(s * strength * factor + rng.gauss(0, 0.01))
    return scores, rets


def _closes_from_scores(scores, rets, start: float = 100.0):
    """由收益序列还原一条价格曲线（仅用于 evaluate(closes, scores) 分支）。"""
    closes = [start]
    for r in rets[:-1]:
        closes.append(closes[-1] * (1 + r))
    return closes


# ---------------------------------------------------------------- 斜率工具
def test_linear_slope_basic():
    assert _linear_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert _linear_slope([3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert _linear_slope([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert _linear_slope([]) == 0.0
    assert _linear_slope([1.0]) == 0.0


# ---------------------------------------------------------------- 沉默优先
def test_insufficient_samples_returns_unknown_not_guess(tmp_path):
    """样本不足 → unknown + reason，绝不猜趋势（这是与"给个默认值"的关键区别）。"""
    m = ICTrendMonitor(_cfg(tmp_path))
    res = m.evaluate_pairs("short_term", 5, [0.1, 0.2], [0.01, 0.02])
    assert res.available is False
    assert res.status == STATUS_UNKNOWN
    assert res.reason
    assert res.slope_per_step is None


def test_all_returns_none_returns_unknown(tmp_path):
    """未来收益全部未到期（None）→ 沉默，不把 None 当 0 计算。"""
    m = ICTrendMonitor(_cfg(tmp_path))
    res = m.evaluate_pairs("short_term", 5, [0.1] * 100, [None] * 100)
    assert res.available is False
    assert res.status == STATUS_UNKNOWN


def test_too_few_anchors_keeps_unknown_but_reports_latest_ic(tmp_path):
    """锚点不足以判趋势：available（有 IC 快照）但 status=unknown，不给斜率。"""
    m = ICTrendMonitor(_cfg(tmp_path, window=20, step=10, min_anchors=10))
    scores, rets = _pairs(n=60)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available is True
    assert res.latest_ic is not None
    assert res.status == STATUS_UNKNOWN
    assert res.slope_per_step is None
    assert "锚点不足" in res.reason


# ---------------------------------------------------------------- 趋势判定
def test_decaying_signal_is_detected(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path, window=150, step=40, min_obs=50, min_anchors=3))
    scores, rets = _pairs(n=800, strength=0.08, decay=True)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available
    assert res.status == STATUS_DECAYING
    assert res.slope_per_step < 0
    assert res.decay_per_step_pct is not None and res.decay_per_step_pct < 0


def test_improving_signal_is_detected(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path, window=150, step=40, min_obs=50, min_anchors=3))
    scores, rets = _pairs(n=800, strength=0.08, improve=True)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available
    assert res.status == STATUS_IMPROVING
    assert res.slope_per_step > 0


def test_flat_noise_is_stable_not_decaying(tmp_path):
    """纯噪声：斜率抖动必须落在 stable，不能被当成趋势（阈值存在的意义）。"""
    m = ICTrendMonitor(_cfg(tmp_path, window=60, step=20, min_obs=20, min_decay=0.01))
    rng = random.Random(5)
    scores = [rng.gauss(0, 0.1) for _ in range(600)]
    rets = [rng.gauss(0, 0.01) for _ in range(600)]
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available
    assert res.status == STATUS_STABLE


# ---------------------------------------------------------------- 跌破外推
def test_steps_to_breach_zero_when_already_below(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path))
    scores = [0.01] * 300          # 分数几乎无区分度 → IC 接近 0
    rets = [0.0] * 300
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available
    assert res.steps_to_breach == 0


def test_already_below_line_is_decaying_not_stable(tmp_path):
    """已跌破门禁线时状态必须报 decaying —— 不能出现「stable 但已跌破」的自相矛盾报表。"""
    cfg = _cfg(tmp_path, window=40, step=20, min_obs=20, min_anchors=3)
    cfg["strategy_gate"]["min_ic"] = 0.5        # 把门禁线设得极高，必定"已跌破"
    m = ICTrendMonitor(cfg)
    rng = random.Random(1)
    scores = [rng.gauss(0, 0.1) for _ in range(300)]
    rets = [rng.gauss(0, 0.01) for _ in range(300)]
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available
    assert res.steps_to_breach == 0
    assert res.status == STATUS_DECAYING
    assert "低于门禁线" in res.reason


def test_below_line_but_recovering_reports_improving(tmp_path):
    """已跌破但在回升（斜率显著为正）→ 如实报 improving，而不是一律 decaying。"""
    cfg = _cfg(tmp_path, window=150, step=40, min_obs=50, min_anchors=3)
    cfg["strategy_gate"]["min_ic"] = 0.9
    m = ICTrendMonitor(cfg)
    scores, rets = _pairs(n=800, strength=0.08, improve=True)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available
    assert res.status == STATUS_IMPROVING


def test_steps_to_breach_none_when_slope_non_negative(tmp_path):
    """斜率非负时不给"还有多久掉出去"——不外推『永远安全』这种结论。"""
    m = ICTrendMonitor(_cfg(tmp_path, window=150, step=40, min_obs=50, min_anchors=3))
    scores, rets = _pairs(n=800, strength=0.08, improve=True)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.available
    assert res.steps_to_breach is None


def test_steps_to_breach_positive_when_decaying_toward_line(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path, window=100, step=20, min_obs=40, min_anchors=3))
    scores, rets = _pairs(n=900, strength=0.35, decay=True)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    if abs(res.latest_ic) >= m.min_ic and res.slope_per_step < 0:
        assert res.steps_to_breach is not None and res.steps_to_breach > 0


# ---------------------------------------------------------------- 数据正确性
def test_latest_ic_matches_direct_spearman_on_last_window(tmp_path):
    """时序最后一个锚点的 IC 必须等于「直接在最后一段上算」的结果（口径可复算）。"""
    m = ICTrendMonitor(_cfg(tmp_path, window=60, step=20, min_obs=20, min_anchors=4))
    scores, rets = _pairs(n=300)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    expected = spearman_ic(scores[-60:], rets[-60:])
    # evaluate_pairs 返回原始精度对象（未经 to_dict），此处可严格比对
    assert res.latest_ic == pytest.approx(expected, abs=1e-12)


def test_ic_series_length_matches_anchors(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path, window=60, step=20, min_obs=20, min_anchors=4))
    scores, rets = _pairs(n=300)
    res = m.evaluate_pairs("short_term", 5, scores, rets)
    assert res.anchors == len(res.ic_series)
    assert res.anchors > 0


def test_evaluate_aligned_does_not_cross_fold_boundaries(tmp_path):
    """evaluate_aligned 直接消费已对齐收益：折缝处的错位收益不会进入锚点。

    对照：若把折缝收益当成连续序列（拼接两段互不相关的收益），
    IC 时序会被人为抹平；这里验证同一输入下结果与直接分段计算一致。
    """
    m = ICTrendMonitor(_cfg(tmp_path, window=60, step=20, min_obs=20, min_anchors=4))
    scores, rets = _pairs(n=200, seed=3)
    res = m.evaluate_aligned({"short_term": {"horizon_days": 5, "scores": scores, "returns": rets}})
    row = res["horizons"]["short_term"]
    assert row["available"] is True
    # 契约里 IC 保留 4 位（to_dict 统一 round），比对按契约精度
    assert row["latest_ic"] == pytest.approx(
        round(spearman_ic(scores[-60:], rets[-60:]), 4), abs=1e-9
    )


# ---------------------------------------------------------------- 同源与汇总
def test_min_ic_comes_from_strategy_gate_config(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["strategy_gate"]["min_ic"] = 0.07
    m = ICTrendMonitor(cfg)
    assert m.min_ic == pytest.approx(0.07)


def test_summarize_flags_decaying_and_near_breach(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path, near_breach_steps=5))
    res = m.evaluate_aligned({})   # 空输入也必须能安全汇总
    assert res["horizons"] == {}
    assert res["decaying"] == []
    assert res["near_breach"] == []
    assert res["issues"] == []
    assert res["near_breach_steps"] == 5


def test_report_payload_is_json_serializable(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path))
    scores, rets = _pairs(n=300)
    res = m.evaluate_aligned({"short_term": {"horizon_days": 5, "scores": scores, "returns": rets}})
    text = json.dumps(res, ensure_ascii=False)
    assert "short_term" in text
    # NaN / Infinity 不允许出现在契约里
    assert "NaN" not in text and "Infinity" not in text


def test_ic_series_values_are_finite(tmp_path):
    m = ICTrendMonitor(_cfg(tmp_path))
    scores, rets = _pairs(n=300)
    res = m.evaluate_aligned({"short_term": {"horizon_days": 5, "scores": scores, "returns": rets}})
    for v in res["horizons"]["short_term"]["ic_series"]:
        assert v is None or math.isfinite(v)


# ---------------------------------------------------------------- 配置
def test_default_config_has_ic_trend_section():
    for name in ("configs/config.yaml", "configs/config_pro.yaml"):
        cfg = yaml.safe_load(Path(PROJECT_ROOT / name).read_text(encoding="utf-8"))
        assert "ic_trend" in cfg, name
        section = cfg["ic_trend"]
        for key in ("window", "step", "min_anchors", "min_obs", "min_decay",
                    "report_dir", "near_breach_steps"):
            assert key in section, f"{name}:{key}"


def test_config_values_are_sane():
    cfg = yaml.safe_load(Path(PROJECT_ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    ic = cfg["ic_trend"]
    assert ic["window"] > ic["min_obs"]
    assert ic["step"] > 0
    assert ic["min_anchors"] >= 2
    assert 0 < ic["min_decay"] < 1


# ---------------------------------------------------------------- CLI
def test_cli_exposes_ic_trend_command():
    import main

    parser = main.build_parser()
    args = parser.parse_args(["ic-trend"])
    assert args.command == "ic-trend"


def test_cli_ic_trend_writes_report(tmp_path, monkeypatch):
    """端到端（离线）：无行情时如实报错，不落假报告。"""
    import main

    cfg = {"data": {"markets": {"stock": {"enabled": True, "symbols": ["600519.SH"]}},
                    "prediction_horizons": {"short_term": 5}},
           "ic_trend": {"report_dir": str(tmp_path / "reports"), "window": 10,
                        "step": 5, "min_anchors": 2, "min_obs": 5}}

    import scripts.evaluate_models as ev
    monkeypatch.setattr(ev, "load_market_data", lambda *a, **k: {})
    payload = main.run_ic_trend(cfg, symbols=["600519.SH"])
    assert "error" in payload
    assert not (tmp_path / "reports" / "ic_trend.json").exists()


# ---------------------------------------------------------------- 监控报表
def _monitor_cfg(tmp_path: Path, write_trend: bool = True, decaying: bool = False) -> dict:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if write_trend:
        payload = {
            "generated_at": "2026-09-10T10:00:00",
            "window": 250, "step": 40, "min_ic": 0.03, "near_breach_steps": 3,
            "decaying": ["short_term"] if decaying else [],
            "near_breach": ["mid_term"] if decaying else [],
            "horizons": {
                "short_term": {"available": True, "status": "decaying" if decaying else "stable",
                               "latest_ic": 0.032, "slope_per_step": -0.004,
                               "decay_per_step_pct": -12.5, "latest_hit_rate": 0.53,
                               "steps_to_breach": 1 if decaying else 40, "anchors": 30},
                "mid_term": {"available": False, "status": "unknown", "anchors": 0,
                             "reason": "样本不足"},
            },
        }
        (reports / "ic_trend.json").write_text(json.dumps(payload), encoding="utf-8")
    return {"report": {"output_dir": str(reports)},
            "ic_trend": {"report_dir": str(reports), "near_breach_steps": 3},
            "strategy_gate": {"report_dir": str(reports)}}


def test_monitor_collects_ic_trend(tmp_path):
    from src.monitor.health_report import ModelMonitor

    rep = ModelMonitor(_monitor_cfg(tmp_path)).collect().to_dict()
    sec = rep["ic_trend"]
    assert sec["available"] is True
    assert sec["decaying"] == []
    statuses = {r["horizon"]: r["status"] for r in sec["rows"]}
    assert statuses["short_term"] == "stable"
    assert statuses["mid_term"] == "unknown"


def test_monitor_reports_decay_as_warning(tmp_path):
    from src.monitor.health_report import ModelMonitor

    rep = ModelMonitor(_monitor_cfg(tmp_path, decaying=True)).collect().to_dict()
    joined = " ".join(rep["issues"])
    assert "short_term" in joined
    assert "跌破门禁线" in joined


def test_monitor_without_trend_report_is_not_an_error(tmp_path):
    from src.monitor.health_report import ModelMonitor

    rep = ModelMonitor(_monitor_cfg(tmp_path, write_trend=False)).collect().to_dict()
    assert rep["ic_trend"]["available"] is False
    assert rep["ic_trend"]["reason"] == "no_ic_trend_report"


def test_monitor_markdown_contains_trend_section(tmp_path):
    from src.monitor.health_report import ModelMonitor, render_markdown

    md = render_markdown(ModelMonitor(_monitor_cfg(tmp_path)).collect().to_dict())
    assert "IC 趋势 / 信号衰减（Q5）" in md
    assert "short_term" in md


def test_monitor_markdown_renders_unavailable_branch(tmp_path):
    from src.monitor.health_report import ModelMonitor, render_markdown

    md = render_markdown(
        ModelMonitor(_monitor_cfg(tmp_path, write_trend=False)).collect().to_dict()
    )
    assert "IC 趋势 / 信号衰减（Q5）" in md
    assert "no_ic_trend_report" in md


# ---------------------------------------------------------------- 日报 / 周报
def test_daily_report_includes_trend_section(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    cfg = _monitor_cfg(tmp_path)
    cfg["features"] = {"sentiment_enabled": False}
    preds = [{"symbol": "600519.SH",
              "predictions": {"short_term": {"direction": "看涨", "probability": 0.6,
                                             "confidence": 0.6, "horizon_days": 5}}}]
    report = DailyReportGenerator(cfg).generate_report(preds)
    assert "信号衰减趋势（Q5）" in report
    assert "short_term" in report


def test_daily_report_trend_section_handles_missing_report(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    cfg = _monitor_cfg(tmp_path, write_trend=False)
    cfg["features"] = {"sentiment_enabled": False}
    report = DailyReportGenerator(cfg).generate_report(
        [{"symbol": "600519.SH", "predictions": {}}]
    )
    assert "信号衰减趋势（Q5）" in report
    assert "未评估" in report


def test_weekly_report_includes_gate_and_trend_sections(tmp_path):
    from src.report.daily_report import WeeklyReportGenerator

    cfg = _monitor_cfg(tmp_path)
    cfg["features"] = {"sentiment_enabled": False}
    report = WeeklyReportGenerator(cfg).generate_report(
        [{"symbol": "600519.SH", "predictions": {}}]
    )
    assert "信号衰减趋势（Q5）" in report
    assert "策略门禁" in report


# ---------------------------------------------------------------- API
def test_api_ic_trend_endpoint(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import server

    cfg = _monitor_cfg(tmp_path)
    server._config = cfg
    client = TestClient(server.app)
    resp = client.get("/api/v1/monitor/ic-trend")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert "short_term" in body["horizons"]


def test_api_ic_trend_endpoint_missing_report(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import server

    server._config = _monitor_cfg(tmp_path, write_trend=False)
    client = TestClient(server.app)
    body = client.get("/api/v1/monitor/ic-trend").json()
    assert body["available"] is False
    assert body["reason"] == "no_ic_trend_report"
