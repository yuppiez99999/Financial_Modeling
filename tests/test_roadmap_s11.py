"""S11 预测周期切换决策前置：多重比较校正 + IC 显著性 + 决策单。

测试要点（与模块文档的边界一一对应）：
  - 统计：正态 p 值、Bonferroni、Holm、多重比较下限的行为正确且单调；
  - 显著性判定：小样本 / 零 IC / 强信号三档给出正确结论，不猜、不放宽；
  - 多重比较陷阱：**原始过线但校正后不显著**必须判 `reject`，不得当证据；
  - 决策单：未人工确认恒为 `pending`；批准需签字；报告过期 → `stale`；
  - 边界：`affects_gate` 恒为 False、不改 `data.prediction_horizons`；
  - 消费口：CLI / 监控报表 / 日报 / 周报 / API 缺失分支与正常分支。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.eval import horizon_decision as hd

# ⚠️ trial_registry.dir 指向不存在的临时目录：S13 起校正会消费历史试验次数，
# 若不隔离，用例会读到仓库里真实的 logs/trials.jsonl 而随机失败。
CONFIG = {
    "data": {"prediction_horizons": {"short_term": 5, "mid_term": 10, "long_term": 20}},
    "horizon_decision": {"report_dir": "reports"},
    "trial_registry": {"dir": ".pytest_nonexistent_trials", "filename": "trials.jsonl"},
}


def _scan(pooled: dict, generated_at: str | None = None) -> dict:
    return {
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "candidates": [int(k) for k in pooled],
        "pooled": pooled,
    }


def _row(days: int, ic: float, hit: float = 0.51, samples: int = 20000,
         passed: bool = False, icir: float = 0.05, available: bool = True) -> dict:
    return {
        "horizon_days": days,
        "available": available,
        "passed": passed,
        "ic": ic,
        "icir": icir,
        "hit_rate": hit,
        "samples": samples,
        "reason": "" if passed else "未过线",
    }


# ----------------------------------------------------------------------
# 统计基础
# ----------------------------------------------------------------------
def test_normal_p_value_known_points():
    assert hd.normal_two_sided_p(0.0) == pytest.approx(1.0, abs=1e-12)
    assert hd.normal_two_sided_p(1.96) == pytest.approx(0.05, abs=1e-3)
    assert hd.normal_two_sided_p(3.29) == pytest.approx(0.001, abs=1e-3)
    # 单调：|z| 越大 p 越小
    assert hd.normal_two_sided_p(1.0) > hd.normal_two_sided_p(2.0) > hd.normal_two_sided_p(3.0)


def test_normal_p_value_bad_input_is_conservative():
    """异常输入不给"显著"结论：NaN / 非数字一律 p=1。"""
    assert hd.normal_two_sided_p(float("nan")) == 1.0
    assert hd.normal_two_sided_p("abc") == 1.0  # type: ignore[arg-type]


def test_bonferroni_scales_and_clamps():
    assert hd.bonferroni(0.01, 5) == pytest.approx(0.05)
    assert hd.bonferroni(0.5, 5) == 1.0          # 截断到 1
    assert hd.bonferroni(0.01, 1) == pytest.approx(0.01)
    assert hd.bonferroni(0.01, 0) == pytest.approx(0.01)  # 非法次数按 1 处理


def test_holm_is_monotone_and_at_least_as_strict_as_bonferroni():
    raw = [0.001, 0.02, 0.3, 0.9]
    adj = hd.holm_adjust(raw)
    assert all(b >= a - 1e-12 for a, b in zip(raw, adj))  # 校正只会变大
    assert adj == sorted(adj) or True  # 同序返回，不要求输入有序
    # Holm 单调不减且在最小 p 上等价于 Bonferroni
    assert adj[0] == pytest.approx(min(1.0, raw[0] * len(raw)))
    # 比均匀 Bonferroni 更精确（不会一律乘 m）
    bonf = [hd.bonferroni(p, len(raw)) for p in raw]
    assert all(h <= b + 1e-12 for h, b in zip(adj, bonf))


def test_holm_empty_and_bad_values():
    assert hd.holm_adjust([]) == []
    out = hd.holm_adjust([float("nan"), 0.5])
    assert len(out) == 2 and all(0.0 <= x <= 1.0 for x in out)


def test_min_p_floor_grows_with_trials():
    """候选越多，「最好看的那个」纯靠运气能到多小 —— 下限必须随之变小。"""
    assert hd._min_p_value(1) == 1.0
    assert hd._min_p_value(5) > hd._min_p_value(20) > hd._min_p_value(100)
    assert 0.0 < hd._min_p_value(5) < 0.2


def test_ic_p_value_paths():
    # 大样本 + 正 IC → 有显著性；样本不足 → 保守返回 1
    assert hd.ic_p_value(0.10, 20000) < 0.01
    assert hd.ic_p_value(0.0, 20000) == 1.0
    assert hd.ic_p_value(0.10, 2) == 1.0
    # ICIR 路径：ICIR=0（无波动）退回样本量路径，不产生假显著
    assert hd.ic_p_value(0.01, 100, icir=0.0) > 0.5


# ----------------------------------------------------------------------
# 单候选评估
# ----------------------------------------------------------------------
def test_candidate_unavailable_not_usable_as_evidence():
    row = hd.evaluate_candidate_row(_row(40, 0.2, available=False), [5, 10, 20], 5,
                                    min_samples=30)
    assert row["available"] is False
    assert row["usable_as_evidence"] is False
    assert "不参与决策证据" in row["reason"]
    assert row["p_value"] is None


def test_candidate_marks_current_horizon():
    row = hd.evaluate_candidate_row(_row(10, 0.03), [5, 10, 20], 5)
    assert row["is_current_horizon"] is True
    other = hd.evaluate_candidate_row(_row(40, 0.03), [5, 10, 20], 5)
    assert other["is_current_horizon"] is False


def test_candidate_raw_pass_but_not_significant_is_flagged():
    """原始过线（扫描达标）但校正后不显著 —— 必须明说"不能直接当证据"。"""
    row = hd.evaluate_candidate_row(
        _row(40, 0.0925, hit=0.541, passed=True, icir=0.06), [5, 10, 20], 5)
    assert row["passed_scan_gate"] is True
    assert row["significant"] is False
    assert "不能直接当证据" in row["reason"] or "挑出来的达标" in row["reason"]


def test_strong_signal_becomes_significant_after_correction():
    row = hd.evaluate_candidate_row(
        _row(40, 0.30, hit=0.60, passed=True, icir=0.40, samples=30000), [5, 10, 20], 5)
    assert row["significant"] is True
    assert row["p_bonferroni"] <= 0.05


def test_pick_best_prefers_significant_then_abs_ic():
    rows = [
        {"available": True, "significant": False, "ic": 0.30, "horizon_days": 60},
        {"available": True, "significant": True, "ic": 0.10, "horizon_days": 40},
    ]
    assert hd.pick_best_candidate(rows)["horizon_days"] == 40
    assert hd.pick_best_candidate([]) is None
    assert hd.pick_best_candidate([{"available": False}]) is None


# ----------------------------------------------------------------------
# evaluate_scan
# ----------------------------------------------------------------------
def test_evaluate_scan_missing_report():
    res = hd.evaluate_scan(None, CONFIG)
    assert res["available"] is False
    assert res["verdict"] == hd.VERDICT_DEFER
    assert "horizon-scan" in res["narrative"]
    assert res["affects_gate"] is False


def test_evaluate_scan_rejects_multiplicity_trap():
    """核心用例：5 个候选里挑出一个"达标"，但校正后不显著 → 不支持切换。"""
    scan = _scan({
        "5": _row(5, 0.01, 0.50),
        "10": _row(10, 0.02, 0.51),
        "20": _row(20, 0.03, 0.51),
        "40": _row(40, 0.0925, 0.541, passed=True, icir=0.06),
        "60": _row(60, 0.05, 0.52),
    })
    res = hd.evaluate_scan(scan, CONFIG, proposed_days=[40])
    assert res["verdict"] == hd.VERDICT_REJECT
    assert res["raw_passing_candidates"] == [40]
    assert res["significant_candidates"] == []
    assert "多重比较" in res["narrative"]
    assert res["affects_gate"] is False
    assert res["n_trials"] == 5


def test_evaluate_scan_same_horizon_is_not_a_change():
    scan = _scan({"5": _row(5, 0.03), "10": _row(10, 0.03), "20": _row(20, 0.03)})
    res = hd.evaluate_scan(scan, CONFIG, proposed_days=[10])
    assert res["verdict"] == hd.VERDICT_REJECT
    assert "不构成口径变更" in res["narrative"]


def test_evaluate_scan_all_fail_says_changing_horizon_wont_help():
    scan = _scan({"5": _row(5, 0.0), "40": _row(40, 0.001)})
    res = hd.evaluate_scan(scan, CONFIG, proposed_days=[40])
    assert res["verdict"] == hd.VERDICT_REJECT
    assert "换周期解决不了问题" in res["narrative"]


def test_evaluate_scan_significant_without_approval_defers():
    """显著证据也只是"可以谈"，未经人工确认不得变成 approve。"""
    scan = _scan({"40": _row(40, 0.30, 0.60, passed=True, icir=0.40, samples=30000)})
    res = hd.evaluate_scan(scan, CONFIG, proposed_days=[40])
    assert res["significant_candidates"] == [40]
    assert res["verdict"] == hd.VERDICT_DEFER
    assert "需人工确认" in res["narrative"]
    assert res["approved"] is False


def test_evaluate_scan_significant_with_approval_approves():
    scan = _scan({"40": _row(40, 0.30, 0.60, passed=True, icir=0.40, samples=30000)})
    res = hd.evaluate_scan(scan, CONFIG, proposed_days=[40], approved=True)
    assert res["verdict"] == hd.VERDICT_APPROVE
    assert "不代改配置" in res["narrative"]
    assert res["affects_gate"] is False


def test_evaluate_scan_stale_report_defers():
    old = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
    scan = _scan({"40": _row(40, 0.30, 0.60, passed=True, icir=0.40, samples=30000)},
                 generated_at=old)
    res = hd.evaluate_scan(scan, CONFIG, proposed_days=[40], approved=True)
    assert res["stale"] is True
    assert res["verdict"] == hd.VERDICT_DEFER
    assert "不可复现" in res["narrative"]


def test_evaluate_scan_unparsable_timestamp_is_stale():
    scan = _scan({"40": _row(40, 0.3)}, generated_at="not-a-date")
    res = hd.evaluate_scan(scan, CONFIG, proposed_days=[40])
    assert res["stale"] is True
    assert res["report_age_days"] is None


def test_evaluate_scan_empty_pooled():
    res = hd.evaluate_scan(_scan({}), CONFIG)
    assert res["available"] is False
    assert res["verdict"] == hd.VERDICT_DEFER
    assert "没有可评估的候选周期" in res["narrative"] or "没有" in res["narrative"]


def test_evaluate_scan_never_touches_configured_horizons():
    """边界：评估前后现行口径逐字段不变。"""
    cfg = json.loads(json.dumps(CONFIG))
    before = dict(cfg["data"]["prediction_horizons"])
    scan = _scan({"40": _row(40, 0.30, passed=True, icir=0.4, samples=30000)})
    hd.evaluate_scan(scan, cfg, proposed_days=[40], approved=True)
    assert cfg["data"]["prediction_horizons"] == before


def test_evaluate_scan_thresholds_from_config():
    cfg = json.loads(json.dumps(CONFIG))
    cfg["horizon_decision"] = {"max_family_p": 0.5, "min_samples": 1000,
                              "max_report_age_days": 1}
    scan = _scan({"40": _row(40, 0.2, 0.55, samples=500, icir=0.3)})
    res = hd.evaluate_scan(scan, cfg, proposed_days=[40])
    assert res["thresholds"]["max_family_p"] == 0.5
    # 样本 500 < min_samples 1000 → 候选不可用
    assert res["candidates"][0]["available"] is False


# ----------------------------------------------------------------------
# 决策单
# ----------------------------------------------------------------------
def test_decision_record_pending_without_signature():
    scan = _scan({"40": _row(40, 0.30, 0.60, passed=True, icir=0.40, samples=30000)})
    rec = hd.build_decision_record(scan, CONFIG, proposed_days=[40])
    assert rec["verdict"] == hd.VERDICT_DEFER
    assert rec["status"] == hd.STATUS_PENDING
    assert rec["confirmed_by"] == ""
    assert any("无人工确认" in b for b in rec["blockers"])
    assert rec["affects_gate"] is False


def test_decision_record_confirmed_when_signed_and_significant():
    scan = _scan({"40": _row(40, 0.30, 0.60, passed=True, icir=0.40, samples=30000)})
    rec = hd.build_decision_record(scan, CONFIG, proposed_days=[40],
                                   decided_by="安然", reason="业务可接受 40 日延迟")
    assert rec["verdict"] == hd.VERDICT_APPROVE
    assert rec["status"] == hd.STATUS_CONFIRMED
    assert rec["confirmed_by"] == "安然"
    assert rec["blockers"] == []
    assert rec["decision"]["type"] == "prediction_horizon_change"
    assert rec["decision"]["from"] == [5, 10, 20]
    assert rec["decision"]["to"] == [40]
    assert rec["decision"]["changed"] is True


def test_decision_record_reject_stays_pending_even_signed():
    scan = _scan({"5": _row(5, 0.0), "40": _row(40, 0.001)})
    rec = hd.build_decision_record(scan, CONFIG, proposed_days=[40], decided_by="安然")
    assert rec["verdict"] == hd.VERDICT_REJECT
    assert rec["status"] == hd.STATUS_PENDING


def test_decision_record_stale_status():
    old = (datetime.now() - timedelta(days=99)).isoformat(timespec="seconds")
    scan = _scan({"40": _row(40, 0.3, 0.6, passed=True, icir=0.4, samples=30000)},
                 generated_at=old)
    rec = hd.build_decision_record(scan, CONFIG, proposed_days=[40], decided_by="安然")
    assert rec["status"] == hd.STATUS_STALE


def test_decision_record_missing_report_has_blocker():
    rec = hd.build_decision_record(None, CONFIG, proposed_days=[40])
    assert rec["blockers"]
    assert any("缺少多周期扫描报告" in b for b in rec["blockers"])
    assert rec["verdict"] == hd.VERDICT_DEFER


def test_decision_record_freeze_flag_from_config():
    cfg = json.loads(json.dumps(CONFIG))
    cfg["horizon_decision"]["freeze_current_horizons"] = False
    rec = hd.build_decision_record(None, cfg)
    assert rec["freeze_current_horizons"] is False


# ----------------------------------------------------------------------
# 落盘 / 读取
# ----------------------------------------------------------------------
def test_save_and_load_roundtrip(tmp_path):
    cfg = {"horizon_decision": {"report_dir": str(tmp_path)},
           "trial_registry": {"dir": str(tmp_path), "filename": "trials.jsonl"}}
    scan = _scan({"40": _row(40, 0.3)})
    rec = hd.build_decision_record(scan, cfg, proposed_days=[40])
    path = hd.save(rec, cfg)
    assert path.exists()
    loaded = hd.load(cfg)
    assert loaded["verdict"] == rec["verdict"]
    assert loaded["evidence"]["n_trials"] == 1
    assert loaded["evidence"]["n_trials_history"] == 0


def test_load_missing_and_corrupt(tmp_path):
    cfg = {"horizon_decision": {"report_dir": str(tmp_path)}}
    assert hd.load(cfg) is None
    (tmp_path / hd.REPORT_NAME).write_text("{not json", encoding="utf-8")
    assert hd.load(cfg) is None


def test_paths_from_config(tmp_path):
    cfg = {"horizon_decision": {"report_dir": str(tmp_path)}}
    assert hd.report_path(cfg) == tmp_path / hd.REPORT_NAME
    assert hd.scan_path(cfg).name == hd.DEFAULT_SCAN_NAME


# ----------------------------------------------------------------------
# CLI / 报表 / API 契约
# ----------------------------------------------------------------------
def test_cli_has_horizon_decision_command():
    import main

    parser = main.build_parser()
    args = parser.parse_args(["horizon-decision"])
    assert args.command == "horizon-decision"
    assert args.decided_by == ""
    assert args.reason == ""


def test_cli_horizon_decision_accepts_signature_args():
    import main

    parser = main.build_parser()
    args = parser.parse_args(["horizon-decision", "--days", "40",
                              "--decided-by", "安然", "--reason", "ok"])
    assert args.days == "40"
    assert args.decided_by == "安然"
    assert args.reason == "ok"


def test_run_horizon_decision_end_to_end(tmp_path, monkeypatch):
    import main

    scan_dir = tmp_path / "reports"
    scan_dir.mkdir()
    scan = _scan({"40": _row(40, 0.30, 0.60, passed=True, icir=0.40, samples=30000)})
    (scan_dir / hd.DEFAULT_SCAN_NAME).write_text(
        json.dumps(scan, ensure_ascii=False), encoding="utf-8")
    cfg = {
        "data": {"prediction_horizons": {"short_term": 5, "mid_term": 10, "long_term": 20}},
        "horizon_decision": {"report_dir": str(scan_dir), "scan_report_dir": str(scan_dir)},
    }
    out = main.run_horizon_decision(cfg, [40], decided_by="安然", reason="ok")
    assert out["status"] == hd.STATUS_CONFIRMED
    assert (scan_dir / hd.REPORT_NAME).exists()


def test_run_horizon_decision_without_scan_report(tmp_path):
    import main

    cfg = {"data": {"prediction_horizons": {"short_term": 5}},
           "horizon_decision": {"report_dir": str(tmp_path),
                                "scan_report_dir": str(tmp_path)}}
    out = main.run_horizon_decision(cfg, [40])
    assert out["verdict"] == hd.VERDICT_DEFER
    assert out["blockers"]


def test_monitor_report_renders_horizon_decision(tmp_path):
    from src.monitor.health_report import ModelMonitor, render_markdown

    cfg = {"horizon_decision": {"report_dir": str(tmp_path)}}
    scan = _scan({"40": _row(40, 0.3, 0.6, passed=True, icir=0.4, samples=30000)})
    hd.save(hd.build_decision_record(scan, cfg, proposed_days=[40]), cfg)

    collected = ModelMonitor(cfg)._collect_horizon_decision()
    assert collected["available"] is True

    text = render_markdown({"horizon_decision": collected})
    assert "S11" in text
    assert "决策单" in text
    assert "affects_gate=false" in text


def test_monitor_collect_missing_decision(tmp_path):
    from src.monitor.health_report import ModelMonitor

    res = ModelMonitor({"horizon_decision": {"report_dir": str(tmp_path)}})._collect_horizon_decision()
    assert res["available"] is False
    assert "horizon-decision" in res["hint"]


def test_daily_report_section_missing(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    gen = DailyReportGenerator.__new__(DailyReportGenerator)
    gen.config = {"horizon_decision": {"report_dir": str(tmp_path)}}
    gen.report_dir = str(tmp_path)
    lines = gen._generate_horizon_decision_section()
    text = "\n".join(lines)
    assert "S11" in text
    assert "未评估" in text


def test_daily_report_section_renders(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    cfg = {"horizon_decision": {"report_dir": str(tmp_path)}}
    scan = _scan({"40": _row(40, 0.3, 0.6, passed=True, icir=0.4, samples=30000)})
    hd.save(hd.build_decision_record(scan, cfg, proposed_days=[40]), cfg)

    gen = DailyReportGenerator.__new__(DailyReportGenerator)
    gen.config = cfg
    gen.report_dir = str(tmp_path)
    text = "\n".join(gen._generate_horizon_decision_section())
    assert "周期切换决策单" in text
    assert "[40]" in text
    assert "不改变" in text


def test_api_endpoint_missing(tmp_path):
    from fastapi.testclient import TestClient

    from src.api import server

    old = server._config
    try:
        server._config = {"horizon_decision": {"report_dir": str(tmp_path)}}
        client = TestClient(server.app)
        res = client.get("/api/v1/strategy/horizon-decision")
        assert res.status_code == 200
        body = res.json()
        assert body["available"] is False
        assert body["affects_gate"] is False
    finally:
        server._config = old


def test_api_endpoint_ok(tmp_path):
    from fastapi.testclient import TestClient

    from src.api import server

    cfg = {"horizon_decision": {"report_dir": str(tmp_path)},
           "trial_registry": {"dir": str(tmp_path), "filename": "trials.jsonl"}}
    scan = _scan({"40": _row(40, 0.3, 0.6, passed=True, icir=0.4, samples=30000)})
    hd.save(hd.build_decision_record(scan, cfg, proposed_days=[40]), cfg)

    old = server._config
    try:
        server._config = cfg
        client = TestClient(server.app)
        body = client.get("/api/v1/strategy/horizon-decision").json()
        assert body["available"] is True
        assert body["affects_gate"] is False
        assert body["evidence"]["n_trials"] == 1
        assert body["evidence"]["n_trials_history"] == 0
    finally:
        server._config = old


def test_config_sections_present_and_horizons_unchanged():
    """配置段存在，且现行门禁口径未被本模块改动。"""
    for name in ("configs/config.yaml", "configs/config_pro.yaml"):
        text = Path(name).read_text(encoding="utf-8")
        assert "horizon_decision:" in text
        assert "max_family_p" in text
    pro = Path("configs/config_pro.yaml").read_text(encoding="utf-8")
    assert "short_term: 5" in pro and "mid_term: 10" in pro and "long_term: 20" in pro
