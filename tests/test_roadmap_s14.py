"""S14 发布态健康检查：把散落运维信号收敛成一条可执行判断。

测试要点：
  - 优先级：**产物可用性 > 指标好坏**（过期/损坏的产物比不达标更危险）；
  - blocking 必须真的阻断（`status=blocked`），缺失产物不得当成健康；
  - 告警路由：blocking 必发；action 只在 keys 变化时发（去重可验证）；
  - 边界：只汇总不重算（`affects_gate=false`）、不自动修复；
  - 消费口：CLI / 监控报表 / 落盘与读取。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.monitor import release_check as rc


def _fresh(days: float = 0.0) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")


def _healthy_inputs() -> dict:
    return {
        "gate": {"state": "readonly"},
        "ic_trend": {"available": True, "decaying": [], "generated_at": _fresh(1)},
        "horizon_scan": {"available": True, "generated_at": _fresh(1)},
        "horizon_decision": {"status": "pending", "verdict": "reject"},
        "feature_experiment": {"verdict": "reject"},
        "trial_registry": {"available": True, "count": 3},
        "models_missing": [],
    }


# ----------------------------------------------------------------------
# 基础结论
# ----------------------------------------------------------------------
def test_healthy_state_is_ready():
    res = rc.check_release_state(None, **_healthy_inputs())
    assert res["status"] == rc.STATUS_READY
    assert res["blocking"] == []
    assert res["affects_gate"] is False


def test_readonly_gate_is_info_not_action():
    res = rc.check_release_state(None, **_healthy_inputs())
    assert any(i["code"] == "gate_readonly" for i in res["info"])
    assert not any(i["code"] == "gate_readonly" for i in res["actions"])


def test_gated_state_is_also_info():
    inputs = _healthy_inputs()
    inputs["gate"] = {"state": "gated"}
    res = rc.check_release_state(None, **inputs)
    assert any(i["code"] == "gate_gated" for i in res["info"])


# ----------------------------------------------------------------------
# 产物可用性优先
# ----------------------------------------------------------------------
def test_missing_scan_blocks_even_if_metrics_fine():
    inputs = _healthy_inputs()
    inputs["horizon_scan"] = None
    res = rc.check_release_state(None, **inputs)
    assert res["status"] == rc.STATUS_BLOCKED
    assert any(i["code"] == "scan_missing" for i in res["blocking"])


def test_stale_scan_blocks():
    inputs = _healthy_inputs()
    inputs["horizon_scan"] = {"available": True, "generated_at": _fresh(30)}
    res = rc.check_release_state(None, **inputs)
    assert res["status"] == rc.STATUS_BLOCKED
    codes = {i["code"] for i in res["blocking"]}
    assert "scan_stale" in codes


def test_unparsable_scan_timestamp_blocks():
    inputs = _healthy_inputs()
    inputs["horizon_scan"] = {"available": True, "generated_at": "garbage"}
    res = rc.check_release_state(None, **inputs)
    assert any(i["code"] == "scan_timestamp_unparsable" for i in res["blocking"])


def test_corrupt_trial_registry_blocks():
    inputs = _healthy_inputs()
    inputs["trial_registry"] = {"available": False, "reason": "损坏"}
    res = rc.check_release_state(None, **inputs)
    assert res["status"] == rc.STATUS_BLOCKED
    item = [i for i in res["blocking"] if i["code"] == "trials_unavailable"][0]
    assert "校正次数可能被低估" in item["message"]


def test_missing_models_blocks():
    inputs = _healthy_inputs()
    inputs["models_missing"] = ["lightgbm_short_term_5d.pkl"]
    res = rc.check_release_state(None, **inputs)
    assert res["status"] == rc.STATUS_BLOCKED
    assert "lightgbm_short_term_5d.pkl" in res["blocking"][0]["message"]


def test_stale_decision_blocks():
    inputs = _healthy_inputs()
    inputs["horizon_decision"] = {"status": "stale", "verdict": "defer"}
    res = rc.check_release_state(None, **inputs)
    assert any(i["code"] == "decision_stale" for i in res["blocking"])


# ----------------------------------------------------------------------
# 建议动作
# ----------------------------------------------------------------------
def test_missing_decision_and_experiment_are_actions():
    inputs = _healthy_inputs()
    inputs["horizon_decision"] = None
    inputs["feature_experiment"] = None
    res = rc.check_release_state(None, **inputs)
    assert res["status"] == rc.STATUS_ATTENTION
    codes = {i["code"] for i in res["actions"]}
    assert {"decision_missing", "feature_experiment_missing"} <= codes


def test_ic_decay_is_action():
    inputs = _healthy_inputs()
    inputs["ic_trend"] = {"available": True, "decaying": ["short_term"],
                          "generated_at": _fresh(1)}
    res = rc.check_release_state(None, **inputs)
    assert res["status"] == rc.STATUS_ATTENTION
    item = [i for i in res["actions"] if i["code"] == "ic_decaying"][0]
    assert "short_term" in item["message"]
    assert "重训练" in item["next_step"]


def test_stale_ic_trend_is_action_not_blocking():
    inputs = _healthy_inputs()
    inputs["ic_trend"] = {"available": True, "decaying": [], "generated_at": _fresh(90)}
    res = rc.check_release_state(None, **inputs)
    assert any(i["code"] == "ic_trend_stale" for i in res["actions"])
    assert not any(i["code"] == "ic_trend_stale" for i in res["blocking"])


def test_adopt_feature_experiment_is_action():
    inputs = _healthy_inputs()
    inputs["feature_experiment"] = {"verdict": "adopt"}
    res = rc.check_release_state(None, **inputs)
    assert any(i["code"] == "feature_experiment_adopt" for i in res["actions"])


def test_blocking_outranks_action():
    inputs = _healthy_inputs()
    inputs["horizon_scan"] = None                 # blocking
    inputs["ic_trend"] = {"available": True, "decaying": ["mid_term"],
                          "generated_at": _fresh(1)}   # action
    res = rc.check_release_state(None, **inputs)
    assert res["status"] == rc.STATUS_BLOCKED
    assert res["counts"]["blocking"] >= 1
    assert res["counts"]["action"] >= 1


# ----------------------------------------------------------------------
# keys / 告警路由
# ----------------------------------------------------------------------
def test_keys_are_stable_and_exclude_info():
    inputs = _healthy_inputs()
    res1 = rc.check_release_state(None, **inputs)
    res2 = rc.check_release_state(None, **inputs)
    assert res1["keys"] == res2["keys"] == []      # 只有 info → 无待处理 key
    inputs["ic_trend"] = {"available": True, "decaying": ["short_term"],
                          "generated_at": _fresh(1)}
    res3 = rc.check_release_state(None, **inputs)
    assert res3["keys"] == ["ic_decaying"]


def test_route_alerts_sends_on_first_action(tmp_path):
    cfg = {"release_check": {"state_file": str(tmp_path / "state.json")}}
    inputs = _healthy_inputs()
    inputs["ic_trend"] = {"available": True, "decaying": ["short_term"],
                          "generated_at": _fresh(1)}
    res = rc.check_release_state(None, **inputs)
    d1 = rc.route_alerts(res, cfg)
    assert d1["should_send"] is True
    assert d1["new_keys"] == ["ic_decaying"]


def test_route_alerts_dedupes_unchanged_actions(tmp_path):
    cfg = {"release_check": {"state_file": str(tmp_path / "state.json")}}
    inputs = _healthy_inputs()
    inputs["ic_trend"] = {"available": True, "decaying": ["short_term"],
                          "generated_at": _fresh(1)}
    res = rc.check_release_state(None, **inputs)
    rc.route_alerts(res, cfg)
    d2 = rc.route_alerts(res, cfg)
    assert d2["should_send"] is False          # 同一句"IC 衰减"不重复播
    assert "静默" in d2["reason"]


def test_route_alerts_always_sends_blocking(tmp_path):
    cfg = {"release_check": {"state_file": str(tmp_path / "state.json")}}
    inputs = _healthy_inputs()
    inputs["horizon_scan"] = None
    res = rc.check_release_state(None, **inputs)
    rc.route_alerts(res, cfg)
    d2 = rc.route_alerts(res, cfg)
    assert d2["should_send"] is True           # blocking 每次都发
    assert "blocking" in d2["reason"]


def test_route_alerts_reports_resolved(tmp_path):
    cfg = {"release_check": {"state_file": str(tmp_path / "state.json")}}
    inputs = _healthy_inputs()
    inputs["ic_trend"] = {"available": True, "decaying": ["short_term"],
                          "generated_at": _fresh(1)}
    rc.route_alerts(rc.check_release_state(None, **inputs), cfg)
    res2 = rc.check_release_state(None, **_healthy_inputs())
    d = rc.route_alerts(res2, cfg)
    assert d["resolved_keys"] == ["ic_decaying"]
    assert d["should_send"] is True            # 恢复也要通知（否则人以为还在坏）


def test_route_alerts_survives_corrupt_state(tmp_path):
    cfg = {"release_check": {"state_file": str(tmp_path / "state.json")}}
    (tmp_path / "state.json").write_text("{broken", encoding="utf-8")
    res = rc.check_release_state(None, **_healthy_inputs())
    d = rc.route_alerts(res, cfg)
    assert "should_send" in d


# ----------------------------------------------------------------------
# collect_and_check（只读采集）
# ----------------------------------------------------------------------
def test_collect_and_check_reads_artifacts(tmp_path):
    cfg = {
        "report_dir": str(tmp_path),
        "horizon_scan": {"report_dir": str(tmp_path)},
        "horizon_decision": {"report_dir": str(tmp_path),
                             "scan_report_dir": str(tmp_path)},
        "feature_experiment": {"report_dir": str(tmp_path)},
        "trial_registry": {"dir": str(tmp_path), "filename": "trials.jsonl"},
        "release_check": {"state_file": str(tmp_path / "state.json")},
        "strategy_gate": {"report_dir": str(tmp_path)},
        "ic_trend": {"report_dir": str(tmp_path)},
        "training": {"save_dir": str(tmp_path / "models")},
        "audit": {"dir": str(tmp_path / "audit")},
    }
    res = rc.collect_and_check(cfg)
    assert res["affects_gate"] is False
    assert res["status"] in (rc.STATUS_READY, rc.STATUS_ATTENTION, rc.STATUS_BLOCKED)
    assert isinstance(res["keys"], list)


def test_collect_and_check_never_raises(tmp_path):
    rc.collect_and_check({})            # 全空配置也不得抛异常


# ----------------------------------------------------------------------
# CLI / 报表 / 落盘
# ----------------------------------------------------------------------
def test_cli_release_check_command():
    import main

    parser = main.build_parser()
    args = parser.parse_args(["release-check"])
    assert args.command == "release-check"
    assert args.notify is False
    args2 = parser.parse_args(["release-check", "--notify", "--json"])
    assert args2.notify is True and args2.as_json is True


def test_run_release_check_writes_report(tmp_path):
    import main

    cfg = {
        "report_dir": str(tmp_path),
        "horizon_scan": {"report_dir": str(tmp_path)},
        "horizon_decision": {"report_dir": str(tmp_path),
                             "scan_report_dir": str(tmp_path)},
        "feature_experiment": {"report_dir": str(tmp_path)},
        "trial_registry": {"dir": str(tmp_path), "filename": "trials.jsonl"},
        "release_check": {"report_dir": str(tmp_path),
                          "state_file": str(tmp_path / "state.json")},
        "strategy_gate": {"report_dir": str(tmp_path)},
        "ic_trend": {"report_dir": str(tmp_path)},
        "training": {"save_dir": str(tmp_path / "models")},
        "audit": {"dir": str(tmp_path / "audit")},
    }
    out = main.run_release_check(cfg, notify=True, as_json=True)
    assert (tmp_path / "release_check.json").exists()
    assert "alert_routing" in out


def test_monitor_collects_release_check(tmp_path):
    from src.monitor.health_report import ModelMonitor

    cfg = {"release_check": {"report_dir": str(tmp_path)}}
    res = ModelMonitor(cfg)._collect_release_check()
    assert res["available"] is False
    assert "release-check" in res["hint"]

    (tmp_path / "release_check.json").write_text(json.dumps({
        "status": "blocked", "blocking": [{"code": "scan_stale", "severity": "blocking",
                                          "message": "过期", "next_step": "重跑"}],
        "actions": [], "info": [], "keys": ["scan_stale"], "counts": {},
    }, ensure_ascii=False), encoding="utf-8")
    res2 = ModelMonitor(cfg)._collect_release_check()
    assert res2["available"] is True
    assert res2["status"] == "blocked"


def test_monitor_section_renders(tmp_path):
    from src.monitor.health_report import render_markdown

    payload = {"release_check": {
        "available": True, "status": "blocked",
        "blocking": [{"code": "scan_stale", "severity": "blocking",
                      "message": "扫描报告过期", "next_step": "重跑 horizon-scan"}],
        "actions": [], "info": [], "counts": {}, "keys": ["scan_stale"],
    }}
    text = render_markdown(payload)
    assert "发布态" in text
    assert "scan_stale" in text
    assert "重跑" in text


def test_config_section_present():
    for name in ("configs/config.yaml", "configs/config_pro.yaml"):
        text = Path(name).read_text(encoding="utf-8")
        assert "release_check:" in text
