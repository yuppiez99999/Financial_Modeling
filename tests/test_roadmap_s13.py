"""S13 评估试验登记：把「试了多少次」变成不可篡改的事实。

测试要点：
  - 追加写（append-only）：登记不覆盖历史，只增加记录；
  - 损坏容忍：文件损坏时**绝不当作 0 次**（那是放纵未登记自由度）；
  - asof 语义：只看当时之前登记的试验（不允许"用未来的次数校正过去"）；
  - 口径指纹：同口径稳定、异口径不同，且只覆盖真正影响结果的关键配置；
  - 消费：S11/S12 的校正次数必须包含历史试验（分母变大 → 更难显著）；
  - 边界：登记失败不影响评估结果；CLI / 监控报表分支。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import trial_registry as tr

BASE_CFG = {
    "data": {"prediction_horizons": {"short_term": 5, "mid_term": 10, "long_term": 20},
             "start_date": "2020-01-01"},
    "features": {"extended_indicators": True, "macro_enabled": True},
    "model": {"type": "lightgbm", "factors": {"enabled": True}},
    "strategy_gate": {"min_ic": 0.03, "min_hit_rate": 0.52, "scope": "all"},
    "trial_registry": {"dir": "logs"},
}


def _cfg(tmp_path: Path) -> dict:
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["trial_registry"] = {"dir": str(tmp_path), "filename": "trials.jsonl"}
    return cfg


# ----------------------------------------------------------------------
# 落盘与追加写
# ----------------------------------------------------------------------
def test_registry_path_from_config(tmp_path):
    assert tr.registry_path(_cfg(tmp_path)) == tmp_path / "trials.jsonl"


def test_record_appends_without_overwriting(tmp_path):
    cfg = _cfg(tmp_path)
    e1 = tr.record("ic", cfg, summary={"symbols": 26})
    e2 = tr.record("ic", cfg, summary={"symbols": 26})
    assert e1["_persisted"] is True and e2["_persisted"] is True
    lines = (tmp_path / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2          # 追加，不覆盖
    assert json.loads(lines[0])["command"] == "ic"


def test_record_creates_directory(tmp_path):
    cfg = _cfg(tmp_path / "nested" / "deep")
    entry = tr.record("horizon-scan", cfg)
    assert entry["_persisted"] is True
    assert (tmp_path / "nested" / "deep" / "trials.jsonl").exists()


def test_make_entry_is_pure_and_has_fingerprint():
    entry = tr.make_entry("ic", BASE_CFG, summary={"x": 1}, note="n",
                          at="2026-09-11T00:00:00")
    assert entry["at"] == "2026-09-11T00:00:00"
    assert entry["command"] == "ic"
    assert entry["note"] == "n"
    assert entry["fingerprint_hash"]
    assert "data.prediction_horizons" in entry["fingerprint"]


# ----------------------------------------------------------------------
# 读取与损坏容忍
# ----------------------------------------------------------------------
def test_load_missing_file_is_empty_not_error(tmp_path):
    assert tr.load_entries(_cfg(tmp_path)) == []


def test_load_skips_bad_lines_but_keeps_good(tmp_path):
    cfg = _cfg(tmp_path)
    tr.record("ic", cfg)
    with (tmp_path / "trials.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{broken\n")
    entries = tr.load_entries(cfg)
    assert len(entries) == 1        # 好记录保留


def test_load_raises_when_all_lines_broken(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "trials.jsonl").write_text("{broken\n{also\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tr.load_entries(cfg)


def test_count_trials_reports_unavailable_on_corrupt(tmp_path):
    """核心：损坏**不得**被当成 0 次（0 次 = 校正失效 = 放纵自由度）。"""
    cfg = _cfg(tmp_path)
    (tmp_path / "trials.jsonl").write_text("garbage\n", encoding="utf-8")
    res = tr.count_trials(cfg)
    assert res["available"] is False
    assert res["count"] == 0
    assert "损坏" in res["reason"]


# ----------------------------------------------------------------------
# 统计与 asof
# ----------------------------------------------------------------------
def test_count_trials_filters_by_command_and_asof(tmp_path):
    cfg = _cfg(tmp_path)
    tr.record("ic", cfg)
    tr.record("ic", cfg)
    tr.record("horizon-scan", cfg)
    allc = tr.count_trials(cfg)
    assert allc["count"] == 3
    assert allc["by_command"] == {"ic": 2, "horizon-scan": 1}
    assert allc["trial_index"] == 4          # 下一次是第 4 次

    only_ic = tr.count_trials(cfg, command="ic")
    assert only_ic["count"] == 2


def test_asof_excludes_future_trials(tmp_path):
    cfg = _cfg(tmp_path)
    tr.record("ic", cfg, at="2026-01-01T00:00:00")
    tr.record("ic", cfg, at="2026-06-01T00:00:00")
    past = tr.count_trials(cfg, asof="2026-03-01T00:00:00")
    assert past["count"] == 1
    assert past["total"] == 2


def test_count_trials_filters_by_fingerprint(tmp_path):
    cfg = _cfg(tmp_path)
    tr.record("ic", cfg)
    other = json.loads(json.dumps(cfg))
    other["strategy_gate"]["min_ic"] = 0.99
    tr.record("ic", other)
    fp = tr.fingerprint_hash(tr.fingerprint(cfg))
    res = tr.count_trials(cfg, fingerprint_hash_value=fp)
    assert res["count"] == 1


def test_summary_includes_fingerprint_and_latest(tmp_path):
    cfg = _cfg(tmp_path)
    tr.record("ic", cfg, at="2026-02-02T00:00:00")
    out = tr.summary(cfg)
    assert out["available"] is True
    assert out["fingerprint_hash"] == tr.fingerprint_hash(out["fingerprint"])
    assert out["latest_at"] == "2026-02-02T00:00:00"


def test_summary_on_empty_registry(tmp_path):
    out = tr.summary(_cfg(tmp_path))
    assert out["available"] is True
    assert out["count"] == 0
    assert out["latest_at"] == ""


# ----------------------------------------------------------------------
# 口径指纹
# ----------------------------------------------------------------------
def test_fingerprint_stable_for_same_config():
    assert tr.fingerprint_hash(tr.fingerprint(BASE_CFG)) == \
        tr.fingerprint_hash(tr.fingerprint(json.loads(json.dumps(BASE_CFG))))


@pytest.mark.parametrize("path,value", [
    ("strategy_gate.min_ic", 0.99),
    ("data.prediction_horizons", {"short_term": 5, "mid_term": 10, "long_term": 40}),
    ("features.sentiment_enabled", True),
    ("model.factors.enabled", False),
])
def test_fingerprint_changes_when_knobs_change(path, value):
    other = json.loads(json.dumps(BASE_CFG))
    cur = other
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    assert tr.fingerprint_hash(tr.fingerprint(other)) != \
        tr.fingerprint_hash(tr.fingerprint(BASE_CFG))


def test_fingerprint_ignores_unrelated_keys():
    other = json.loads(json.dumps(BASE_CFG))
    other["logging"] = {"level": "DEBUG"}
    assert tr.fingerprint_hash(tr.fingerprint(other)) == \
        tr.fingerprint_hash(tr.fingerprint(BASE_CFG))


def test_fingerprint_custom_keys():
    fp = tr.fingerprint(BASE_CFG, keys=("model.type",))
    assert fp == {"model.type": "lightgbm"}


def test_fingerprint_missing_keys_are_skipped():
    fp = tr.fingerprint({"data": {}})
    assert fp == {}


# ----------------------------------------------------------------------
# 消费：S11 / S12 的校正必须包含历史试验
# ----------------------------------------------------------------------
def test_horizon_decision_uses_history_as_extra_trials(tmp_path):
    from src.eval import horizon_decision as hd

    cfg = _cfg(tmp_path)
    for _ in range(4):
        tr.record("horizon-scan", cfg)

    budget = hd.trial_budget(cfg)
    assert budget["available"] is True
    assert budget["extra_trials"] == 4

    scan = {"generated_at": "2099-01-01T00:00:00", "pooled": {
        "40": {"horizon_days": 40, "available": True, "passed": True,
               "ic": 0.0925, "icir": 0.06, "hit_rate": 0.541,
               "samples": 31000, "reason": ""},
    }}
    res = hd.evaluate_scan(scan, cfg, proposed_days=[40])
    assert res["n_trials"] == 1 + 4        # 本次 1 个候选 + 历史 4 次
    assert res["n_trials_history"] == 4
    assert res["min_p_floor"] < 1.0


def test_horizon_decision_trial_budget_unavailable_is_reported(tmp_path):
    from src.eval import horizon_decision as hd

    cfg = _cfg(tmp_path)
    (tmp_path / "trials.jsonl").write_text("broken\n", encoding="utf-8")
    budget = hd.trial_budget(cfg)
    assert budget["available"] is False
    assert budget["extra_trials"] == 0     # 但如实标注不可用，不静默当 0


def test_feature_experiment_counts_history(tmp_path):
    from src.eval import feature_experiment as fe

    cfg = _cfg(tmp_path)
    for _ in range(3):
        tr.record("feature-experiment", cfg)

    res = {
        "thresholds": {"max_family_p": 0.05},
        "_config": cfg,
        "horizons": {"short_term": {"arms": [
            {"arm": "cross_sectional", "horizon": "short_term", "available": True,
             "ic_delta": 0.05, "hit_delta": 0.01, "features_added": ["a"],
             "fold_ic_deltas": [0.04, 0.05, 0.045, 0.055]},
        ]}},
    }
    fe.decide(res)
    assert res["n_trials_history"] == 3
    assert res["n_trials"] == 1 + 3


def test_feature_experiment_without_config_degrades_gracefully():
    from src.eval import feature_experiment as fe

    res = {"thresholds": {"max_family_p": 0.05}, "horizons": {"s": {"arms": [
        {"arm": "macro", "horizon": "s", "available": True, "ic_delta": 0.05,
         "hit_delta": 0.0, "features_added": [], "fold_ic_deltas": [0.04, 0.05, 0.045]},
    ]}}}
    fe.decide(res)
    assert res["n_trials_history"] == 0    # 无 config → 如实降级为 0 增量


# ----------------------------------------------------------------------
# CLI / 报表
# ----------------------------------------------------------------------
def test_cli_trials_command():
    import main

    parser = main.build_parser()
    args = parser.parse_args(["trials"])
    assert args.command == "trials"
    args2 = parser.parse_args(["trials", "--note", "试了 40 日", "--asof", "2026-01-01"])
    assert args2.note == "试了 40 日"
    assert args2.asof == "2026-01-01"


def test_run_trials_records_and_reads(tmp_path, capsys):
    import main

    cfg = _cfg(tmp_path)
    out = main.run_trials(cfg, note="试了 40 日周期")
    assert out["available"] is True
    assert out["count"] == 1
    out2 = main.run_trials(cfg)
    assert out2["count"] == 1               # 第二次不带 note，不新增
    out3 = main.run_trials(cfg, note="再试一次")
    assert out3["count"] == 2


def test_run_trials_corrupt_registry(tmp_path):
    import main

    cfg = _cfg(tmp_path)
    (tmp_path / "trials.jsonl").write_text("nope\n", encoding="utf-8")
    out = main.run_trials(cfg)
    assert out["available"] is False


def test_monitor_collects_trials(tmp_path):
    from src.monitor.health_report import ModelMonitor

    cfg = _cfg(tmp_path)
    res = ModelMonitor(cfg)._collect_trial_registry()
    assert res["available"] is True
    assert res["count"] == 0

    tr.record("ic", cfg)
    res2 = ModelMonitor(cfg)._collect_trial_registry()
    assert res2["count"] == 1


def test_monitor_section_renders(tmp_path):
    from src.monitor.health_report import render_markdown

    cfg = _cfg(tmp_path)
    tr.record("ic", cfg)
    payload = {"trial_registry": {"available": True, "count": 1, "total": 1,
                                  "by_command": {"ic": 1}, "fingerprint_hash": "abc",
                                  "latest_at": "2026-09-11T00:00:00"}}
    text = render_markdown(payload)
    assert "试验登记" in text
    assert "abc" in text


def test_record_failure_does_not_raise(tmp_path):
    """登记失败不得让评估失败（fail-soft）。"""
    cfg = _cfg(tmp_path / "file-as-dir")
    (tmp_path / "file-as-dir").write_text("x", encoding="utf-8")  # 目录位置被文件占用
    entry = tr.record("ic", cfg)
    assert entry["_persisted"] is False


def test_config_section_present():
    for name in ("configs/config.yaml", "configs/config_pro.yaml"):
        text = Path(name).read_text(encoding="utf-8")
        assert "trial_registry:" in text
