"""S15 / G5 T15.3 测试：置信度子集门禁双指标决策单（须人工签字）。

设计要点（与 S11 horizon_decision / S12 / S13 测试同构）：
  - 全部离线（合成报告 dict + tmp 目录），CI 不触网、不重训；
  - 边界优先：不改门禁结构、不代选阈值、不签字不放行、样本不足不猜；
  - 双指标必须**两条腿同时过线**才算通过（单腿过线一律不合格）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import confidence_gate as cg  # noqa: E402


# ----------------------------------------------------------------------
# 构造合成报告
# ----------------------------------------------------------------------
def _row(thr, hit, cov, samples=5000, avail=True):
    return {"threshold": thr, "available": avail, "coverage": cov,
            "samples": samples, "hit_rate": hit, "ic": 0.05}


def _holdout(generated_at="2026-09-11T18:00:00", horizons=None):
    """构造保留期报告：默认在 thr∈[0.2,0.3] 内双指标过线。"""
    if horizons is None:
        # 0.2 命中率 52.9% 覆盖 17.4%；0.3 命中率 55.2% 覆盖 5.6% → 都过线
        horizons = {
            "5d": {"kind": "confidence_curve", "generated_at": generated_at,
                   "rows": [_row(0.0, 0.512, 1.0), _row(0.1, 0.522, 0.437),
                            _row(0.2, 0.529, 0.174), _row(0.3, 0.552, 0.056),
                            _row(0.4, 0.577, 0.013)]},
            "10d": {"kind": "confidence_curve", "generated_at": generated_at,
                    "rows": [_row(0.0, 0.527, 1.0), _row(0.2, 0.551, 0.223),
                             _row(0.3, 0.558, 0.079)]},
        }
    return {"generated_at": generated_at, "horizons": horizons}


_CFG = {"strategy_gate": {"confidence_gate": {
    "thr_min": 0.2, "thr_max": 0.3, "min_hit_rate": 0.52,
    "min_coverage": 0.05, "min_samples": 50}}}


# ----------------------------------------------------------------------
# 双指标判定
# ----------------------------------------------------------------------
class TestDualMetric:
    def test_both_legs_pass(self):
        r = cg.evaluate_threshold_row(_row(0.2, 0.529, 0.174), 0.52, 0.05)
        assert r["hit_rate_passed"] and r["coverage_passed"] and r["passed"]
        assert r["blocked_by"] == []

    def test_hit_pass_but_coverage_too_low_fails(self):
        """只过命中率、覆盖率过低 → 不合格（防"只留几个样本 100%"陷阱）。"""
        r = cg.evaluate_threshold_row(_row(0.3, 0.90, 0.01), 0.52, 0.05)
        assert r["hit_rate_passed"] and not r["coverage_passed"]
        assert not r["passed"]
        assert any("覆盖率" in b for b in r["blocked_by"])

    def test_coverage_pass_but_hit_too_low_fails(self):
        """只过覆盖率、命中率不够 → 不合格（就是现行全样本口径，没有改善）。"""
        r = cg.evaluate_threshold_row(_row(0.0, 0.49, 1.0), 0.52, 0.05)
        assert r["coverage_passed"] and not r["hit_rate_passed"]
        assert not r["passed"]
        assert any("命中率" in b for b in r["blocked_by"])

    def test_insufficient_samples_not_guessed(self):
        r = cg.evaluate_threshold_row(_row(0.2, 0.99, 0.99, samples=10), 0.52, 0.05,
                                      min_samples=50)
        assert r["available"] is False and r["passed"] is False
        assert r["blocked_by"] == ["insufficient_samples"]
        assert r["hit_rate"] is None  # 不猜

    def test_unavailable_row_flagged(self):
        r = cg.evaluate_threshold_row(_row(0.4, 0.0, 0.0, avail=False), 0.52, 0.05)
        assert r["available"] is False and r["passed"] is False


# ----------------------------------------------------------------------
# 候选区间扫描
# ----------------------------------------------------------------------
class TestCurveScan:
    def test_candidate_range_only_within_bounds(self):
        curve = {"kind": "confidence_curve", "rows": [
            _row(0.0, 0.512, 1.0), _row(0.1, 0.522, 0.437),
            _row(0.2, 0.529, 0.174), _row(0.3, 0.552, 0.056),
            _row(0.4, 0.577, 0.013),  # 过双指标但超区间 → 排除
        ]}
        ev = cg.evaluate_curve(curve, 0.52, 0.05, thr_min=0.2, thr_max=0.3)
        assert ev["available"] is True
        assert ev["candidate_thresholds"] == [0.2, 0.3]
        assert ev["candidate_range"] == [0.2, 0.3]

    def test_no_candidate_when_none_pass_both(self):
        curve = {"kind": "confidence_curve", "rows": [
            _row(0.0, 0.49, 1.0), _row(0.2, 0.60, 0.01), _row(0.3, 0.51, 0.08)]}
        ev = cg.evaluate_curve(curve, 0.52, 0.05, thr_min=0.2, thr_max=0.3)
        assert ev["candidate_thresholds"] == []
        assert ev["candidate_range"] is None
        assert "没有阈值同时满足双指标" in ev["reason"]

    def test_missing_curve_not_fabricated(self):
        ev = cg.evaluate_curve(None, 0.52, 0.05)
        assert ev["available"] is False and ev["curve_found"] is False
        assert "缺少置信度曲线" in ev["reason"]

    def test_rows_sorted_and_in_range_flag(self):
        curve = {"kind": "confidence_curve", "rows": [_row(0.3, 0.55, 0.06),
                                                      _row(0.1, 0.5, 0.4)]}
        ev = cg.evaluate_curve(curve, 0.52, 0.05, thr_min=0.2, thr_max=0.3)
        thrs = [r["threshold"] for r in ev["rows"]]
        assert thrs == sorted(thrs)
        assert {r["threshold"]: r["in_candidate_range"] for r in ev["rows"]}[0.1] is False


# ----------------------------------------------------------------------
# 保留期证据
# ----------------------------------------------------------------------
class TestHoldoutEvidence:
    def test_evidence_from_holdout(self):
        ev = cg.evaluate_holdout_evidence(_holdout(), _CFG)
        assert ev["available"] is True
        assert ev["evidence_source"] == "holdout"
        assert ev["affects_gate"] is False
        assert ev["candidate_rank"]
        assert "单一时段" in ev["single_period_warning"]

    def test_flat_holdout_layout_supported(self):
        flat = {"generated_at": "2026-09-11T18:00:00",
                "5d": {"rows": [_row(0.2, 0.53, 0.2)]}}
        ev = cg.evaluate_holdout_evidence(flat, _CFG)
        assert "5d" in ev["horizons"]

    def test_stale_report(self):
        ev = cg.evaluate_holdout_evidence(_holdout(generated_at="2020-01-01T00:00:00"),
                                          _CFG)
        assert ev["stale"] is True

    def test_missing_report(self):
        ev = cg.evaluate_holdout_evidence(None, _CFG)
        assert ev["report_found"] is False and ev["available"] is False

    def test_holdout_all_below_threshold_rejects(self):
        h = _holdout(horizons={"5d": {"rows": [_row(0.2, 0.51, 0.2)]}})
        ev = cg.evaluate_holdout_evidence(h, _CFG)
        assert ev["candidate_rank"] == []
        assert "均无双指标同时过线" in ev["reason"]


# ----------------------------------------------------------------------
# 决策单（须签字）
# ----------------------------------------------------------------------
class TestDecisionRecord:
    def test_unsigned_is_pending_defer(self):
        rec = cg.build_decision_record(_holdout(), config=_CFG,
                                       chosen_threshold=0.3)
        assert rec["verdict"] == cg.VERDICT_DEFER
        assert rec["status"] == cg.STATUS_PENDING
        assert rec["affects_gate"] is False
        assert any("签字" in b for b in rec["blockers"])

    def test_signed_but_no_threshold_still_defer(self):
        """有签字但没挑阈值 → 不能 approve（候选是区间不是单点）。"""
        rec = cg.build_decision_record(_holdout(), config=_CFG, decided_by="安然")
        assert rec["verdict"] == cg.VERDICT_DEFER
        assert rec["status"] == cg.STATUS_PENDING

    def test_signed_with_valid_threshold_approves(self):
        rec = cg.build_decision_record(_holdout(), config=_CFG,
                                       decided_by="安然", chosen_threshold=0.3,
                                       reason="保留期双指标过线，批准")
        assert rec["verdict"] == cg.VERDICT_APPROVE
        assert rec["status"] == cg.STATUS_CONFIRMED
        assert rec["confirmed_by"] == "安然"
        assert rec["affects_gate"] is False
        # 即便 confirmed，也不代改配置
        assert "不代改配置" in rec["narrative"] or "人工落配置" in rec["gate_note"]

    def test_threshold_outside_candidates_rejects(self):
        rec = cg.build_decision_record(_holdout(), config=_CFG,
                                       decided_by="安然", chosen_threshold=0.1)
        assert rec["verdict"] == cg.VERDICT_REJECT

    def test_no_signature_never_confirmed(self):
        """无论证据多好，没有签字就绝不 confirmed（fail-close）。"""
        rec = cg.build_decision_record(_holdout(), config=_CFG, chosen_threshold=0.2)
        assert rec["status"] != cg.STATUS_CONFIRMED

    def test_stale_holdout_blocks(self):
        rec = cg.build_decision_record(_holdout(generated_at="2020-01-01T00:00:00"),
                                       config=_CFG, decided_by="安然",
                                       chosen_threshold=0.3)
        assert rec["status"] == cg.STATUS_STALE
        assert rec["verdict"] == cg.VERDICT_DEFER

    def test_missing_holdout_defers(self):
        rec = cg.build_decision_record(None, config=_CFG, decided_by="安然")
        assert rec["verdict"] == cg.VERDICT_DEFER
        assert any("保留期" in b for b in rec["blockers"])

    def test_walk_forward_only_supplement_not_evidence(self):
        curve = {"kind": "confidence_curve", "rows": [_row(0.3, 0.9, 0.5)]}
        rec = cg.build_decision_record(None, curve=curve, config=_CFG)
        assert rec["walk_forward_supplement"]["role"] == "walk_forward_supplement"
        assert "walk-forward" in rec["walk_forward_supplement"]["note"]
        # 补充曲线不能替代保留期证据 → 仍然 defer
        assert rec["verdict"] == cg.VERDICT_DEFER

    def test_dual_metric_rule_stated(self):
        rec = cg.build_decision_record(_holdout(), config=_CFG)
        assert "同时过线" in rec["dual_metric"]["rule"]

    def test_freeze_structure_default(self):
        rec = cg.build_decision_record(_holdout(), config=_CFG)
        assert rec["freeze_structure"] is True

    def test_record_serializable(self):
        rec = cg.build_decision_record(_holdout(), config=_CFG,
                                       decided_by="安然", chosen_threshold=0.3)
        json.dumps(rec, ensure_ascii=False)  # 不抛异常 = 可落盘


# ----------------------------------------------------------------------
# 落盘 / 读取
# ----------------------------------------------------------------------
class TestIO:
    def test_load_json_missing_returns_none(self, tmp_path):
        assert cg.load_json(tmp_path / "nope.json") is None

    def test_load_json_corrupt_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        assert cg.load_json(p) is None

    def test_paths_respect_report_dir(self):
        cfg = {"strategy_gate": {"confidence_gate": {"report_dir": "out"}}}
        assert cg.report_path(cfg) == Path("out") / cg.REPORT_NAME
        assert cg.holdout_path(cfg) == Path("out") / cg.DEFAULT_HOLDOUT_NAME


# ----------------------------------------------------------------------
# CLI 接线
# ----------------------------------------------------------------------
class TestCliWiring:
    def test_parser_accepts_confidence_gate(self):
        from main import build_parser
        ns = build_parser().parse_args(["confidence-gate"])
        assert ns.command == "confidence-gate"

    def test_parser_chosen_threshold_and_decided_by(self):
        from main import build_parser
        ns = build_parser().parse_args(
            ["confidence-gate", "--chosen-threshold", "0.25",
             "--decided-by", "安然", "--reason", "ok"])
        assert ns.chosen_threshold == "0.25"
        assert ns.decided_by == "安然"

    def test_run_confidence_gate_no_report_defers(self, tmp_path, monkeypatch):
        from main import run_confidence_gate
        monkeypatch.chdir(tmp_path)
        cfg = {"strategy_gate": {"confidence_gate": {"report_dir": "reports"}}}
        rec = run_confidence_gate(cfg, decided_by="安然")
        assert rec["verdict"] == cg.VERDICT_DEFER
        assert rec["affects_gate"] is False
        assert (tmp_path / "reports" / cg.REPORT_NAME).exists()

    def test_run_confidence_gate_end_to_end(self, tmp_path, monkeypatch):
        from main import run_confidence_gate
        monkeypatch.chdir(tmp_path)
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / cg.DEFAULT_HOLDOUT_NAME).write_text(
            json.dumps(_holdout(), ensure_ascii=False), encoding="utf-8")
        cfg = {"strategy_gate": {"confidence_gate": {
            "report_dir": "reports", "thr_min": 0.2, "thr_max": 0.3,
            "min_hit_rate": 0.52, "min_coverage": 0.05, "min_samples": 50}}}
        rec = run_confidence_gate(cfg, decided_by="安然", chosen_threshold=0.3)
        assert rec["verdict"] == cg.VERDICT_APPROVE
        assert rec["status"] == cg.STATUS_CONFIRMED
