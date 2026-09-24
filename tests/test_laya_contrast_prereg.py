"""S26 / J1 · T26.7 对照判据预注册：纪律守卫（全离线，无外部依赖）。

设计要点（这是**纪律测试**，不是功能测试）：
  - **规则先于跑数**：判定规则在 T26.1 之前冻结，指纹（sha256）钉死规则内容；
    规则一改 → 指纹变 ⇒ 该次读数不得再当「预注册结果」引用；
  - **fail-close**：无预注册规则 ⇒ 任何读数都不能判 `pass`；
  - **不编造**：真实权重未接入 / 读数缺失 ⇒ `unverifiable`，不给通过与否；
  - **结构性只读**：全部产出 `affects_gate` / `affects_signal` 恒 False；
  - **顺序纪律**：T26.7 与 T26.6 一样，先于 T26.1 人工签字。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import laya_contrast_prereg as P  # noqa: E402

PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
DOC_PATH = PROJECT_ROOT / "00_kickoff" / "s26_laya_decision_conclusion.md"
CAIRN_DOC = PROJECT_ROOT / "cairn" / "laya-readonly-admission.md"

FORBIDDEN_FIELDS = {"probability", "direction", "signal", "net_up_probability",
                    "composite_score"}


def _good_contrast(**over):
    c = {
        "kind": "laya_contrast",
        "available": True,
        "verdict": "measured",
        "stop_loss": False,
        "delta_hit_rate": 0.05,
        "laya_high_conf_samples": 30,
        "spearman_vs_baseline": 0.40,
        "confidence_return_bands": [
            {"band": [0.0, 0.25], "mean_return": -0.010},
            {"band": [0.25, 0.5], "mean_return": 0.000},
            {"band": [0.5, 0.75], "mean_return": 0.015},
            {"band": [0.75, 1.0], "mean_return": 0.030},
        ],
    }
    c.update(over)
    return c


# ----------------------------------------------------------------------
# 规则冻结 / 指纹
# ----------------------------------------------------------------------
class TestCriterionFrozen:
    def test_default_criterion_has_all_three_gates(self):
        c = P.default_criterion()
        r = c["rules"]
        assert r["min_delta_hit_rate"] == 0.02
        assert r["max_spearman_vs_baseline"] == 0.85
        assert r["require_band_monotonicity"] is True

    def test_criterion_is_preregistered_before_admission(self):
        c = P.default_criterion()
        assert "T26.1" in c["frozen_before"]

    def test_fingerprint_is_deterministic(self):
        assert P.criterion_fingerprint() == P.criterion_fingerprint(P.default_criterion())

    def test_fingerprint_changes_when_rule_changes(self):
        """改规则 ⇒ 指纹必变（事后改规则可被识别）。"""
        c = P.default_criterion()
        base = P.criterion_fingerprint(c)
        c2 = P.default_criterion()
        c2["rules"]["min_delta_hit_rate"] = 0.01  # 悄悄放松
        assert P.criterion_fingerprint(c2) != base

    def test_version_present(self):
        assert P.default_criterion()["version"] == P.CRITERION_VERSION


# ----------------------------------------------------------------------
# fail-close：无规则 / 无读数
# ----------------------------------------------------------------------
class TestFailClose:
    def test_no_criterion_never_passes(self):
        v = P.apply_criterion(_good_contrast(), None)
        assert v["verdict"] == P.VERDICT_NO_CRITERION
        assert v["verdict"] != P.VERDICT_PASS

    def test_unverifiable_contrast_stays_unverifiable(self):
        v = P.apply_criterion({"available": True, "verdict": "unverifiable",
                               "stop_loss": True}, P.default_criterion())
        assert v["verdict"] == P.VERDICT_UNVERIFIABLE

    def test_missing_contrast_is_unverifiable(self):
        v = P.apply_criterion(None, P.default_criterion())
        assert v["verdict"] == P.VERDICT_UNVERIFIABLE

    def test_partial_gate_unreadable_is_not_pass(self):
        v = P.apply_criterion(_good_contrast(spearman_vs_baseline=None),
                              P.default_criterion())
        assert v["verdict"] == P.VERDICT_UNVERIFIABLE
        assert v["verdict"] != P.VERDICT_PASS


# ----------------------------------------------------------------------
# 三条门各自可独立证伪
# ----------------------------------------------------------------------
class TestGates:
    def test_all_pass(self):
        v = P.apply_criterion(_good_contrast(), P.default_criterion())
        assert v["verdict"] == P.VERDICT_PASS

    def test_delta_gate_fails(self):
        v = P.apply_criterion(_good_contrast(delta_hit_rate=0.001), P.default_criterion())
        assert v["verdict"] == P.VERDICT_FAIL
        assert any("命中率增量" in f for f in v["failures"])

    def test_spearman_gate_fails_on_duplicate_sources(self):
        v = P.apply_criterion(_good_contrast(spearman_vs_baseline=0.97),
                              P.default_criterion())
        assert v["verdict"] == P.VERDICT_FAIL
        assert any("重复" in f for f in v["failures"])

    def test_spearman_gate_is_symmetric(self):
        v = P.apply_criterion(_good_contrast(spearman_vs_baseline=-0.97),
                              P.default_criterion())
        assert v["verdict"] == P.VERDICT_FAIL

    def test_band_monotonicity_gate_fails(self):
        bad = _good_contrast(confidence_return_bands=[
            {"mean_return": 0.030}, {"mean_return": 0.000}, {"mean_return": -0.020}])
        v = P.apply_criterion(bad, P.default_criterion())
        assert v["verdict"] == P.VERDICT_FAIL
        assert any("单调" in f for f in v["failures"])

    def test_band_monotonicity_flat_is_ok(self):
        flat = _good_contrast(confidence_return_bands=[
            {"mean_return": 0.0}, {"mean_return": 0.0}, {"mean_return": 0.0}])
        v = P.apply_criterion(flat, P.default_criterion())
        # 全同值 ⇒ 单调性不可判定 ⇒ 不判 pass（fail-close）
        assert v["verdict"] == P.VERDICT_UNVERIFIABLE

    def test_insufficient_high_conf_samples_not_pass(self):
        v = P.apply_criterion(_good_contrast(laya_high_conf_samples=3),
                              P.default_criterion())
        assert v["verdict"] != P.VERDICT_PASS


# ----------------------------------------------------------------------
# 结构性只读 / 无信号字段
# ----------------------------------------------------------------------
class TestReadonly:
    def test_report_is_readonly(self):
        r = P.build_prereg_report(_good_contrast())
        assert r["affects_gate"] is False
        assert r["affects_signal"] is False

    def test_verdict_is_readonly(self):
        v = P.apply_criterion(_good_contrast(), P.default_criterion())
        assert v["affects_gate"] is False
        assert v["affects_signal"] is False

    def test_prereg_report_has_no_signal_fields_recursively(self):
        r = P.build_prereg_report(_good_contrast())
        blob = json.dumps(r, ensure_ascii=False)
        for field in FORBIDDEN_FIELDS:
            assert f'"{field}"' not in blob, f"产出不应含信号字段 {field}"

    def test_report_carries_fingerprint(self):
        r = P.build_prereg_report(_good_contrast())
        assert r["criterion_fingerprint"] == P.criterion_fingerprint(r["criterion"])

    def test_n_trials_is_recorded(self):
        v = P.apply_criterion(_good_contrast(), P.default_criterion(), n_trials=7)
        assert v["n_trials"] == 7


# ----------------------------------------------------------------------
# 排期 / 顺序纪律
# ----------------------------------------------------------------------
class TestScheduleOrdering:
    def _stage(self):
        return next(s for s in json.loads(PLAN_PATH.read_text(encoding="utf-8"))["stages"]
                    if s["id"] == "S26")

    def test_t26_7_registered_and_delivered(self):
        s = self._stage()
        assert "T26.7" in s["auto_acceptable"]
        t = next(x for x in s["tasks"] if x["id"] == "T26.7")
        assert t["status"] == "completed"
        assert t.get("completed_at") and t.get("result")

    def test_prereg_also_precedes_admission_checkpoint(self):
        """顺序纪律：判据预注册（T26.7）属自动任务、先于准入签字（T26.1）。

        （tasks 列表里人工检查点排在前，故用 auto_acceptable 归属判定交付顺序——
        与 T26.6 的同类守卫一致。）
        """
        s = self._stage()
        assert "T26.7" in s["auto_acceptable"]
        assert "T26.1" in s["manual_checkpoint"]

    def test_manual_checkpoints_stay_pending(self):
        s = self._stage()
        for tid in s["manual_checkpoint"]:
            t = next(x for x in s["tasks"] if x["id"] == tid)
            assert t["status"] == "pending", f"{tid} 不应为 {t['status']}"
            assert t["auto_run"] is False

    def test_conclusion_doc_mentions_prereg(self):
        text = DOC_PATH.read_text(encoding="utf-8")
        assert "T26.7" in text
        assert "laya-prereg" in text
        assert "affects_gate" in text

    def test_cairn_doc_mentions_prereg(self):
        text = CAIRN_DOC.read_text(encoding="utf-8")
        assert "T26.7" in text
        assert "laya-prereg" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
