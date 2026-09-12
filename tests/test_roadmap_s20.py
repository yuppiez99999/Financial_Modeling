"""S20 / H5 路线测试：投研辅助只读接入（T20.1~T20.4）。

设计要点：
  - 全部离线（合成数据 + 注入附注），CI 不触网、不调用任何 LLM；
  - **只读纪律是结构性保证**：附注字段不可进入信号/概率/门禁；
  - 不编造效果：附注不参与打分 → 对照结论必须是 unverifiable + 止损；
  - 不签字不放行：无 decided_by → 结论只能 defer / cancel（status=pending）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.research_assist import (  # noqa: E402
    CANDIDATES,
    VERDICT_CANCEL,
    VERDICT_DEFER,
    VERDICT_PROCEED,
    attach_research_note,
    build_decision,
    build_research_evaluation,
    offline_contrast,
)


# ----------------------------------------------------------------------
# T20.1 调研评估
# ----------------------------------------------------------------------
class TestResearchEvaluation:
    def test_all_three_candidates_covered(self):
        ev = build_research_evaluation()
        repos = [c["repo"] for c in ev["candidates"]]
        assert "TauricResearch/TradingAgents" in repos
        assert "hsliuping/TradingAgents-CN" in repos
        assert "qusong0627/QuantMind" in repos

    def test_no_candidate_passes_all_three_gates(self):
        """现状：无候选同时满足三项准入（如实结论，不美化）。"""
        ev = build_research_evaluation()
        assert ev["n_worth_introducing"] == 0
        assert "整阶段取消" in ev["recommendation"]

    def test_license_unclear_is_blocker(self):
        ev = build_research_evaluation()
        qm = next(c for c in ev["candidates"] if "QuantMind" in c["repo"])
        assert qm["license_clear"] is False
        assert "许可证不明确" in qm["blockers"]
        assert qm["worth_introducing"] is False

    def test_offline_not_reproducible_blocks_introduction(self):
        ev = build_research_evaluation()
        cn = next(c for c in ev["candidates"] if "TradingAgents-CN" in c["repo"])
        assert cn["a_share_adapted"] is True
        assert cn["worth_introducing"] is False
        assert any("离线不可复现" in b for b in cn["blockers"])

    def test_evaluation_never_affects_signal_or_gate(self):
        ev = build_research_evaluation()
        assert ev["affects_gate"] is False and ev["affects_signal"] is False

    def test_hypothetical_fully_qualified_candidate_can_pass(self):
        """反向验证：三项都满足时 worth_introducing 应为真（判定不是恒假）。"""
        ev = build_research_evaluation([{
            "repo": "demo/qualified", "license": "MIT", "stars_hint": "-",
            "a_share_adaptation": "有", "dependency_cost": "低",
            "offline_reproducible": True, "verdict": "pending", "reason": "",
        }])
        assert ev["n_worth_introducing"] == 1


# ----------------------------------------------------------------------
# T20.3 离线对照
# ----------------------------------------------------------------------
class TestOfflineContrast:
    def _data(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.5 + 0.3 * rng.random(n)
        r = 0.5 * (s - 0.5) + rng.normal(0, 0.05, n)
        return s, r

    def test_notes_not_in_scoring_is_unverifiable_and_stops(self):
        """核心边界：附注不参与打分 → 差异结构性为 0 → 如实止损，不编造效果。"""
        s, r = self._data()
        res = offline_contrast(s, r, notes=[{"note": "x"}] * len(s))
        assert res["verdict"] == "unverifiable"
        assert res["stop_loss"] is True
        assert res["delta_hit_rate"] == 0.0
        assert "不可量化" in res["conclusion"]
        assert res["affects_signal"] is False and res["affects_gate"] is False

    def test_no_notes_also_unverifiable(self):
        s, r = self._data()
        res = offline_contrast(s, r)
        assert res["verdict"] == "unverifiable" and res["stop_loss"] is True

    def test_external_scores_enable_real_measurement(self):
        """只有提供独立信号时才算真对照（且仍标注不进信号路径）。"""
        s, r = self._data()
        notes = [{"scores": v} for v in s]     # 与基准同分 → delta 0，但路径不同
        res = offline_contrast(s, r, notes=notes)
        assert res["verdict"] == "measured"
        assert res["delta_hit_rate"] == pytest.approx(0.0, abs=1e-9)
        assert res["affects_signal"] is False

    def test_insufficient_samples_not_guessed(self):
        res = offline_contrast([0.5] * 5, [0.01] * 5)
        assert res["available"] is False and "样本不足" in res["reason"]

    def test_empty_not_guessed(self):
        res = offline_contrast([], [])
        assert res["available"] is False and res["reason"] == "无样本"


# ----------------------------------------------------------------------
# T20.2 只读附注
# ----------------------------------------------------------------------
class TestReadonlyNote:
    def test_note_does_not_touch_signal_fields(self):
        pred = {"probability": 0.62, "direction": "看涨", "confidence": 0.62}
        out = attach_research_note(pred, note="研报附注", source="demo")
        assert out["probability"] == 0.62
        assert out["direction"] == "看涨"
        assert out["confidence"] == 0.62
        assert out["research_note"] == "研报附注"
        assert out["research_note_readonly"] is True

    def test_original_prediction_not_mutated(self):
        pred = {"probability": 0.5}
        attach_research_note(pred, note="x")
        assert "research_note" not in pred   # 不原地修改

    def test_non_dict_passthrough(self):
        assert attach_research_note(None, note="x") is None


# ----------------------------------------------------------------------
# T20.4 决策单
# ----------------------------------------------------------------------
class TestDecision:
    def _eval(self):
        return build_research_evaluation()

    def test_no_signature_means_pending_not_confirmed(self):
        d = build_decision(self._eval(), offline_contrast([0.5] * 100, [0.01] * 100))
        assert d["status"] == "pending"
        assert d["verdict"] in (VERDICT_DEFER, VERDICT_CANCEL)
        assert any("缺少人工签字" in b for b in d["blockers"])

    def test_zero_qualified_candidates_cancels_by_default(self):
        d = build_decision(self._eval(), None)
        assert d["verdict"] == VERDICT_CANCEL

    def test_signed_proceed_becomes_confirmed(self):
        """有合格候选 + 人工签字 + proceed → confirmed/proceed。"""
        qualified = build_research_evaluation([{
            "repo": "demo/qualified", "license": "MIT", "stars_hint": "-",
            "a_share_adaptation": "有", "dependency_cost": "低",
            "offline_reproducible": True, "verdict": "pending", "reason": "",
        }])
        d = build_decision(qualified, None, decided_by="human", proceed=True)
        assert d["verdict"] == VERDICT_PROCEED and d["status"] == "confirmed"

    def test_signed_proceed_blocked_when_no_qualified_candidate(self):
        """0 个合格候选时，即便签字也不得 proceed（不能靠签字绕过准入）。"""
        d = build_decision(self._eval(), None, decided_by="human", proceed=True)
        assert d["verdict"] == VERDICT_CANCEL

    def test_signed_without_proceed_cancels(self):
        d = build_decision(self._eval(), None, decided_by="human", proceed=False)
        assert d["verdict"] == VERDICT_CANCEL and d["status"] == "confirmed"

    def test_decision_never_affects_signal_layers(self):
        d = build_decision(self._eval(), None, decided_by="human", proceed=True)
        assert d["affects_gate"] is False and d["affects_signal"] is False
        assert "不进信号路径" in d["note"]

    def test_unverifiable_contrast_is_a_blocker(self):
        s, r = np.linspace(0.4, 0.6, 200), np.zeros(200)
        c = offline_contrast(s, r, notes=[{"note": "x"}] * 200)
        d = build_decision(self._eval(), c, decided_by="")
        assert any("不可量化" in b for b in d["blockers"])

    def test_candidates_constant_documented(self):
        """候选清单本身就是排期交付物，必须可审计。"""
        assert len(CANDIDATES) >= 3
        for c in CANDIDATES:
            assert c["repo"] and c["license"] and c["reason"]
