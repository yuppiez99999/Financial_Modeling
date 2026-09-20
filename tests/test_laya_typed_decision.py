"""S26 / J1 · Laya 类型化决策只读接入：纪律守卫（fail-close，不是功能测试）。

设计要点（这是**纪律测试**，不是功能测试）：
  - **全部离线**：合成概率 + 合成数据，CI 不触网、不下载权重、不重训；
  - **结构性只读**：全部产出 `readonly=True`，且**不含** `probability` /
    `direction` / `signal` 字段，`affects_gate` / `affects_signal` 恒 False；
  - **不编造效果**：真实 Laya 权重未接入 ⇒ 对照必须记 `unverifiable` + 止损；
  - **只留 Laya**：级联无 Jev 升级臂，零网络调用；
  - **不代签**：无 `decided_by` 时决策只能是 `defer` / `cancel`；
  - **排期一致**：S26 阶段与 T26.1/T26.5 人工检查点双向一致、priority 唯一连续；
  - **文档不悬空**：结论文档存在且写明边界。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import laya_typed_decision as L  # noqa: E402

PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
MANIFEST_PATH = PROJECT_ROOT / "schedule" / "manual_checkpoints.json"
DOC_PATH = PROJECT_ROOT / "00_kickoff" / "s26_laya_decision_conclusion.md"

# 结构性禁止字段：任何 Laya 产出都不得出现（否则就可能被信号路径消费）
FORBIDDEN_FIELDS = {"probability", "direction", "signal", "net_up_probability",
                    "composite_score"}


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# T26.1 准入评估
# ----------------------------------------------------------------------
class TestEvaluation:
    def test_laya_offline_ok_but_heavy_needs_manual(self):
        ev = L.build_laya_evaluation()
        laya = next(r for r in ev["candidates"] if r["repo"].startswith("Convai"))
        assert laya["offline_ok"] is True
        assert laya["heavy_dependency"] is True
        assert laya["needs_manual_admission"] is True

    def test_jev_rejected_offline_not_reproducible(self):
        ev = L.build_laya_evaluation()
        jev = next(r for r in ev["candidates"] if r["repo"].startswith("Jev"))
        assert jev["offline_ok"] is False
        assert jev["worth_introducing"] is False
        assert "离线不可复现" in " ".join(jev["blockers"])

    def test_evaluation_is_readonly(self):
        ev = L.build_laya_evaluation()
        assert ev["affects_gate"] is False
        assert ev["affects_signal"] is False


# ----------------------------------------------------------------------
# T26.2 只读适配器
# ----------------------------------------------------------------------
class TestAdapter:
    def test_missing_weights_degrades_without_error(self):
        a = L.LayaLocalAdapter(weights_path="")
        assert a.available is False
        st = a.status()
        assert st["affects_gate"] is False and st["affects_signal"] is False
        assert "未提供" in st["reason"]

    def test_nonexistent_weights_degrades(self, tmp_path):
        a = L.LayaLocalAdapter(weights_path=str(tmp_path / "nope.bin"))
        assert a.available is False
        assert "不存在" in a.status()["reason"]

    def test_decide_returns_noul_when_unavailable(self):
        d = L.LayaLocalAdapter().decide()
        assert d["kind"] == L.DECISION_NOUL
        assert d["available"] is False
        assert d["readonly"] is True
        assert d["affects_gate"] is False and d["affects_signal"] is False

    def test_typed_decision_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            L.typed_decision("bogus")

    def test_typed_decision_rejects_out_of_range_confidence(self):
        with pytest.raises(ValueError):
            L.typed_decision(L.DECISION_CHOICE, choice="up", confidence=1.5)

    def test_noul_clears_choice_and_score(self):
        d = L.typed_decision(L.DECISION_NOUL, choice="up", score=0.9, confidence=0.9)
        assert d["choice"] is None and d["score"] is None

    def test_no_forbidden_fields_in_outputs(self):
        outputs = [
            L.LayaLocalAdapter().decide(),
            L.LayaLocalAdapter().status(),
            L.typed_decision(L.DECISION_CHOICE, choice="up", score=0.7, confidence=0.7),
            L.cascade_smoke(adapter_available=False),
        ]
        for o in outputs:
            assert not (set(o.keys()) & FORBIDDEN_FIELDS), o

    def test_attach_note_touches_no_signal_fields(self):
        pred = {"probability": 0.62, "direction": "看涨"}
        out = L.attach_laya_note(pred, L.typed_decision(L.DECISION_CHOICE, choice="up"))
        # 原字段逐字保留（不被覆盖/污染）
        assert out["probability"] == 0.62 and out["direction"] == "看涨"
        assert out["laya_readonly"] is True
        assert "laya_decision" in out


# ----------------------------------------------------------------------
# T26.3 离线对照
# ----------------------------------------------------------------------
class TestContrast:
    def test_no_laya_output_is_unverifiable_and_stops_loss(self):
        rng = np.random.default_rng(42)
        proba = rng.uniform(0.4, 0.6, 200)
        ret = rng.normal(0, 0.02, 200)
        r = L.offline_contrast(proba, ret)
        assert r["verdict"] == "unverifiable"
        assert r["stop_loss"] is True
        assert r["laya_hit_rate"] is None
        assert r["affects_gate"] is False and r["affects_signal"] is False

    def test_insufficient_samples_does_not_guess(self):
        r = L.offline_contrast([0.5] * 5, [0.01] * 5)
        assert r["available"] is False

    def test_with_laya_output_measures_and_reports_spearman(self):
        rng = np.random.default_rng(7)
        proba = rng.uniform(0.4, 0.6, 200)
        ret = rng.normal(0, 0.02, 200)
        laya_conf = rng.uniform(0, 1, 200)
        r = L.offline_contrast(proba, ret, laya_confidence=laya_conf)
        assert r["verdict"] == "measured"
        assert r["stop_loss"] is False
        assert r["spearman_vs_baseline"] is not None
        assert isinstance(r["confidence_return_bands"], list)

    def test_contrast_never_affects_gate_or_signal(self):
        rng = np.random.default_rng(1)
        r = L.offline_contrast(rng.uniform(0.4, 0.6, 100), rng.normal(0, 0.01, 100),
                               laya_confidence=rng.uniform(0, 1, 100))
        assert r["affects_gate"] is False and r["affects_signal"] is False


# ----------------------------------------------------------------------
# T26.4 级联冒烟（只留 Laya，零网络）
# ----------------------------------------------------------------------
class TestCascadeSmoke:
    def test_local_only_no_network_no_raise(self):
        s = L.cascade_smoke()
        assert s["network_calls"] == 0
        assert s["no_network_dependency"] is True
        assert s["raises"] is False
        assert s["architecture"].startswith("laya_local_only")
        assert s["verdict"] == "pass"

    def test_unavailable_all_noul(self):
        s = L.cascade_smoke(adapter_available=False, confidences=[0.9, 0.8, 0.7])
        assert s["n_noul"] == 3 and s["n_choice"] == 0

    def test_low_confidence_degrades_high_confidence_accepts(self):
        s = L.cascade_smoke(adapter_available=True, confidences=[0.9, 0.4],
                            threshold=0.6)
        assert s["n_choice"] == 1 and s["n_noul"] == 1
        assert all(d["readonly"] is True for d in s["decisions"])

    def test_no_jev_upgrade_arm_mentioned(self):
        s = L.cascade_smoke(adapter_available=True, confidences=[0.1])
        assert "Jev" in s["architecture"] or "Jev" in s["conclusion"]
        assert "无 Jev 升级臂" in s["decisions"][0]["reason"]


# ----------------------------------------------------------------------
# T26.5 决策单：不代签
# ----------------------------------------------------------------------
class TestDecision:
    def _decision(self, **kw):
        ev = L.build_laya_evaluation()
        contrast = L.offline_contrast([0.5] * 50, [0.01] * 50)
        return L.build_decision(ev, contrast, **kw)

    def test_no_signature_is_defer_or_cancel(self):
        d = self._decision()
        assert d["verdict"] in (L.VERDICT_DEFER, L.VERDICT_CANCEL)
        assert d["status"] == L.STATUS_PENDING
        assert any("decided_by" in b for b in d["blockers"])

    def test_signed_proceed_requires_worth(self):
        d = self._decision(decided_by="安然", proceed=True)
        assert d["verdict"] == L.VERDICT_PROCEED
        assert d["status"] == L.STATUS_CONFIRMED

    def test_unverifiable_contrast_blocks_proceed(self):
        d = self._decision(decided_by="安然", proceed=True)
        # 对照 unverifiable ⇒ 止损说明必须留下
        assert any("不可量化" in b for b in d["blockers"])

    def test_decision_is_readonly(self):
        d = self._decision()
        assert d["affects_gate"] is False and d["affects_signal"] is False


# ----------------------------------------------------------------------
# 排期与人工检查点一致性
# ----------------------------------------------------------------------
class TestScheduleConsistency:
    def _stage(self):
        return next(s for s in _plan()["stages"] if s["id"] == "S26")

    def test_stage_exists_with_expected_shape(self):
        s = self._stage()
        assert s["round"] == "J1"
        assert s["affects_gate"] is False
        assert set(s["auto_acceptable"]) == {"T26.2", "T26.3", "T26.4"}
        assert set(s["manual_checkpoint"]) == {"T26.1", "T26.5"}

    def test_manual_checkpoint_not_in_auto_acceptable(self):
        s = self._stage()
        assert not (set(s["manual_checkpoint"]) & set(s["auto_acceptable"]))

    def test_auto_tasks_completed_and_auditable(self):
        s = self._stage()
        for tid in s["auto_acceptable"]:
            t = next(x for x in s["tasks"] if x["id"] == tid)
            assert t["status"] == "completed", f"{tid} 未交付"
            assert t.get("completed_at") and t.get("result"), f"{tid} 缺审计字段"

    def test_manual_tasks_are_not_auto_completed(self):
        s = self._stage()
        for tid in s["manual_checkpoint"]:
            t = next(x for x in s["tasks"] if x["id"] == tid)
            assert t["status"] in ("pending", "confirmed"), f"{tid} 被代签: {t['status']}"
            assert t["auto_run"] is False

    def test_stage_not_prematurely_completed(self):
        s = self._stage()
        assert s["status"] != "completed", "人工检查点未签前 S26 不得收官"

    def test_checkpoints_in_manifest_and_pending(self):
        manifest = {c["id"]: c for c in _manifest()["checkpoints"]}
        for tid in ("T26.1", "T26.5"):
            assert tid in manifest, f"{tid} 未进决策包"
            assert manifest[tid]["status"] == "pending", f"{tid} 不应为已签"
            assert manifest[tid]["stage"] == "S26"

    def test_manifest_priorities_unique_and_sequential(self):
        prios = [c["priority"] for c in _manifest()["checkpoints"]]
        assert prios == list(range(1, len(prios) + 1))

    def test_manifest_covers_j_round(self):
        assert any("J1" in r for r in _manifest()["rounds_covered"])

    @pytest.mark.parametrize("path", ["reports/laya_evaluation.json",
                                      "00_kickoff/s26_laya_decision_conclusion.md"])
    def test_evidence_paths_are_repo_relative_and_registered(self, path):
        """证据路径必须仓库相对；reports/ 运行产物须登记进 known_optional_evidence。"""
        p = Path(path)
        assert not p.is_absolute()
        optional = set(_manifest().get("known_optional_evidence") or [])
        if p.exists() or (PROJECT_ROOT / p).exists():
            return
        assert path in optional or path.split("#", 1)[0] in optional, \
            f"证据路径悬空且未登记为可选: {path}"

    def test_conclusion_doc_states_boundary(self):
        assert DOC_PATH.exists(), "缺少 S26 结论文档"
        text = DOC_PATH.read_text(encoding="utf-8")
        assert "affects_gate" in text and "只读" in text
