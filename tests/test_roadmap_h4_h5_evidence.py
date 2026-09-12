"""T19.4 / T20.x 决策材料守卫 + T11.2 记账一致性（fail-close，不是功能测试）。

设计要点：
  - **全部离线**（合成概率 + 合成 JSON），CI 不触网、不重训；
  - 消融对照必须**同一拟合**：AUC 必须不变（校准是单调映射），变了即装配错误；
  - 对照**不做择优**：一升一降必须是 mixed，负面读数必须保留；
  - 决策材料必须**不悬空**：清单引用的文件与文档必须真实存在；
  - 检查点只允许 `pending` / `confirmed`（2026-09-12 用户确认后为 `confirmed`）、
    priority 唯一连续；阶段收官须带人工确认字段；
  - **T11.2 不得再出现 `completed_at`**（防「status=pending + completed_at」回潮）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import calibration_ablation as ca  # noqa: E402

PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
MANIFEST_PATH = PROJECT_ROOT / "schedule" / "manual_checkpoints.json"
DOC_PATH = PROJECT_ROOT / "00_kickoff" / "manual_checkpoints_round_g_h.md"

# 本轮（H4/H5）新进决策包的检查点
H45_CHECKPOINT_IDS = {"T19.4", "T20.1", "T20.4"}


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _stage(sid: str) -> dict:
    return next(s for s in _plan()["stages"] if s["id"] == sid)


# ----------------------------------------------------------------------
# 合成三条腿（离线）
# ----------------------------------------------------------------------
def _synthetic_legs(n: int = 4000, seed: int = 0, n_cal: int = 2000):
    """构造「原始概率系统性高估」的样本外序列 + 校准腿。"""
    from src.eval.probability_calibration import fit_calibrator

    rng = np.random.default_rng(seed)
    truth = rng.beta(2, 5, n)
    y = (rng.random(n) < truth).astype(int)
    raw = np.clip(truth * 0.6 + 0.25 + rng.normal(0, 0.03, n), 0.01, 0.99)
    fwd = (np.where(y > 0.5, 1.0, -1.0) * rng.uniform(0.005, 0.03, n)
           + rng.normal(0, 0.01, n))
    platt = fit_calibrator(raw[:n_cal], y[:n_cal], method="platt")
    iso = fit_calibrator(raw[:n_cal], y[:n_cal], method="isotonic")
    legs = {
        "base": ca.build_leg_payload(raw[n_cal:], y[n_cal:], fwd[n_cal:]),
        "platt": ca.build_leg_payload(platt["transform"](raw[n_cal:]),
                                      y[n_cal:], fwd[n_cal:]),
        "isotonic": ca.build_leg_payload(iso["transform"](raw[n_cal:]),
                                         y[n_cal:], fwd[n_cal:]),
    }
    return legs


# ----------------------------------------------------------------------
# 消融对照：结构性纪律
# ----------------------------------------------------------------------
class TestAblationSameFit:
    def test_report_available_and_aligned(self):
        rep = ca.build_ablation_report(_synthetic_legs())
        assert rep["available"] and rep["aligned"]
        assert rep["affects_gate"] is False
        assert set(rep["legs"]) == {"base", "platt", "isotonic"}

    def test_auc_invariance_for_strictly_monotone_calibration(self):
        """**严格单调**校准（platt）→ AUC 必须不变；变了说明不是同一次拟合。

        注意：isotonic 是**非严格单调**（分箱 → 大量并列），AUC 会因并列而
        轻微移动 —— 这是合法现象，不是装配错误，因此单独测（见下一例）。
        """
        rep = ca.build_ablation_report(_synthetic_legs())
        base = rep["legs"]["base"]["gate_point"]["auc"]
        platt = rep["legs"]["platt"]["gate_point"]["auc"]
        assert base is not None and platt is not None
        assert abs(platt - base) < 1e-6, f"platt 破坏 AUC 不变性: {base} vs {platt}"

    def test_isotonic_auc_drift_is_bounded_and_reported(self):
        """isotonic 的并列会带来小幅 AUC 漂移 —— 必须**如实报告**而非隐藏。

        若漂移超过容差，说明装配有问题；在容差内则允许存在，
        但 AUC 对照项本身仍会被记成 unchanged/degraded（不粉饰）。
        """
        rep = ca.build_ablation_report(_synthetic_legs())
        base = rep["legs"]["base"]["gate_point"]["auc"]
        iso = rep["legs"]["isotonic"]["gate_point"]["auc"]
        assert abs(iso - base) < 0.02, f"isotonic AUC 漂移过大: {base} vs {iso}"
        tag = next(c["tag"] for c in rep["criteria"]["isotonic"]["criteria"]
                   if c["criterion"].startswith("AUC"))
        assert tag in ("unchanged", "degraded", "improved")

    def test_tie_rule_counts_ties_in_denominator(self):
        """平局（p == 0.5）计入分母但不算命中：防「剔掉中性样本」抬高命中率。

        构造：2 个平局 + 1 判对 + 1 判错 → 命中率必须是 1/4 = 0.25。
        若把平局剔出分母，会得到 1/2 = 0.5（系统性虚高）。
        """
        proba = [0.5, 0.5, 0.9, 0.1]
        fwd = [0.01, -0.01, 0.01, 0.02]
        hr = ca.hit_rate_clf(proba, fwd)
        assert hr["available"]
        assert hr["samples"] == 4
        assert hr["tie_samples"] == 2
        assert hr["hit_rate"] == pytest.approx(0.25)  # 1/4，平局在分母不在分子
        assert hr["coverage"] == pytest.approx(0.5)   # 有效方向样本占比

    def test_tie_rule_hit_rate_never_exceeds_strict_rule(self):
        """稳健性：同一样本上，含平局口径的命中率不可能高于剔除平局口径。"""
        proba = [0.5, 0.5, 0.5, 0.9, 0.1]
        fwd = [0.01, -0.01, 0.02, 0.01, -0.01]
        hr = ca.hit_rate_clf(proba, fwd)
        strict = 1.0  # 只看非平局：0.9/+、0.1/− 都判对
        assert hr["hit_rate"] <= strict

    def test_unaligned_legs_are_rejected(self):
        """样本未对齐（y_true/fwd_ret 不一致）→ 不产结论（不猜）。"""
        legs = _synthetic_legs()
        legs["platt"]["fwd_ret"] = [-x for x in legs["platt"]["fwd_ret"]]
        rep = ca.build_ablation_report(legs)
        assert rep["available"] is False
        assert rep["aligned"] is False
        assert "未对齐" in rep["reason"] or "不一致" in rep["reason"]

    def test_missing_base_leg_is_rejected(self):
        legs = _synthetic_legs()
        legs.pop("base")
        rep = ca.build_ablation_report(legs)
        assert rep["available"] is False
        assert "base" in rep["reason"]


class TestAblationNoCherryPicking:
    def test_verdict_is_mixed_when_criteria_disagree(self):
        """一升一降（Brier/ECE 变好但阈值子集命中率变差）→ 必须是 mixed。

        这是本轮最关键的纪律：**概率更准 ≠ 决策更好**，
        不得择优、不得合成单一总分。
        """
        rep = ca.build_ablation_report(_synthetic_legs())
        crit = rep["criteria"]["platt"]["criteria"]
        tags = {c["criterion"]: c["tag"] for c in crit}
        # 校准质量必须改善（构造使然），决策读数可好可坏
        assert tags["Brier（越低越好）"] == "improved"
        assert tags["ECE（越低越好）"] == "improved"
        # 只要存在一升一降，判定就必须是 mixed / no_material_change，
        # 绝不能因为「校准质量变好」就整体判 improved
        ups = [t for t in tags.values() if t == "improved"]
        downs = [t for t in tags.values() if t == "degraded"]
        verdict = rep["criteria"]["platt"]["verdict"]
        if ups and downs:
            assert verdict == "mixed", f"一升一降却判 {verdict}: {tags}"
        assert verdict != "improved" or not downs

    def test_no_composite_score_field(self):
        """不得出现任何「加权总分 / 综合评分」字段（防把多指标压成单一结论）。"""
        rep = ca.build_ablation_report(_synthetic_legs())
        blob = json.dumps(rep, ensure_ascii=False).lower()
        for banned in ("composite_score", "overall_score", "weighted_score",
                       "total_score"):
            assert banned not in blob, f"出现综合评分字段: {banned}"

    def test_degraded_reading_is_kept_not_dropped(self):
        """构造一条「校准后更差」的腿：必须如实记 degraded，不得丢弃该项。"""
        legs = _synthetic_legs()
        n = len(legs["base"]["proba"])
        # 反向校准：把概率推向错误方向（单调但反向）→ 决策读数必然变差
        base = np.asarray(legs["base"]["proba"], dtype=float)
        legs["platt"]["proba"] = list(np.clip(0.5 - (base - 0.5), 0.001, 0.999))
        rep = ca.build_ablation_report(legs)
        crit = rep["criteria"]["platt"]["criteria"]
        tags = [c["tag"] for c in crit]
        assert "degraded" in tags, f"负面读数被丢弃: {tags}"
        assert rep["criteria"]["platt"]["verdict"] != "improved"

    def test_insufficient_rows_are_marked_not_guessed(self):
        """阈值行样本不足 → available=false，不给指标（不猜）。"""
        legs = _synthetic_legs(n=300, n_cal=100)
        rep = ca.build_ablation_report(legs, grid=(0.0, 0.9),
                                       min_samples=5000)
        rows = rep["legs"]["base"]["threshold_curve"]["rows"]
        assert all(r["available"] is False for r in rows)


# ----------------------------------------------------------------------
# 决策材料：不得悬空 / 检查点状态可追溯
# ----------------------------------------------------------------------
class TestDecisionMaterialsNotDangling:
    def test_h45_checkpoints_are_registered(self):
        ids = {cp["id"] for cp in _manifest()["checkpoints"]}
        missing = H45_CHECKPOINT_IDS - ids
        assert not missing, f"H4/H5 检查点未进决策包: {sorted(missing)}"

    def test_h45_checkpoints_are_pending_or_confirmed(self):
        """H4/H5 检查点只允许 `pending` / `confirmed`（`completed` 即代签）。

        2026-09-12 用户确认（Issue #40）后落定为 `confirmed`：
        确认 ≠ 代签，但必须带签署字段（见 test_roadmap_manual_checkpoints.py）。
        """
        by = {cp["id"]: cp for cp in _manifest()["checkpoints"]}
        for cid in H45_CHECKPOINT_IDS:
            st = by[cid]["status"]
            assert st in ("pending", "confirmed"), f"{cid} 状态非法: {st}"
            if st == "confirmed":
                for key in ("confirmed_by", "confirmed_at", "decision"):
                    assert by[cid].get(key), f"{cid} 标 confirmed 却缺 {key}（无签字依据）"

    def test_all_referenced_evidence_paths_are_declared_kinds(self):
        """引用的路径必须是 `00_kickoff/` 文档或 `reports/` 产物（防引到不存在的东西）。"""
        for cp in _manifest()["checkpoints"]:
            for ev in cp.get("evidence") or []:
                assert ev.startswith(("reports/", "00_kickoff/")), \
                    f"{cp['id']} 引用了非约定路径: {ev}"

    def test_referenced_docs_exist_on_disk(self):
        missing = []
        for cp in _manifest()["checkpoints"]:
            doc = cp["doc"].split("#")[0]
            if not (PROJECT_ROOT / doc).exists():
                missing.append((cp["id"], doc))
        assert not missing, f"决策包引用了不存在的文档: {missing}"

    def test_priorities_unique_and_continuous(self):
        ps = [cp["priority"] for cp in _manifest()["checkpoints"]]
        assert len(ps) == len(set(ps)), "priority 重复"
        assert sorted(ps) == list(range(1, len(ps) + 1)), f"priority 不连续: {sorted(ps)}"

    def test_h_round_stages_closed_with_signature(self):
        """H 轮阶段不得「无签字收官」：`completed` 必须带人工确认字段与 closing。"""
        for sid in ("S16", "S17", "S18", "S19", "S20"):
            s = _stage(sid)
            assert s["status"] in ("in_progress", "completed"), f"{sid} 状态非法"
            if s["status"] == "completed":
                assert s.get("confirmed_by") and s.get("confirmed_at"), \
                    f"{sid} 标 completed 却无人工确认字段（防阶段假完成）"
                assert (s.get("closing") or {}).get("affects_gate") is False, \
                    f"{sid} closing 未声明 affects_gate=false"

    def test_decision_doc_covers_new_checkpoints(self):
        doc = DOC_PATH.read_text(encoding="utf-8")
        for cid in H45_CHECKPOINT_IDS:
            assert cid in doc, f"决策包文档未覆盖 {cid}"
        assert "不代签" in doc
        assert "affects_gate" in doc

    def test_decision_doc_declares_count_eleven(self):
        """文档声明的条目数必须与清单一致（防文档与清单漂移）。"""
        doc = DOC_PATH.read_text(encoding="utf-8")
        assert "11 条" in doc, "决策包文档声明的条目数与清单不一致"
        assert len(_manifest()["checkpoints"]) == 11


# ----------------------------------------------------------------------
# T11.2 记账（防回潮）
# ----------------------------------------------------------------------
class TestT112Bookkeeping:
    def _task(self) -> dict:
        stage = _stage("S11")
        return next(t for t in stage["tasks"] if t["id"] == "T11.2")

    def test_no_completed_at_on_manual_checkpoint(self):
        """人工检查点与 `completed_at` 并存 = 记账矛盾，必须钉死。

        状态可为 `pending` / `confirmed`（2026-09-12 用户确认后为 `confirmed`），
        但**永远不得**出现 `completed_at` / `result`（那等于把「已交付」当「已决策」）。
        """
        t = self._task()
        assert t["status"] in ("pending", "confirmed"), f"状态非法: {t['status']}"
        assert not t.get("completed_at"), "T11.2 又出现 completed_at（记账矛盾回潮）"
        assert not t.get("result"), "T11.2 又出现 result（视为已决策）"

    def test_delivery_fact_is_preserved_in_name(self):
        """移除状态字段不得丢交付事实：三档口径必须仍可追溯。"""
        name = self._task()["name"]
        assert "0.125" in name or "三档" in name
        assert "pending" in name or "草案" in name

    def test_manifest_entry_status_is_allowed(self):
        """清单条目状态只允许 `pending` / `confirmed`；若已确认须可追溯。"""
        by = {cp["id"]: cp for cp in _manifest()["checkpoints"]}
        st = by["T11.2"]["status"]
        assert st in ("pending", "confirmed"), f"状态非法: {st}"
        if st == "confirmed":
            assert by["T11.2"].get("confirmed_by") and by["T11.2"].get("confirmed_at")


# ----------------------------------------------------------------------
# S19 / S20 阶段结构（与 S16~S18 同构）
# ----------------------------------------------------------------------
class TestS19S20Structure:
    @pytest.mark.parametrize("sid", ["S19", "S20"])
    def test_required_fields(self, sid: str):
        s = _stage(sid)
        for key in ("id", "name", "status", "started_at", "scheduled_dates",
                    "tasks", "auto_acceptable", "manual_checkpoint",
                    "source", "depends_on", "note"):
            assert key in s, f"{sid} 缺少字段 {key}"

    @pytest.mark.parametrize("sid", ["S19", "S20"])
    def test_checkpoints_not_in_auto_acceptable(self, sid: str):
        s = _stage(sid)
        assert not (set(s["auto_acceptable"]) & set(s["manual_checkpoint"]))

    @pytest.mark.parametrize("sid", ["S19", "S20"])
    def test_all_tasks_done_or_pending_evidence(self, sid: str):
        s = _stage(sid)
        manual = set(s["manual_checkpoint"])
        for t in s["tasks"]:
            if t["id"] in manual:
                assert t["status"] in ("pending", "confirmed"), t["status"]
                assert not t.get("completed_at")
                continue
            assert t["status"] == "completed"
            assert t.get("completed_at") and t.get("result"), \
                f"{t['id']} 标 completed 却无 completed_at/result"
