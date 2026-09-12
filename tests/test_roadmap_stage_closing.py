"""阶段收官守卫（fail-close）：「确认即收官」口径固化，防两种漂移。

背景（Issue #40）：
  2026-09-12 用户指令「这11项确认 并继续按排期计划开发」——11 项人工检查点由
  `pending` → `confirmed`。此前阶段 `status` 存在两种等价口径：
    - 确认侧：置 `completed`；
    - 收口侧：按「自动交付 ≠ 阶段完成」保留 `in_progress`。
  本轮统一为**「确认即收官」**：确认收集齐后阶段收官为 `completed`。

本守卫钉死两条**方向相反**的漂移：
  1. **假进度**：阶段标 `completed`，但人工检查点未 `confirmed`、或缺 `completed_at`
     / `confirmed_by` / `confirmed_at` —— 无签字却收官；
  2. **假保守**：检查点已全部 `confirmed`、阶段却既不 `completed` 也无任何交付依据
     —— 口径悬空（既不是「未开工」也不是「已收官」）。

同时钉死：**人工检查点自身永不 `completed`**（确认 ≠ 代签），
且收官**不得**顺带改 `strategy_gate` 配置（确认对象是现状默认值）。

全部离线：只读 `plan.json` 与 `configs/config.yaml`，不触网、不重训。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 2026-09-12 确认后收官的阶段（G1~G5 + H1~H5）
CLOSED_STAGE_IDS = ["S11", "S12", "S13", "S14", "S15",
                    "S16", "S17", "S18", "S19", "S20"]


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _stages() -> list:
    return [s for s in _plan()["stages"] if s["id"] in CLOSED_STAGE_IDS]


def _manual_tasks(stage: dict) -> list:
    manual = set(stage.get("manual_checkpoint") or [])
    return [t for t in stage["tasks"] if t["id"] in manual]


class TestConfirmedStagesAreClosed:
    def test_all_confirmed_stages_are_completed(self):
        """11 项确认后，G/H 轮阶段必须已收官为 `completed`。"""
        bad = [s["id"] for s in _stages() if s["status"] != "completed"]
        assert not bad, f"确认后仍未收官（口径悬空）: {bad}"

    @pytest.mark.parametrize("sid", CLOSED_STAGE_IDS)
    def test_completed_stage_has_completed_at(self, sid: str):
        """收官必须可审计：`completed_at` 不得缺失，也不得为 null。"""
        s = next(x for x in _plan()["stages"] if x["id"] == sid)
        assert s.get("completed_at"), f"{sid} 标 completed 却无 completed_at"

    @pytest.mark.parametrize("sid", CLOSED_STAGE_IDS)
    def test_completed_stage_carries_signature(self, sid: str):
        """收官必须带人工签署背书（防自动流程代签）。"""
        s = next(x for x in _plan()["stages"] if x["id"] == sid)
        assert s.get("confirmed_by"), f"{sid} 收官却无 confirmed_by"
        assert s.get("confirmed_at"), f"{sid} 收官却无 confirmed_at"
        assert s.get("confirmed_in", "").endswith("/issues/40"), (
            f"{sid} 收官未指向人工确认来源")

    @pytest.mark.parametrize("sid", CLOSED_STAGE_IDS)
    def test_completed_stage_manual_checkpoints_confirmed(self, sid: str):
        """阶段收官的前提：本阶段人工检查点必须已 `confirmed`。"""
        s = next(x for x in _plan()["stages"] if x["id"] == sid)
        manual = _manual_tasks(s)
        assert manual, f"{sid} 收官却无人工检查点（无签字依据）"
        for t in manual:
            assert t["status"] == "confirmed", (
                f"{t['id']} 未确认却让 {s['id']} 收官（假进度）")


class TestManualCheckpointsNeverCompleted:
    @pytest.mark.parametrize("sid", CLOSED_STAGE_IDS)
    def test_manual_checkpoint_stays_out_of_completed(self, sid: str):
        """确认 ≠ 代签：人工检查点永远不得是 `completed`。"""
        s = next(x for x in _plan()["stages"] if x["id"] == sid)
        bad = [t["id"] for t in _manual_tasks(s) if t["status"] == "completed"]
        assert not bad, f"人工检查点被标 completed（代签）: {bad}"

    @pytest.mark.parametrize("sid", CLOSED_STAGE_IDS)
    def test_manual_checkpoint_has_no_delivery_fields(self, sid: str):
        s = next(x for x in _plan()["stages"] if x["id"] == sid)
        for t in _manual_tasks(s):
            assert t.get("auto_run") is False, f"{t['id']} 标 auto_run"
            assert not t.get("completed_at"), f"{t['id']} 有 completed_at"
            assert not t.get("result"), f"{t['id']} 有 result"


class TestClosingDeclaresNoGateChange:
    @pytest.mark.parametrize("sid", CLOSED_STAGE_IDS)
    def test_closing_affects_gate_false(self, sid: str):
        s = next(x for x in _plan()["stages"] if x["id"] == sid)
        c = s.get("closing") or {}
        assert c.get("affects_gate") is False, f"{sid} closing 未声明 affects_gate=false"
        assert c.get("auto_scope"), f"{sid} closing 缺 auto_scope"
        assert c.get("manual_scope"), f"{sid} closing 缺 manual_scope"

    def test_strategy_gate_config_is_frozen_readonly(self):
        """收官不得顺带改门禁：`confidence_gate` 必须仍是冻结只读态。"""
        if not CONFIG_PATH.exists():
            pytest.skip("无配置")
        cfg = CONFIG_PATH.read_text(encoding="utf-8")
        assert 'mode: "report_only"' in cfg, "门禁被改动（不是 report_only）"
        assert "freeze_structure: true" in cfg, "冻结标志丢失"
        # 关键开关仍为关闭：确认的是现状默认值
        for toggle in ("conformal:", "regime:"):
            seg = cfg.split(toggle, 1)[1].split("\n", 1)[1]
            head = seg[:400]
            assert "enabled: false" in head, f"{toggle} 在收官中被意外启用"
