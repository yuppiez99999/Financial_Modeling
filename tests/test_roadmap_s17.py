"""S17 / H2 路线测试：CPCV 净化交叉验证 + 过拟合审计（T17.1~T17.3）。

设计要点：
  - 全部离线（合成索引 + tmp 目录），CI 不触网；
  - 边界优先：不改门禁、不改写历史报告、不代下结论；
  - 无前视：purge/embargo 只能**剔除**样本；分段不重叠；
  - 不猜：路径不足 / 矩阵不齐 / 报告缺失 → available=false + 原因。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.cpcv import (  # noqa: E402
    cpcv_evaluate,
    cpcv_paths,
    cpcv_splits,
    deflated_shrinkage,
    pbo,
    purge_embargo_audit,
)
from src.eval.overfit_audit import (  # noqa: E402
    annotate_selection_freedom,
    build_report,
    effective_trial_budget,
    recompute_history,
)


# ----------------------------------------------------------------------
# T17.1 CPCV + purge/embargo
# ----------------------------------------------------------------------
class TestCpcvPaths:
    def test_combination_count_and_non_overlapping_test(self):
        paths = cpcv_paths(600, n_blocks=6, k_test=2, horizon_days=5)
        assert len(paths) == 15  # C(6,2)
        for p in paths:
            # 测试索引不得与训练索引相交
            assert not set(p["test_idx"].tolist()) & set(p["train_idx"].tolist())
            assert p["test_idx"].size > 0 and p["train_idx"].size > 0

    def test_insufficient_samples_returns_empty_not_guessed(self):
        assert cpcv_paths(40, n_blocks=6, k_test=2) == []
        assert cpcv_paths(600, n_blocks=6, k_test=6) == []

    def test_splits_interface_compatible(self):
        splits = cpcv_splits(600, n_blocks=6, k_test=2, horizon_days=5)
        assert len(splits) == 15
        tr, te = splits[0]
        assert isinstance(tr, np.ndarray) and isinstance(te, np.ndarray)

    def test_purge_removes_leaking_train_samples(self):
        """训练样本若紧贴测试区间（标签窗口重叠）必须被剔除。"""
        paths = cpcv_paths(600, n_blocks=6, k_test=2, horizon_days=5, embargo=5)
        for p in paths:
            tr = set(p["train_idx"].tolist())
            for lo, hi in p["test_intervals"]:
                # 测试区间前 h 天 与 后 embargo 天 内不得有训练样本
                for t in range(max(0, lo - 4), lo):
                    assert t not in tr, f"{t} 应因 purge 被剔除"
                for t in range(hi, min(600, hi + 5)):
                    assert t not in tr, f"{t} 应因 embargo 被剔除"

    def test_audit_detects_leak_and_passes_clean(self):
        leak = purge_embargo_audit(600, [95, 96, 300], [100, 101, 102], 5,
                                   embargo=5, test_blocks=[(100, 103)])
        assert leak["ok"] is False and leak["overlaps"] > 0
        clean = purge_embargo_audit(600, [50, 51, 300], [100, 101, 102], 5,
                                    embargo=5, test_blocks=[(100, 103)])
        assert clean["ok"] is True and clean["overlaps"] == 0

    def test_multi_block_audit_not_confused_by_gap(self):
        """多块测试时中间的训练样本不得被误判为泄漏（审计核心边界）。"""
        paths = cpcv_paths(600, n_blocks=6, k_test=2, horizon_days=5, embargo=5)
        for p in paths:
            audit = purge_embargo_audit(600, p["train_idx"], p["test_idx"], 5,
                                        embargo=5,
                                        test_blocks=[tuple(x) for x in p["test_intervals"]])
            assert audit["ok"] is True, audit["reason"]


class TestOverfitMetrics:
    def test_shrinkage_penalizes_best(self):
        r = deflated_shrinkage([0.01, 0.02, 0.03, 0.04, 0.05], n_trials=50)
        assert r["available"] is True
        assert r["best_shrunk"] < r["best_raw"]
        assert r["method"] == "cpcv_shrinkage"  # 不冒充 Sharpe DSR 精确解

    def test_shrinkage_survival_flag(self):
        r = deflated_shrinkage([0.4, 0.41, 0.39], n_trials=2)
        assert r["survives"] is True

    def test_shrinkage_empty_not_guessed(self):
        assert deflated_shrinkage([], n_trials=5)["available"] is False

    def test_pbo_counts_underperformance(self):
        r = pbo([[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
                 [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]])
        assert r["available"] is True
        assert r["pbo"] == pytest.approx(1.0)  # 最优策略每折都跑输

    def test_pbo_insufficient_candidates_not_guessed(self):
        r = pbo([[[0.5], [0.1]]])
        assert r["available"] is False and r["pbo"] is None

    def test_report_shape_and_boundaries(self):
        r = cpcv_evaluate([0.05] * 15, n_samples=600, n_blocks=6, k_test=2,
                          horizon_days=5, n_trials=20)
        assert r["kind"] == "cpcv_evaluation"
        assert r["affects_gate"] is False
        assert r["n_paths"] == 15 and r["purge_audit"]["ok"] is True
        assert r["shrinkage"]["available"] is True

    def test_insufficient_paths_reason(self):
        """每块样本过少（100 样本 / 6 组 ≈ 17 个）→ 路径不足，不猜。"""
        r = cpcv_evaluate([0.1], n_samples=100, n_blocks=6, k_test=2)
        assert r["available"] is False and "路径不足" in r["reason"]

    def test_too_few_blocks_for_k(self):
        """k 不小于组数时不生成路径（组合无意义）。"""
        assert cpcv_paths(600, n_blocks=6, k_test=6) == []


# ----------------------------------------------------------------------
# T17.2 统一试验预算
# ----------------------------------------------------------------------
class TestTrialBudget:
    def _cfg(self, tmp_path):
        return {"trial_registry": {"dir": str(tmp_path), "filename": "trials.jsonl"}}

    def test_missing_registry_is_zero_not_failure(self, tmp_path):
        cfg = self._cfg(tmp_path)
        b = effective_trial_budget(cfg)
        assert b["available"] is True and b["count"] == 0

    def test_cross_command_counting(self, tmp_path):
        from src.eval.trial_registry import record

        cfg = self._cfg(tmp_path)
        for cmd in ("ic", "ic", "tune", "conformal"):
            record(cmd, cfg)
        b = effective_trial_budget(cfg)
        assert b["count"] == 4
        assert b["by_command"]["ic"] == 2 and b["by_command"]["tune"] == 1

    def test_command_filter(self, tmp_path):
        from src.eval.trial_registry import record

        cfg = self._cfg(tmp_path)
        for cmd in ("ic", "tune"):
            record(cmd, cfg)
        b = effective_trial_budget(cfg, commands=["tune"])
        assert b["count"] == 1

    def test_asof_only_counts_known_trials(self, tmp_path):
        from src.eval.trial_registry import record

        cfg = self._cfg(tmp_path)
        record("ic", cfg, at="2026-01-01T00:00:00")
        record("ic", cfg, at="2026-06-01T00:00:00")
        b = effective_trial_budget(cfg, asof="2026-03-01T00:00:00")
        assert b["count"] == 1  # 无前视：后来那次不算

    def test_corrupt_registry_not_treated_as_zero(self, tmp_path):
        cfg = self._cfg(tmp_path)
        (Path(tmp_path) / "trials.jsonl").write_text("{bad json}\n", encoding="utf-8")
        b = effective_trial_budget(cfg)
        assert b["available"] is False and b["count"] == 0 and b["reason"]

    def test_annotation_mentions_total(self, tmp_path):
        from src.eval.trial_registry import record

        cfg = self._cfg(tmp_path)
        record("ic", cfg)
        ann = annotate_selection_freedom(effective_trial_budget(cfg), n_this_run=1)
        assert ann["n_trials_effective"] == 2
        assert "已扫描 2 次" in ann["note"]

    def test_unavailable_budget_warns_underestimated(self):
        ann = annotate_selection_freedom({"available": False, "reason": "文件损坏"})
        assert ann["available"] is False
        assert "可能被低估" in ann["note"]


# ----------------------------------------------------------------------
# T17.3 历史读数回算
# ----------------------------------------------------------------------
class TestHistoryRecompute:
    def _cfg(self, tmp_path):
        return {"trial_registry": {"dir": str(tmp_path), "filename": "trials.jsonl"}}

    def test_missing_report_marked_not_guessed(self, tmp_path):
        cfg = self._cfg(tmp_path)
        rows = recompute_history(cfg, reports_dir=str(tmp_path / "none"))
        assert all(r["report_found"] is False for r in rows)
        assert all(r["p_adjusted"] is None for r in rows)
        assert all("报告缺失" in r["verdict_note"] for r in rows)

    def test_recompute_shrinks_significance(self, tmp_path):
        from src.eval.trial_registry import record

        cfg = self._cfg(tmp_path)
        for _ in range(30):
            record("horizon-decision", cfg)
        rep = tmp_path / "reports"
        rep.mkdir()
        (rep / "horizon_decision.json").write_text(
            json.dumps({"rows": [{"p_value": 0.01}]}), encoding="utf-8")
        rows = recompute_history(cfg, reports_dir=str(rep))
        row = next(r for r in rows if r["command"] == "horizon-decision")
        assert row["raw_min_p"] == pytest.approx(0.01)
        assert row["p_adjusted"] >= 0.01
        assert row["still_significant"] is False  # 0.01*30 = 0.3
        assert "不显著" in row["verdict_note"]

    def test_z_to_p_fallback(self, tmp_path):
        """报告只有 z 统计量时也能换算 p（不因缺 p 字段而放弃回算）。"""
        cfg = self._cfg(tmp_path)
        rep = tmp_path / "reports"
        rep.mkdir()
        (rep / "label_ab.json").write_text(
            json.dumps({"horizons": {"5d": {"delta": {"hit_rate_z_new": 4.0}}}}),
            encoding="utf-8")
        rows = recompute_history(cfg, reports_dir=str(rep))
        row = next(r for r in rows if r["command"] == "label-ab")
        assert row["raw_min_p"] is not None and row["raw_min_p"] < 0.001

    def test_corrupt_report_not_crashed(self, tmp_path):
        cfg = self._cfg(tmp_path)
        rep = tmp_path / "reports"
        rep.mkdir()
        (rep / "label_ab.json").write_text("{not json", encoding="utf-8")
        rows = recompute_history(cfg, reports_dir=str(rep))
        assert next(r for r in rows if r["command"] == "label-ab")["report_found"] is False

    def test_history_reports_not_rewritten(self, tmp_path):
        """回算只读：既有报告内容必须逐字节不变。"""
        cfg = self._cfg(tmp_path)
        rep = tmp_path / "reports"
        rep.mkdir()
        payload = '{"rows": [{"p_value": 0.02}]}'
        f = rep / "feature_experiment.json"
        f.write_text(payload, encoding="utf-8")
        recompute_history(cfg, reports_dir=str(rep))
        assert f.read_text(encoding="utf-8") == payload

    def test_build_report_boundaries(self, tmp_path):
        cfg = self._cfg(tmp_path)
        r = build_report(cfg, reports_dir=str(tmp_path / "none"))
        assert r["kind"] == "overfit_audit"
        assert r["affects_gate"] is False
        assert r["cpcv"]["available"] is False
        assert "T17.4" in r["note"]
