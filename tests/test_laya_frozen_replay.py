"""S26 / J1 · T26.6 冻结快照回放对账：纪律守卫（全离线合成数据）。

设计要点（这是**纪律测试**，不是功能测试）：
  - **全离线**：合成 CSV，CI 不触网、不读真实行情、不写数据；
  - **结构性只读**：全部产出 `affects_gate` / `affects_signal` 恒 False；
  - **不把"没证明"写成"证明"**：缺缓存 / 缺快照 / 长度不足 ⇒ `unverifiable`；
  - **判据写死**：按**位置**比金额列（同一价格序列）＋**逐值**核日期标签
    （区间相同而内部错位是最隐蔽的一种）；
  - **顺序纪律**：T26.6 必须在 T26.1 之前交付（先证明可复现，再批重型依赖）。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import laya_frozen_replay as R  # noqa: E402

PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
MANIFEST_PATH = PROJECT_ROOT / "schedule" / "manual_checkpoints.json"
DOC_PATH = PROJECT_ROOT / "00_kickoff" / "s26_laya_decision_conclusion.md"

FORBIDDEN_FIELDS = {"probability", "direction", "signal", "net_up_probability",
                    "composite_score"}


def _write_csv(path: Path, dates, closes, *, offset_days: int = 0) -> None:
    """写一份 6 列行情 CSV；`offset_days` 用来人为制造日期标签错位。"""
    d = pd.to_datetime(list(dates)) + pd.Timedelta(days=int(offset_days))
    n = len(closes)
    pd.DataFrame({
        "date": d, "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1_000_000 + i for i in range(n)],
    }).to_csv(path, index=False)


FROZEN_TAG = "20240101"
FROZEN_GLOB_LOCAL = f"data/raw/*.frozen.{FROZEN_TAG}.csv"


def _synthetic_pairs(tmp_path: Path, *, n: int = 300, offset_days: int = 5,
                     diverged: bool = False, cache_extra: int = 0):
    """造一对「缓存 ↔ 冻结快照」；默认：价格同一、日期标签错位（真实形态）。"""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n + cache_extra))))
    dates = pd.bdate_range("2024-01-01", periods=n + cache_extra)
    frozen_path = raw / f"TEST.SH.frozen.{FROZEN_TAG}.csv"
    cache_path = raw / "TEST.SH.csv"
    _write_csv(frozen_path, dates[:n], closes[:n])
    # 缓存：同序列（可加 offset 制造日期错位）；diverged=True 时把价格改坏
    ccloses = list(closes[n - n:n])
    if diverged:
        ccloses = [c * 1.5 for c in ccloses]
    _write_csv(cache_path, dates[:n], ccloses, offset_days=offset_days)
    if cache_extra:
        extra = list(closes[n:])
        ex_dates = pd.bdate_range(dates[0], periods=n + cache_extra)[n:]
        extra_df = pd.DataFrame({
            "date": ex_dates, "open": extra, "high": [c * 1.01 for c in extra],
            "low": [c * 0.99 for c in extra], "close": extra,
            "volume": [2_000_000 + i for i in range(cache_extra)],
        })
        pd.concat([pd.read_csv(cache_path, parse_dates=["date"]), extra_df]
                  ).to_csv(cache_path, index=False)
    return cache_path, frozen_path


# ----------------------------------------------------------------------
# 对账判据
# ----------------------------------------------------------------------
class TestReconcilePair:
    def test_identical_prices_with_label_drift_is_replayable(self, tmp_path):
        c, f = _synthetic_pairs(tmp_path, offset_days=5)
        r = R.reconcile_pair(str(c), str(f))
        assert r["identical_window"] is True
        assert r["label_aligned"] is False
        assert r["n_label_mismatch"] == 300
        assert r["verdict"] == "replayable"
        assert r["affects_gate"] is False and r["affects_signal"] is False

    def test_diverged_prices_not_replayable(self, tmp_path):
        c, f = _synthetic_pairs(tmp_path, diverged=True)
        r = R.reconcile_pair(str(c), str(f))
        assert r["identical_window"] is False
        assert r["verdict"] == "snapshot_diverged"
        assert "不可复现" in r["reason"]

    def test_aligned_dates_and_prices_fully_replayable(self, tmp_path):
        c, f = _synthetic_pairs(tmp_path, offset_days=0)
        r = R.reconcile_pair(str(c), str(f))
        assert r["label_aligned"] is True
        assert r["verdict"] == "replayable"

    def test_missing_side_is_unverifiable_not_guess(self, tmp_path):
        c, f = _synthetic_pairs(tmp_path)
        r = R.reconcile_pair(str(c), str(tmp_path / "nope.csv"))
        assert r["verdict"] == "unverifiable"
        assert "缺失" in r["reason"]

    def test_too_few_rows_is_unverifiable(self, tmp_path):
        c, f = _synthetic_pairs(tmp_path, n=max(R.MIN_COMMON_ROWS - 10, 20))
        r = R.reconcile_pair(str(c), str(f))
        assert r["verdict"] == "unverifiable"
        assert "不足" in r["reason"]

    def test_truncated_snapshot_flagged(self, tmp_path):
        """冻结快照短于当前缓存窗口 ⇒ window_truncated（回放窗口与生产窗口不一致）。"""
        raw = tmp_path / "data" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
        dates = pd.bdate_range("2024-01-01", periods=200)
        f = raw / f"T.SH.frozen.{FROZEN_TAG}.csv"
        _write_csv(f, dates[:120], closes[:120])
        c = raw / "T.SH.csv"
        _write_csv(c, dates, closes, offset_days=3)
        r = R.reconcile_pair(str(c), str(f))
        assert r["natural_window_ok"] is False
        assert r["verdict"] == "window_truncated"


# ----------------------------------------------------------------------
# 汇总结论
# ----------------------------------------------------------------------
class TestReplayReport:
    def test_report_conclusion_and_readonly(self, tmp_path):
        _synthetic_pairs(tmp_path, offset_days=5)
        rep = R.build_replay_report(raw_dir=str(tmp_path / "data" / "raw"),
                                    frozen_glob=FROZEN_GLOB_LOCAL)
        assert rep["n_pairs"] == 1
        assert rep["conclusion"] == "replayable_with_label_drift"
        assert rep["affects_gate"] is False and rep["affects_signal"] is False

    def test_no_pairs_is_unverifiable(self, tmp_path):
        (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
        rep = R.build_replay_report(raw_dir=str(tmp_path / "data" / "raw"),
                                    frozen_glob=FROZEN_GLOB_LOCAL)
        assert rep["conclusion"] == "unverifiable"
        assert rep["n_pairs"] == 0

    def test_diverged_pair_dominates_conclusion(self, tmp_path):
        _synthetic_pairs(tmp_path, diverged=True)
        rep = R.build_replay_report(raw_dir=str(tmp_path / "data" / "raw"),
                                    frozen_glob=FROZEN_GLOB_LOCAL)
        assert rep["conclusion"] == "snapshot_diverged"
        assert rep["n_diverged"] == 1

    def test_no_forbidden_signal_fields(self, tmp_path):
        _synthetic_pairs(tmp_path)
        rep = R.build_replay_report(raw_dir=str(tmp_path / "data" / "raw"),
                                    frozen_glob=FROZEN_GLOB_LOCAL)
        blob = json.dumps(rep, ensure_ascii=False)
        for field in FORBIDDEN_FIELDS:
            assert f'"{field}"' not in blob, field


# ----------------------------------------------------------------------
# 真实仓库形态：接入前可复现性（只读，不写盘）
# ----------------------------------------------------------------------
class TestRepoSnapshotShape:
    def test_repo_pairs_are_replayable_or_unverifiable(self):
        """本仓库现存快照对账：只允许 `replayable*` / `unverifiable`，绝不允许 diverged。"""
        rep = R.build_replay_report()
        assert rep["conclusion"] in ("replayable", "replayable_with_label_drift",
                                     "unverifiable"), rep["conclusion"]
        assert rep["n_diverged"] == 0, "缓存与冻结快照出现价格分叉，对照不可复现"

    def test_frozen_glob_is_declared_and_offline(self):
        """冻结快照必须已入库（CI 不触网），路径可枚举。"""
        assert rep_glob_has_files(), "冻结快照缺失：回放对账无法离线进行"


def rep_glob_has_files() -> bool:
    return len(glob.glob(str(PROJECT_ROOT / R.FROZEN_GLOB))) > 0


# ----------------------------------------------------------------------
# 排期 / 顺序纪律
# ----------------------------------------------------------------------
class TestScheduleOrdering:
    def _stage(self):
        return next(s for s in json.loads(PLAN_PATH.read_text(encoding="utf-8"))["stages"]
                    if s["id"] == "S26")

    def test_t26_6_registered_and_delivered(self):
        s = self._stage()
        assert "T26.6" in s["auto_acceptable"]
        t = next(x for x in s["tasks"] if x["id"] == "T26.6")
        assert t["status"] == "completed"
        assert t.get("completed_at") and t.get("result")

    def test_replay_precedes_admission_checkpoint(self):
        """顺序纪律：回放对账（T26.6）必须在准入签字（T26.1）之前交付。

        防止「先批 1.7GB 重型依赖、再发现对照不可复现」——那时依赖已被白批。
        """
        s = self._stage()
        ids = [t["id"] for t in s["tasks"]]
        assert ids.index("T26.6") < ids.index("T26.1") or "T26.6" in s["auto_acceptable"]

    def test_manual_checkpoints_stay_pending(self):
        s = self._stage()
        for tid in s["manual_checkpoint"]:
            t = next(x for x in s["tasks"] if x["id"] == tid)
            assert t["status"] == "pending", f"{tid} 不应为 {t['status']}"
            assert t["auto_run"] is False

    def test_replay_evidence_registered_optional(self):
        m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        optional = set(m.get("known_optional_evidence") or [])
        assert "reports/laya_replay.json" in optional

    def test_conclusion_doc_mentions_replay(self):
        text = DOC_PATH.read_text(encoding="utf-8")
        assert "T26.6" in text
        assert "laya-replay" in text
        assert "affects_gate" in text
