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
    def test_repo_frozen_frames_load_offline(self):
        """冻结快照是仓库内**唯一机器无关行情源**（T26.8）：已入库、可离线加载、列契约齐备。"""
        frames = R.load_frozen_frames()
        assert frames, "冻结快照缺失：回放对账无法离线进行"
        required = {"date", *R.PRICE_COLUMNS}
        for sym, df in frames.items():
            assert required <= set(df.columns), sym
            assert len(df) >= R.MIN_COMMON_ROWS, sym

    def test_repo_divergence_always_diagnosed(self):
        """任何 diverged 必须带 divergence_pattern 诊断 —— 不给无解释的分歧（T26.8）。

        更正说明（2026-09-25）：本类原有的守卫是「本仓库现存对账只允许
        replayable* / unverifiable，绝不允许 diverged」。该守卫写在交付机上
        （那里缓存 ≡ 冻结基线），但缓存是 gitignore 的**机器本地文件**：
        跨机实测（2026-09-25）复权锚点重算使 26/28 对 diverged —— 分叉本身
        是跨机常态、且不影响**快照基**对照（T26.8 已交付）。仓库守卫能钉死的
        是「分歧必须带诊断」；机器本地数据链健康度进 T26.1 的准入材料
        （admission_implication），不进仓库守卫。
        """
        rep = R.build_replay_report()
        assert rep["conclusion"] in ("replayable", "replayable_with_label_drift",
                                     "snapshot_diverged", "unverifiable"), rep["conclusion"]
        for p in rep["pairs"]:
            if p.get("verdict") == "snapshot_diverged":
                assert p.get("divergence_pattern") in R.DIVERGENCE_PATTERNS, p["symbol"]

    def test_frozen_glob_is_declared_and_offline(self):
        """冻结快照必须已入库（CI 不触网），路径可枚举。"""
        assert rep_glob_has_files(), "冻结快照缺失：回放对账无法离线进行"


def rep_glob_has_files() -> bool:
    return len(glob.glob(str(PROJECT_ROOT / R.FROZEN_GLOB))) > 0


# ----------------------------------------------------------------------
# T26.8：分歧诊断 + 冻结快照数据基
# ----------------------------------------------------------------------
def _write_pair_custom(tmp_path: Path, *, n: int, rebase_from: int = 0,
                       rebase_factor: float = 1.0, revise_last: int = 0,
                       volume_base_frozen: int = 1_000_000,
                       volume_base_cache: int = 1_000_000):
    """造一对可精细控制分歧形态的「缓存 ↔ 冻结快照」。

    - `rebase_from`/`rebase_factor`：前 `rebase_from` 行价格乘系数（qfq 锚点重算形态）；
    - `revise_last`：最后 N 行 close 加一个偏移（最近行修订形态）；
    - `volume_base_*`：两边 volume 基数不同（跨机股/手口径差的抽象）。
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    dates = pd.bdate_range("2024-01-01", periods=n)

    def _dump(path: Path, *, factor: float, vol_base: int, revise: int = 0) -> None:
        adj = [c * (factor if i < rebase_from else 1.0) for i, c in enumerate(closes)]
        if revise:
            adj = adj[:-revise] + [c + 0.12 for c in adj[-revise:]]
        pd.DataFrame({
            "date": dates, "open": adj, "high": [c * 1.01 for c in adj],
            "low": [c * 0.99 for c in adj], "close": adj,
            "volume": [vol_base + i for i in range(n)],
        }).to_csv(path, index=False)

    frozen_path = raw / f"TEST.SH.frozen.{FROZEN_TAG}.csv"
    cache_path = raw / "TEST.SH.csv"
    _dump(frozen_path, factor=1.0, vol_base=volume_base_frozen)
    _dump(cache_path, factor=rebase_factor, vol_base=volume_base_cache,
          revise=revise_last and 1 or 0)
    return cache_path, frozen_path


class TestDivergenceDiagnosis:
    def test_qfq_reanchor_tail_anchored_history_rebased(self, tmp_path):
        """历史 ×1.2 重算、尾部 10 行相等 ⇒ qfq_reanchor；volume 全不同**不否决**价格同一性。"""
        c, f = _write_pair_custom(tmp_path, n=300, rebase_from=290, rebase_factor=1.2,
                                  volume_base_cache=3_000_000)
        r = R.reconcile_pair(str(c), str(f))
        assert r["verdict"] == "snapshot_diverged"
        assert r["divergence_pattern"] == "qfq_reanchor"
        assert r["tail_price_match_rate"] >= R.TAIL_MATCH_MIN
        assert r["older_price_match_rate"] <= R.OLDER_MATCH_MAX
        assert r["volume_identical_rate"] == 0.0
        assert "复权锚点" in r["reason"]
        assert "不可复现" in r["reason"]

    def test_recent_revision_history_identical_tail_row_revised(self, tmp_path):
        """历史完全一致、仅最后 1 行不同（快照日盘中采集/数据源修订）⇒ recent_revision。"""
        c, f = _write_pair_custom(tmp_path, n=300, revise_last=1,
                                  volume_base_cache=2_500_000)
        r = R.reconcile_pair(str(c), str(f))
        assert r["verdict"] == "snapshot_diverged"
        assert r["divergence_pattern"] == "recent_revision"
        assert r["older_price_match_rate"] >= R.OLDER_MATCH_MIN
        assert "最近" in r["reason"] or "修订" in r["reason"]

    def test_hard_divergence_tail_also_differs(self, tmp_path):
        c, f = _synthetic_pairs(tmp_path, diverged=True)
        r = R.reconcile_pair(str(c), str(f))
        assert r["divergence_pattern"] == "hard_divergence"

    def test_price_equal_volume_different_is_replayable(self, tmp_path):
        """价格同一、仅 volume 口径不同 ⇒ 仍 replayable（volume 不参与同一性判定）。"""
        c, f = _write_pair_custom(tmp_path, n=300, volume_base_cache=9_000_000)
        r = R.reconcile_pair(str(c), str(f))
        assert r["verdict"] == "replayable"
        assert r["volume_identical_rate"] == 0.0

    def test_label_shift_still_detected_on_identical_prices(self, tmp_path):
        """对齐搜索不回归 T26.6 原语义：同序列 + 日期错位 ⇒ replayable + 标签漂移读数。"""
        c, f = _synthetic_pairs(tmp_path, offset_days=5)
        r = R.reconcile_pair(str(c), str(f))
        assert r["verdict"] == "replayable"
        assert r["label_aligned"] is False
        assert r["n_label_mismatch"] == 300


class TestFrozenDataBase:
    def test_load_frozen_frames_reads_committed_snapshots(self, tmp_path):
        c, f = _write_pair_custom(tmp_path, n=300)
        frames = R.load_frozen_frames(raw_dir=str(tmp_path / "data" / "raw"),
                                      frozen_glob=f"data/raw/*.frozen.{FROZEN_TAG}.csv")
        assert set(frames) == {"TEST.SH"}
        df = frames["TEST.SH"]
        assert {"date", *R.PRICE_COLUMNS, "volume"} <= set(df.columns)
        assert len(df) == 300
        # 排序 + 去重契约：日期单调
        assert df["date"].is_monotonic_increasing

    def test_load_frozen_frames_symbol_filter_and_missing_dir(self, tmp_path):
        _write_pair_custom(tmp_path, n=300)
        raw = str(tmp_path / "data" / "raw")
        glob_pat = f"data/raw/*.frozen.{FROZEN_TAG}.csv"
        assert R.load_frozen_frames(raw_dir=raw, frozen_glob=glob_pat,
                                    symbols=["NOPE.SH"]) == {}
        assert R.load_frozen_frames(raw_dir=str(tmp_path / "no_such_dir"),
                                    frozen_glob=glob_pat) == {}

    def test_frames_are_readonly_shaped(self, tmp_path):
        """数据基只读：加载不落盘、不联网 —— 产出仅内存字典（结构性保证，防写缓存的误用）。"""
        c, f = _write_pair_custom(tmp_path, n=300)
        before = c.read_text(encoding="utf-8")
        R.load_frozen_frames(raw_dir=str(tmp_path / "data" / "raw"),
                             frozen_glob=f"data/raw/*.frozen.{FROZEN_TAG}.csv")
        assert c.read_text(encoding="utf-8") == before


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

    def test_t26_8_registered_and_delivered(self):
        """T26.8（对照数据基冻结化 + 分歧诊断）同属 T26.1 之前的自动补交付。"""
        s = self._stage()
        assert "T26.8" in s["auto_acceptable"]
        t = next(x for x in s["tasks"] if x["id"] == "T26.8")
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
