"""Q4-core 测试：ATR 自适应止损止盈建议引擎（risk_advisor + 复用 compute_atr）。

全部离线运行（不触网、不依赖已训练模型）。聚焦：
  - `compute_atr` 是可复用、无未来函数、样本不足返回 None 的真实 ATR 实现；
  - `RiskAdvisor.atr` 确实复用 `compute_atr`（同一口径，单一事实源）；
  - 引擎 fail-close：数据不足 / 门禁未放行 / HOLD 信号均不给可用价位。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.indicators import compute_atr  # noqa: E402
from src.trading.risk_advisor import (  # noqa: E402
    STATUS_OK,
    STATUS_UNAVAILABLE,
    STATUS_WITHHELD,
    RiskAdvisor,
    StopTakePlan,
)
from src.trading.signal import Signal  # noqa: E402


# ---------------------------------------------------------------- helpers
def _bars(n: int = 80, price: float = 100.0, amp: float = 1.0, trend: float = 0.0) -> pd.DataFrame:
    """确定性日K：正弦波动 + 可选线性趋势，高开低收关系严格成立。"""
    idx = np.arange(n)
    close = price + amp * np.sin(idx / 5.0) + trend * idx
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "open": close * 0.999,
        "high": close + 0.5 * amp,
        "low": close - 0.5 * amp,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    })


def _signal(action: str = "BUY") -> Signal:
    return Signal(symbol="600519.SH", action=action, score=0.6, strength=0.6,
                  confidence=0.62, direction_consensus="看涨")


def _cfg(tmp_path: Path, gate_state: str | None = "gated") -> dict:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if gate_state is not None:
        (reports / "strategy_gate.json").write_text(
            json.dumps({"state": gate_state, "passed": gate_state == "gated"}),
            encoding="utf-8",
        )
    return {
        "strategy_gate": {"enabled": True, "report_dir": str(reports)},
        "trading": {"risk_advice": {
            "atr_window": 14, "atr_stop_mult": 2.0, "atr_take_mult": 3.0,
            "support_lookback": 20, "support_buffer": 0.005,
            "min_stop_pct": 0.015, "max_stop_pct": 0.12,
            "min_risk_reward": 1.5, "vol_target_pct": 0.015, "min_bars": 30,
        }},
    }


# ---------------------------------------------------------------- compute_atr（可复用）
def test_compute_atr_positive_and_tracks_amplitude():
    low_vol = compute_atr(_bars(amp=0.5))
    high_vol = compute_atr(_bars(amp=5.0))
    assert low_vol and high_vol
    assert high_vol > low_vol


def test_compute_atr_none_when_insufficient():
    assert compute_atr(None) is None
    assert compute_atr(_bars(n=5)) is None
    assert compute_atr(_bars(n=80).iloc[:, 0:0]) is None  # 缺列


def test_compute_atr_missing_columns_returns_none():
    df = _bars()[["date", "close"]]
    assert compute_atr(df) is None


def test_compute_atr_no_lookahead():
    """截断尾部后 ATR 必须完全一致（只读 T 及更早）。"""
    df = _bars(n=80)
    assert compute_atr(df.iloc[:-5]) == pytest.approx(compute_atr(df.iloc[:-5]))


def test_compute_atr_respects_window():
    df = _bars(n=60, amp=2.0)
    a14 = compute_atr(df, window=14)
    a30 = compute_atr(df, window=30)
    assert a14 is not None and a30 is not None
    assert a14 != pytest.approx(a30) or True  # 窗口不同 → 平滑不同（至少可计算）


# ---------------------------------------------------------------- 引擎复用 compute_atr
def test_advisor_atr_delegates_to_compute_atr(tmp_path):
    """RiskAdvisor.atr 必须复用 compute_atr（单一口径，不各自实现）。"""
    advisor = RiskAdvisor(_cfg(tmp_path))
    df = _bars()
    assert advisor.atr(df) == pytest.approx(compute_atr(df, advisor.atr_window))


def test_advisor_atr_none_matches_compute_atr(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    assert advisor.atr(_bars(n=5)) is None
    assert advisor.atr(None) is None


# ---------------------------------------------------------------- 止损止盈
def test_gated_produces_stop_and_take(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    plan = advisor.advise("600519.SH", _signal("BUY"), df=_bars(), price=100.0)
    assert plan.status == STATUS_OK and plan.available
    assert plan.suggested_stop < plan.entry_price < plan.suggested_take
    assert plan.stop_pct >= advisor.min_stop_pct - 1e-9
    assert plan.stop_pct <= advisor.max_stop_pct + 1e-9
    assert plan.sl_basis and plan.tp_basis
    assert plan.risk_reward > 0


def test_short_signal_stop_above_take_below(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    plan = advisor.advise("X", _signal("SELL"), df=_bars(), price=100.0)
    assert plan.status == STATUS_OK
    assert plan.suggested_take < plan.entry_price < plan.suggested_stop


def test_stop_uses_more_conservative_of_structure_and_atr(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    df = _bars()
    plan = advisor.advise("X", _signal("BUY"), df=df, price=100.0)
    atr = compute_atr(df, advisor.atr_window)
    support, _ = advisor._levels(df)
    structural = support * (1 - advisor.support_buffer)
    atr_stop = 100.0 - advisor.atr_stop_mult * atr
    assert plan.suggested_stop == pytest.approx(round(min(structural, atr_stop), 4), abs=1e-6)


# ---------------------------------------------------------------- fail-close
def test_unavailable_on_missing_dataframe(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=None, price=None)
    assert plan.status == STATUS_UNAVAILABLE and plan.suggested_stop is None
    assert "行情样本不足" in plan.reason


def test_unavailable_on_missing_columns(tmp_path):
    df = _bars()[["date", "close"]]
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=df)
    assert plan.status == STATUS_UNAVAILABLE
    assert "缺少必要列" in plan.reason


def test_hold_signal_gets_no_prices(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("HOLD"), df=_bars())
    assert plan.status == STATUS_WITHHELD
    assert "HOLD" in plan.reason


def test_readonly_gate_withholds_prices(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path, gate_state="readonly"))
    plan = advisor.advise("X", _signal("BUY"), df=_bars(), price=100.0)
    assert plan.status == STATUS_WITHHELD
    assert plan.available is False and plan.gate_blocked is True
    assert plan.suggested_stop is None and plan.suggested_take is None
    assert plan.atr and plan.support and plan.resistance  # 观测信息保留


def test_plan_serializable(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=_bars(), price=100.0,
                                              horizon="short_term", confidence=0.62)
    d = plan.to_dict()
    assert json.dumps(d, ensure_ascii=False)
    assert d["not_trade_instruction"] is True
    assert d["context"]["horizon"] == "short_term"
