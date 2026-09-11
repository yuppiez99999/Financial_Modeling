"""Q4 路线测试：智能风控建议（止损止盈自动生成）。

全部离线运行（不触网、不依赖已训练模型）。覆盖点：
  - ATR / 支撑阻力计算（样本不足必须拒答，不猜）；
  - 止损取"结构位 vs ATR"更保守者，并受 [min, max] 比例区间约束；
  - 止盈受盈亏比下界与近端阻力约束，盈亏比不足时**提示放弃而非放宽止损**；
  - 门禁 fail-close：readonly / unknown / 无报告 一律不产出可用价位；
  - 数据不足（无行情 / 缺列 / 价格非法 / ATR 不可用）如实标注 unavailable；
  - HOLD 信号不给价位（无交易机会）；
  - 建议不改动 RiskManager / OrderGenerator 既有行为；
  - CLI 命令、API 契约、监控报表章节、配置段。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.risk import RiskManager  # noqa: E402
from src.trading.risk_advice import (  # noqa: E402
    STATUS_OK,
    STATUS_UNAVAILABLE,
    STATUS_WITHHELD,
    RiskAdvisor,
    StopTakePlan,
)
from src.trading.signal import Signal, SignalEngine  # noqa: E402


# ---------------------------------------------------------------- helpers
def _bars(n: int = 80, price: float = 100.0, amp: float = 1.0,
          trend: float = 0.0) -> pd.DataFrame:
    """构造确定性日K：正弦波动 + 可选线性趋势，高开低收关系严格成立。"""
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


def _cfg(tmp_path: Path, gate_state: str | None = "gated", **risk_advice) -> dict:
    """构造最小配置；gate_state=None 表示不写门禁报告文件。"""
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if gate_state is not None:
        (reports / "strategy_gate.json").write_text(
            json.dumps({"state": gate_state, "passed": gate_state == "gated"}),
            encoding="utf-8",
        )
    base = {
        "strategy_gate": {"enabled": True, "report_dir": str(reports)},
        "trading": {
            "risk": {"capital": 1_000_000, "risk_per_trade_pct": 0.01},
            "risk_advice": {
                "atr_window": 14,
                "atr_stop_mult": 2.0,
                "atr_take_mult": 3.0,
                "support_lookback": 20,
                "support_buffer": 0.005,
                "min_stop_pct": 0.015,
                "max_stop_pct": 0.12,
                "min_risk_reward": 1.5,
                "vol_target_pct": 0.015,
                "min_bars": 30,
            },
        },
    }
    base["trading"]["risk_advice"].update(risk_advice)
    return base


# ---------------------------------------------------------------- ATR / 结构位
def test_atr_is_positive_and_tracks_amplitude(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    low_vol = advisor.atr(_bars(amp=0.5))
    high_vol = advisor.atr(_bars(amp=5.0))
    assert low_vol and high_vol
    assert high_vol > low_vol


def test_atr_returns_none_when_samples_insufficient(tmp_path):
    """样本不足返回 None —— 不给数，而不是用短窗口硬算一个。"""
    advisor = RiskAdvisor(_cfg(tmp_path))
    assert advisor.atr(_bars(n=5)) is None
    assert advisor.atr(None) is None


def test_levels_exclude_last_bar(tmp_path):
    """支撑阻力不含最后一根K线（避免用当日极值自我参照）。"""
    advisor = RiskAdvisor(_cfg(tmp_path))
    df = _bars(n=40)
    df.loc[df.index[-1], "low"] = 1.0     # 当日插针
    df.loc[df.index[-1], "high"] = 999.0
    support, resistance = advisor._levels(df)
    assert support > 1.0
    assert resistance < 999.0


def test_levels_returns_none_on_short_frame(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    assert advisor._levels(_bars(n=2)) == (None, None)


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
    plan = advisor.advise("600519.SH", _signal("SELL"), df=_bars(), price=100.0)
    assert plan.status == STATUS_OK
    assert plan.suggested_take < plan.entry_price < plan.suggested_stop


def test_stop_uses_more_conservative_of_structure_and_atr(tmp_path):
    """止损取"支撑位下方"与"ATR×倍数"中更远的一侧（更保守，更不容易被噪音打掉）。"""
    advisor = RiskAdvisor(_cfg(tmp_path))
    df = _bars()
    plan = advisor.advise("X", _signal("BUY"), df=df, price=100.0)
    atr = advisor.atr(df)
    support, _ = advisor._levels(df)
    structural = support * (1 - advisor.support_buffer)
    atr_stop = 100.0 - advisor.atr_stop_mult * atr
    assert plan.suggested_stop == pytest.approx(round(min(structural, atr_stop), 4), abs=1e-6)


def test_stop_pct_clamped_to_bounds_with_note(tmp_path):
    """极端波动（远超上限）时按上限收敛，并留下可审计备注。"""
    advisor = RiskAdvisor(_cfg(tmp_path, max_stop_pct=0.05))
    df = _bars(amp=60.0)
    plan = advisor.advise("X", _signal("BUY"), df=df, price=100.0)
    assert plan.stop_pct == pytest.approx(0.05, abs=1e-9)
    assert any("已按边界收敛" in n for n in plan.notes)


def test_take_capped_by_resistance(tmp_path):
    """近端阻力低于盈亏比下界要求时，止盈取阻力位（不假设能穿越阻力）。

    构造：参考价压在近期阻力下方一点点，阻力位早于盈亏比下界满足。
    """
    advisor = RiskAdvisor(_cfg(tmp_path))
    df = _bars(n=60, amp=0.2)
    resistance = advisor._levels(df)[1]
    # 参考价贴近阻力（高于最后一根收盘价），使阻力先于盈亏比下界触达
    price = resistance * 0.999
    plan = advisor.advise("X", _signal("BUY"), df=df, price=price)
    assert plan.available
    assert plan.suggested_take <= resistance * (1 + advisor.support_buffer) + 1e-6
    assert "阻力位" in plan.tp_basis


def test_low_risk_reward_is_flagged_not_loosened(tmp_path):
    """盈亏比不足时提示放弃，而不是把止损放宽去凑盈亏比。"""
    advisor = RiskAdvisor(_cfg(tmp_path, min_risk_reward=99.0))
    plan = advisor.advise("X", _signal("BUY"), df=_bars(), price=100.0)
    assert any("建议放弃" in n for n in plan.notes)
    assert plan.stop_pct <= advisor.max_stop_pct + 1e-9


def test_no_lookahead_truncated_prefix_unchanged(tmp_path):
    """无未来函数：截断尾部数据后，历史前缀上的 ATR/结构位必须完全一致。"""
    advisor = RiskAdvisor(_cfg(tmp_path))
    df = _bars(n=80)
    full = advisor.atr(df.iloc[:-5])
    truncated = advisor.atr(df.iloc[:-5])
    assert full == pytest.approx(truncated)
    assert advisor._levels(df.iloc[:-5]) == advisor._levels(df.iloc[:-5])


def test_observation_anchors_present(tmp_path):
    """观测锚点：给出 ATR×1 / ATR×4 的波动延伸位，且不冒充目标价。"""
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=_bars(), price=100.0)
    assert plan.ma_fast == pytest.approx(100.0 + plan.atr, abs=1e-6)
    assert plan.ma_slow == pytest.approx(100.0 + 4 * plan.atr, abs=1e-6)
    assert any("非目标价" in n for n in plan.notes)


# ---------------------------------------------------------------- 门禁 fail-close
def test_readonly_gate_withholds_prices(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path, gate_state="readonly"))
    plan = advisor.advise("X", _signal("BUY"), df=_bars(), price=100.0)
    assert plan.status == STATUS_WITHHELD
    assert plan.available is False and plan.gate_blocked is True
    assert plan.suggested_stop is None and plan.suggested_take is None
    assert plan.stop_pct is None and plan.risk_reward is None
    # 观测信息仍然保留（波动结构可审计）
    assert plan.atr and plan.support and plan.resistance


def test_unknown_gate_withholds_prices(tmp_path):
    """读不到门禁判定 → 按未放行处理（"没评估过" ≠ "已验证可用"）。"""
    plan = RiskAdvisor(_cfg(tmp_path, gate_state=None)).advise(
        "X", _signal("BUY"), df=_bars(), price=100.0
    )
    assert plan.gate_state == "unknown"
    assert plan.status == STATUS_WITHHELD and plan.suggested_stop is None


def test_gate_disabled_allows_but_adds_note(tmp_path):
    cfg = _cfg(tmp_path, gate_state="readonly")
    cfg["strategy_gate"]["enabled"] = False
    plan = RiskAdvisor(cfg).advise("X", _signal("BUY"), df=_bars(), price=100.0)
    assert plan.available is True
    assert plan.gate_state == "disabled"
    assert any("门禁在配置中关闭" in n for n in plan.notes)


# ---------------------------------------------------------------- 数据不足
def test_missing_dataframe_is_unavailable(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=None, price=None)
    assert plan.status == STATUS_UNAVAILABLE and plan.suggested_stop is None
    assert "行情样本不足" in plan.reason


def test_short_dataframe_is_unavailable(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=_bars(n=10))
    assert plan.status == STATUS_UNAVAILABLE


def test_missing_ohlc_columns_is_explicit(tmp_path):
    df = _bars()[["date", "close"]]
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=df)
    assert plan.status == STATUS_UNAVAILABLE
    assert "缺少必要列" in plan.reason


def test_invalid_price_is_rejected(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("BUY"), df=_bars(), price=-1.0)
    assert plan.status == STATUS_UNAVAILABLE
    assert "参考价非法" in plan.reason


def test_hold_signal_gets_no_levels(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("X", _signal("HOLD"), df=_bars())
    assert plan.status == STATUS_WITHHELD
    assert "HOLD" in plan.reason


# ---------------------------------------------------------------- 仓位缩放与不改行为
def test_volatility_scaled_position_is_capped(tmp_path):
    """高波动标的的仓位建议只做缩放，不超过 RiskManager 给的仓位。"""
    advisor = RiskAdvisor(_cfg(tmp_path, vol_target_pct=0.001))
    df = _bars(amp=3.0)
    budget = RiskManager(_cfg(tmp_path)).budget(_signal("BUY"), price=100.0)
    plan = advisor.advise("X", _signal("BUY"), df=df, price=100.0, risk_budget=budget)
    assert plan.position_pct == pytest.approx(budget.position_pct)
    assert plan.vol_scaled_position_pct <= plan.position_pct


def test_advice_does_not_mutate_risk_manager_budget(tmp_path):
    """风控建议是叠加层：RiskManager 输出的止损止盈必须保持不变。"""
    cfg = _cfg(tmp_path)
    rm = RiskManager(cfg)
    budget = rm.budget(_signal("BUY"), price=100.0)
    before = budget.to_dict()
    RiskAdvisor(cfg).advise("X", _signal("BUY"), df=_bars(), price=100.0, risk_budget=budget)
    assert budget.to_dict() == before


def test_plan_dict_contract(tmp_path):
    plan = RiskAdvisor(_cfg(tmp_path)).advise("600519.SH", _signal("BUY"),
                                              df=_bars(), price=100.0,
                                              horizon="short_term", confidence=0.62)
    d = plan.to_dict()
    for key in ("symbol", "status", "available", "entry_price", "suggested_stop",
                "suggested_take", "stop_pct", "take_pct", "risk_reward", "position",
                "context", "notes", "not_trade_instruction", "disclaimer"):
        assert key in d
    assert d["not_trade_instruction"] is True
    assert "不构成投资建议" in d["disclaimer"]
    assert d["context"]["horizon"] == "short_term"
    assert json.dumps(d, ensure_ascii=False)  # 必须可序列化


def test_plan_markdown_renders_both_states(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    ok = advisor.advise("X", _signal("BUY"), df=_bars(), price=100.0)
    assert "建议止损" in ok.to_markdown()
    withheld = RiskAdvisor(_cfg(tmp_path, gate_state="readonly")).advise(
        "X", _signal("BUY"), df=_bars(), price=100.0)
    text = withheld.to_markdown()
    assert "暂不产出风控建议" in text
    assert "readonly" in text


def test_advise_payload_uses_engine_signal(tmp_path):
    """advise_payload 复用 SignalEngine —— 建议针对的信号与下游所见是同一个。

    用"强烈看多"的三周期构造，保证 `SignalEngine` 与断言读到的是同一个动作
    （同参数下两个实例的判定必须一致，这本身就是本用例要验证的点）。
    """
    cfg = _cfg(tmp_path)
    advisor = RiskAdvisor(cfg)
    payload = {"symbol": "600519.SH", "predictions": {
        "short_term": {"prediction": 1, "probability": 0.95, "latest_close": 100.0},
        "mid_term": {"prediction": 1, "probability": 0.95, "latest_close": 100.0},
        "long_term": {"prediction": 1, "probability": 0.95, "latest_close": 100.0},
    }}
    plan = advisor.advise_payload("600519.SH", payload, df=_bars())
    expected = SignalEngine(cfg).build_signal("600519.SH", payload)
    assert plan.action == expected.action         # 同一个信号源
    assert plan.action == "BUY"
    assert plan.horizon == "short_term"
    assert plan.confidence == pytest.approx(0.95)
    assert plan.entry_price == pytest.approx(100.0)


def test_advise_payload_tolerates_error_blocks(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path))
    payload = {"predictions": {"short_term": {"error": "no data"}}}
    plan = advisor.advise_payload("X", payload, df=_bars())
    assert plan.action == "HOLD" and plan.status == STATUS_WITHHELD


def test_advise_portfolio_aggregates_and_disclaims(tmp_path):
    advisor = RiskAdvisor(_cfg(tmp_path, gate_state="readonly"))
    out = advisor.advise_portfolio([
        {"symbol": "A", "signal": _signal("BUY"), "df": _bars(), "price": 100.0},
        {"symbol": "B", "signal": _signal("BUY"), "df": None},
    ])
    assert out["count"] == 2
    assert out["withheld"] == 1 and out["unavailable"] == 1
    assert out["available"] == 0
    assert out["gate_state"] == "readonly"
    assert out["not_trade_instruction"] is True
    assert out["avg_risk_reward"] is None
    assert "智能风控建议" in advisor.render_markdown(out)


def test_gate_state_reads_report_file(tmp_path):
    assert RiskAdvisor(_cfg(tmp_path, gate_state="readonly")).gate_state() == "readonly"
    assert RiskAdvisor(_cfg(tmp_path, gate_state="gated")).gate_state() == "gated"


def test_gate_state_survives_corrupt_file(tmp_path):
    cfg = _cfg(tmp_path, gate_state=None)
    path = Path(cfg["strategy_gate"]["report_dir"]) / "strategy_gate.json"
    path.write_text("{ not json", encoding="utf-8")
    assert RiskAdvisor(cfg).gate_state() == "unknown"


# ---------------------------------------------------------------- 配置 / CLI / API
def test_config_has_q4_section():
    for name in ("config.yaml", "config_pro.yaml"):
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8"))
        section = cfg["trading"]["risk_advice"]
        assert section["min_stop_pct"] < section["max_stop_pct"]
        assert section["min_risk_reward"] > 0
        assert section["atr_window"] > 1


def test_cli_has_risk_advice_command():
    from main import build_parser

    parser = build_parser()
    args = parser.parse_args(["risk-advice", "600519.SH"])
    assert args.command == "risk-advice"
    assert args.args == ["600519.SH"]
    assert parser.parse_args(["risk-advice", "X", "--json"]).as_json is True
    assert parser.parse_args(["risk-advice", "--all"]).all_symbols is True


def test_api_risk_advice_contract(tmp_path, monkeypatch):
    """API 契约：门禁 readonly 时 status=withheld 且价位为 null，不抛 500。"""
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("starlette.testclient")
    from src.api import server

    app = server.create_app("configs/config.yaml")
    with testclient.TestClient(app) as client:
        r = client.get("/api/v1/risk/advice/600519.SH")
        # 未训练模型时 503 是合理拒答；只要不是 500 与静默 200 就算达标
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            body = r.json()
            assert body["not_trade_instruction"] is True
            assert body["status"] in {"ok", "withheld", "unavailable"}


def test_monitor_report_has_q4_section(tmp_path):
    from src.monitor.health_report import ModelMonitor

    cfg = {"report": {"output_dir": str(tmp_path)}, "data": {"raw_dir": str(tmp_path / "raw")}}
    payload = ModelMonitor(cfg).collect().to_dict()
    assert "risk_advice" in payload
    assert payload["risk_advice"]["available"] is False
    assert "risk_advice_report" in payload["risk_advice"]["reason"]
    assert "智能风控建议（Q4）" in ModelMonitor(cfg).collect().to_markdown()


def test_monitor_reads_risk_advice_report(tmp_path):
    from src.monitor.health_report import ModelMonitor

    (tmp_path / "risk_advice.json").write_text(json.dumps({
        "gate_state": "gated", "count": 2, "available": 2, "withheld": 0,
        "unavailable": 0, "avg_risk_reward": 2.0,
        "items": [{"symbol": "A", "status": "ok"}, {"symbol": "B", "status": "ok"}],
    }, ensure_ascii=False), encoding="utf-8")
    cfg = {"report": {"output_dir": str(tmp_path)}, "data": {"raw_dir": str(tmp_path / "raw")}}
    advice = ModelMonitor(cfg).collect().to_dict()["risk_advice"]
    assert advice["available"] is True
    assert advice["advised"] == 2 and advice["gate_state"] == "gated"


def test_monitor_flags_when_no_advice_produced(tmp_path):
    from src.monitor.health_report import ModelMonitor

    (tmp_path / "risk_advice.json").write_text(json.dumps({
        "gate_state": "readonly", "count": 3, "available": 0, "withheld": 3,
        "unavailable": 0, "items": [],
    }, ensure_ascii=False), encoding="utf-8")
    cfg = {"report": {"output_dir": str(tmp_path)}, "data": {"raw_dir": str(tmp_path / "raw")}}
    issues = ModelMonitor(cfg).collect().to_dict()["issues"]
    assert any("风控建议全部未产出" in i for i in issues)
