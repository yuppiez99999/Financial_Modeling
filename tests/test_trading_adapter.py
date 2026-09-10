"""测试量化交易适配层 (src/trading)。"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as main_cli
from src.trading.signal import SignalEngine, BUY, SELL, HOLD
from src.trading.risk import RiskManager
from src.trading.orders import OrderGenerator
from src.trading.adapter import TradingAdapter
from src.trading.backtest import Backtester


def make_bullish_predictions(prob=0.85):
    """构造三周期一致看多的预测结果。"""
    def one(pred):
        return {
            "prediction": 1, "probability": prob,
            "confidence": prob, "direction": "看涨",
            "latest_close": 120.0,
        }
    return {
        "symbol": "600519.SH",
        "predictions": {
            "short_term": one(prob),
            "mid_term": one(prob),
            "long_term": one(prob),
        },
    }


def make_bearish_predictions(prob=0.8):
    def one(pred):
        return {
            "prediction": 0, "probability": prob,
            "confidence": prob, "direction": "看跌",
            "latest_close": 120.0,
        }
    return {
        "symbol": "600519.SH",
        "predictions": {
            "short_term": one(prob),
            "mid_term": one(prob),
            "long_term": one(prob),
        },
    }


class FakePredictor:
    """替身预测器：直接返回预置预测结果。"""
    def __init__(self, result):
        self._result = result

    def predict_all_horizons(self, symbol):
        r = dict(self._result)
        r["symbol"] = symbol
        return r


def _cfg():
    return main_cli.load_config()


def test_signal_engine_bullish():
    cfg = _cfg()
    sig = SignalEngine(cfg).build_signal("600519.SH", make_bullish_predictions())
    assert sig.action == BUY
    assert sig.score > 0
    assert sig.confidence >= 0.8


def test_signal_engine_bearish_hold_on_no_short():
    cfg = _cfg()
    # 关闭做空
    cfg["trading"]["signal"]["allow_short"] = False
    sig = SignalEngine(cfg).build_signal("600519.SH", make_bearish_predictions())
    assert sig.action == HOLD  # 不允许做空，跌势观望


def test_signal_engine_bearish_sell_when_short_allowed():
    cfg = _cfg()
    sig = SignalEngine(cfg).build_signal("600519.SH", make_bearish_predictions())
    assert sig.action == SELL
    assert sig.score < 0


def test_risk_manager_budget_buy():
    cfg = _cfg()
    sig = SignalEngine(cfg).build_signal("600519.SH", make_bullish_predictions())
    budget = RiskManager(cfg).budget(sig, price=120.0)
    assert budget.action == "BUY"
    assert budget.position_amount > 0
    assert budget.suggested_qty is not None and budget.suggested_qty > 0
    assert budget.stop_price < 120.0 < budget.take_price


def test_risk_manager_hold_zero():
    cfg = _cfg()
    sig = SignalEngine(cfg).build_signal("600519.SH", {
        "predictions": {
            "short_term": {"prediction": 1, "probability": 0.51, "confidence": 0.51},
            "mid_term": {"prediction": 0, "probability": 0.51, "confidence": 0.51},
            "long_term": {"prediction": 1, "probability": 0.5, "confidence": 0.5},
        }
    })
    budget = RiskManager(cfg).budget(sig, price=120.0)
    assert budget.action == "HOLD"
    assert budget.position_amount == 0.0


def test_order_generator_buy_with_sl_tp():
    cfg = _cfg()
    sig = SignalEngine(cfg).build_signal("600519.SH", make_bullish_predictions())
    budget = RiskManager(cfg).budget(sig, price=120.0)
    orders = OrderGenerator(cfg).generate(budget)
    # 主单 + 止损 + 止盈
    assert len(orders) >= 1
    kinds = [o.meta.get("kind") for o in orders]
    assert "stop_loss" in kinds and "take_profit" in kinds


def test_adapter_end_to_end_bullish():
    cfg = _cfg()
    result = make_bullish_predictions()
    adapter = TradingAdapter(cfg, predictor=FakePredictor(result))
    out = adapter.process_symbol("600519.SH")
    assert out["signal"]["action"] == "BUY"
    assert out["orders"], "应生成订单"
    assert out["risk_budget"]["suggested_qty"] > 0


def test_adapter_export_feed(tmp_path):
    cfg = _cfg()
    adapter = TradingAdapter(cfg, predictor=FakePredictor(make_bullish_predictions()))
    results = adapter.run(["600519.SH"])
    p = adapter.export_feed(results, tmp_path / "feed.json")
    assert p.exists()
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert len(data["items"]) == 1


def test_backtester_winning_trend():
    cfg = _cfg()
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    signals = [{"action": "HOLD", "horizon": 1} for _ in closes]
    signals[0] = {"action": "BUY", "horizon": 5}
    result = Backtester(cfg).run(closes, signals)
    assert result.n_trades == 1
    assert result.win_rate == 1.0
    assert result.total_return > 0


def test_backtest_no_trades():
    cfg = _cfg()
    closes = [100, 101, 102]
    signals = [{"action": "HOLD"} for _ in closes]
    result = Backtester(cfg).run(closes, signals)
    assert result.n_trades == 0
