"""非有限值（NaN / ±inf）与越界值**不得**污染信号 → 风控 → 下单链路。

背景
----
前一（代码质量加固）轮修掉了 `ic.py` 里「只用 `value != value` 认 NaN、认不出
`inf`」的缺陷。本轮把同一类缺陷沿**下游链路**继续查了一遍：

1. `signal._direction_value` 直接把 `probability` 代入 `(proba - 0.5) * 2.0`，
   `prob |> 1` 会放大出越界方向分，`probability = -inf` 会算出 `score = -inf`
   并被 `score <= sell_threshold` 判成 **SELL** ——
   即「一个坏掉的概率把看涨预测悄悄翻成做空单」。
2. `risk._sizing_fraction` 的 `min(strength, 1.0)` 只有上界：
   `strength = inf` 会被夹到 1.0（= 满档仓位），`NaN` 则让比较全部为 False。
3. `predictor.load_models` 的 `_instantiate` 闭包捕获循环变量 `horizon_days`，
   一旦被延迟调用，三个周期会统一拿到循环结束后的最后一个值。

三条修复都遵循同一纪律：**非有限值不是"极端信号"，而是数据已损坏**，
必须 fail-safe（退化为中性 / 显式 clamp / 显式绑定），并留痕。

每条断言都做过双向验证：回退实现 → 断言失败 → 恢复 → 断言通过。
"""
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.risk import RiskManager
from src.trading.signal import BUY, HOLD, SELL, Signal, SignalEngine

NONFINITE = [float("inf"), float("-inf"), float("nan")]


def _predictions(prob, prediction=1, confidence=0.9):
    one = {"prediction": prediction, "probability": prob, "confidence": confidence}
    return {"predictions": {h: dict(one) for h in ("short_term", "mid_term", "long_term")}}


def _cfg(capital=1_000_000, method="fixed_fractional"):
    return {"trading": {"risk": {"capital": capital, "sizing_method": method}}}


class TestSignalProbabilityContract:
    """概率契约：越界 / 非有限值不得放大方向，更不得反转方向。"""

    def test_score_never_exceeds_unit_interval_on_overflow(self):
        eng = SignalEngine()
        for prob in (10.0, 100.0, -5.0, 2.0):
            sig = eng.build_signal("X", _predictions(prob))
            assert math.isfinite(sig.score), f"prob={prob} 产出非有限 score"
            assert -1.0 <= sig.score <= 1.0, f"prob={prob} 产出越界 score={sig.score}"
            assert -1.0 <= sig.strength <= 1.0

    def test_inf_does_not_flip_direction(self):
        """核心回归：probability = -inf 曾被算成 score = -inf → SELL。"""
        eng = SignalEngine()
        sig = eng.build_signal("X", _predictions(float("-inf"), prediction=1))
        assert sig.score == 0.0, "非有限概率必须退化为中性方向分"
        assert sig.action == HOLD, "坏掉的概率不得把看涨预测翻成 SELL"
        assert sig.direction_consensus == "中性"

    def test_inf_positive_does_not_saturate_to_buy(self):
        eng = SignalEngine()
        sig = eng.build_signal("X", _predictions(float("inf"), prediction=1))
        assert sig.action == HOLD and sig.score == 0.0

    def test_nan_probability_is_neutral_hold(self):
        eng = SignalEngine()
        sig = eng.build_signal("X", _predictions(float("nan")))
        assert sig.action == HOLD and sig.score == 0.0

    def test_non_numeric_probability_does_not_raise(self):
        eng = SignalEngine()
        for bad in ("bad", None, object()):
            sig = eng.build_signal("X", _predictions(bad))
            assert sig.action == HOLD and sig.score == 0.0

    def test_valid_probability_semantics_unchanged(self):
        """回归保护：合法概率的方向语义与修复前一致。"""
        eng = SignalEngine()
        bullish = eng.build_signal("X", _predictions(0.9, prediction=1))
        assert bullish.action == BUY and bullish.score > 0
        bearish = eng.build_signal("X", _predictions(0.9, prediction=0))
        assert bearish.action == SELL and bearish.score < 0

    def test_boundary_probabilities_are_accepted(self):
        eng = SignalEngine()
        for prob in (0.0, 0.5, 1.0):
            sig = eng.build_signal("X", _predictions(prob))
            assert -1.0 <= sig.score <= 1.0

    def test_confidence_nonfinite_is_contained(self):
        eng = SignalEngine()
        for conf in NONFINITE:
            sig = eng.build_signal("X", _predictions(0.9, confidence=conf))
            assert math.isfinite(sig.confidence)
            assert 0.0 <= sig.confidence <= 1.0


class TestRiskSizingFiniteness:
    """仓位缩放：非有限值不得被 min() 夹成"满档"或被静默当作强信号。"""

    def _budget(self, strength, confidence, method="fixed_fractional", proba_win=None):
        sig = Signal(symbol="X", action=BUY, score=0.5, strength=strength,
                     confidence=confidence, direction_consensus="看涨")
        return RiskManager(_cfg(method=method)).budget(sig, price=100.0, proba_win=proba_win)

    def test_infinite_strength_is_not_treated_as_max_conviction(self):
        """核心回归：strength = inf 曾被夹到 1.0 → 满档仓位。"""
        strong = self._budget(1.0, 0.9)
        infimum = self._budget(float("inf"), 0.9)
        assert math.isfinite(infimum.position_pct)
        assert infimum.position_pct < strong.position_pct, (
            "非有限 strength 的仓位必须低于合法满档 strength，而不是被夹到上界"
        )

    def test_nan_strength_is_contained(self):
        b = self._budget(float("nan"), 0.9)
        assert math.isfinite(b.position_pct) and b.position_pct >= 0.0

    def test_nonfinite_confidence_is_contained(self):
        for conf in NONFINITE:
            b = self._budget(0.8, conf)
            assert math.isfinite(b.position_pct)
            assert 0.0 <= b.position_pct <= 0.20 + 1e-9

    def test_position_pct_never_exceeds_cap(self):
        for strength in list(NONFINITE) + [0.5, 1.0, 10.0]:
            b = self._budget(strength, 0.9)
            assert b.position_pct <= 0.20 + 1e-9

    def test_kelly_nonfinite_win_rate_is_contained(self):
        for pw in NONFINITE:
            b = self._budget(0.5, 0.9, method="kelly", proba_win=pw)
            assert math.isfinite(b.position_pct)
            assert 0.0 <= b.position_pct <= 0.25 + 1e-9

    def test_hold_produces_no_position(self):
        sig = Signal(symbol="X", action=HOLD, score=0.0, strength=0.0,
                     confidence=0.0, direction_consensus="中性")
        b = RiskManager(_cfg()).budget(sig, price=100.0)
        assert b.position_pct == 0.0 and b.suggested_qty is None


class TestPredictorClosureBinding:
    """`_instantiate` 必须按值绑定当前周期的 horizon_days。"""

    def _engine(self):
        import src.inference.predictor as P
        from src.inference.predictor import PredictionEngine

        engine = PredictionEngine.__new__(PredictionEngine)
        engine.config = {"model": {}}
        engine.horizons = {"short_term": 5, "mid_term": 10, "long_term": 20}
        engine.models = {}
        engine.save_dir = Path("models")
        return P, engine

    def test_deferred_instantiate_keeps_each_horizon(self, monkeypatch):
        """核心回归：闭包捕获循环变量 → 三个周期统一拿到 20。"""
        captured = []

        class FakePredictor:
            def __init__(self, horizon_days=None, context_days=None, verbose=None):
                captured.append(horizon_days)
                self.horizon_days = horizon_days

        P, engine = self._engine()
        monkeypatch.setattr(P, "TimesFMFinancePredictor", FakePredictor)

        def deferred_load(save_dir, model_key, lgb_file, instantiate_fn):
            # 故意不在循环内调用，模拟「收集回调、循环结束后统一实例化」
            return {"lightgbm": None, "timesfm": instantiate_fn}

        monkeypatch.setattr(P, "load_ensemble_entry", deferred_load)
        engine.load_models("ensemble")

        resolved = {k: v["ensemble"]["timesfm"]().horizon_days for k, v in engine.models.items()}
        assert resolved == {"short_term_5d": 5, "mid_term_10d": 10, "long_term_20d": 20}

    def test_immediate_instantiate_keeps_each_horizon(self, monkeypatch):
        captured = []

        class FakePredictor:
            def __init__(self, horizon_days=None, context_days=None, verbose=None):
                captured.append(horizon_days)
                self.horizon_days = horizon_days

        P, engine = self._engine()
        monkeypatch.setattr(P, "TimesFMFinancePredictor", FakePredictor)
        engine.load_models("timesfm")

        assert captured == [5, 10, 20]
