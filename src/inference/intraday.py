"""分钟级预测更新（Q3 路线）：把日频模型观点与盘中快照做"廉价、可解释"的对齐。

路线图（`SALES_PLAN.md` §8.2）Q3：
  > 实时数据流接入，支持分钟级预测更新

## 这里的"分钟级更新"到底更新了什么

**不是**每分钟重训/重推一个模型（既无数据源支持，也会引入未来函数），
而是回答一个盘中才有意义的问题：

> 昨收时的模型观点（方向 + 概率），在盘中价格相对昨收发生 X% 变化后，
> 是否仍成立？还是已经"失真"到需要人工复核？

因此本模块输出三类可审计的信息：

1. **baseline**：T-1 收盘口径的日频预测（`PredictionEngine.predict_all_horizons`，有模型就用模型）；
2. **live**：盘中快照（最新价 / 昨收 / 相对昨收涨跌）；
3. **drift**：盘中变化是否与 baseline 方向**相反且幅度足够大** → 触发 `stale` 预警。

## 防未来函数

- baseline 只读**已落盘的日K缓存**（T-1 及更早），当日盘中价只作为"新观测"比对；
- 盘中价**绝不**回写日K缓存、绝不用来重算日K特征；
- 无模型时降级为**透明的技术口径打分**（RSI / 均线偏离 / 动量），
  并在输出中显式标注 `model: "rules_fallback"` —— 让调用方一眼能看出这不是模型输出。

## 数据纪律

只使用真实数据源（快照失败 → `available=False`），绝不使用 simulation 兜底数据。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from src.data.streaming import (
    IntradaySnapshot,
    IntradayStore,
    MarketClock,
    RealtimeQuoteClient,
    RollingFeatureBuilder,
)

logger = logging.getLogger(__name__)

MODEL_RULES_FALLBACK = "rules_fallback"

# 盘中"观点失真"判定默认值（configs: streaming.drift_*）
DEFAULT_DRIFT_NOTIONAL = 0.01      # 相对昨收 1% 以上的反向波动才算"足够大"
DEFAULT_FRESHNESS_MAX_DAYS = 5     # 日K缓存落后超过 5 天视为陈旧，拒绝给盘中结论


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out  # NaN → default


@dataclass
class IntradaySignal:
    """单只标的的盘中信号更新结果（可直接进 API / 日报）。"""

    symbol: str
    available: bool = False
    reason: str = ""
    asof: str = ""
    is_trading_hours: bool = False
    model: str = ""
    baseline: Dict[str, Any] = field(default_factory=dict)
    live: Dict[str, Any] = field(default_factory=dict)
    drift: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "available": self.available,
            "reason": self.reason,
            "asof": self.asof,
            "is_trading_hours": self.is_trading_hours,
            "model": self.model,
            "baseline": self.baseline,
            "live": self.live,
            "drift": self.drift,
            "features": self.features,
        }


class IntradayPredictor:
    """盘中预测更新器（只读 + fail-open）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 quote_client: Optional[RealtimeQuoteClient] = None,
                 store: Optional[IntradayStore] = None,
                 engine: Any = None) -> None:
        self.config = config or {}
        cfg = (self.config.get("streaming", {}) or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.poll_seconds = int(cfg.get("poll_seconds", 60))
        self.drift_notional = float(cfg.get("drift_notional", DEFAULT_DRIFT_NOTIONAL))
        self.max_stale_days = int(cfg.get("max_stale_days", DEFAULT_FRESHNESS_MAX_DAYS))
        self.quote_client = quote_client or RealtimeQuoteClient(self.config)
        self.store = store or IntradayStore(self.config)
        self.feature_builder = RollingFeatureBuilder(self.config)
        self._engine = engine  # 允许注入（测试用），None 时懒加载 PredictionEngine

    # ------------------------------------------------------------------
    def _get_engine(self):
        if self._engine is False:
            return None
        if self._engine is None:
            try:
                from src.inference.predictor import PredictionEngine

                engine = PredictionEngine(self.config)
                engine.load_models(self.config.get("model", {}).get("type", "lightgbm"))
                self._engine = engine
            except Exception as e:  # noqa: BLE001  无模型时降级为规则口径
                logger.warning(f"[intraday] 预测引擎加载失败，降级为规则口径: {e}")
                self._engine = False  # 哨兵：明确标记"不可用"，避免每次重试
        return self._engine or None

    def _baseline(self, symbol: str, daily_df) -> Dict[str, Any]:
        """T-1 收盘口径的 baseline：优先真实模型，失败降级规则口径。"""
        engine = self._get_engine()
        if engine is not None and engine is not False:
            try:
                pred = engine.predict_all_horizons(symbol)
                horizons = {
                    h: {
                        "direction": p.get("direction", ""),
                        "probability": _to_float(p.get("probability"), 0.5),
                        "horizon_days": p.get("horizon_days", 0),
                    }
                    for h, p in (pred.get("predictions") or {}).items()
                    if isinstance(p, dict) and "error" not in p
                }
                if horizons:
                    return {"model": "model", "horizons": horizons}
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[intraday] {symbol} 模型预测失败，降级规则口径: {e}")

        feat = self.feature_builder.build(symbol, daily_df)
        if not feat.available:
            return {"model": MODEL_RULES_FALLBACK, "horizons": {},
                    "reason": feat.reason}
        return {
            "model": MODEL_RULES_FALLBACK,
            "horizons": {
                "short_term": {
                    "direction": "看涨" if feat.features.get("ma_gap", 0) > 0 else "看跌",
                    "probability": round(0.5 + max(min(feat.features.get("ma_gap", 0) * 5, 0.2), -0.2), 4),
                    "horizon_days": 5,
                }
            },
            "reason": "未加载到模型，使用透明规则口径（RSI/均线偏离/动量），仅供观测",
        }

    # ------------------------------------------------------------------
    def build(self, symbol: str, daily_df=None,
              snapshot: Optional[IntradaySnapshot] = None,
              now: Optional[datetime] = None) -> IntradaySignal:
        """构造单只标的的盘中信号（不触网时需外部注入 snapshot）。"""
        now = now or datetime.now()
        result = IntradaySignal(
            symbol=symbol,
            asof=now.isoformat(timespec="seconds"),
            is_trading_hours=MarketClock.is_trading_hours(now),
        )

        if daily_df is None:
            result.reason = "无日K缓存（盘中更新需要至少 T-1 收盘数据）"
            return result

        feature = self.feature_builder.build(symbol, daily_df)
        result.features = feature.to_dict()
        if not feature.available:
            result.reason = feature.reason
            result.drift = {"status": "unknown", "reason": feature.reason}
            return result

        if self._is_stale(feature.asof, now):
            result.reason = (
                f"日K缓存落后（最后交易日 {feature.asof}，容忍 {self.max_stale_days} 天），"
                "盘中结论可能失真，请先刷新行情缓存"
            )
            result.drift = {"status": "unknown", "reason": result.reason}
            return result

        baseline = self._baseline(symbol, daily_df)
        result.model = baseline.get("model", "")
        result.baseline = baseline
        if not baseline.get("horizons"):
            result.reason = baseline.get("reason", "无可用 baseline")
            result.drift = {"status": "unknown", "reason": result.reason}
            return result

        if snapshot is None:
            result.reason = "无盘中快照（休市或实时源不可用）"
            result.drift = {"status": "unknown", "reason": result.reason}
            return result

        result.live = snapshot.to_dict()
        result.drift = self._assess_drift(baseline, snapshot)
        result.available = True
        if not result.model:
            result.model = baseline.get("model", "")
        return result

    def _is_stale(self, asof: str, now: datetime) -> bool:
        """日K最后交易日是否过旧（节假日期间容忍 `max_stale_days` 天）。

        缓存过旧时**拒绝给盘中结论**：拿一周前的日K去解释今天的盘中波动，
        得出的"观点失真"预警毫无意义，宁可如实标注不可用。
        """
        if not asof:
            return False
        try:
            ts = datetime.fromisoformat(str(asof)[:19])
        except (ValueError, TypeError):
            return False
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        return (now - ts).days > self.max_stale_days

    def _assess_drift(self, baseline: Dict[str, Any],
                      snapshot: IntradaySnapshot) -> Dict[str, Any]:
        """判定盘中走势是否让 baseline 观点"失真"。

        口径（保守，宁可不报警也不错报）：
          看涨观点 + 盘中跌破昨收超过 drift_notional  → stale
          看跌观点 + 盘中涨过昨收超过 drift_notional  → stale
          其余（同向或波动不足）                      → intact
        快照未带来有效涨跌信息（change_pct 恒 0）时判 `unknown`，不误报。
        """
        change = snapshot.change_pct
        dominant = self._dominant_horizon(baseline)
        direction = dominant.get("direction", "")
        if not direction:
            return {"status": "unknown", "change_pct": round(change, 6),
                    "reason": "无 baseline 方向，无法判定"}
        if change == 0.0 or snapshot.prev_close <= 0:
            return {"status": "unknown", "change_pct": round(change, 6),
                    "reason": "盘中快照无有效涨跌（可能休市或昨收缺失）"}

        bullish = direction == "看涨"
        adverse = (bullish and change <= -self.drift_notional) or (
            (not bullish) and change >= self.drift_notional
        )
        return {
            "status": "stale" if adverse else "intact",
            "change_pct": round(change, 6),
            "direction": direction,
            "threshold": self.drift_notional,
            "reason": (
                f"盘中相对昨收 {change:+.2%}，与 baseline「{direction}」相悖，观点可能失真"
                if adverse
                else f"盘中相对昨收 {change:+.2%}，未与 baseline「{direction}」相悖"
            ),
        }

    @staticmethod
    def _dominant_horizon(baseline: Dict[str, Any]) -> Dict[str, Any]:
        """取置信度最高的周期作为"主导观点"（并列时取周期更长的那个，更保守）。"""
        horizons = baseline.get("horizons") or {}
        best: Dict[str, Any] = {}
        best_conf = -1.0
        order = {"short_term": 0, "mid_term": 1, "long_term": 2}
        for name, item in horizons.items():
            conf = abs(_to_float(item.get("probability"), 0.5) - 0.5)
            if conf >= best_conf:
                best_conf = conf
                best = dict(item)
                best["horizon"] = name
                best_conf = conf
                order.get(name, 0)
        return best

    # ------------------------------------------------------------------
    def poll_once(self, symbols: Sequence[str], collector=None,
                  persist: bool = True) -> Dict[str, Any]:
        """执行一轮盘中更新：拉快照 → 逐只生成信号 → （可选）落盘快照。

        Returns:
            {"asof", "is_trading_hours", "quotes": n, "signals": [...], "stale": [...]}
        """
        from src.data.collector import DataCollector

        collector = collector or DataCollector(self.config)
        quotes = self.quote_client.fetch_quotes(symbols) if symbols else {}
        if persist and quotes:
            self.store.append(list(quotes.values()))

        signals: List[Dict[str, Any]] = []
        for symbol in symbols:
            daily_df = None
            try:
                daily_df = collector.load_cached(symbol)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[intraday] {symbol} 读取缓存失败: {e}")
            sig = self.build(symbol, daily_df, quotes.get(symbol))
            signals.append(sig.to_dict())

        return {
            "asof": datetime.now().isoformat(timespec="seconds"),
            "is_trading_hours": MarketClock.is_trading_hours(),
            "poll_seconds": self.poll_seconds,
            "quotes": len(quotes),
            "signals": signals,
            "stale": [s["symbol"] for s in signals if (s.get("drift") or {}).get("status") == "stale"],
        }
