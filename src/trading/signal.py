"""信号聚合：将 TrendCast Pro 多周期预测结果聚合为可执行的交易信号。

信号引擎综合 短/中/长 三个周期的方向预测及其概率，按周期权重计算一个
综合多空得分（score ∈ [-1, 1]），据此映射为离散交易动作：
  - BUY  强烈看多 → 做多
  - SELL 强烈看空 → 做空 / 卖出
  - HOLD 方向不明或处于观望区间
并输出信号强度、平均置信度、主要矛盾方向等供上层风控与下单使用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 周期默认权重（可被 config 覆盖）：短期噪音大、长期置信更高
DEFAULT_HORIZON_WEIGHTS = {
    "short_term": 0.30,
    "mid_term": 0.35,
    "long_term": 0.35,
}

# 动作
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


@dataclass
class Signal:
    """一条归一化后的交易信号。

    - action     : BUY / SELL / HOLD
    - score      : 综合多空得分 ∈ [-1, 1]（>0 看多，<0 看空）
    - strength   : 信号强度 0~1（|score| 折算）
    - confidence : 多周期平均置信度
    - position   : 建议方向仓位比例 0~1（多头为正、空头为负，按风险后在 order 层确定手数）
    """

    symbol: str
    action: str
    score: float
    strength: float
    confidence: float
    direction_consensus: str
    horizon_signals: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "score": round(self.score, 4),
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "direction_consensus": self.direction_consensus,
            "horizons": self.horizon_signals,
        }


class SignalEngine:
    """根据预测结果生成交易信号。"""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("trading", {}).get("signal", {})
        # 周期权重
        weights = dict(DEFAULT_HORIZON_WEIGHTS)
        weights.update(cfg.get("horizon_weights", {}))
        # 归一化权重
        total = sum(weights.values()) or 1.0
        self.horizon_weights = {k: v / total for k, v in weights.items()}

        # 动作阈值：score > buy_threshold 视为 BUY，< sell_threshold 视为 SELL
        self.buy_threshold = float(cfg.get("buy_threshold", 0.25))
        self.sell_threshold = float(cfg.get("sell_threshold", -0.25))
        # 是否允许做空（SELL）
        self.allow_short = bool(cfg.get("allow_short", True))

    def _direction_value(self, horizon_pred: dict[str, Any]) -> float:
        """把单个周期的预测折算为 [-1, 1] 的方向分。

        0/1 分类：1(看涨)→+1，0(看跌)→-1；概率作为幅度权。无法预测返回 0。
        """
        if not isinstance(horizon_pred, dict) or "error" in horizon_pred:
            return 0.0
        pred = horizon_pred.get("prediction")
        if pred is None:
            return 0.0
        proba = float(horizon_pred.get("probability", 0.5))
        # 将 0.5 中性点映射到 0，向两端线性放大
        mag = (proba - 0.5) * 2.0  # [-1,1]
        return 1.0 * mag if pred == 1 else -1.0 * mag

    def build_signal(self, symbol: str, predictions: dict[str, Any]) -> Signal:
        """输入引擎预测结果（predict_all_horizons 返回结构），输出 Signal。

        Args:
            symbol: 标的代码
            predictions: {symbol, predictions:{short_term:{...}, mid_term:{...}, long_term:{...}}}
        """
        horizon_map = predictions.get("predictions", predictions)
        if not isinstance(horizon_map, dict):
            horizon_map = {}

        contrib: dict[str, dict[str, Any]] = {}
        weighted_sum = 0.0
        conf_sum = 0.0
        conf_weight = 0.0
        resolved_count = 0

        for h_name, w in self.horizon_weights.items():
            hp = horizon_map.get(h_name)
            dv = self._direction_value(hp) if isinstance(hp, dict) else 0.0
            weighted_sum += dv * w
            contrib[h_name] = {"direction_score": round(dv, 4), "raw": hp}
            if isinstance(hp, dict) and "error" not in hp and hp.get("prediction") is not None:
                # 只用有实际预测的周期统计置信度
                conf_sum += float(hp.get("confidence", 0.5)) * w
                conf_weight += w
                resolved_count += 1

        score = weighted_sum  # ∈ [-1,1]
        confidence = (conf_sum / conf_weight) if conf_weight > 0 else 0.0
        strength = abs(score)

        # 动作映射
        if score >= self.buy_threshold:
            action = BUY
        elif score <= self.sell_threshold:
            action = SELL if self.allow_short else HOLD
        else:
            action = HOLD

        # 共识方向
        pos = [h for h, c in contrib.items() if c["direction_score"] > 0]
        neg = [h for h, c in contrib.items() if c["direction_score"] < 0]
        if len(pos) > len(neg):
            consensus = "看涨"
        elif len(neg) > len(pos):
            consensus = "看跌"
        elif score > 0:
            consensus = "偏多"
        elif score < 0:
            consensus = "偏空"
        else:
            consensus = "中性"

        from datetime import datetime
        sig = Signal(
            symbol=symbol,
            action=action,
            score=score,
            strength=strength,
            confidence=confidence,
            direction_consensus=consensus,
            horizon_signals=contrib,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )
        logger.info(
            "信号 %s -> %s score=%.3f confidence=%.2f consensus=%s",
            symbol, action, score, confidence, consensus,
        )
        return sig
