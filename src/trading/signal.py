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
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 周期默认权重（可被 config 覆盖）：短期噪音大、长期置信更高
DEFAULT_HORIZON_WEIGHTS = {
    "short_term": 0.30,
    "mid_term": 0.35,
    "long_term": 0.35,
}

# 概率契约：概率必须落在 [0, 1]。
# 越界 / NaN / ±inf 一律**不可作方向依据** —— 既不得放大方向分，也不得反转方向。
# 这类值不是"极端看多/看空"，而是**数据坏了**；按 0.5（中性）处理并留痕。
PROB_FLOOR = 0.0
PROB_CEIL = 1.0
NEUTRAL_PROB = 0.5


def _clamp_unit(value: float) -> float:
    """把得分 clamp 到 [-1, 1]；非有限值退化为 0（中性）。"""
    if not math.isfinite(value):
        return 0.0
    return min(max(value, -1.0), 1.0)


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

    @staticmethod
    def _sanitize_confidence(raw: Any) -> float:
        """把置信度规整到 [0, 1]。

        置信度只作仓位缩放因子（`risk._sizing_fraction` 里虽有 `min()`
        兜底上界，但**负数与 NaN / inf 没有下界保护**），故此处统一 fail-safe。
        """
        try:
            conf = float(raw)
        except (TypeError, ValueError):
            logger.warning("置信度不可解析 %r，按中性 %.1f 处理", raw, NEUTRAL_PROB)
            return NEUTRAL_PROB
        if not math.isfinite(conf):
            logger.warning("置信度为非有限值 %r，按中性 %.1f 处理", raw, NEUTRAL_PROB)
            return NEUTRAL_PROB
        if conf < PROB_FLOOR or conf > PROB_CEIL:
            logger.warning("置信度越界 %r，已 clamp 到 [0,1]", raw)
        return min(max(conf, PROB_FLOOR), PROB_CEIL)

    @staticmethod
    def _sanitize_proba(raw: Any) -> float:
        """把原始概率规整到 [0, 1] 契约内。

        非有限值（NaN / ±inf）与越界值**不是"极端方向"**，而是数据已损坏：
        若原样进入方向分计算，`(proba - 0.5) * 2.0` 会被放大成远超 [-1, 1] 的
        数值（越界）甚至变成 ±inf（非有限值），进而：

        - `score` 突破 [-1, 1] 契约，污染日报 / 审计 / API；
        - **符号反转**：`probability = -inf` 时 `mag = -inf`，
          在 `prediction == 1`（看涨）下算出 `score = -inf` → 被判为 SELL，
          即**一个坏掉的概率悄悄把看涨信号翻成做空单**。

        因此此处 fail-safe 到中性 0.5（方向分为 0），并显式留痕。
        缺失 / 不可解析同样按中性处理（与既有 `prediction is None → 0.0` 一致）。
        """
        try:
            proba = float(raw)
        except (TypeError, ValueError):
            logger.warning("概率不可解析 %r，按中性 %.1f 处理", raw, NEUTRAL_PROB)
            return NEUTRAL_PROB
        if not math.isfinite(proba):
            logger.warning("概率为非有限值 %r，按中性 %.1f 处理（不作方向依据）", raw, NEUTRAL_PROB)
            return NEUTRAL_PROB
        if proba < PROB_FLOOR or proba > PROB_CEIL:
            logger.warning("概率越界 %r，按中性 %.1f 处理（不作方向依据）", raw, NEUTRAL_PROB)
            return NEUTRAL_PROB
        return proba

    def _direction_value(self, horizon_pred: dict[str, Any]) -> float:
        """把单个周期的预测折算为 [-1, 1] 的方向分。

        0/1 分类：1(看涨)→+1，0(看跌)→-1；概率作为幅度权。无法预测返回 0。
        概率先过 `_sanitize_proba` 契约，越界 / 非有限值一律退化为中性，
        **不会**被放大成越界方向分，也不会反转方向。
        """
        if not isinstance(horizon_pred, dict) or "error" in horizon_pred:
            return 0.0
        pred = horizon_pred.get("prediction")
        if pred is None:
            return 0.0
        proba = self._sanitize_proba(horizon_pred.get("probability", NEUTRAL_PROB))
        # 将 0.5 中性点映射到 0，向两端线性放大（proba ∈ [0,1] ⇒ mag ∈ [-1,1]）
        mag = (proba - NEUTRAL_PROB) * 2.0  # [-1,1]
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
                # 只用有实际预测的周期统计置信度；置信度同样受 [0,1] 契约约束
                conf_sum += self._sanitize_confidence(hp.get("confidence", NEUTRAL_PROB)) * w
                conf_weight += w
                resolved_count += 1

        # 方向分已逐周期收敛到 [-1,1]，加权和（权重归一化且非负）亦落在 [-1,1]。
        # 仍做一次 clamp：这是**契约硬校验**，一旦越界说明上游装配出了问题，
        # 与其把越界值放行到风控/下单，不如在此钉死在契约边界并留痕。
        score = _clamp_unit(weighted_sum)
        if not math.isfinite(weighted_sum) or abs(weighted_sum) > 1.0 + 1e-9:
            logger.warning("综合得分越界 %r，已 clamp 到 [-1,1]（上游装配异常）", weighted_sum)
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
