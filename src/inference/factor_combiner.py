"""多因子加权组合预测（Q2 路线图：多因子模型集成，支持因子加权组合预测）。

把「模型因子」与「特征因子」统一成同一套加权框架：

  score = Σ (weight_i × factor_score_i)   ，factor_score ∈ [-1, 1]

- **模型因子**：lightgbm / timesfm / lstm 各分量模型给出的概率，按周期加权融合；
- **特征因子**：技术指标构造的横截面因子（动量 / 波动 / 量价 / 均线偏离），
  由 ``build_feature_factors`` 归一化到 [-1, 1]，权重可由配置显式指定或按 IC 自适应分配；
- **无 net 依赖**：纯 numpy/pandas 实现，缺失分量自动重归一化权重（fail-soft），
  绝不静默把缺失分量当作 0 分（那会引入方向性偏差，而非中性）。

与 ``src/trading/signal.SignalEngine`` 的分工：
- SignalEngine：把「三周期预测」聚合为一个交易信号（周期维度）；
- FactorCombiner：把「多个因子」聚合为一个周期内的综合得分（因子维度）。
两者可串联：因子得分 → 周期预测概率 → 信号。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# 默认因子权重（可由 config.model.factors.weights 覆盖）
DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "lightgbm": 0.45,
    "timesfm": 0.25,
    "momentum": 0.10,
    "trend": 0.10,
    "volume": 0.05,
    "volatility": 0.05,
}


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _squash(value: Optional[float], scale: float = 1.0) -> float:
    """把任意实数压到 [-1, 1]（tanh 平滑，无硬截断）。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return _clip(math.tanh(float(value) / (scale or 1.0)))


def proba_to_score(probability: Optional[float]) -> float:
    """概率 → 方向分：0.5 中性映射为 0，向两端线性放大到 [-1, 1]。"""
    if probability is None or (isinstance(probability, float) and math.isnan(probability)):
        return 0.0
    return _clip((float(probability) - 0.5) * 2.0)


@dataclass
class FactorScore:
    """单个因子的得分明细。"""

    name: str
    score: float
    weight: float
    contribution: float
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.name,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 4),
        }


@dataclass
class CombinedScore:
    """因子组合结果。"""

    symbol: str
    score: float
    direction: str
    probability: float
    confidence: float
    factors: List[FactorScore] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    weighter: str = "static"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "direction": self.direction,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4),
            "weighter": self.weighter,
            "missing": list(self.missing),
            "factors": [f.to_dict() for f in self.factors],
        }


class FactorCombiner:
    """多因子加权组合器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 ic_calculator: Any = None):
        cfg = ((config or {}).get("model_factors")
               or ((config or {}).get("model", {}) or {}).get("factors", {}) or {})
        weights = dict(DEFAULT_FACTOR_WEIGHTS)
        weights.update({k: float(v) for k, v in (cfg.get("weights") or {}).items()})
        self.weights = weights
        # weighter: static(静态权重) / ic(按 IC 自适应权重)
        self.weighter = str(cfg.get("weighter", "static")).lower()
        self.ic_calculator = ic_calculator

    # ------------------------------------------------------------------
    # 权重
    # ------------------------------------------------------------------
    def resolve_weights(self, available: Sequence[str],
                        ic_by_factor: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """按可用因子解析权重（自动重归一化）。

        - static：直接用配置权重，缺失因子的权重被移除后重归一化；
        - ic    ：权重 = |IC|，并按配置权重作先验缩放（IC 缺失则回落到 static）。
        """
        base = {name: float(self.weights.get(name, 0.0)) for name in available}
        if self.weighter == "ic" and ic_by_factor:
            base = {
                name: abs(float(ic_by_factor.get(name, 0.0))) * float(self.weights.get(name, 0.0))
                for name in available
            }
        total = sum(v for v in base.values() if v > 0)
        if total <= 0:
            # 全零权重 → 等权兜底，避免除零
            n = len(available) or 1
            return {name: 1.0 / n for name in available}
        return {name: (v / total if v > 0 else 0.0) for name, v in base.items()}

    # ------------------------------------------------------------------
    # 组合
    # ------------------------------------------------------------------
    def combine(self, symbol: str, factor_scores: Dict[str, Optional[float]],
                ic_by_factor: Optional[Dict[str, float]] = None) -> CombinedScore:
        """把各因子得分加权组合为综合得分。"""
        available = [name for name, val in factor_scores.items() if val is not None]
        missing = [name for name, val in factor_scores.items() if val is None]
        weights = self.resolve_weights(available, ic_by_factor)

        details: List[FactorScore] = []
        score = 0.0
        for name, val in factor_scores.items():
            w = float(weights.get(name, 0.0))
            s = _clip(float(val)) if val is not None else 0.0
            contribution = s * w
            score += contribution
            details.append(FactorScore(name=name, score=s, weight=w,
                                       contribution=contribution, raw=val))

        score = _clip(score)
        probability = _clip(0.5 + score / 2.0, 0.0, 1.0)
        if score > 0.05:
            direction = "看涨"
        elif score < -0.05:
            direction = "看跌"
        else:
            direction = "中性"
        confidence = abs(score) if missing else abs(score) * (len(available) / max(len(factor_scores), 1))

        return CombinedScore(
            symbol=symbol,
            score=score,
            direction=direction,
            probability=probability,
            confidence=_clip(confidence, 0.0, 1.0),
            factors=details,
            missing=missing,
            weighter=self.weighter,
        )

    # ------------------------------------------------------------------
    # 模型因子
    # ------------------------------------------------------------------
    @staticmethod
    def model_factor_scores(horizon_predictions: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """从 predict_all_horizons 的单周期结果中抽取模型因子得分。

        支持结构：
        - ``{"probability": 0.62, "model": "lightgbm_..."}``
        - ``{"components": {"lightgbm": 0.62, "timesfm": 0.55}}``（扩展）
        """
        out: Dict[str, Optional[float]] = {}
        if not isinstance(horizon_predictions, dict) or "error" in horizon_predictions:
            return out
        comps = horizon_predictions.get("components")
        if isinstance(comps, dict):
            for name, proba in comps.items():
                out[str(name)] = proba_to_score(proba)
        elif horizon_predictions.get("probability") is not None:
            model_name = str(horizon_predictions.get("model", "lightgbm")).split("_")[0]
            out[model_name] = proba_to_score(horizon_predictions.get("probability"))
        return out


# ----------------------------------------------------------------------
# 特征因子（由行情 DataFrame 构造，[0,1] 归一化到 [-1,1]）
# ----------------------------------------------------------------------
def _sigmoid_normalize(series: Any, window: int = 60) -> float:
    """把序列最新值转为滚动分位得分 ∈ [-1, 1]（+1 = 历史高位）。"""
    try:
        tail = series.dropna().tail(window)
        if len(tail) < 5:
            return 0.0
        latest = float(tail.iloc[-1])
        lo, hi = float(tail.min()), float(tail.max())
        if hi <= lo:
            return 0.0
        return _clip((latest - lo) / (hi - lo) * 2.0 - 1.0)
    except Exception:  # noqa: BLE001
        return 0.0


def build_feature_factors(df: Any) -> Dict[str, float]:
    """由行情 DataFrame（含 close/volume）构造特征因子得分。

    纯函数：不触网、不落盘；缺列自动跳过（返回 0 分），fail-soft。
    """
    factors: Dict[str, float] = {"momentum": 0.0, "trend": 0.0, "volume": 0.0, "volatility": 0.0}
    try:
        if df is None or len(df) == 0:
            return factors
        close = df["close"] if "close" in df.columns else None
        if close is None:
            return factors
        close = close.astype(float)

        # 动量：近 20 日收益率在滚动窗口中的相对分位
        if len(close) > 20:
            mom = close.pct_change(20).dropna()
            factors["momentum"] = _sigmoid_normalize(mom)
        # 趋势：收盘价 / MA20 - 1 的相对分位（均线偏离）
        if len(close) >= 20:
            ma = close.rolling(20).mean()
            dev = (close / ma - 1.0).dropna()
            factors["trend"] = _sigmoid_normalize(dev)
        # 量价：近 5 日成交量均值 / 近 20 日均值 - 1
        if "volume" in df.columns and len(df) >= 20:
            vol = df["volume"].astype(float)
            ratio = float(vol.tail(5).mean() / vol.tail(20).mean() - 1.0) if vol.tail(20).mean() else 0.0
            factors["volume"] = _squash(ratio, scale=0.5)
        # 波动：近 20 日波动率相对分位取负（高波动不一定是坏信号，但降权更稳）
        if len(close) > 20:
            vola = close.pct_change().rolling(20).std().dropna()
            factors["volatility"] = -_sigmoid_normalize(vola)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[factors] 特征因子构造失败，按 0 分处理: {e}")
    return factors


def combine_horizon(combiner: FactorCombiner, symbol: str, df: Any,
                    horizon_predictions: Dict[str, Any],
                    ic_by_factor: Optional[Dict[str, float]] = None) -> CombinedScore:
    """便捷入口：模型因子 + 特征因子 → 单周期综合得分。"""
    factor_scores = combiner.model_factor_scores(horizon_predictions)
    for name, value in build_feature_factors(df).items():
        factor_scores.setdefault(name, value)
    return combiner.combine(symbol, factor_scores, ic_by_factor=ic_by_factor)
