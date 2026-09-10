"""信号一致性校验（Q3）：跨周期 / 跨模型 / 跨口径的自洽性体检。

## 为什么需要它

Q2 之后仓库里同时存在多条"给出方向"的路径：

| 来源 | 方向来源 |
|------|---------|
| `PredictEngine.predict_all_horizons` | 各周期独立模型（lightgbm / lstm / ensemble） |
| `FactorCombiner` | 模型概率 + 技术特征因子加权 |
| `FactorModel` | IC 加权因子模型 |
| `IntradayPredictor` | T-1 baseline + 盘中快照 |

它们**各自都可能跑通**，但没人回答一个更基础的问题：**它们互相矛盾吗？**
一个"中期看涨 0.62、长期看跌 0.55"的组合，和"多因子看跌 0.58"是同一套系统给出的，
放进日报里客户一眼就会问"到底看涨还是看跌"。

本模块把这类矛盾显式化：

1. **跨周期一致性**：三周期方向是否同向；短期与长期相悖时给 `divergent`；
2. **跨模型一致性**：可用模型族（tree / factor / sequence）方向是否一致；
3. **跨口径一致性**：模型口径 vs 多因子口径得分差是否超容忍带。

## 设计原则

- **只读、纯计算**：不训练、不触网、不落盘（除显式导出）；
- **样本不足即沉默**：无法判定的项标 `unknown`，绝不臆测"一致"；
- **可解释**：每个不一致都给 `reason`，可直接进日报与监控报表。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATUS_CONSISTENT = "consistent"
STATUS_DIVERGENT = "divergent"
STATUS_UNKNOWN = "unknown"

# 跨口径得分差的默认容忍带（|两个口径得分之差| 超过该值视为口径分歧）
DEFAULT_TOLERANCE = 0.15
# 概率偏离 0.5 的最小幅度：小于该值视为"无观点"，不参与一致性判定
DEFAULT_NEUTRAL_BAND = 0.02


def _direction_of(probability: float, neutral_band: float) -> str:
    """概率 → 方向；落在中性带内返回 ""（无观点，不参与一致性判定）。"""
    if probability > 0.5 + neutral_band:
        return "看涨"
    if probability < 0.5 - neutral_band:
        return "看跌"
    return ""


@dataclass
class ConsistencyReport:
    """一致性体检报告（可直接序列化进 API / 日报 / 监控）。"""

    symbol: str = ""
    asof: str = ""
    status: str = STATUS_UNKNOWN
    horizon_consistency: Dict[str, Any] = field(default_factory=dict)
    model_consistency: Dict[str, Any] = field(default_factory=dict)
    cross_caliber: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asof": self.asof or datetime.now().isoformat(timespec="seconds"),
            "status": self.status,
            "horizon_consistency": self.horizon_consistency,
            "model_consistency": self.model_consistency,
            "cross_caliber": self.cross_caliber,
            "notes": list(self.notes),
        }


class SignalConsistencyChecker:
    """跨周期 / 跨模型 / 跨口径一致性校验器（纯读）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = ((config or {}).get("consistency", {}) or {})
        self.tolerance = float(cfg.get("tolerance", DEFAULT_TOLERANCE))
        self.neutral_band = float(cfg.get("neutral_band", DEFAULT_NEUTRAL_BAND))
        # 允许"短期与中长周期相悖"（短周期噪声大是已知事实），
        # 只有中/长周期之间相悖才算 divergent。
        self.ignore_short_term = bool(cfg.get("ignore_short_term", True))

    # ------------------------------------------------------------------
    def check_horizons(self, predictions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """跨周期一致性：三周期方向是否同向。"""
        horizons = (predictions or {}).get("predictions") or {}
        dirs: Dict[str, str] = {}
        probs: Dict[str, float] = {}
        for name, item in horizons.items():
            if not isinstance(item, dict) or "error" in item:
                continue
            prob = float(item.get("probability", 0.5) or 0.5)
            prob = 0.5 if prob != prob else prob  # NaN → 中性
            probs[name] = prob
            d = _direction_of(prob, self.neutral_band)
            if d:
                dirs[name] = d

        detail: Dict[str, Any] = {
            "directions": dirs,
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        }
        if not dirs:
            detail.update({"status": STATUS_UNKNOWN, "reason": "所有周期均无明确方向（中性带内）"})
            return detail

        considered = {
            k: v for k, v in dirs.items()
            if not (self.ignore_short_term and k == "short_term")
        } or dirs  # 只剩短期时仍以短期为准，避免"沉默"

        unique = set(considered.values())
        if len(unique) == 1:
            detail.update({
                "status": STATUS_CONSISTENT,
                "reason": f"参与判定的周期方向一致（{'/'.join(considered)} → {next(iter(unique))}）",
                "considered": sorted(considered),
            })
        else:
            detail.update({
                "status": STATUS_DIVERGENT,
                "reason": "周期方向相悖：" + "，".join(f"{k}={v}" for k, v in sorted(considered.items())),
                "considered": sorted(considered),
            })
        return detail

    # ------------------------------------------------------------------
    def check_models(self, components: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """跨模型一致性：各类分量（tree / factor / sequence）方向是否一致。"""
        comps = components or {}
        dirs: Dict[str, str] = {}
        scores: Dict[str, float] = {}
        for name, value in comps.items():
            if name.startswith("_") or value is None:
                continue
            score = float(value)
            if score != score:  # NaN
                continue
            scores[name] = score
            if score > self.neutral_band:
                dirs[name] = "看涨"
            elif score < -self.neutral_band:
                dirs[name] = "看跌"

        detail: Dict[str, Any] = {
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "directions": dirs,
        }
        if len(dirs) < 2:
            detail.update({
                "status": STATUS_UNKNOWN,
                "reason": f"有效分量不足（{len(dirs)} < 2），无法比较模型间接一致性",
            })
            return detail
        if len(set(dirs.values())) == 1:
            detail.update({
                "status": STATUS_CONSISTENT,
                "reason": f"各分量方向一致（{'/'.join(sorted(dirs))} → {next(iter(set(dirs.values())))}）",
            })
        else:
            detail.update({
                "status": STATUS_DIVERGENT,
                "reason": "分量方向相悖：" + "，".join(f"{k}={v}" for k, v in sorted(dirs.items())),
            })
        return detail

    # ------------------------------------------------------------------
    def check_calibers(self, model_probability: Optional[float],
                       factor_score: Optional[float]) -> Dict[str, Any]:
        """跨口径一致性：模型概率（未中性化）与因子得分（已中性化）是否同向。

        两者量纲不同，先各自做符号判定，再用**概率偏离 0.5 的幅度**与
        |factor_score| 的差距做容忍带比较，避免拿不同量纲直接相减。
        """
        if model_probability is None or factor_score is None:
            return {"status": STATUS_UNKNOWN, "reason": "缺少模型或因子口径数据"}
        try:
            prob = float(model_probability)
            score = float(factor_score)
        except (TypeError, ValueError):
            return {"status": STATUS_UNKNOWN, "reason": "口径数据无法解析为数值"}
        if prob != prob or score != score:
            return {"status": STATUS_UNKNOWN, "reason": "口径数据含 NaN"}

        model_dir = _direction_of(prob, self.neutral_band)
        factor_dir = "看涨" if score > self.neutral_band else ("看跌" if score < -self.neutral_band else "")
        detail: Dict[str, Any] = {
            "model_probability": round(prob, 4),
            "model_direction": model_dir,
            "factor_score": round(score, 4),
            "factor_direction": factor_dir,
            "tolerance": self.tolerance,
        }
        if not model_dir or not factor_dir:
            detail.update({"status": STATUS_UNKNOWN, "reason": "任一口径无明确方向（中性带内）"})
            return detail

        # 同向但幅度差距过大，也算分歧（一个"强烈看涨"配一个"微弱看涨"需人工复核）
        gap = abs((prob - 0.5) - abs(score))
        if model_dir != factor_dir:
            detail.update({
                "status": STATUS_DIVERGENT,
                "reason": f"口径方向相悖：模型={model_dir}（{prob:.2%}），因子={factor_dir}（{score:+.3f}）",
            })
        elif gap > self.tolerance:
            detail.update({
                "status": STATUS_DIVERGENT,
                "reason": f"口径方向相同但强度差距 {gap:.3f} > 容忍带 {self.tolerance}",
            })
        else:
            detail.update({
                "status": STATUS_CONSISTENT,
                "reason": f"两口径方向一致（{model_dir}），强度差距 {gap:.3f} 在容忍带内",
            })
        return detail

    # ------------------------------------------------------------------
    def check(self, symbol: str,
              predictions: Optional[Dict[str, Any]] = None,
              components: Optional[Dict[str, Any]] = None,
              model_probability: Optional[float] = None,
              factor_score: Optional[float] = None,
              asof: Optional[str] = None) -> ConsistencyReport:
        """综合体检：任一项 divergent → 整体 divergent；全部 unknown → unknown。"""
        report = ConsistencyReport(symbol=symbol, asof=asof or "")
        report.horizon_consistency = self.check_horizons(predictions)
        report.model_consistency = self.check_models(components)
        report.cross_caliber = self.check_calibers(model_probability, factor_score)

        statuses = [
            report.horizon_consistency.get("status"),
            report.model_consistency.get("status"),
            report.cross_caliber.get("status"),
        ]
        if STATUS_DIVERGENT in statuses:
            report.status = STATUS_DIVERGENT
        elif any(s == STATUS_CONSISTENT for s in statuses):
            report.status = STATUS_CONSISTENT
        else:
            report.status = STATUS_UNKNOWN

        # 一致性判定不参与策略门禁（门禁只看 IC / 命中率），此处明确免责，
        # 避免下游把它误当成"第二道门禁"而产生隐式行为变更。
        report.notes.append("一致性校验为观测项，不参与策略门禁（门禁口径见 strategy_gate）")
        if report.status == STATUS_DIVERGENT:
            report.notes.append("存在口径/周期分歧，建议人工复核后再据此解读信号")
        return report


def check_from_engine(config: Dict[str, Any], engine, symbol: str) -> Dict[str, Any]:
    """便捷入口：对已加载的引擎跑一次一致性体检（fail-soft）。"""
    try:
        pred = engine.predict_all_horizons(symbol)
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "status": STATUS_UNKNOWN, "reason": f"预测失败: {e}"}
    return SignalConsistencyChecker(config).check(symbol, predictions=pred).to_dict()
