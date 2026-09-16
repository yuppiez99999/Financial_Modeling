"""决策源契约适配层（DecisionFeed）：把 TrendCast Pro 的原始预测翻译成
**下游可直接消费 + 可审计**的决策字段。

## 为什么需要这一层

16_ 的既定定位是「只出方向与概率，决策权归下游（tradingview / 28 系统）」。
但下游真正落地时反复遇到同一个问题：**契约里给的都是「原材料」，
每个消费方都要自己再写一遍同样的换算**，于是同一份预测在不同链路上
被翻译成不同的口径 —— 28 侧 `trendcast_signal_source._aggregate` 就是
一例（自己定 0.2/0.5/0.3 权重、自己定 0.6/0.4 动作阈值、只用
`|p-0.5|×2` 当置信度）。

本模块把这些换算**收到生产方一侧，做成显式、可复算、可审计的字段**：

1. **每周期净看涨概率**（`net_up_probability`）—— 下游不再需要自己
   写 ``direction=="看涨" ? p : 1-p`` 这类条件分支（这是信号口径不一致的
   主要来源；方向与概率打架时下游的行为完全取决于它读的是哪个字段）；
2. **多周期综合得分**（`composite_score`）—— 按显式权重聚合，
   三周期齐备/部分缺失如实标注，不静默补 0.5；
3. **校准后概率 / 不确定性**（`calibrated_probability` / `uncertainty`）
   —— 复用 S19 已固化的校准参数（缺失时不猜，原样标注
   ``calibration_applied=false``）；
4. **建议门槛与「是否建议采信」**（`recommended_threshold` /
   `advisory_consumable`）—— 把 S15「置信度子集命中率」结论落成
   对下游友好的**采纳建议**，同时在结构中声明
   ``advisory_only=true`` / ``affects_gate=false``：**本层不改变任何
   门禁判定，也不产出仓位/下单建议**。
5. **审计披露**（`audit`）—— 24h 内新增的**已回溯命中率**随 feed 一起下发，
   下游可据此自行决定是否采信，不必回查 16_ 内部文件。

## 边界（比功能重要）

- **不改门禁**：`affects_gate` 恒为 False，不写 `strategy_gate` 结论；
- **不产出仓位**：只给方向 / 概率 / 采纳建议，不给权重、不给手数；
- **不猜**：校准参数缺失、审计样本不足、周期缺失 —— 一律如实标注
  ``available=false`` + ``reason``，绝不用 0.5 冒充真实读数；
- **不择优**：门槛推荐只取「已配置的现行值」，不在代码里挑一个好看的阈值。

无前视说明：本模块只消费推理输出与已落盘审计记录，不重训模型、
不重新构造目标、不接触未来价格。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FEED_VERSION = "decision-feed/1"

# 多周期聚合默认权重（与 16_ 侧 SignalEngine.DEFAULT_HORIZON_WEIGHTS 同源口径：
# 短期噪音大、长期置信更高）。可被 config.decision_feed.horizon_weights 覆盖。
DEFAULT_HORIZON_WEIGHTS: Dict[str, float] = {
    "short_term": 0.30,
    "mid_term": 0.35,
    "long_term": 0.35,
}

# 建议门槛的来源：S15/G5 置信度子集门禁的**已配置现行值**（config
# strategy_gate.confidence_gate.thr 区间中人工确认过的默认），不是本模块自选。
DEFAULT_ADVISORY_THRESHOLD = 0.20

# 未校准时的兜底口径说明（下游据此知道 confidence 是怎么来的）
CONFIDENCE_SOURCE_PROBA = "proba_distance(|p-0.5|*2)"

DIRECTION_UP = "看涨"
DIRECTION_DOWN = "看跌"


# ----------------------------------------------------------------------
# 纯函数：单周期 / 多周期换算
# ----------------------------------------------------------------------
def net_up_probability(direction: Any, probability: Any) -> Optional[float]:
    """把「方向 + 该方向概率」归一为**净看涨概率** ∈ [0, 1]。

    这是本层存在的首要理由：下游不必再判断 direction 字符串，
    因此**不存在**「方向说看涨、但下游按概率 0.42 当看空」这类口径分裂。

    非有限值 / 缺失 → None（**不返回 0.5**：0.5 是有效的中性读数，
    不能与「数据坏了」混为一谈）。
    """
    import math

    if probability is None:
        return None
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p):
        return None
    p = min(max(p, 0.0), 1.0)
    d = str(direction or "").strip()
    if d == DIRECTION_UP:
        return round(p, 6)
    if d == DIRECTION_DOWN:
        return round(1.0 - p, 6)
    # 方向缺失/未知：概率本身仍可作为净看涨读数（模型输出的是 P(up)），
    # 但调用方会通过 direction_known=false 看到这个事实。
    return round(p, 6)


def _is_up(direction: Any) -> bool:
    return str(direction or "").strip() == DIRECTION_UP


def _is_down(direction: Any) -> bool:
    return str(direction or "").strip() == DIRECTION_DOWN


def aggregate_horizons(horizons: Dict[str, Any],
                       weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """多周期净看涨概率加权聚合 → 综合分（可复算、可解释）。

    与 28 侧 `_aggregate` 的关键差别：
      - 只对**可用周期**归一（缺失周期不补 0.5，避免把"没有观点"稀释成
        "中性观点"—— 这两者在门槛子集口径下行为完全不同）；
      - 显式回报 `missing_horizons` / `coverage`，下游可据覆盖率决定是否采信；
      - 权重缺失周期上的部分被剔除后**重新归一**，而不是把权重白扔掉。
    """
    w_cfg = dict(weights or DEFAULT_HORIZON_WEIGHTS)
    total_w = 0.0
    num = 0.0
    per: Dict[str, Any] = {}
    missing: List[str] = []

    for h, w in w_cfg.items():
        info = horizons.get(h) if isinstance(horizons, dict) else None
        if not isinstance(info, dict) or "error" in info:
            missing.append(h)
            continue
        p_up = net_up_probability(info.get("direction"), info.get("probability"))
        if p_up is None:
            missing.append(h)
            continue
        per[h] = {
            "net_up_probability": p_up,
            "direction": info.get("direction"),
            "direction_known": _is_up(info.get("direction")) or _is_down(info.get("direction")),
            "probability": info.get("probability"),
            "model": info.get("model"),
            "weight": round(float(w), 6),
        }
        total_w += float(w)
        num += float(w) * p_up

    if total_w <= 0:
        return {
            "available": False,
            "reason": "三周期均无可用概率，拒绝输出综合分（不猜）",
            "per_horizon": per,
            "missing_horizons": missing,
        }

    score = num / total_w
    return {
        "available": True,
        "composite_score": round(score, 6),          # 净看涨概率 ∈ [0,1]
        "composite_signed": round((score - 0.5) * 2.0, 6),  # ∈ [-1,1]，正=看多
        "coverage": round(total_w / sum(float(w) for w in w_cfg.values()), 6),
        "per_horizon": per,
        "missing_horizons": missing,
        "weights_used": {h: float(w_cfg[h]) for h in per},
        "weights_config": {h: float(v) for h, v in w_cfg.items()},
    }


def confidence_from_score(score: float) -> float:
    """综合分 → 置信度 = |score − 0.5| × 2（0=无观点，1=极端自信）。

    与 `src.eval.confidence_curve.confidence_from_proba`、S15 置信度曲线、
    `uncertainty_from_probability` 同一口径 —— 全仓库只有这一种换算。
    """
    s = min(max(float(score), 0.0), 1.0)
    return round(abs(s - 0.5) * 2.0, 6)


def advisory_verdict(confidence: float, threshold: float) -> Dict[str, Any]:
    """按置信度门槛给出**采纳建议**（不改变门禁，不产出仓位）。

    ``advisory_only`` 与 ``affects_gate=false`` 是本结构的硬声明：
    它只回答「这条信号够不够格进下游的打分参考」，不回答「买多少」。
    """
    consumable = confidence >= float(threshold)
    return {
        "confidence": round(float(confidence), 6),
        "recommended_threshold": round(float(threshold), 6),
        "advisory_consumable": bool(consumable),
        "advisory_only": True,
        "affects_gate": False,
        "note": ("建议门槛取自 16_ 已配置的置信度子集口径（S15/G5，人工检查点 T15.3 "
                 "= defer：口径已定、未切换门禁）。本字段仅为下游提供统一的采信建议，"
                 "**不改变** 16_ 侧任何放行判定，也不构成仓位/下单建议。"),
    }


# ----------------------------------------------------------------------
# 校准 / 审计 辅助
# ----------------------------------------------------------------------
def _calibration_dir(config: Dict[str, Any]) -> str:
    return str(((config or {}).get("training", {}) or {}).get("save_dir", "models"))


def _enrich(pred: Dict[str, Any], horizon: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """追加校准字段（失败一律降级为「未校准」，绝不因缺参数而报错）。"""
    try:
        from src.inference.probability_calibrator import enrich_prediction

        got = enrich_prediction(pred, horizon, directory=_calibration_dir(config))
        return {
            "calibrated_probability": got.get("calibrated_probability"),
            "calibration_applied": bool(got.get("calibration_applied", False)),
            "calibration_method": got.get("calibration_method"),
            "uncertainty": got.get("uncertainty"),
            "calibration_reason": got.get("calibration_reason", ""),
        }
    except Exception as e:  # noqa: BLE001 - 校准层不可用不得影响主 feed
        logger.warning("[decision-feed] 校准字段追加失败（按未校准处理）: %s", e)
        return {
            "calibrated_probability": None,
            "calibration_applied": False,
            "calibration_method": None,
            "uncertainty": None,
            "calibration_reason": f"校准层不可用: {e}",
        }


def _load_audit_records() -> tuple[List[Dict[str, Any]], Optional[str]]:
    """读一次审计记录，供 audit 摘要与 analytics 共用（避免重复读盘/口径漂移）。"""
    try:
        from src.audit.prediction_audit import PredictionAudit

        return list(PredictionAudit().load_records()), None
    except Exception as e:  # noqa: BLE001
        logger.warning("[decision-feed] 审计记录读取失败: %s", e)
        return [], f"审计不可用: {e}"


def _audit_window(config: Dict[str, Any], hours: int = 24,
                  records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """把审计窗口内的**已回溯命中率**摘要进 feed（下游可自行判断是否采信）。

    只读既有审计记录（``PredictionAudit._load_records``）；文件缺失 /
    样本不足一律 ``available=false`` + reason，绝不用 0 或 0.5 冒充读数。
    """
    if records is None:
        records, err = _load_audit_records()
        if err:
            return {"available": False, "reason": err}

    verified = [r for r in records if r.get("verified")]
    total = len(records)
    hits = sum(1 for r in verified if r.get("hit"))
    hit_rate = (hits / len(verified)) if verified else None

    cutoff = datetime.now() - timedelta(hours=int(hours))
    recent = []
    for r in verified:
        try:
            if datetime.fromisoformat(str(r.get("timestamp"))) >= cutoff:
                recent.append(r)
        except (TypeError, ValueError):
            continue
    recent_hits = sum(1 for r in recent if r.get("hit"))
    recent_rate = (recent_hits / len(recent)) if recent else None

    return {
        "available": True,
        "window_hours": int(hours),
        "total_records": total,
        "verified": len(verified),
        "pending": total - len(verified),
        "hit_rate_all": round(hit_rate, 6) if hit_rate is not None else None,
        "recent_verified": len(recent),
        "recent_hit_rate": round(recent_rate, 6) if recent_rate is not None else None,
        "note": ("hit_rate 为 16_ 侧用真实行情回溯的**已到期**预测命中率；"
                 "``recent_*`` 只覆盖窗口内新增验证，样本少时不具备统计意义，"
                 "下游不应据此单独采信（放行结论见 16_ 侧 gate）。"),
    }


def _analytics(config: Dict[str, Any], threshold: float,
               records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """衰减/有效性分析摘要（含在 feed 内，失败一律降级为 available=false）。"""
    try:
        from src.eval.decision_analytics import analyze

        if records is None:
            records, err = _load_audit_records()
            if err:
                return {"available": False, "reason": err}
        return analyze(records, threshold=threshold)
    except Exception as e:  # noqa: BLE001 - 分析失败不得影响主 feed
        logger.warning("[decision-feed] 有效性分析失败: %s", e)
        return {"available": False, "reason": f"分析不可用: {e}"}


# ----------------------------------------------------------------------
# 主构建函数
# ----------------------------------------------------------------------
def build_decision_feed(payload: Dict[str, Any], config: Optional[Dict[str, Any]] = None,
                        audit_hours: int = 24) -> Dict[str, Any]:
    """把 `/api/v1/portfolio/summary` 的原始契约升级为决策源契约。

    Args:
        payload: ``build_portfolio_summary`` / REST 端点的原始返回（含 predictions）。
        config:  TrendCast Pro 配置（读门槛 / 权重 / 校准目录）。
        audit_hours: 审计窗口（小时）。

    Returns:
        决策源契约 dict：``{contract_version, generated_at, model_type,
        advisory_config, predictions:[...], meta:{...}}``。

    **向后兼容**：本函数不修改传入 payload，原契约字段一个不少；
    新增字段全部命名空间化为显式名字，下游可渐进消费。
    """
    cfg = config or {}
    df_cfg = (cfg.get("decision_feed", {}) or {})
    weights = df_cfg.get("horizon_weights") or DEFAULT_HORIZON_WEIGHTS
    threshold = float(df_cfg.get("advisory_threshold", DEFAULT_ADVISORY_THRESHOLD))

    _records, _rec_err = _load_audit_records()
    preds_in = list((payload or {}).get("predictions") or [])
    out: List[Dict[str, Any]] = []
    consumable = 0
    scored = 0

    for item in preds_in:
        if not isinstance(item, dict):   # 契约崩坏不得拖垮端点
            continue
        symbol = item.get("symbol")
        horizons_in = item.get("horizons") or {}
        enriched: Dict[str, Any] = {}
        for hname, info in horizons_in.items():
            if not isinstance(info, dict) or "error" in info:
                enriched[hname] = info
                continue
            merged = dict(info)
            merged.update(_enrich(info, hname, cfg))
            # 每周期也给出净看涨概率：下游逐周期消费时同样不必判断方向
            merged["net_up_probability"] = net_up_probability(
                info.get("direction"), info.get("probability"))
            enriched[hname] = merged

        agg = aggregate_horizons(enriched, weights)
        entry: Dict[str, Any] = {
            "symbol": symbol,
            "sector": item.get("sector", ""),
            "horizons": enriched,
            "aggregate": agg,
        }
        if agg.get("available"):
            scored += 1
            conf = confidence_from_score(agg["composite_score"])
            verdict = advisory_verdict(conf, threshold)
            # 校准后的置信度（参数缺失时不猜：回落到未校准口径并标注）
            cal_scores = []
            for hname, per in agg["per_horizon"].items():
                hinfo = enriched.get(hname) or {}
                cp = hinfo.get("calibrated_probability")
                if cp is None:
                    continue
                p_up_cal = net_up_probability(hinfo.get("direction"), cp)
                if p_up_cal is not None:
                    cal_scores.append((hname, p_up_cal, float(per.get("weight", 0.0))))
            if cal_scores:
                tw = sum(w for _h, _p, w in cal_scores) or 1.0
                cal_score = sum(p * w for _h, p, w in cal_scores) / tw
                verdict["calibrated_composite_score"] = round(cal_score, 6)
                verdict["calibrated_confidence"] = confidence_from_score(cal_score)
                verdict["calibrated_consumable"] = (
                    verdict["calibrated_confidence"] >= threshold)
            else:
                verdict["calibrated_composite_score"] = None
                verdict["calibrated_confidence"] = None
                verdict["calibrated_consumable"] = None
            entry["advisory"] = verdict
            if verdict["advisory_consumable"]:
                consumable += 1
        else:
            entry["advisory"] = {
                "advisory_consumable": False,
                "advisory_only": True,
                "affects_gate": False,
                "reason": agg.get("reason", "无可用聚合"),
            }
        out.append(entry)

    meta_in = dict((payload or {}).get("meta") or {})
    meta_in.update({
        "symbol_count": len(out),
        "scored_count": scored,
        "advisory_consumable_count": consumable,
        "confidence_source": CONFIDENCE_SOURCE_PROBA,
        "horizon_weights": {h: float(w) for h, w in dict(weights).items()},
        "affects_gate": False,
        "readonly": True,
    })

    return {
        "contract_version": FEED_VERSION,
        "generated_at": (payload or {}).get("generated_at")
        or datetime.now().isoformat(timespec="seconds"),
        "model_type": (payload or {}).get("model_type"),
        "role": "decision_source_readonly",
        "position_role": "observer",   # 结构性声明：本 feed 不产出仓位
        "advisory_config": {
            "recommended_threshold": round(threshold, 6),
            "threshold_source": ("strategy_gate.confidence_gate 现行配置（S15/G5；"
                                 "T15.3=defer，口径已定未切换门禁）"),
            "confidence_source": CONFIDENCE_SOURCE_PROBA,
            "horizon_weights": {h: float(w) for h, w in dict(weights).items()},
            "advisory_only": True,
            "affects_gate": False,
        },
        "predictions": out,
        "audit": _audit_window(cfg, hours=audit_hours, records=_records),
        "analytics": _analytics(cfg, threshold, records=_records),
        "meta": meta_in,
        "note": ("本 feed 是 16_ 对下游决策源（tradingview / 28）的只读契约："
                 "只出方向、概率、校准值与**采纳建议**，不产出仓位/下单建议；"
                 "所有新增字段均为增量，原始 direction/probability/model 逐字段保留。"),
    }
