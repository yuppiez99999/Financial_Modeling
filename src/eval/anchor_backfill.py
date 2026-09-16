"""锚点回填：用本地真实日K + 已训练模型，产出「分数 × 命中 × 已实现收益」锚点序列。

## 为什么单独一个模块

`decision_analytics` 回答的是「审计记录里高置信子集有没有优势」，而审计记录
依赖系统**在线跑过**一段时间的预测。冷启动 / 换池之后审计是空的 ——
下游（TradingView）拿不到任何 `analytics.overall` 读数，只能看到
`available=false`。本模块补的是这一段：

**离线回填**：沿历史时间轴每 ``step`` 个交易日取一个锚点，用**当时可见**的
特征行推理，再用**之后** ``horizon`` 日的真实收益回填结果。

## 无前视纪律（这是本模块唯一的正确性红线）

- 特征行只取到 ``t``（``FeatureEngineer.transform`` 全部指标都是滚动/后视窗，
  见 `src/data/preprocessor.py`）→ 不存在未来信息；
- 结果只在 ``t+h`` 已收线时才回填（``close[t+h] / close[t] - 1``），
  未到期锚点标 ``pending=None`` 并**不参与**命中率 / 收益统计；
- 模型是**当下这一个**（不是逐锚点重训）→ 结论必须一起读
  「同模型回看历史」的局限，故所有读数标 ``validated=False``，
  下游与卡片一起显示 ``NOT VALIDATED``。

## 锚点重叠

``step < horizon`` 时锚点视窗互相重叠，水平读数不可当独立样本
（与 `decision_analytics` 的边界声明同源），故锚点里带 ``overlap`` 标记，
下游可自行决定是否降权。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_LABELS = ("short_term", "mid_term", "long_term")


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def backfill_symbol(engine: Any, symbol: str, horizon: str = "short_term",
                    step: int = 5, max_anchors: int = 120,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
    """单标的锚点回填。

    Args:
        engine: ``PredictionEngine`` 实例（模型已 ``load_models``）。
        symbol: 标的代码。
        horizon: 用哪个周期的模型出分（缺省 5 日）。
        step: 锚点间隔（交易日）。``step < horizon_days`` 即视窗重叠。
        max_anchors: 锚点上限（取**最近**的这么多段，避免长历史把小样本口径放大）。
        end_date: 截断日期（回填到该日为止）；缺省用数据末端。

    Returns:
        ``{symbol, horizon, horizon_days, step, anchors:[...], available, reason}``。
        每锚点：``{date, ts, score, net_up_probability, direction, probability,
        hit, actual_return, forward_days, pending, overlap}``。
    """
    # 数据采集器走**模块级名字**导入：生产路径不变，测试可按名替换为回放替身
    # （直接 `from ... import` 到函数内会让 monkeypatch 打不中，见 tests/test_tv_export.py）
    from src.data.collector import DataCollector

    cfg = getattr(engine, "config", {}) or {}
    horizon_days = int((cfg.get("data", {}).get("prediction_horizons", {})
                        or {}).get(horizon, 5))
    out: Dict[str, Any] = {
        "symbol": symbol, "horizon": horizon, "horizon_days": horizon_days,
        "step": int(step), "anchors": [], "available": False, "validated": False,
    }

    model_key = f"{horizon}_{horizon_days}d"
    if model_key not in getattr(engine, "models", {}):
        out["reason"] = f"模型 {model_key} 未加载，无法回填"
        return out

    try:
        collector = DataCollector(cfg)  # noqa: F821 - 见上文模块级名字说明
        raw = collector.load_cached(symbol)
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"缓存读取失败: {e}"
        return out
    if raw is None or len(raw) == 0:
        out["reason"] = "无本地日K缓存（data/raw/ 缺该标的）"
        return out

    try:
        feats = engine.feature_engineer.transform(raw, horizon_days)
        feature_cols = engine.feature_engineer.get_feature_columns(feats, horizon_days)
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"特征工程失败: {e}"
        return out

    close = pd.to_numeric(raw["close"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d").tolist()
    n = len(feats)
    if n <= horizon_days + 1:
        out["reason"] = f"样本不足（{n} 行 ≤ horizon {horizon_days} + 1）"
        return out

    # 最后一个可完整回填的锚点下标（t + horizon <= n-1）
    last_idx = n - 1 - horizon_days
    if end_date:
        try:
            cut = pd.Timestamp(end_date)
            while last_idx > 0 and pd.Timestamp(dates[last_idx]) > cut:
                last_idx -= 1
        except Exception:  # noqa: BLE001
            pass
    if last_idx < 1:
        out["reason"] = "回填窗口为空（end_date 太早或数据太短）"
        return out

    # 锚点从后往前取，保证 max_anchors 保留的是最近一段
    idxs: List[int] = []
    i = last_idx
    while i >= 1 and len(idxs) < int(max_anchors):
        idxs.append(i)
        i -= max(1, int(step))
    idxs.reverse()

    from src.export.decision_feed import net_up_probability

    anchors: List[Dict[str, Any]] = []
    for t in idxs:
        score = _predict_score(engine, model_key, feats, feature_cols, t)
        if score is None:
            continue
        entry = {
            "date": dates[t],
            "index": int(t),
            "score": round(score["net_up_probability"], 6),
            "probability": round(score["probability"], 6),
            "direction": score["direction"],
        }
        c0, c1 = close[t], close[t + horizon_days]
        if _finite(c0) and _finite(c1) and c0 > 0:
            ret = c1 / c0 - 1.0
            direction_up = score["net_up_probability"] >= 0.5
            entry.update({
                "hit": bool(direction_up == (ret > 0)),
                "actual_return": round(float(ret), 8),
                "forward_days": horizon_days,
                "pending": False,
            })
        else:
            entry.update({"hit": None, "actual_return": None,
                          "forward_days": horizon_days, "pending": True})
        entry["overlap"] = bool(int(step) < horizon_days)
        anchors.append(entry)

    out["anchors"] = anchors

    scored = [a for a in anchors if not a.get("pending") and a.get("hit") is not None]
    if not scored:
        out["reason"] = "无到期锚点（全部 pending 或推理失败）"
        return out

    rets = [a["actual_return"] for a in scored if _finite(a.get("actual_return"))]
    out.update({
        "available": True,
        "reason": "",
        "anchor_count": len(anchors),
        "scored_count": len(scored),
        "hit_rate": round(sum(1 for a in scored if a["hit"]) / len(scored), 6),
        "mean_return": round(sum(rets) / len(rets), 8) if rets else None,
        "overlap": bool(int(step) < horizon_days),
        "limit_note": ("同模型回看历史、非逐锚点重训；锚点视窗重叠时水平读数"
                       "不可当独立样本。validated=false。"),
    })
    return out


def _predict_score(engine: Any, model_key: str, feats: pd.DataFrame,
                   feature_cols: Sequence[str], row: int) -> Optional[Dict[str, Any]]:
    """对 ``feats`` 的第 ``row`` 行取分（按模型特征名对齐，与推理链路口径一致）。

    直接复用 `_select_by_names`：位置截断会静默错位，且「用值当列名」会静默
    全 0 填充（见 `_align_by_feature_names` 的调用契约注释）—— 回填链路
    绝不能绕开这条修复，否则产出的锚点序列会是一条常数（实测踩过）。
    """
    from src.inference.predictor import _select_by_names

    model_data = engine.models.get(model_key)
    if not isinstance(model_data, dict) or "model" not in model_data:
        return None
    model = model_data["model"]
    scaler = model_data.get("scaler")
    try:
        model_feats = list(getattr(model, "feature_name_", []) or [])
        X = _select_by_names(feats, row, model_feats)
        if X is None:
            X = feats[list(feature_cols)].iloc[row:row + 1].to_numpy(dtype=float)
        X_pred = scaler.transform(X) if scaler is not None else X
        proba = float(model.predict_proba(X_pred)[0][1])
    except Exception as e:  # noqa: BLE001
        logger.debug("[anchor-backfill] 第 %s 行推理失败: %s", row, e)
        return None
    if not math.isfinite(proba):
        return None
    return {
        "probability": proba,
        "direction": "看涨" if proba > 0.5 else "看跌",
        "net_up_probability": proba,
    }


def backfill(engine: Any, symbols: Sequence[str], horizon: str = "short_term",
             step: int = 5, max_anchors: int = 120,
             end_date: Optional[str] = None) -> Dict[str, Any]:
    """多标的锚点回填汇总（逐标失败如实登记，不静默丢弃）。"""
    results: Dict[str, Any] = {}
    ok, skipped = [], []
    pooled_scores: List[float] = []
    pooled_hits: List[bool] = []
    pooled_rets: List[float] = []

    for sym in symbols:
        r = backfill_symbol(engine, sym, horizon=horizon, step=step,
                           max_anchors=max_anchors, end_date=end_date)
        results[str(sym)] = r
        if r.get("available"):
            ok.append(str(sym))
            for a in r["anchors"]:
                if a.get("pending") or a.get("hit") is None:
                    continue
                pooled_scores.append(float(a["score"]))
                pooled_hits.append(bool(a["hit"]))
                if _finite(a.get("actual_return")):
                    pooled_rets.append(float(a["actual_return"]))
        else:
            skipped.append({"symbol": str(sym), "reason": r.get("reason", "未知")})

    return {
        "kind": "trendcast_anchor_backfill/1",
        "horizon": horizon,
        "step": int(step),
        "validated": False,
        "affects_gate": False,
        "readonly": True,
        "symbols_ok": ok,
        "symbols_skipped": skipped,
        "pooled": {
            "anchor_count": len(pooled_scores),
            "hit_rate": (round(sum(pooled_hits) / len(pooled_hits), 6)
                         if pooled_hits else None),
            "mean_return": (round(sum(pooled_rets) / len(pooled_rets), 8)
                            if pooled_rets else None),
        },
        "by_symbol": results,
        "note": ("离线回填读数，用于给下游提供「分数 × 命中 × 已实现收益」的对照；"
                 "同模型回看历史且锚点视窗可能重叠，**不可当独立样本**，"
                 "亦不作为任何放行 / 仓位依据（affects_gate=false）。"),
    }
