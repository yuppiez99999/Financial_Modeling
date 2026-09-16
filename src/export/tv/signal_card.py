"""TradingView 图片信号卡（PNG）：**像素即契约**。

## 设计取向（为什么是「单图承载全部数值 + 元数据 + 锚点」）

TradingView 的图片导入是通过内置「图片仪表盘 / drawing 导入」读**一张静态图**，
它不会被喂 JSON，也不会跑本项目的代码。所以交付物必须做到：

1. **一张图里能读到全部关键读数** —— 三个周期的净看涨概率、综合分、置信度、
   门禁结论、采纳权重、以及可选的锚点走势小图；
2. **元数据不靠肉眼** —— 同一张 PNG 的 ``tEXt`` 块里放一份机器可读 JSON
   （``signal_contract`` 等键），自动化消费方（脚本 / 以后接 Pine 读取）
   可以直接解析，中文按 UTF-8 落盘；
3. **锚点序列折进图里** —— 回测锚点（日期 / 分数 / 命中 / 已实现收益）在图上
   既画成走势小图，也以全精度 JSON 落在 ``anchors_json`` 键，
   让「图看着对」与「数看着对」不会各说各话。

⚠️ 本模块**只画图，不改任何判定**：门禁结论、采纳建议全部来自
``decision_feed`` 契约，逐字段复用（``affects_gate`` 恒为 False）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .png_writer import Canvas, RGB

CARD_W = 1120
INK: RGB = (17, 24, 39)
MUTED: RGB = (90, 102, 120)
LINE: RGB = (206, 214, 226)
BG: RGB = (255, 255, 255)
PANEL: RGB = (247, 249, 252)
UP: RGB = (198, 40, 40)      # A股口径：红涨
DOWN: RGB = (21, 128, 61)    # 绿跌
GATE_OFF: RGB = (146, 64, 14)
GATE_ON: RGB = (21, 128, 61)
ANCHOR_POS: RGB = (198, 40, 40)
ANCHOR_NEG: RGB = (21, 128, 61)


def _fmt_pct(v: Any, digits: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "N/A"
    return f"{f * 100:.{digits}f}%"


def _fmt_num(v: Any, digits: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "N/A"
    return f"{f:.{digits}f}"


def _direction_word(p_up: Optional[float]) -> str:
    if p_up is None:
        return "N/A"
    return "UP" if float(p_up) >= 0.5 else "DOWN"


def _color_for(p_up: Optional[float]) -> RGB:
    if p_up is None:
        return MUTED
    return UP if float(p_up) >= 0.5 else DOWN


# ----------------------------------------------------------------------
# 锚点 -> 坐标
# ----------------------------------------------------------------------
def _anchor_points(anchors: Sequence[Dict[str, Any]], x0: int, y0: int,
                   w: int, h: int) -> List[Tuple[int, int]]:
    """把锚点分数（net up prob ∈ [0,1]）映射到小图坐标（0.5 = 中线）。"""
    usable = [a for a in anchors if isinstance(a, dict)
              and isinstance(a.get("score"), (int, float))]
    if len(usable) < 2:
        return []
    span = max(1, len(usable) - 1)
    pts: List[Tuple[int, int]] = []
    for i, a in enumerate(usable):
        s = min(max(float(a["score"]), 0.0), 1.0)
        x = x0 + int(round(w * i / span))
        y = y0 + h - int(round(h * s))
        pts.append((x, y))
    return pts


def _anchor_stats(anchors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [a for a in anchors if isinstance(a, dict)]
    hits = [a for a in usable if a.get("hit") is True]
    rets = [float(a["actual_return"]) for a in usable
            if isinstance(a.get("actual_return"), (int, float))]
    return {
        "count": len(usable),
        "hit_rate": round(len(hits) / len(usable), 6) if usable else None,
        "mean_return": round(sum(rets) / len(rets), 8) if rets else None,
    }


# ----------------------------------------------------------------------
# 主构建
# ----------------------------------------------------------------------
def build_signal_card(feed: Dict[str, Any], symbol: str,
                      anchors: Optional[Sequence[Dict[str, Any]]] = None,
                      banner: str = "",
                      extra_text: Optional[Dict[str, str]] = None) -> Tuple[bytes, Dict[str, Any]]:
    """构建单标的信号卡 PNG。

    Args:
        feed: ``build_decision_feed`` 的产出（决策源契约）。
        symbol: 要出卡的标的。
        anchors: 回测锚点序列 ``[{date, score, hit, actual_return, horizon}]``。
        banner: 卡片顶部的自定义提示行（例如「NOT VALIDATED」）。
        extra_text: 额外写入 tEXt 的键值对。

    Returns:
        ``(png_bytes, card_meta)`` —— card_meta 是写入 tEXt 的那份 JSON。
    """
    anchors = list(anchors or [])
    preds = list((feed or {}).get("predictions") or [])
    entry = next((p for p in preds if isinstance(p, dict)
                  and str(p.get("symbol")) == str(symbol)), None)

    horizons = ["short_term", "mid_term", "long_term"]
    h_zh = {"short_term": "SHORT 5D", "mid_term": "MID 10D", "long_term": "LONG 20D"}

    agg = (entry or {}).get("aggregate") or {}
    advisory = (entry or {}).get("advisory") or {}
    audit = (feed or {}).get("audit") or {}
    analytics = (feed or {}).get("analytics") or {}
    overall_verdict = ((analytics.get("overall") or {}).get("verdict") or {})

    meta = (feed or {}).get("meta") or {}
    weights = (meta.get("horizon_weights") or {})

    # 高度按锚点区自适应
    anchor_h = 150 if len(anchors) >= 2 else 0
    height = 236 + 78 + anchor_h + 74

    c = Canvas(CARD_W, height, background=BG)

    # ---- 表头
    c.rect(0, 0, CARD_W, 52, (15, 23, 42))
    c.text(20, 12, "TRENDCAST PRO / DECISION SOURCE", (255, 255, 255))
    c.text(20, 30, f"SYMBOL {symbol}   CONTRACT {(feed or {}).get('contract_version', 'N/A')}",
           (148, 163, 184))
    role = str((feed or {}).get("position_role") or "observer").upper()
    c.text(CARD_W - 20 - c.text_width(f"ROLE={role} GATE=OFF"), 12,
           f"ROLE={role} GATE=OFF", GATE_OFF)
    gen = str((feed or {}).get("generated_at") or "")[:19]
    c.text(CARD_W - 20 - c.text_width(f"GENERATED {gen}"), 30,
           f"GENERATED {gen}", (148, 163, 184))

    y = 68
    if banner:
        c.rect(20, y, CARD_W - 40, 22, (254, 243, 199))
        c.text(28, y + 8, banner, GATE_OFF)
        y += 32

    # ---- 三周期行
    c.text(20, y, "HORIZON   NET-UP-PROB   DIR   CALIBRATED   UNCERT   WEIGHT", MUTED)
    y += 14
    c.hline(20, y, CARD_W - 40, LINE)
    y += 8

    for h in horizons:
        info = ((entry or {}).get("horizons") or {}).get(h) or {}
        p_up = info.get("net_up_probability")
        per = (agg.get("per_horizon") or {}).get(h) or {}
        w = per.get("weight", weights.get(h))
        c.text(20, y, h_zh.get(h, h.upper()), INK)
        c.text(120, y, f"{_fmt_pct(p_up)}  ({_fmt_num(p_up)})", _color_for(p_up))
        c.text(300, y, _direction_word(p_up), _color_for(p_up))
        c.text(370, y, ("YES" if info.get("calibration_applied")
                        else "NO") + f"  {_fmt_num(info.get('calibrated_probability'))}", MUTED)
        c.text(540, y, _fmt_num(info.get("uncertainty")), MUTED)
        c.text(660, y, _fmt_num(w, 2), MUTED)
        if info.get("error") or (info == {}):
            c.text(720, y, "MISSING (NOT BACKFILLED WITH 0.5)", GATE_OFF)
        y += 16

    y += 6
    c.hline(20, y, CARD_W - 40, LINE)
    y += 10

    # ---- 聚合 + 采纳建议
    c.rect(20, y, CARD_W - 40, 74, PANEL)
    comp = agg.get("composite_score")
    c.text(32, y + 10, "COMPOSITE SCORE", MUTED)
    c.text(32, y + 26, f"{_fmt_num(comp)}   SIGNED {_fmt_num(agg.get('composite_signed'))}"
                       f"   {_direction_word(comp)}", _color_for(comp))
    cov = agg.get("coverage")
    c.text(32, y + 46, f"COVERAGE {_fmt_num(cov, 2)}   WEIGHTS "
                       f"{_fmt_num(weights.get('short_term'), 2)}/"
                       f"{_fmt_num(weights.get('mid_term'), 2)}/"
                       f"{_fmt_num(weights.get('long_term'), 2)}", MUTED)
    miss = agg.get("missing_horizons") or []
    if miss:
        c.text(32, y + 60, "MISSING " + ",".join(str(m).upper() for m in miss)[:48],
               GATE_OFF)

    consumable = advisory.get("advisory_consumable")
    verdict_txt = "CONSUMABLE (ADVISORY ONLY)" if consumable else "NOT CONSUMABLE"
    verdict_col = (GATE_ON if consumable else GATE_OFF)
    c.text(600, y + 10, "ADVISORY VERDICT", MUTED)
    c.text(600, y + 26, verdict_txt, verdict_col)
    c.text(600, y + 46, f"CONF {_fmt_num(advisory.get('confidence'))}  THR "
                        f"{_fmt_num(advisory.get('recommended_threshold'), 2)}", MUTED)
    c.text(600, y + 60, "AFFECTS-GATE FALSE  POSITION-OBSERVER", GATE_OFF)
    y += 84

    # ---- 证据栏（审计 + 有效性判定）
    c.text(20, y, "EVIDENCE / AUDIT", MUTED)
    c.text(460, y, "EFFECTIVENESS VERDICT", MUTED)
    y += 14
    c.hline(20, y, CARD_W - 40, LINE)
    y += 8
    if audit.get("available"):
        c.text(20, y, f"VERIFIED {audit.get('verified')} / TOTAL {audit.get('total_records')}"
                      f"   HIT {_fmt_pct(audit.get('hit_rate_all'))}", INK)
        c.text(20, y + 14, f"RECENT {audit.get('recent_verified')}  HIT "
                           f"{_fmt_pct(audit.get('recent_hit_rate'))}", MUTED)
    else:
        c.text(20, y, f"AUDIT UNAVAILABLE: {str(audit.get('reason') or '')[:52]}", GATE_OFF)
        c.text(20, y + 14, "NO PODIUM NUMBERS - NOT FABRICATED", GATE_OFF)
    st = str(overall_verdict.get("status") or "insufficient").upper()
    c.text(460, y, f"STATUS {st}  N={overall_verdict.get('n_total', 'N/A')}", 
           GATE_ON if st == "EFFECTIVE" else GATE_OFF)
    c.text(460, y + 14, f"SUBSET HIT {_fmt_pct(overall_verdict.get('subset_hit_rate'))}"
                        f"  RET {_fmt_num(overall_verdict.get('subset_mean_return'))}", MUTED)
    y += 34

    # ---- 锚点小图
    if anchor_h:
        st_anchor = _anchor_stats(anchors)
        c.rect(20, y, CARD_W - 40, anchor_h - 12, PANEL)
        c.text(32, y + 8, "ANCHOR SERIES (BACKFILLED, NOT VALIDATED)", MUTED)
        c.text(660, y + 8, f"N {st_anchor['count']}  HIT {_fmt_pct(st_anchor['hit_rate'])}"
                           f"  MEAN-RET {_fmt_num(st_anchor['mean_return'])}", MUTED)
        gx, gy = 40, y + 30
        gw, gh = CARD_W - 240, anchor_h - 56
        c.rect(gx, gy, gw, gh, (255, 255, 255))
        c.rect(gx, gy, gw, gh, LINE, fill=False)
        mid = gy + gh // 2
        c.hline(gx, mid, gw, LINE)
        pts = _anchor_points(anchors, gx, gy, gw, gh)
        if pts:
            c.polyline(pts, ANCHOR_POS)
            c.rect(pts[-1][0] - 2, pts[-1][1] - 2, 5, 5, INK)
        c.text(gx, gy + gh + 4, "SCORE 0.0-1.0", MUTED)
        # 收益条（正红负绿），画在右侧
        bx = gx + gw + 20
        bar_w = 150
        usable = [a for a in anchors if isinstance(a.get("actual_return"), (int, float))]
        if usable:
            m = max(abs(float(a["actual_return"])) for a in usable) or 1.0
            c.text(bx, y + 30, "ANCHOR RETURNS", MUTED)
            for i, a in enumerate(usable[: max(1, (anchor_h - 60) // 10)]):
                r = float(a["actual_return"])
                half = int(round(bar_w / 2 * min(abs(r) / m, 1.0)))
                by = y + 42 + i * 10
                c.rect(bx + bar_w // 2, by, max(1, half), 6,
                       ANCHOR_POS if r >= 0 else ANCHOR_NEG)
                c.rect(bx + bar_w // 2 - max(1, half), by, max(1, half), 6,
                       ANCHOR_POS if r < 0 else ANCHOR_NEG)
            c.text(bx, y + 42 + max(1, (anchor_h - 60) // 10) * 10 + 2,
                   f"SCALE +/-{_fmt_pct(m)}", MUTED)
        y += anchor_h

    # ---- 页脚
    c.hline(20, y, CARD_W - 40, LINE)
    y += 8
    c.text(20, y, "READ-ONLY OBSERVER OUTPUT. ADVISORY ONLY. NO POSITION SIZING. "
                  "NOT INVESTMENT ADVICE.", GATE_OFF)
    c.text(20, y + 14, "HIGH CONFIDENCE != CONSUMABLE (HIT-RATE AND REALIZED RETURN "
                       "MOVE IN OPPOSITE DIRECTIONS).", MUTED)

    card_meta = {
        "kind": "trendcast_tv_signal_card/1",
        "symbol": symbol,
        "contract_version": (feed or {}).get("contract_version"),
        "generated_at": (feed or {}).get("generated_at"),
        "position_role": (feed or {}).get("position_role"),
        "affects_gate": False,
        "advisory_only": True,
        "net_up_probability": {h: (((entry or {}).get("horizons") or {}).get(h) or {})
                               .get("net_up_probability") for h in horizons},
        "composite_score": comp,
        "composite_signed": agg.get("composite_signed"),
        "coverage": cov,
        "horizon_weights": {h: weights.get(h) for h in horizons},
        "missing_horizons": miss,
        "advisory": {"consumable": consumable,
                     "confidence": advisory.get("confidence"),
                     "recommended_threshold": advisory.get("recommended_threshold")},
        "audit": {"available": audit.get("available"),
                  "verified": audit.get("verified"),
                  "hit_rate_all": audit.get("hit_rate_all")},
        "effectiveness_verdict": st,
        "anchors": {
            "count": _anchor_stats(anchors)["count"],
            "hit_rate": _anchor_stats(anchors)["hit_rate"],
            "mean_return": _anchor_stats(anchors)["mean_return"],
            "validated": False,
            "note": ("锚点为历史回填读数，水平读数不可当独立样本；"
                     "以本卡为交易依据未经人工检查点批准。"),
        },
    }

    texts: Dict[str, str] = {
        # 机器可读：契约摘要（UTF-8）
        "signal_contract": json.dumps(card_meta, ensure_ascii=False, separators=(",", ":")),
        # 机器可读：锚点全精度序列
        "anchors_json": json.dumps(list(anchors), ensure_ascii=False,
                                   separators=(",", ":")),
        # 人读：平台预览里直接能看到
        "Title": f"TrendCast {symbol}",
        "Description": (f"contract={card_meta['contract_version']} "
                        f"composite={comp} verdict={st} "
                        f"role=observer affects_gate=false"),
        "Software": "TrendCast Pro tv-export",
        # 纪律字段：即使调用方（handoff）不再追加，卡片自身也带边界声明
        "Source": "TrendCast Pro decision-feed/1",
        "Boundary": ("position_role=observer; affects_gate=false; "
                     "advisory_only=true; no position sizing; "
                     "not investment advice"),
        "AnchorEvidence": ("validated=false; backfilled history; "
                           "not an independent sample"),
    }
    texts.update(extra_text or {})
    return c.to_png(texts), card_meta


def save_signal_card(feed: Dict[str, Any], symbol: str, path: str | Path,
                     anchors: Optional[Sequence[Dict[str, Any]]] = None,
                     banner: str = "",
                     extra_text: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    png, card_meta = build_signal_card(feed, symbol, anchors=anchors,
                                       banner=banner, extra_text=extra_text)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    card_meta["png_path"] = str(out)
    card_meta["png_bytes"] = len(png)
    return card_meta
