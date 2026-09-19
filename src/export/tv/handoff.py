"""TradingView 一次交付编排：契约 → 图片信号卡 + Pine 数据层 + JSON 报告。

一条命令产出下游「拿来即用」的三件套：

```
outputs/tv/exports/
├── signals/000001.SZ.png      # 图片信号卡（tEXt 内嵌机器可读契约 + 锚点）
├── signals/index.json         # 卡片清单（symbol → png → 内嵌键）
├── pine/index.json            # Pine 数据层清单（seed_key 一览）
├── pine/trendcast/000001.SZ.json   # request.seed 直接读的 JSON
└── handoff_report.json        # 全量报告（数值 / 路径 / 纪律声明）
```

设计约束：
- **不重算**：所有数值来自同一个 ``decision_feed`` 产出，卡片与 JSON 必然一致；
- **不触网**：数据来自 ``data/raw/`` 本地缓存（真实日K），无缓存即如实标注缺口；
- **不改门禁**：``affects_gate`` 恒 False，报告里逐字段声明。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .pine_json import PINE_CONTRACT, write_pine_payloads
from .signal_card import save_signal_card

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "outputs/tv/exports"

NOT_VALIDATED_BANNER = ("NOT VALIDATED - READ-ONLY OBSERVER OUTPUT / "
                        "ADVISORY ONLY / NO POSITION SIZING")


def build_handoff(feed: Dict[str, Any],
                  anchors_by_symbol: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
                  out_dir: str | Path = DEFAULT_OUT_DIR,
                  symbols: Optional[Sequence[str]] = None,
                  banner: str = NOT_VALIDATED_BANNER,
                  card_limit: Optional[int] = None,
                  cards_for_unavailable: bool = True) -> Dict[str, Any]:
    """生成 TradingView 交付三件套并落盘。

    Args:
        feed: ``build_decision_feed`` 产出。
        anchors_by_symbol: ``{symbol: [anchor, ...]}`` 锚点序列（可空）。
        out_dir: 输出根目录。
        symbols: 只出这些标的（缺省 = 契约内全部）。
        banner: 卡片顶部提示行。
        card_limit: 卡片数量上限（给「只出重点持仓」的场景；None = 全出）。
        cards_for_unavailable: 契约里「三周期全缺 / 聚合不可用」的标的是否也出卡。
            缺省 True —— 出一张**如实标注缺失**的卡（下游能看到"这条没数据"，
            比整条消失更难被误读）；置 False 则只出有读数的卡。

    Returns:
        交付报告 dict（同时落盘 ``handoff_report.json``）。
    """
    anchors_by_symbol = anchors_by_symbol or {}
    base = Path(out_dir)
    sig_dir = base / "signals"
    pine_dir = base / "pine"
    sig_dir.mkdir(parents=True, exist_ok=True)
    pine_dir.mkdir(parents=True, exist_ok=True)

    preds = [p for p in ((feed or {}).get("predictions") or []) if isinstance(p, dict)]
    available = {str(p.get("symbol")) for p in preds
                 if isinstance(p.get("aggregate"), dict) and p["aggregate"].get("available")}
    have = {str(p.get("symbol")) for p in preds}
    if symbols:
        want = [str(s) for s in symbols]
    elif cards_for_unavailable:
        want = [str(p.get("symbol")) for p in preds if p.get("symbol")]
    else:
        want = [str(p.get("symbol")) for p in preds if p.get("symbol")
                and str(p.get("symbol")) in available]
    if card_limit is not None:
        want = want[: int(card_limit)]

    cards: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for sym in want:
        if not cards_for_unavailable and sym not in available:
            skipped.append({"symbol": sym,
                            "reason": "无可用聚合（三周期全缺），按配置不出卡"})
            continue
        anchors = anchors_by_symbol.get(sym) or []
        try:
            meta = save_signal_card(
                feed, sym, sig_dir / f"{sym}.png", anchors=anchors, banner=banner,
                extra_text={
                    "Source": "TrendCast Pro decision-feed/1",
                    "Boundary": ("observer output; affects_gate=false; "
                                 "no position sizing; not investment advice"),
                    "AnchorEvidence": ("validated=false; backfilled history; "
                                       "not an independent sample"),
                })
            cards.append(meta)
        except Exception as e:  # noqa: BLE001 - 单卡失败不得打断整批
            logger.warning("[tv-handoff] 信号卡生成失败 %s: %s", sym, e)
            skipped.append({"symbol": sym, "reason": f"卡片生成失败: {e}"})

    # 请求但不可交付的标的，如实列出（不静默丢弃）
    for sym in want:
        if sym not in have:
            skipped.append({"symbol": sym,
                            "reason": "契约中无该标的预测（缺数据或未加载模型）"})
        elif sym not in available and not cards_for_unavailable:
            skipped.append({"symbol": sym,
                            "reason": "无可用聚合（三周期全缺），按配置不出卡"})

    pine_written: List[Dict[str, Any]] = []
    try:
        pine_written = write_pine_payloads(feed, pine_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("[tv-handoff] Pine 数据层写出失败: %s", e)

    (sig_dir / "index.json").write_text(json.dumps({
        "kind": "trendcast_tv_signal_cards/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cards": [{"symbol": c["symbol"], "png": c.get("png_path"),
                   "bytes": c.get("png_bytes"),
                   "embedded_keys": ["signal_contract", "anchors_json",
                                     "Title", "Description", "Source", "Boundary"],
                   "composite_score": c.get("composite_score"),
                   "advisory": c.get("advisory")} for c in cards],
        "meta": {"affects_gate": False, "advisory_only": True,
                 "position_role": "observer"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = (feed or {}).get("meta") or {}
    report = {
        "kind": "trendcast_tv_handoff/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_contract": (feed or {}).get("contract_version"),
        "pine_contract": PINE_CONTRACT,
        "out_dir": str(base),
        "card_count": len(cards),
        "cards": cards,
        "skipped": skipped,
        "pine_files": pine_written,
        "pine_dir": str(pine_dir),
        "signals_dir": str(sig_dir),
        "feed_summary": {
            "symbol_count": meta.get("symbol_count"),
            "scored_count": meta.get("scored_count"),
            "advisory_consumable_count": meta.get("advisory_consumable_count"),
        },
        "boundary": {
            "position_role": "observer",
            "affects_gate": False,
            "advisory_only": True,
            "no_position_sizing": True,
            "not_investment_advice": True,
            "note": ("卡片与 Pine 数据层均为只读观测；高置信 ≠ 可采信 "
                     "（命中率与平均已实现收益反向）。以本交付为交易依据"
                     "须经人工检查点批准。"),
        },
    }
    (base / "handoff_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[tv-handoff] 信号卡 %d 张 / Pine 文件 %d 个 → %s",
                len(cards), len(pine_written), base)
    return report
