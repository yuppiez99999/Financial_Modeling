"""Pine 外挂 JSON 契约层：``tv-pine/1``。

## 为什么是「按时间对齐的列 + 列字典」而不是原契约直出

Pine Script 消费外部 JSON 的**唯一近原生**路径是「变量 × 时序」的表结构：

```pine
//@version=6
import TradingView/ta/8 as ta
var matrix<float> m = request.seed("TRENDCAST_000001_SZ", "trendcast/000001.SZ.json")
```

``request.seed`` 把 JSON 读成一个矩阵：**每列一个变量，单数列 = 时间戳**。
因此契约层的三个硬约束：

1. ``ret_*`` 列**优先**（``ret_composite`` / ``ret_short_term`` / ...），
   下游 Pine 公式直接读；
2. 每列都有对应的 ``columns[]`` 条目（``id`` 必须与列名逐字一致）——
   列字典是 Pine 侧对齐的依据，缺了就只能靠位置猜；
3. **门禁类读数刻意不出口**（``is_trade`` / 仓位 / 权重都不在列里）：
   本层的用途是「把 16_ 工厂搬到 TradingView 的同一张图上做对照」，
   不是「让 TradingView 按这个下单」。纪律写在 ``meta`` 里而不是写在文档里 ——
   下游图里能看见 ``affects_gate: false``。

与决策源契约的关系：本层是 ``decision-feed/1`` 的**投影**，不是新口径。
同一个 ``build_decision_feed`` 产出 → 投影成列，数值逐字段一致
（``tests/test_tv_export.py`` 用契约值反查列值）。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PINE_CONTRACT = "tv-pine/1"

# 列顺序即矩阵列序；ret_* 在前，Pine 侧按名取列时顺序无关但人读更直观
_BASE_COLUMNS: Sequence[str] = (
    "ret_composite",      # 综合分（净看涨概率 ∈ [0,1]）
    "ret_composite_signed",  # (score-0.5)*2 ∈ [-1,1]，Pine 画零轴的图更省事
    "ret_short_term",
    "ret_mid_term",
    "ret_long_term",
    "ret_confidence",
    "ret_coverage",
    "ret_calibrated_composite",
)

_HORIZON_KEYS = ("short_term", "mid_term", "long_term")


def _ts_slug(symbol: str) -> str:
    """标的 → 文件名安全的 slug（Pine ``request.seed`` 的键）。"""
    return str(symbol).replace("/", "_").replace("\\", "_")


def _num(v: Any, digits: int = 6) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, digits)


def build_pine_payload(feed: Dict[str, Any],
                       as_of: Optional[str] = None) -> Dict[str, Any]:
    """把决策源契约投影成 ``tv-pine/1`` 载荷。

    每个标的一条记录：``{symbol, data: [{ts, ret_*...}], meta}``。
    合约缺失的周期**不写列值**（``None``），filler 不补 0.5 ——
    与契约层同一条纪律（0.5 是有效中性读数，不能冒充缺失）。
    """
    preds = list((feed or {}).get("predictions") or [])
    ts = as_of or str((feed or {}).get("generated_at") or "") or \
        datetime.now(timezone.utc).isoformat(timespec="seconds")

    records: List[Dict[str, Any]] = []
    columns: List[Dict[str, str]] = [
        {"id": "ts", "name": "ts", "type": "string",
         "desc": "读数时间（契约 generated_at），单数列，Pine 对齐用"},
    ]
    desc = {
        "ret_composite": "综合分 = 多周期净看涨概率加权（∈[0,1]）",
        "ret_composite_signed": "(composite-0.5)*2 ∈[-1,1]，正=看多",
        "ret_short_term": "5 日周期净看涨概率（∈[0,1]）",
        "ret_mid_term": "10 日周期净看涨概率（∈[0,1]）",
        "ret_long_term": "20 日周期净看涨概率（∈[0,1]）",
        "ret_confidence": "置信度 = |composite-0.5|*2（与 16_ 同口径）",
        "ret_coverage": "可用周期权重覆盖率（缺失周期不补 0.5）",
        "ret_calibrated_composite": "校准后综合分（未校准时缺列，不猜）",
    }
    columns += [{"id": c, "name": c, "type": "float", "desc": desc[c]}
                for c in _BASE_COLUMNS]

    for item in preds:
        if not isinstance(item, dict):
            continue
        agg = item.get("aggregate") or {}
        advisory = item.get("advisory") or {}
        horizons = item.get("horizons") or {}
        if not agg.get("available"):
            continue    # 无可用聚合 → 不出口该标的（不造 0.5）
        row = {
            "ts": ts,
            "ret_composite": _num(agg.get("composite_score")),
            "ret_composite_signed": _num(agg.get("composite_signed")),
            "ret_short_term": _num((horizons.get("short_term") or {}).get("net_up_probability")),
            "ret_mid_term": _num((horizons.get("mid_term") or {}).get("net_up_probability")),
            "ret_long_term": _num((horizons.get("long_term") or {}).get("net_up_probability")),
            "ret_confidence": _num(advisory.get("confidence")),
            "ret_coverage": _num(agg.get("coverage")),
            "ret_calibrated_composite": _num(advisory.get("calibrated_composite_score")),
        }
        records.append({
            "symbol": item.get("symbol"),
            "seed_key": f"trendcast/{_ts_slug(item.get('symbol'))}.json",
            "data": [row],
            "meta": {
                "consumable": bool(advisory.get("advisory_consumable")),
                "affects_gate": False,
                "advisory_only": True,
                "missing_horizons": agg.get("missing_horizons") or [],
            },
        })

    return {
        "contract_version": PINE_CONTRACT,
        "source_contract": (feed or {}).get("contract_version"),
        "generated_at": (feed or {}).get("generated_at"),
        "seeded_at": ts,
        "columns": columns,
        "symbols": records,
        "meta": {
            "position_role": "observer",
            "affects_gate": False,
            "advisory_only": True,
            "ret_primacy": ("以下列名以 ret_ 前缀可被 Pine 侧按前缀批量绑定；"
                            "列值全部来自 decision-feed/1，未做二次换算。"),
            "consumption": (
                "import TradingView/ta/8 as ta\n"
                "var matrix<float> m = request.seed(\"TRENDCAST_000001_SZ\", "
                "\"trendcast/000001.SZ.json\")\n"
                "if m.size() > 0\n"
                "    float comp = m.get(1, m.rows() - 1)\n"
                "    plot(comp * 100, \"composite %\", color.blue)"
            ),
            "boundary": ("本层不出口 is_trade / 仓位 / 权重——"
                         "TradingView 侧只做同图对照，不做下单依据。"),
        },
    }


def build_pine_payloads(feed: Dict[str, Any],
                        as_of: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """按标的拆分的载荷：``{seed_key: payload}``，一个文件一个 seed。"""
    whole = build_pine_payload(feed, as_of=as_of)
    out: Dict[str, Dict[str, Any]] = {}
    for rec in whole["symbols"]:
        out[rec["seed_key"]] = {
            "contract_version": PINE_CONTRACT,
            "source_contract": whole["source_contract"],
            "generated_at": whole["generated_at"],
            "seed_key": rec["seed_key"],
            "columns": whole["columns"],
            "symbol": rec["symbol"],
            "data": rec["data"],
            "meta": dict(whole["meta"], symbol_meta=rec["meta"]),
        }
    return out


def write_pine_payloads(feed: Dict[str, Any], out_dir: str | Path,
                        as_of: Optional[str] = None) -> List[Dict[str, Any]]:
    """落盘 ``trendcast/<symbol>.json`` + ``trendcast/index.json``。

    Returns: ``[{symbol, seed_key, path, rows, columns}]``（供报告引用）。
    """
    base = Path(out_dir)
    (base / "trendcast").mkdir(parents=True, exist_ok=True)
    payloads = build_pine_payloads(feed, as_of=as_of)
    written: List[Dict[str, Any]] = []
    for seed_key, payload in payloads.items():
        path = base / seed_key
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        written.append({
            "symbol": payload["symbol"],
            "seed_key": seed_key,
            "path": str(path),
            "rows": len(payload["data"]),
            "columns": [c["id"] for c in payload["columns"]],
        })
    index = {
        "contract_version": PINE_CONTRACT,
        "source_contract": (feed or {}).get("contract_version"),
        "generated_at": (feed or {}).get("generated_at"),
        "columns_contract": {
            "ts": "单数列（Pine 时间对齐）",
            "ret_*": "数值列（可被 Pine 按前缀批量绑定）",
        },
        "symbols": [{"symbol": w["symbol"], "seed_key": w["seed_key"],
                     "rows": w["rows"]} for w in written],
        "meta": {
            "position_role": "observer",
            "affects_gate": False,
            "advisory_only": True,
        },
    }
    (base / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return written
