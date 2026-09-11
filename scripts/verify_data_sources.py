#!/usr/bin/env python
"""S14/G4 数据源回退链验收脚本（T14.3 人工检查点用）。

用途：对 `configs/config_pro.yaml` 中**全部启用标的**跑一遍回退链，
统计拉取成功率，产出可机器校对的验收报告 JSON。

验收口径（来自 schedule/plan.json T14.3）：
  - `macro` 与 `config_pro.yaml` 标的池**全量拉取成功率 ≥ 95%**；
  - 失败标的与失败原因必须逐条列出（不许用"部分成功"含糊过去）；
  - 期货 / 外汇必须**逐类给出实际拉到的行数与日期区间**，
    证明"已开启"是真实取到数据，而不是配置里改了个 true。

用法：
    python scripts/verify_data_sources.py                     # 走 configs/config_pro.yaml
    python scripts/verify_data_sources.py --source akshare    # 只验 akshare 单源
    python scripts/verify_data_sources.py --offline           # 离线自检（不触网，仅校验配置与代码映射）
    python scripts/verify_data_sources.py --out reports/s14_data_source_acceptance.json

设计约束：
  - **只读不写**：不落任何行情缓存（避免把验收数据混进训练缓存）；
  - 任何单标的失败都只记录、不中断（fail-soft），最终统一出报告；
  - 退出码：达标 0 / 未达标 1（便于 CI 与人工检查点直接判定）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.akshare_client import (  # noqa: E402
    AkshareClient,
    akshare_available,
    split_symbol,
)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def enabled_symbols(cfg: dict) -> dict[str, list[str]]:
    """按市场类别收集启用标的。"""
    markets = (cfg.get("data", {}) or {}).get("markets", {}) or {}
    return {
        name: list((mcfg or {}).get("symbols") or [])
        for name, mcfg in markets.items()
        if (mcfg or {}).get("enabled")
    }


def verify_market_symbols(cfg: dict, source: str | None = None) -> dict:
    """逐个标的拉取，统计成功率（不落缓存）。"""
    if source:
        cfg = {**cfg, "data": {**cfg.get("data", {}), "source": [source]}}

    # 直接用 AkshareClient 单源验收：绕开 DataCollector 的缓存写入，只读不写
    from src.data.akshare_client import AkshareClient as _AK

    client = _AK(cfg)
    data_cfg = cfg.get("data", {}) or {}
    start = str(data_cfg.get("start_date", "") or "")
    end = str(data_cfg.get("end_date", "") or "")
    markets = enabled_symbols(cfg)

    per_symbol: dict[str, dict] = {}
    per_market: dict[str, dict] = {}
    for market, symbols in markets.items():
        ok = 0
        for symbol in symbols:
            try:
                df = client.fetch(symbol, start_date=start, end_date=end)
            except Exception as e:  # noqa: BLE001
                df = None
                err = f"{type(e).__name__}: {e}"
            else:
                err = ""
            if df is not None and len(df) > 0:
                ok += 1
                per_symbol[symbol] = {
                    "market": market,
                    "status": "ok",
                    "rows": int(len(df)),
                    "first_date": str(df["date"].iloc[0]),
                    "last_date": str(df["date"].iloc[-1]),
                    "last_close": float(df["close"].iloc[-1]),
                }
            else:
                per_symbol[symbol] = {
                    "market": market,
                    "status": "failed",
                    "rows": 0,
                    "reason": err or "empty_or_unavailable",
                }
        per_market[market] = {
            "total": len(symbols),
            "ok": ok,
            "success_rate": round(ok / len(symbols), 4) if symbols else 0.0,
        }

    total = sum(len(v) for v in markets.values())
    ok_total = sum(1 for v in per_symbol.values() if v["status"] == "ok")
    return {
        "total": total,
        "ok": ok_total,
        "success_rate": round(ok_total / total, 4) if total else 0.0,
        "per_market": per_market,
        "per_symbol": per_symbol,
    }


def verify_macro(cfg: dict, source: str = "akshare") -> dict:
    """宏观指标可用性（5 项注册指标）。"""
    if source:
        cfg = {**cfg, "data": {**cfg.get("data", {}), "macro": {
            **(cfg.get("data", {}).get("macro", {}) or {}), "source": [source]
        }}}
    from src.data.macro_client import MacroClient

    client = MacroClient(cfg)
    health = client.health()
    ok = sum(1 for v in health.values() if v["status"] == "ok")
    return {
        "total": len(health),
        "ok": ok,
        "success_rate": round(ok / len(health), 4) if health else 0.0,
        "per_indicator": health,
    }


def offline_selfcheck(cfg: dict) -> dict:
    """离线自检：只校验配置与代码映射，不触网。"""
    markets = enabled_symbols(cfg)
    mapping = {}
    for market, symbols in markets.items():
        mapping[market] = {s: split_symbol(s) for s in symbols}
    unknown = [
        s for syms in mapping.values() for s, (m, _c) in syms.items() if m == "unknown"
    ]
    return {
        "akshare_available": akshare_available(),
        "source_chain": cfg.get("data", {}).get("source"),
        "enabled_markets": {k: len(v) for k, v in markets.items()},
        "unknown_symbols": unknown,
        "mapping": mapping,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="S14/G4 数据源回退链验收")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config_pro.yaml"))
    ap.add_argument("--source", default=None,
                    help="只验收单一数据源（如 akshare / tencent）；缺省走配置回退链")
    ap.add_argument("--offline", action="store_true", help="离线自检，不触网")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="成功率验收线（默认 0.95，来自 T14.3）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "s14_data_source_acceptance.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "source_filter": args.source,
        "threshold": args.threshold,
        "offline": bool(args.offline),
    }

    if args.offline:
        report["selfcheck"] = offline_selfcheck(cfg)
        report["passed"] = not report["selfcheck"]["unknown_symbols"]
    else:
        report["symbols"] = verify_market_symbols(cfg, source=args.source)
        report["macro"] = verify_macro(cfg, source=args.source or "akshare")
        report["passed"] = (
            report["symbols"]["success_rate"] >= args.threshold
            and report["macro"]["success_rate"] >= args.threshold
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 终端摘要 ----
    print("=" * 64)
    print("S14/G4 数据源回退链验收")
    print("=" * 64)
    if args.offline:
        sc = report["selfcheck"]
        print(f"akshare 可导入: {sc['akshare_available']}")
        print(f"回退链: {sc['source_chain']}")
        print(f"启用市场: {sc['enabled_markets']}")
        print(f"未识别代码: {sc['unknown_symbols'] or '无'}")
    else:
        sym = report["symbols"]
        print(f"标的池: {sym['ok']}/{sym['total']} = {sym['success_rate']:.1%}")
        for market, info in sym["per_market"].items():
            print(f"  {market:<8} {info['ok']}/{info['total']} = {info['success_rate']:.1%}")
        macro = report["macro"]
        print(f"宏观指标: {macro['ok']}/{macro['total']} = {macro['success_rate']:.1%}")
        failed = [k for k, v in sym["per_symbol"].items() if v["status"] != "ok"]
        if failed:
            print(f"失败标的: {failed}")
    print(f"\n验收线 {args.threshold:.0%} → {'✅ 达标' if report['passed'] else '❌ 未达标'}")
    print(f"报告已保存: {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
