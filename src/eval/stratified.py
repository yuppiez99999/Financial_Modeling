"""按资产类别分池评估（S9）：把「池化口径被木桶短板绑架」这件事真正解决掉。

问题（S7 实测结论的落地）：
  `strategy_gate` 默认 `scope=all`，判定的是**整池 IC 序列**。
  当池子里混着波动结构完全不同的标的（个股 vs 宽基 ETF）时，
  一条序列的 IC 是两类互不相干信号的混合，方向性互相抵消 ——
  表现最好的子池被表现最差的子池稀释掉，**门禁永远卡在门槛线上**。

  S7 已用 `--stratify` 拿到实测证据（个股分层后轻松通过、拖后腿的是宽基 ETF），
  并把「按标的分层建模 / 按资产类别分别设门禁」写成下一步建议。
  S9 把这条建议落成一个**可复算、可审计、无前视**的分池评估层。

本模块做什么：
  - 按资产类别（个股 / ETF / 期货 / 外汇 / 可转债 / 未识别）拆分标的池；
  - 每个分池**独立**跑同一套 walk-forward 口径，各自得 IC / 命中率 / 门禁判定；
  - 产出 `reports/stratified_gate.json`，供监控报表 / API / 日报消费。

本模块**不做什么**（边界比功能重要）：
  - **不改 `strategy_gate` 结论**：默认放行判定仍看整池口径（`scope=all`）。
    分池是"补充证据"，不是"绕过门槛"。想按分池放行必须显式配置
    （`strategy_gate.pool_scope.enabled: true` + `mode: per_class`），
    且**每个分池各自过各自的线**，绝不"取最好看的那个池子"。
  - **不下单、不给仓位**：只标注分池信号的准入状态（`gated` / `readonly`）；
  - **不猜**：分池样本不足 → `available=false`，绝不用"最宽的池子"凑数。

无前视保证：
  walk-forward 切分在每个分池**内部**独立进行（同一套 ``walk_forward_splits``），
  测试折严格在训练折之后；分池只决定"哪些标的/哪些样本进哪条序列"，
  不含任何未来信息（标的分类是静态元数据，不随时间变化）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.eval.asset_class import (
    CLASS_ORDER,
    CLASS_UNKNOWN,
    group_symbols,
    label,
    normalize_symbol,
    resolve_pool,
    summarize_classification,
)
from src.inference.ic import ICCalculator

logger = logging.getLogger(__name__)

REPORT_NAME = "stratified_gate.json"

# 单个分池的最少标的数：1 只也能算 IC（样本来自时间维度），
# 但**门禁判定**需要跨标的的多样性才可信，低于该值只评估不判定。
DEFAULT_MIN_SYMBOLS = 2
# 单分池最少样本数（与 ICCalculator 的 min_samples 无关，后者管单周期）
DEFAULT_MIN_SAMPLES = 30


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (config or {}).get("pool_gate", {}) or {}


class StratifiedEvaluator:
    """按资产类别分池评估 + 门禁判定（纯读、无网络、无副作用）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = _cfg(config)
        gate_cfg = (self.config.get("strategy_gate", {}) or {})
        self.min_symbols = int(cfg.get("min_symbols", DEFAULT_MIN_SYMBOLS))
        self.min_samples = int(cfg.get("min_samples", DEFAULT_MIN_SAMPLES))
        self.ic_calc = ICCalculator(self.config)
        self.gate_dir = Path(gate_cfg.get("report_dir", "reports"))
        self.report_dir = Path(cfg.get("report_dir", self.gate_dir))
        # 显式声明覆盖表：{symbol: asset_class}（识别不了的自定义代码手工归类）
        self.symbol_class_map: Dict[str, str] = dict(cfg.get("symbol_class_map", {}) or {})
        # 分池门禁是否替代整池判定（默认 false = 只做补充证据，不改放行结论）
        pool_scope = (gate_cfg.get("pool_scope", {}) or {})
        self.pool_scope_enabled = bool(pool_scope.get("enabled", False))
        self.pool_scope_mode = str(pool_scope.get("mode", "report_only")).lower()

    # ------------------------------------------------------------------
    def available_classes(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """当前行情数据里实际可评估的分池（空池不出现）。"""
        return group_symbols(list((data or {}).keys()))

    def split(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """按资产类别拆分行情字典（保序，空池不出现）。"""
        return {
            cls: resolve_pool(data, cls, self.symbol_class_map)
            for cls in self.available_classes(data)
        }

    # ------------------------------------------------------------------
    def evaluate_pool(
        self,
        asset_class: str,
        data: Dict[str, Any],
        per_horizon: Dict[str, Dict[str, Any]],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """对单个分池做门禁判定。

        Args:
            per_horizon: ``{horizon: {"horizon_days": int, "scores": [...], "returns": [...],
                           "window_size": int|None}}`` —— 已由调用方按**该分池**的标的
                           构造好的 walk-forward 序列（本模块不训练、不读行情）。

        Returns:
            可直接进 JSON 的分池结果（含每周期指标、门禁判定、reason）。
        """
        symbols = list(symbols if symbols is not None else (data or {}).keys())
        entry: Dict[str, Any] = {
            "asset_class": asset_class,
            "label": label(asset_class),
            "symbols": symbols,
            "symbol_count": len(symbols),
            "horizons": {},
            "passed": False,
            "state": "readonly",
            "available": False,
            "blocked_by": [],
            "reason": "",
        }

        if not symbols:
            entry["reason"] = "该分池当前没有可用标的（分析池为空），不做判定"
            return entry
        if len(symbols) < self.min_symbols:
            entry["reason"] = (
                f"分池标的数 {len(symbols)} < {self.min_symbols}，"
                "只记录指标不判定门禁（单标的的池化 IC 不具备跨标的代表性）"
            )

        horizons_out: Dict[str, Any] = {}
        enough = len(symbols) >= self.min_symbols
        blocked_by: List[str] = []
        total_samples = 0

        for hname, payload in (per_horizon or {}).items():
            scores = list(payload.get("scores") or [])
            returns = list(payload.get("returns") or [])
            res = self.ic_calc.evaluate(
                hname,
                int(payload.get("horizon_days", 0)),
                scores,
                returns,
                window_size=payload.get("window_size"),
                neutral_band=payload.get("neutral_band", 0.0) or 0.0,
            ).to_dict()
            total_samples += int(res.get("samples", 0) or 0)
            horizons_out[hname] = res
            if not res.get("passed"):
                blocked_by.append(f"{hname}: {res.get('reason') or '未达门禁'}")

        entry["horizons"] = horizons_out
        # 报表直接消费的精简列（不塞原始数组，报表只读产物即可渲染）
        entry["ic"] = {h: v.get("ic") for h, v in horizons_out.items()}
        entry["hit_rate"] = {h: v.get("hit_rate") for h, v in horizons_out.items()}
        entry["samples"] = total_samples
        entry["available"] = bool(horizons_out) and total_samples >= self.min_samples

        if not entry["available"]:
            entry["reason"] = entry["reason"] or (
                f"分池有效样本 {total_samples} < {self.min_samples}，样本不足不作判定"
            )
            return entry

        # fail-close：分池样本足够 **且** 每周期都过线，才认为该分池可用
        all_passed = bool(horizons_out) and all(
            h.get("passed") for h in horizons_out.values()
        )
        if not enough:
            entry["passed"] = False
            entry["state"] = "readonly"
            entry["blocked_by"] = blocked_by + [
                f"分池标的数 {len(symbols)} < {self.min_symbols}"
            ]
        else:
            entry["passed"] = all_passed
            entry["state"] = "gated" if all_passed else "readonly"
            entry["blocked_by"] = blocked_by
        entry["reason"] = (
            f"分池已过门禁（{len(horizons_out)} 周期全部达标，标的 {len(symbols)} 只）"
            if entry["passed"]
            else "分池未过门禁，保持只读观测（fail-close）"
        )
        return entry

    # ------------------------------------------------------------------
    def decide(
        self,
        data: Dict[str, Any],
        sequences: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """汇总所有分池的门禁判定。

        Args:
            data      : ``{symbol: DataFrame}``（仅用其键做分池）；
            sequences : ``{asset_class: {horizon: {horizon_days, scores, returns, ...}}}``
                        —— 由 CLI 层用既有 walk-forward 口径构造。

        Returns:
            ``reports/stratified_gate.json`` 的完整 payload。
        """
        groups = self.available_classes(data)
        pools: Dict[str, Any] = {}
        for cls in CLASS_ORDER:
            if cls not in groups:
                continue
            pools[cls] = self.evaluate_pool(
                cls, data, sequences.get(cls, {}) or {}, symbols=groups[cls]
            )

        passed_pools = [c for c, p in pools.items() if p.get("passed")]
        failed = [c for c, p in pools.items() if p.get("available") and not p.get("passed")]
        unavailable = [c for c, p in pools.items() if not p.get("available")]

        # 分池放行：只有显式开启且模式为 per_class 时才影响判定；
        # 即便开启，也是"每个分池各自过线"，不取最好看的那个池子代表全体。
        per_class_enabled = self.pool_scope_enabled and self.pool_scope_mode == "per_class"
        pooled_state = "readonly"
        if per_class_enabled and pools:
            # 所有**可判定**的分池都过线才算整体放行；存在不可判定分池 → fail-close
            pooled_state = (
                "gated" if not failed and not unavailable and passed_pools == list(pools)
                else "readonly"
            )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "src/eval/stratified.py",
            "classification": summarize_classification(list((data or {}).keys())),
            "min_symbols": self.min_symbols,
            "min_samples": self.min_samples,
            "thresholds": {
                "min_ic": self.ic_calc.min_ic,
                "min_hit_rate": self.ic_calc.min_hit_rate,
                "min_samples": self.ic_calc.min_samples,
                "min_windows": self.ic_calc.min_windows,
            },
            "pools": pools,
            "passed_pools": passed_pools,
            "failed_pools": failed,
            "unavailable_pools": unavailable,
            # ⚠️ 默认 report_only：分池结果**不改变** `strategy_gate` 的放行结论
            "pool_scope": {
                "enabled": self.pool_scope_enabled,
                "mode": self.pool_scope_mode,
                "affects_gate": per_class_enabled,
                "state": pooled_state if per_class_enabled else None,
            },
            "affects_gate": per_class_enabled,
        }

    # ------------------------------------------------------------------
    def save(self, payload: Dict[str, Any], path: Optional[str] = None) -> Path:
        """落盘分池报告（监控报表 / API 只读消费）。"""
        out = Path(path) if path else (self.report_dir / REPORT_NAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """读取既有分池报告（纯读，缺失返回 available=False，绝不臆测）。"""
        src = Path(path) if path else (self.report_dir / REPORT_NAME)
        if not src.exists():
            return {
                "available": False,
                "reason": "no_stratified_report",
                "hint": "先运行 `python main.py ic --pool` 生成 reports/stratified_gate.json",
            }
        try:
            payload = json.loads(src.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[pool] 分池报告读取失败: {e}")
            return {"available": False, "error": str(e)}
        payload["available"] = True
        payload["report_path"] = str(src)
        return payload

    # ------------------------------------------------------------------
    def summarize_pools(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """把分池 payload 压成报表行（不塞原始序列）。"""
        rows: List[Dict[str, Any]] = []
        for cls, pool in sorted(
            (payload.get("pools") or {}).items(),
            key=lambda kv: CLASS_ORDER.index(kv[0]) if kv[0] in CLASS_ORDER else 99,
        ):
            fail_horizons = [
                h for h, v in (pool.get("horizons") or {}).items() if not v.get("passed")
            ]
            rows.append({
                "asset_class": cls,
                "label": pool.get("label") or label(cls),
                "symbol_count": pool.get("symbol_count", 0),
                "samples": pool.get("samples", 0),
                "available": bool(pool.get("available")),
                "passed": bool(pool.get("passed")),
                "state": pool.get("state", "readonly"),
                "failed_horizons": fail_horizons,
                "reason": pool.get("reason", ""),
                "ic": {
                    h: v.get("ic")
                    for h, v in (pool.get("horizons") or {}).items()
                },
                "hit_rate": {
                    h: v.get("hit_rate")
                    for h, v in (pool.get("horizons") or {}).items()
                },
            })
        return rows


def compare_with_pooled(payload: Dict[str, Any],
                        pooled: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把分池结果与整池口径对照，回答「是不是被木桶短板拖死的」。

    Args:
        payload: 分池 payload；
        pooled : 整池 IC payload（``reports/ic_report.json`` 的内容）。

    Returns:
        ``{"available": bool, "pooled_passed": bool|None, "best_pool": str|None,
           "passed_pools": [...], "failed_pools": [...], "narrative": str}``
        —— `narrative` 是**如实**的一句话结论，不做任何"已解锁"的暗示。
    """
    pools = payload.get("pools") or {}
    passed = list(payload.get("passed_pools") or [])
    failed = list(payload.get("failed_pools") or [])
    pooled_passed = None
    if pooled:
        pooled_passed = bool(
            (pooled.get("all_passed") if "all_passed" in pooled
             else all(h.get("passed") for h in (pooled.get("horizons") or {}).values()))
        )
    if not pools:
        return {
            "available": False, "pooled_passed": pooled_passed,
            "best_pool": None, "passed_pools": [], "failed_pools": [],
            "narrative": "分池报告为空，无法对照（先跑 `python main.py ic --pool`）",
        }

    best = None
    best_key = None
    for cls, pool in pools.items():
        if not pool.get("available"):
            continue
        key = min(
            [float(v.get("hit_rate", 0.0) or 0.0)
             for v in (pool.get("horizons") or {}).values()] or [0.0]
        )
        if best_key is None or key > best_key:
            best, best_key = cls, key

    parts: List[str] = []
    if pooled_passed is False and passed:
        parts.append(
            f"整池口径未放行，但 {len(passed)} 个分池各自过关"
            f"（{'、'.join(label(c) for c in passed)}）—— 说明整池被拖后腿的分池稀释"
        )
    elif pooled_passed is False and not passed:
        parts.append("整池与分池口径均未放行：信号强度不足的问题不是分池造成的")
    elif pooled_passed is True:
        parts.append("整池口径已放行，分池仅作结构核对")
    if failed:
        parts.append(f"仍未过关的分池：{'、'.join(label(c) for c in failed)}")
    if not parts:
        parts.append("分池结果不足以给出对照结论")

    return {
        "available": True,
        "pooled_passed": pooled_passed,
        "best_pool": best,
        "best_pool_label": label(best) if best else None,
        "passed_pools": passed,
        "failed_pools": failed,
        "narrative": "；".join(parts) + "。分池结论不改变整池门禁判定（默认 report_only）。",
    }


def build_sequences_for_pools(
    data: Dict[str, Any],
    splits: Dict[str, Dict[str, Any]],
    builder: Callable[[Dict[str, Any]], Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """按分池逐池构造 walk-forward 序列。

    Args:
        data    : ``{symbol: DataFrame}``；
        splits  : ``{asset_class: {symbol: DataFrame}}``（``StratifiedEvaluator.split`` 的输出）；
        builder : 单池 → 序列的回调（由 CLI 注入既有 ``build_walkforward_sequences``，
                  保证与整池门禁**完全同源**，不会出现"两套数字互相打架"）。

    Returns:
        ``{asset_class: {horizon: {...}}}`` —— 空池返回空 dict（不伪造序列）。
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cls, pool_data in (splits or {}).items():
        if not pool_data:
            continue
        try:
            out[cls] = builder(pool_data)
        except Exception as e:  # noqa: BLE001 - 单池失败不得拖垮其它分池
            logger.warning(f"[pool] 分池 {cls} 序列构造失败: {e}")
            out[cls] = {}
    return out


def classify_unknown_symbols(symbols) -> List[str]:
    """返回未能识别的标的（供 CLI/报表提示人工补充 ``pool_gate.symbol_class_map``）。"""
    return [normalize_symbol(s) for s in (symbols or [])
            if group_symbols([s]).get(CLASS_UNKNOWN)]
