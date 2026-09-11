"""门禁阻塞诊断：把「未放行」拆成可执行的短板清单。

背景（Q2 路线图遗留 → S7 门禁解锁攻坚）：
  实测门禁为 `readonly` 时，`strategy_gate.json` 只给一句「命中率 51.37% < 52%」。
  人看到的是一个数字，但不知道**还差多少、哪个周期是约束、改动值不值得**。
  本模块把阻塞原因结构化：

  - `shortfall`  ：每个未达标指标距离门槛的绝对差距（可排序、可追踪）；
  - `binding`    ：当前真正卡住门禁的周期/指标（scope=all 下最差的那条腿）；
  - `headroom`   ：已达标指标超出门槛的余量（判断"要不要动它"）；
  - `total_gap`  ：各指标归一化差距之和，作为"离放行还有多远"的单一标量。

设计原则（与 `StrategyGate` 一致）：
  - **纯读**：不训练、不触网、不改门禁结论，只解释既有 payload；
  - **可复算**：同一 payload 无论何时执行结论一致（无随机、无时间依赖）；
  - **不猜**：数据缺失 / 样本不足一律如实标注 `available: false`，不给假门槛差。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 门禁实际校验的三个指标（与 src/inference/ic.py 的 ICResult.passed 判定一致）。
# icir 当前不参与门禁，故不进短板清单，避免"诊断出要优化一个不影响放行的指标"。
GATED_METRICS = ("ic", "hit_rate", "windows")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # 过滤 NaN


def _horizon_thresholds(horizon: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, float]:
    """取单周期阈值：优先用 payload 内嵌的 thresholds，缺省回落到门禁全局阈值。"""
    raw = horizon.get("thresholds") or {}
    return {
        "min_ic": abs(_num(raw.get("min_ic", defaults.get("min_ic", 0.03)))),
        "min_hit_rate": _num(raw.get("min_hit_rate", defaults.get("min_hit_rate", 0.52))),
        "min_windows": _num(raw.get("min_windows", defaults.get("min_windows", 3))),
    }


def diagnose_horizon(name: str, horizon: Dict[str, Any],
                     defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """诊断单个周期的门禁短板（纯函数）。

    返回结构（可直接进日报 / API / 监控报表）::

        {
          "horizon": "short_term",
          "passed": false,
          "available": true,
          "binding_metric": "hit_rate",     # 归一化差距最大的指标 = 真正的短板
          "shortfall": {"hit_rate": 0.0063}, # 还差多少（绝对量）
          "headroom": {"ic": 0.01},          # 超出门槛多少
          "gaps": {"hit_rate": 0.31, ...},   # 归一化差距（0 = 刚好达标）
          "total_gap": 0.31,
          "reason": "命中率 51.37% < 52%",
        }
    """
    defaults = defaults or {}
    th = _horizon_thresholds(horizon, defaults)
    available = bool(horizon.get("available", True))
    passed = bool(horizon.get("passed"))

    out: Dict[str, Any] = {
        "horizon": name,
        "passed": passed,
        "available": available,
        "horizon_days": int(_num(horizon.get("horizon_days"), 0)),
        "binding_metric": None,
        "shortfall": {},
        "headroom": {},
        "gaps": {},
        "total_gap": 0.0,
        "reason": horizon.get("reason", ""),
    }
    if not available:
        out["reason"] = out["reason"] or "样本不足，无法判定（先补样本再谈门槛）"
        return out
    if passed:
        return out

    samples = int(_num(horizon.get("samples"), 0))
    if samples <= 0:
        out["available"] = False
        out["reason"] = out["reason"] or "无有效样本，门禁不可判定"
        return out

    # --- IC：看 |IC| 与门槛的差距 ---
    ic = abs(_num(horizon.get("ic")))
    if ic >= th["min_ic"]:
        out["headroom"]["ic"] = round(ic - th["min_ic"], 6)
    else:
        out["shortfall"]["ic"] = round(th["min_ic"] - ic, 6)
        out["gaps"]["ic"] = (th["min_ic"] - ic) / th["min_ic"] if th["min_ic"] else 0.0

    # --- 命中率 ---
    hr = _num(horizon.get("hit_rate"))
    if hr >= th["min_hit_rate"]:
        out["headroom"]["hit_rate"] = round(hr - th["min_hit_rate"], 6)
    else:
        out["shortfall"]["hit_rate"] = round(th["min_hit_rate"] - hr, 6)
        out["gaps"]["hit_rate"] = (
            (th["min_hit_rate"] - hr) / th["min_hit_rate"] if th["min_hit_rate"] else 0.0
        )

    # --- 有效窗口数（仅当调用方请求了窗口统计，即 windows>0 才有意义）---
    windows = int(_num(horizon.get("windows"), 0))
    if windows > 0:
        if windows >= th["min_windows"]:
            out["headroom"]["windows"] = windows - th["min_windows"]
        else:
            out["shortfall"]["windows"] = th["min_windows"] - windows
            meta = horizon.get("thresholds") or {}
            # 只有 payload 明确带窗口门槛时才算短板，避免把"没算 ICIR"误判成短板
            if _num(meta.get("min_windows"), 0) > 0:
                out["gaps"]["windows"] = (
                    (th["min_windows"] - windows) / th["min_windows"] if th["min_windows"] else 0.0
                )

    if out["gaps"]:
        binding = max(out["gaps"], key=lambda k: out["gaps"][k])
        out["binding_metric"] = binding
        out["total_gap"] = round(sum(out["gaps"].values()), 6)
    return out


def diagnose(ic_payload: Optional[Dict[str, Any]],
             gate_config: Optional[Dict[str, Any]] = None,
             gate_decision: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """对整份 IC 评估结果做门禁阻塞诊断。

    Args:
        ic_payload    : ``ICCalculator.evaluate_all`` 输出（含 ``horizons``）；
        gate_config   : ``config["strategy_gate"]``（取阈值与 scope 口径）；
        gate_decision : 可选，``StrategyGate.decide().to_dict()`` 结果，用于核对状态。
    """
    gate_config = gate_config or {}
    defaults = {
        "min_ic": _num(gate_config.get("min_ic", 0.03), 0.03),
        "min_hit_rate": _num(gate_config.get("min_hit_rate", 0.52), 0.52),
        "min_windows": int(_num(gate_config.get("min_windows", 3), 3)),
    }
    scope = str(gate_config.get("scope", "all")).lower()

    horizons = ((ic_payload or {}).get("horizons") or {})
    details = [diagnose_horizon(name, h, defaults) for name, h in horizons.items()]

    failing = [d for d in details if not d["passed"]]
    # scope=all：任一周期不过就不放行，短板 = 最难补的那条腿（total_gap 最大）
    # scope=any：只要有一个周期过就放行，短板 = 最接近达标的那条腿（total_gap 最小）
    if scope == "any":
        binding = min(failing, key=lambda d: d["total_gap"]) if failing else None
    else:
        binding = max(details, key=lambda d: d["total_gap"]) if details else None
        if not failing:
            binding = None

    metric_counter: Dict[str, int] = {}
    for d in failing:
        if d["binding_metric"]:
            metric_counter[d["binding_metric"]] = metric_counter.get(d["binding_metric"], 0) + 1

    return {
        "scope": scope,
        "thresholds": defaults,
        "gate_state": (gate_decision or {}).get("state"),
        "horizons": {d["horizon"]: d for d in details},
        "failing_horizons": [d["horizon"] for d in failing],
        "passed_count": len(details) - len(failing),
        "total_count": len(details),
        # 当前约束：scope=any 看最易过的一条，scope=all 看最难补的一条
        "binding_horizon": binding["horizon"] if binding else None,
        "binding_metric": binding["binding_metric"] if binding else None,
        "binding_shortfall": binding["shortfall"] if binding else {},
        "blocking_metric_counts": metric_counter,
        "note": _note(details, binding, scope),
    }


def _fmt_shortfall(shortfall: Dict[str, Any]) -> str:
    """把 {'hit_rate': 0.0151} 渲染成可读的 '命中率 1.51 个百分点' 之类。"""
    if not shortfall:
        return "无"
    parts = []
    for k, v in shortfall.items():
        if k == "hit_rate":
            parts.append(f"{float(v) * 100:.2f} 个百分点")
        elif k == "ic":
            parts.append(f"{float(v):.4f}")
        elif k == "windows":
            parts.append(f"{int(v)} 个窗口")
        else:
            parts.append(f"{k} {v}")
    return "、".join(parts)


def _note(details: List[Dict[str, Any]], binding: Optional[Dict[str, Any]],
          scope: str) -> str:
    if not details:
        return "无可用的 IC 评估结果，无法诊断（先跑 `python main.py ic`）"
    if binding is None:
        return "所有周期均已过门槛，无需诊断（若门禁仍为 readonly，请检查 force_readonly / enabled）"
    gap = _fmt_shortfall(binding.get("shortfall") or {})
    metric = binding.get("binding_metric")
    if scope == "any":
        return (f"scope=any 任一周期达标即可放行，当前最接近达标的是 "
                f"{binding['horizon']}（{metric} 还差 {gap}）")
    return (f"scope=all 要求所有周期达标，当前最难补的是 {binding['horizon']}"
            f"（{metric} 还差 {gap}）；优先补齐该周期即可解锁")
