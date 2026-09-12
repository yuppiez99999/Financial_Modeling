"""统一评估量尺（S11 / G1）：把单一 IC 数值升级为完整因子健康度报告。

问题从哪来（Issue #29 集成方案 G1）：
  现有 ``src/inference/ic.py`` 的门禁输出只有**一个 IC 快照 + 一个命中率**。
  项目历史已经因此吃过两次亏：
    1. 「目标泄漏 → AUC 0.84」事故 —— 缺乏分层 IC / 胜率置信区间，
       单一数字虚高无法被交叉验证戳穿；
    2. 门禁长期卡在 52% 命中率门槛线 —— 却无法回答「信号是全局失效，
       还是只在某段区间 / 某类样本上失效」。

  参照 alphalens（Apache-2.0，quantopian/alphalens）的因子绩效分析框架
  （IC 时序 / 按周期衰减 / 分位数收益 / 换手率），给本项目补一套**标准量尺**，
  但**零新增依赖**：纯 numpy + 复用 ``src/inference/ic.py`` 的既有实现，
  保证与门禁口径完全同源（不同实现会出现「诊断说好、门禁说坏」的错位）。

本模块做什么（四个维度，全部 report_only）：
  1. **IC 置信区间**    ：Fisher 变换 + 正态近似，给出 IC 的 95% CI；
                          CI 不含 0 → 显著；含 0 → 如实标注不显著；
  2. **分位数分层收益**  ：按信号分 5 层（quantile spread），输出每层平均
                          未来收益 + 多空差（Q5-Q1 spread）；信号有区分度时
                          层间收益应单调；
  3. **信号换手率**      ：相邻样本信号方向翻转的频率；高换手 + 低 IC
                          = 交易成本会吃掉全部理论收益；
  4. **按周期衰减**      ：IC 在多个前视周期（5/10/20…日）上的衰减曲线，
                          回答「信号优势能维持多久」。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁结论**：``affects_gate`` 恒为 False，所有输出只是诊断证据；
  - **不下单、不给仓位**：不产生任何交易信号；
  - **不猜**：样本不足 / 序列退化时 ``available=False`` + reason，绝不凑数；
  - **无前视**：输入必须是与 ``ic`` 门禁同源构造的 (scores, forward_returns)
    对齐序列（forward return 由未来价格显式计算，未到期样本为 None 已剔除）。

统计口径备注：
  - Fisher 变换对 Spearman IC 的正态近似要求 n ≥ 10，低于该值直接拒答；
  - 分层收益在**层内样本 < min_samples_per_quantile** 时该层标记 unavailable；
  - 换手率只统计有效（非 None、非 NaN）样本对。
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.inference.ic import hit_rate, spearman_ic

logger = logging.getLogger(__name__)

REPORT_NAME = "factor_metrics.json"

# Fisher 变换的最小样本量：n < 10 时 z 分布近似失效，直接拒答
DEFAULT_MIN_SAMPLES = 10
# 分层（quantile）数：5 层 = 底部 20% … 顶部 20%
DEFAULT_QUANTILES = 5
# 单层最少样本：低于该值的层不出结论（防止单层 2 个样本的噪声层污染判断）
DEFAULT_MIN_SAMPLES_PER_QUANTILE = 5
# 置信水平（写死 95%：这是诊断口径，不做多口径扫描以免引入选择自由度）
CONF_LEVEL = 0.95
Z_95 = 1.959963984540054


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (config or {}).get("factor_metrics", {}) or {}


def _clean_pairs(scores: Sequence[Any], returns: Sequence[Any]) -> List[tuple]:
    """清洗对齐序列：剔除 None / NaN，保持顺序（时序顺序是换手率的前提）。"""
    out: List[tuple] = []
    for s, r in zip(scores, returns):
        if s is None or r is None:
            continue
        try:
            sf, rf = float(s), float(r)
        except (TypeError, ValueError):
            continue
        if sf != sf or rf != rf:  # NaN
            continue
        out.append((sf, rf))
    return out


def ic_confidence_interval(ic: float, n: int, conf_level: float = CONF_LEVEL) -> Dict[str, Any]:
    """Fisher 变换法计算 Spearman IC 的置信区间。

    z = artanh(ic) ~ N(artanh(rho), 1/(n-3))，变换回原域得到 IC 的 CI。
    CI 不含 0 → 统计显著；含 0 → 如实说明（诊断不等于放行，但显著性
    是「这个 IC 是不是噪声」的最基本事实）。
    """
    if n < DEFAULT_MIN_SAMPLES:
        return {
            "available": False,
            "reason": f"样本不足（{n} < {DEFAULT_MIN_SAMPLES}），Fisher 近似失效",
            "ci_low": None,
            "ci_high": None,
            "significant": None,
        }
    if abs(ic) >= 1.0:  # 完全相关时 artanh 发散，退化处理
        return {
            "available": True,
            "ci_low": round(ic, 6),
            "ci_high": round(ic, 6),
            "significant": True,
            "note": "完全相关（|IC|=1），区间退化为点估计",
        }
    z = math.atanh(ic)
    se = 1.0 / math.sqrt(max(n - 3, 1))
    # 置信水平目前只支持 95%（见模块 docstring：不做多口径扫描）
    z_crit = Z_95 if conf_level == CONF_LEVEL else Z_95
    low, high = math.tanh(z - z_crit * se), math.tanh(z + z_crit * se)
    return {
        "available": True,
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "significant": bool(low > 0 or high < 0),
        "level": conf_level,
    }


def quantile_returns(scores: Sequence[float], returns: Sequence[float],
                     n_quantiles: int = DEFAULT_QUANTILES,
                     min_per_quantile: int = DEFAULT_MIN_SAMPLES_PER_QUANTILE) -> Dict[str, Any]:
    """按信号分位数的分层未来收益（alphalens quantile return 分析的最小实现）。

    - 层按信号升序等频划分（Q1 = 信号最弱 20%，Q5 = 信号最强 20%）；
    - 信号有区分度时，层均收益应随分位单调；
    - ``monotonic`` = Spearman(分位序号, 层均收益) = ±1 的严格单调判定
      （放宽为秩相关显著同向，用既有 spearman_ic 实现）；
    - ``spread`` = 顶减底（Q_max - Q_min）的平均收益差：做多顶层做空底层的
      理论毛收益（**未扣成本**，成本敏感性见 ``cost_sensitivity``）。
    """
    pairs = _clean_pairs(scores, returns)
    n = len(pairs)
    if n < n_quantiles * min_per_quantile:
        return {
            "available": False,
            "reason": f"样本不足（{n} < {n_quantiles}层 × {min_per_quantile}样本/层）",
        }
    ordered = sorted(pairs, key=lambda p: p[0])
    rows: List[Dict[str, Any]] = []
    for q in range(n_quantiles):
        start = q * n // n_quantiles
        end = (q + 1) * n // n_quantiles if q < n_quantiles - 1 else n
        chunk = ordered[start:end]
        rets = [p[1] for p in chunk]
        rows.append({
            "quantile": q + 1,
            "samples": len(chunk),
            "mean_return": round(sum(rets) / len(rets), 6) if rets else None,
            "hit_rate": round(hit_rate([p[0] for p in chunk], rets), 4) if rets else None,
            "available": len(chunk) >= min_per_quantile,
        })
    mean_rets = [r["mean_return"] for r in rows if r["mean_return"] is not None]
    available_rows = [r for r in rows if r["available"]]
    spread = None
    if len(available_rows) >= 2:
        spread = round(
            available_rows[-1]["mean_return"] - available_rows[0]["mean_return"], 6
        )
    # 层间单调性：分位序号 vs 层均收益的秩相关（±1 = 完全单调）
    monotonic_ic = None
    if len(mean_rets) == len(rows) and len(rows) >= 3:
        monotonic_ic = round(
            spearman_ic([r["quantile"] for r in rows], mean_rets), 4
        )
    return {
        "available": True,
        "quantiles": rows,
        "spread_top_minus_bottom": spread,
        "monotonic_ic": monotonic_ic,
        "monotonic": bool(monotonic_ic is not None and abs(monotonic_ic) >= 0.8),
        "note": "spread 为毛收益（未扣交易成本），成本口径见 cost_sensitivity",
    }


def turnover_rate(scores: Sequence[Any]) -> Dict[str, Any]:
    """信号换手率：相邻有效样本方向翻转的频率。

    高换手本身不是错，但「高换手 + 低 IC」意味着交易成本会吃掉全部理论
    收益 —— 这是 eval_criteria.md「交易类指标」承诺但未落地的部分。

    口径：sign(s_t) != sign(s_{t-1}) 的比例；|s| <= 1e-12 视为观望，
    观望态不翻转也不计入翻转，但计入分母（保守估计成本冲击下界）。
    """
    cleaned = [
        float(s) for s in scores
        if s is not None and not (isinstance(s, float) and s != s)
    ]
    if len(cleaned) < 2:
        return {"available": False, "reason": "样本不足（< 2）"}
    flips = 0
    for i in range(1, len(cleaned)):
        prev, cur = cleaned[i - 1], cleaned[i]
        if abs(prev) <= 1e-12 or abs(cur) <= 1e-12:
            continue  # 观望态不算翻转
        if (prev > 0) != (cur > 0):
            flips += 1
    return {
        "available": True,
        "turnover": round(flips / (len(cleaned) - 1), 4),
        "samples": len(cleaned),
        "flips": flips,
    }


def horizon_decay(scores: Sequence[float],
                  returns_by_days: Dict[int, Sequence[Any]]) -> Dict[str, Any]:
    """按前视周期衰减曲线：同一信号在不同持有期下的 IC。

    Args:
        scores          : 信号序列（t 时点的观点）；
        returns_by_days : {days: forward_returns(days)}，各周期与 scores 对齐，
                          未到期为 None（由 ``forward_returns`` 产出）。
    """
    rows: List[Dict[str, Any]] = []
    for days in sorted(returns_by_days):
        pairs = _clean_pairs(scores, returns_by_days[days])
        if len(pairs) < DEFAULT_MIN_SAMPLES:
            rows.append({
                "days": int(days), "samples": len(pairs),
                "available": False,
                "reason": f"样本不足（{len(pairs)} < {DEFAULT_MIN_SAMPLES}）",
            })
            continue
        ss = [p[0] for p in pairs]
        rr = [p[1] for p in pairs]
        ic = spearman_ic(ss, rr)
        ci = ic_confidence_interval(ic, len(pairs))
        rows.append({
            "days": int(days),
            "samples": len(pairs),
            "available": True,
            "ic": round(ic, 4),
            "ic_ci": ci,
        })
    available_rows = [r for r in rows if r.get("available")]
    best_days = None
    if available_rows:
        best = max(available_rows, key=lambda r: abs(r["ic"]))
        best_days = best["days"]
    return {
        "available": bool(available_rows),
        "curve": rows,
        "best_ic_days": best_days,
        "note": "衰减曲线为诊断证据，best_ic_days 不构成周期切换依据"
                "（切换属口径变更，须走 horizon-decision 人工决策）",
    }


# T11.2 成本三档（单一事实源）：佣金 + 滑点 = 单边成本。
# 2026-09-12 经人工检查点定稿（schedule/manual_checkpoints.json T11.2 confirmed）；
# 消费方：cost_sensitivity（本模块）与 src.eval.portfolio_backtest（S21 组合回测）。
# 三档参数写成常量、不得被配置覆盖 —— 否则「换一组参数就能翻盘」的自由度又回来了。
T112_COST_TIERS = [
    {"name": "conservative", "commission": 0.00025, "slippage": 0.0010,
     "one_side": 0.00125, "note": "保守档：小盘流动性差的场景"},
    {"name": "base", "commission": 0.00025, "slippage": 0.0005,
     "one_side": 0.00075, "note": "基准档：大盘常规场景"},
    {"name": "aggressive", "commission": 0.00010, "slippage": 0.0002,
     "one_side": 0.00030, "note": "乐观档：流动性充裕 + 低佣场景"},
]


def cost_sensitivity(quantile_spread: Optional[float], turnover: Optional[float],
                     horizons: Sequence[Dict[str, Any]],
                     levels: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """成交成本敏感性扫描（T11.2）：三档 佣金×滑点×换手 组合下的净收益下界。

    口径（保守估计，落在 eval_criteria.md「成交假设要显式记录」的框架内）：

      净收益 ≈ spread × 参与率 − 单边成本 × 2 × 调仓频率

    其中调仓频率 = turnover / horizon_days（每期翻转率折算到持有期内的
    实际翻转次数上界）。三档成本见模块常量 ``T112_COST_TIERS``（单一事实源）。

    ⚠️ 三档参数为定稿常量（T11.2 confirmed），不得被配置覆盖。
    """
    levels = list(levels) if levels is not None else [dict(t) for t in T112_COST_TIERS]
    if quantile_spread is None or turnover is None:
        return {
            "available": False,
            "reason": "缺少分层 spread 或换手率（前置指标不可用）",
            "levels": [lv["name"] for lv in levels],
        }
    rows: List[Dict[str, Any]] = []
    for lv in levels:
        one_side = float(lv["commission"]) + float(lv["slippage"])
        per_horizon: Dict[str, Any] = {}
        for h in horizons:
            days = int(h.get("days", 0))
            if not days or not h.get("available"):
                continue
            # 每个持有期内最多翻转 turnover × days 次（线性近似上界）
            rebalances = max(float(turnover) * days, 0.0)
            cost = one_side * 2.0 * rebalances
            per_horizon[str(days)] = {
                "gross_spread": round(quantile_spread, 6),
                "cost": round(cost, 6),
                "net": round(quantile_spread - cost, 6),
                "net_positive": bool(quantile_spread - cost > 0),
            }
        rows.append({
            "name": lv["name"],
            "one_side_cost": round(one_side, 6),
            "per_horizon": per_horizon,
            "note": lv.get("note", ""),
        })
    return {
        "available": bool(rows),
        "spread": round(float(quantile_spread), 6),
        "turnover": round(float(turnover), 4),
        "levels": rows,
        "method": "net = spread − 2 × 单边成本 × (turnover × days)（保守线性上界）",
        "manual_checkpoint": "T11.2 口径定稿前为草案：三档参数为常量，不允许配置覆盖",
    }


class FactorMetricsCalculator:
    """因子健康度报告（纯读、无网络、无副作用，report_only）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = _cfg(config)
        gate_cfg = (self.config.get("strategy_gate", {}) or {})
        self.min_samples = int(cfg.get("min_samples", DEFAULT_MIN_SAMPLES))
        self.quantiles = int(cfg.get("quantiles", DEFAULT_QUANTILES))
        self.min_per_quantile = int(
            cfg.get("min_samples_per_quantile", DEFAULT_MIN_SAMPLES_PER_QUANTILE)
        )
        self.report_dir = Path(cfg.get("report_dir", gate_cfg.get("report_dir", "reports")))

    # ------------------------------------------------------------------
    def evaluate(self, horizon: str, horizon_days: int,
                 scores: Sequence[Any], returns: Sequence[Any],
                 returns_by_days: Optional[Dict[int, Sequence[Any]]] = None) -> Dict[str, Any]:
        """单周期因子健康度报告。

        Args:
            scores / returns   : 与 ``ic`` 门禁同源的 (信号, 未来收益) 对齐序列；
            returns_by_days    : 可选，多周期前视收益（用于衰减曲线）。
        """
        pairs = _clean_pairs(scores, returns)
        n = len(pairs)
        base: Dict[str, Any] = {
            "horizon": horizon,
            "horizon_days": int(horizon_days),
            "samples": n,
            "source": "src/eval/factor_metrics.py",
        }
        if n < self.min_samples:
            return {
                **base,
                "available": False,
                "reason": f"样本不足（{n} < {self.min_samples}）",
                "affects_gate": False,
            }
        ss = [p[0] for p in pairs]
        rr = [p[1] for p in pairs]
        ic = spearman_ic(ss, rr)
        ic_ci = ic_confidence_interval(ic, n)
        quant = quantile_returns(ss, rr, n_quantiles=self.quantiles,
                                 min_per_quantile=self.min_per_quantile)
        # 换手率基于原始顺序序列（非排序），剔除无效样本
        turn = turnover_rate(scores)
        decay = horizon_decay(scores, returns_by_days) if returns_by_days else {
            "available": False, "reason": "未提供多周期收益序列", "curve": [],
        }
        cost = cost_sensitivity(
            quant.get("spread_top_minus_bottom") if quant.get("available") else None,
            turn.get("turnover") if turn.get("available") else None,
            decay.get("curve") or [],
        )
        return {
            **base,
            "available": True,
            "ic": round(ic, 4),
            "ic_ci": ic_ci,
            "hit_rate": round(hit_rate(ss, rr), 4),
            "quantile_returns": quant,
            "turnover": turn,
            "horizon_decay": decay,
            "cost_sensitivity": cost,
            # ⚠️ 恒为 False：诊断证据，不改变 strategy_gate 结论
            "affects_gate": False,
        }

    # ------------------------------------------------------------------
    def evaluate_all(self, sequences: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估：{horizon_name: {horizon_days, scores, returns, returns_by_days?}}。"""
        horizons: Dict[str, Any] = {}
        for hname, payload in (sequences or {}).items():
            horizons[hname] = self.evaluate(
                hname,
                int(payload.get("horizon_days", 0)),
                payload.get("scores", []),
                payload.get("returns", []),
                returns_by_days=payload.get("returns_by_days"),
            )
        available = [h for h in horizons.values() if h.get("available")]
        significant = [h["horizon"] for h in available
                       if (h.get("ic_ci") or {}).get("significant")]
        healthy = [h["horizon"] for h in available
                   if ((h.get("quantile_returns") or {}).get("monotonic"))]
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "src/eval/factor_metrics.py",
            "horizons": horizons,
            "summary": {
                "available_horizons": [h["horizon"] for h in available],
                "ic_significant_horizons": significant,
                "quantile_monotonic_horizons": healthy,
                "n_available": len(available),
            },
            "affects_gate": False,
            "gate_note": (
                "因子健康度报告只提供诊断证据（IC 置信区间 / 分层收益 / 换手 / "
                "成本敏感性），不参与 strategy_gate 放行判定。"
            ),
        }

    # ------------------------------------------------------------------
    def save(self, payload: Dict[str, Any], path: Optional[str] = None) -> Path:
        out = Path(path) if path else self.report_dir / REPORT_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def load(self, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        p = Path(path) if path else self.report_dir / REPORT_NAME
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[factor-metrics] 报告读取失败: {e}")
            return None
