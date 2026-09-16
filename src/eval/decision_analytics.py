"""决策源有效性分析（DecisionFeedAnalytics）：回答下游最关心的那个问题。

## 问题从哪来

16_ 给下游的是一组**因子读数**：概率、置信度、采纳建议。但下游真正要判断的
是「**我该不该信这个因子**」，而这需要**因子 × 已实现收益**的联合分布，
不是单独的因子分布 —— 后者再漂亮也可能与收益无关。

仓库里已有两半，但从未拼起来过：

- `src/eval/confidence_curve.sweep_confidence` 有「阈值 → 覆盖率 / 命中率 / IC」，
  但它用的是 walk-forward **在样本内**的打分，不含跨标的的已实现收益检验；
- `src/eval/portfolio_backtest.consistency_contrast` 有「IC/命中率 vs PnL」，
  但它检验的是**机械 MA 基线**，与模型输出无关。

本模块把两边接上：用**已落盘的真实审计记录**（预测 + 回填的真实收益 + 命中）
按「置信度分档」聚合，产出下游可直接引用的三条读数：

1. `by_confidence_bin`：分档覆盖率 / 命中率 / **平均已实现收益**（含符号）
   —— 回答「高置信是不是真的更准、而且更赚钱」；
2. `threshold_scan`：沿采纳门槛扫描子集命中率与覆盖率 —— 直接对应
   `advisory.recommended_threshold`，让下游门槛有实证支撑；
3. `verdict`：`effective` / `ineffective` / `insufficient` 三态**保守判定**
   （样本不足即 `insufficient`，不拿 n=7 的读数当结论）。

## 边界

- **只读**：不重训、不触网、不写审计；
- **不猜**：样本不足一律 `available=false` + reason；
- **不择优**：不做多重比较后的「最优阈值」推荐，只呈现扫描结果；
- **诚实标注**：审计样本是**历史预测**（含模拟数据时代的记录已被清理的
  前提），不等于未来表现。
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

STATUS_EFFECTIVE = "effective"
STATUS_INEFFECTIVE = "ineffective"
STATUS_INSUFFICIENT = "insufficient"

DEFAULT_BINS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
DEFAULT_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
MIN_SAMPLES_BIN = 20
MIN_SAMPLES_VERDICT = 60


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def extract_samples(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从审计记录中抽出可用于有效性分析的样本（**已回填**真实收益 + 命中）。

    只保留：``verified=True`` 且 ``actual_return`` / ``hit`` 可解析的记录。
    未到期 / 未验证的预测**不参与统计**（不拿 None 当 0）。
    """
    out: List[Dict[str, Any]] = []
    for r in records or []:
        if not isinstance(r, dict):
            continue
        if not r.get("verified"):
            continue
        ret = _finite(r.get("actual_return"))
        if ret is None:
            continue
        hit = r.get("hit")
        if hit is None:
            continue
        prob = _finite(r.get("probability"))
        conf = _finite(r.get("confidence"))
        if conf is None and prob is not None:
            conf = abs(prob - 0.5) * 2.0
        if conf is None:
            continue
        out.append({
            "symbol": r.get("symbol"),
            "horizon": r.get("horizon"),
            "source": r.get("source") or "unspecified",
            "confidence": min(max(float(conf), 0.0), 1.0),
            "probability": prob,
            "actual_return": ret,
            "hit": bool(hit),
        })
    return out


def by_confidence_bin(samples: Sequence[Dict[str, Any]],
                      bins: Sequence[float] = DEFAULT_BINS,
                      min_samples: int = MIN_SAMPLES_BIN) -> List[Dict[str, Any]]:
    """置信度分档 → 覆盖率 / 命中率 / 平均已实现收益（**含符号**）。

    平均收益带符号是关键：一个"命中率高但平均收益为负"的因子（涨小跌大）
    对下游是有害的，任何只看命中率的评估都发现不了。
    """
    rows: List[Dict[str, Any]] = []
    n = len(samples)
    edges = list(bins)
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        last = (i == len(edges) - 2)
        sub = [s for s in samples
               if (s["confidence"] >= lo and (s["confidence"] <= hi if last
                                              else s["confidence"] < hi))]
        row: Dict[str, Any] = {
            "bin": f"[{lo:.1f},{hi:.1f}{']' if last else ')'}",
            "lo": lo, "hi": hi, "samples": len(sub),
            "coverage": round(len(sub) / n, 6) if n else 0.0,
        }
        if len(sub) < max(min_samples, 2):
            row.update({"available": False,
                        "reason": f"样本不足（{len(sub)} < {min_samples}）"})
        else:
            row.update({
                "available": True,
                "hit_rate": round(sum(1 for s in sub if s["hit"]) / len(sub), 6),
                "mean_return": round(sum(s["actual_return"] for s in sub) / len(sub), 8),
                "mean_abs_return": round(
                    sum(abs(s["actual_return"]) for s in sub) / len(sub), 8),
            })
        rows.append(row)
    return rows


def threshold_scan(samples: Sequence[Dict[str, Any]],
                   thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                   min_samples: int = MIN_SAMPLES_BIN) -> List[Dict[str, Any]]:
    """沿采纳门槛扫描：置信度 ≥ thr 的子集命中率 / 平均收益 / 覆盖率。

    与 `advisory.recommended_threshold` 一一对应 —— 下游门槛从此**有据可依**，
    而不是生产方拍一个数、消费方再拍一个数。
    """
    rows: List[Dict[str, Any]] = []
    n = len(samples)
    for thr in thresholds:
        sub = [s for s in samples if s["confidence"] >= float(thr)]
        row: Dict[str, Any] = {
            "threshold": round(float(thr), 6),
            "samples": len(sub),
            "coverage": round(len(sub) / n, 6) if n else 0.0,
        }
        if len(sub) < max(min_samples, 2):
            row.update({"available": False,
                        "reason": f"样本不足（{len(sub)} < {min_samples}）"})
        else:
            row.update({
                "available": True,
                "hit_rate": round(sum(1 for s in sub if s["hit"]) / len(sub), 6),
                "mean_return": round(sum(s["actual_return"] for s in sub) / len(sub), 8),
            })
        rows.append(row)
    return rows


def verdict(samples: Sequence[Dict[str, Any]],
            threshold: float,
            baseline_hit_rate: float = 0.5,
            min_samples: int = MIN_SAMPLES_VERDICT) -> Dict[str, Any]:
    """保守判定：高置信子集是否**同时**在命中率与平均收益上优于基线。

    判定规则（**必须两条腿都过**，只过一条一律不算有效）：
      - 子集命中率 > 基线；**且**
      - 子集平均已实现收益 > 全体样本平均已实现收益。
    样本不足 → `insufficient`（不是"无效"，是不该下结论）。
    结论为 `ineffective` 时如实入库，**不调整门槛去凑一个好看的结果**。
    """
    sub = [s for s in samples if s["confidence"] >= float(threshold)]
    if len(samples) < min_samples or len(sub) < MIN_SAMPLES_BIN:
        return {
            "status": STATUS_INSUFFICIENT,
            "reason": (f"样本不足（全体 {len(samples)} < {min_samples} 或 "
                       f"子集 {len(sub)} < {MIN_SAMPLES_BIN}），不下结论"),
            "n_total": len(samples), "n_subset": len(sub),
        }
    hit = sum(1 for s in sub if s["hit"]) / len(sub)
    mr = sum(s["actual_return"] for s in sub) / len(sub)
    base_mr = sum(s["actual_return"] for s in samples) / len(samples)
    ok = hit > float(baseline_hit_rate) and mr > base_mr
    return {
        "status": STATUS_EFFECTIVE if ok else STATUS_INEFFECTIVE,
        "threshold": round(float(threshold), 6),
        "n_total": len(samples),
        "n_subset": len(sub),
        "subset_hit_rate": round(hit, 6),
        "baseline_hit_rate": round(float(baseline_hit_rate), 6),
        "subset_mean_return": round(mr, 8),
        "baseline_mean_return": round(base_mr, 8),
        "note": ("判定需命中率与平均已实现收益**同时**优于基线；"
                 "`ineffective` 表示当前样本下高置信子集没有优势 —— 这是结论，"
                 "不是缺陷，不得通过调低门槛去规避。"),
    }


def analyze(records: Sequence[Dict[str, Any]],
            threshold: float = 0.2,
            bins: Sequence[float] = DEFAULT_BINS,
            thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
            horizons: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """整体 + 逐周期有效性分析（下游可据此按周期差异化采信）。"""
    samples = extract_samples(records)
    if horizons:
        samples = [s for s in samples if s.get("horizon") in set(horizons)]

    per_horizon: Dict[str, Any] = {}
    for h in sorted({str(s.get("horizon")) for s in samples}):
        sub = [s for s in samples if str(s.get("horizon")) == h]
        per_horizon[h] = {
            "available": True,
            "n_samples": len(sub),
            "by_confidence_bin": by_confidence_bin(sub, bins),
            "verdict": verdict(sub, threshold),
        }

    return {
        "kind": "decision_feed_analytics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available": bool(samples),
        "reason": "" if samples else "无已验证审计样本（预测未到期或审计为空）",
        "n_samples": len(samples),
        "confidence_threshold": round(float(threshold), 6),
        "overall": {
            "by_confidence_bin": by_confidence_bin(samples, bins),
            "threshold_scan": threshold_scan(samples, thresholds),
            "verdict": verdict(samples, threshold),
        },
        "per_horizon": per_horizon,
        "affects_gate": False,
        "readonly": True,
        "note": ("基于**历史**已到期预测 + 真实行情回填收益的联合分布；"
                 "不重训、不改门槛、不择优。判定为 insufficient 时下游应继续"
                 "把信号当只读观测，而不是自行下调门槛。"),
    }
