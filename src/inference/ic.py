"""前视 IC（Information Coefficient）计算与多周期门禁评估。

IC = 预测信号与「未来真实收益」的秩相关（Spearman），衡量信号的单调区分度。
本模块不训练、不落盘、不触网，输入为对齐后的 (score, forward_return) 序列。

设计要点：
- **无前视**：forward return 由未来价格显式计算，调用方须保证只对「已到期」样本评估；
- **样本门槛**：样本数不足（< min_samples）时返回 available=False，不下合格结论；
- **分周期**：short/mid/long 三周期分别统计，门禁按周期判定（见 §Q2 路线图）。

口径：
- IC   : Spearman 秩相关，∈ [-1, 1]，绝对值越大区分度越强
- ICIR : IC 的均值 / 标准差（多窗口滚动时序下的稳定性），无窗口时为 0
- 命中率: sign(score) 与 sign(forward_return) 一致的占比
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 门禁默认阈值（Q2 路线图：IC 与命中率双指标过线才允许从「只读」升级为「打分因子」）
DEFAULT_MIN_SAMPLES = 30
DEFAULT_MIN_IC = 0.03
DEFAULT_MIN_HIT_RATE = 0.52
# 周期内至少需要多少个有效窗口才认为窗口统计可信
DEFAULT_MIN_WINDOWS = 3


def _rank(values: Sequence[float]) -> List[float]:
    """平均秩（处理并列值），返回与输入等长的秩列表。"""
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_ic(scores: Sequence[float], returns: Sequence[float]) -> float:
    """计算 Spearman 秩相关（IC）。样本不足或方差为 0 时返回 0.0。"""
    n = min(len(scores), len(returns))
    if n < 2:
        return 0.0
    a: List[float] = [float(scores[i]) for i in range(n)]
    b: List[float] = [float(returns[i]) for i in range(n)]
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n))
    vb = sum((rb[i] - mb) ** 2 for i in range(n))
    if va <= 0 or vb <= 0:
        return 0.0
    return float(cov / (va ** 0.5 * vb ** 0.5))


def hit_rate(scores: Sequence[float], returns: Sequence[float], neutral_band: float = 0.0) -> float:
    """方向命中率：sign(score) 与 sign(return) 一致的比例。

    neutral_band > 0 时，|score| <= neutral_band 的样本视为观望、不计入分母。
    """
    hits = 0
    total = 0
    n = min(len(scores), len(returns))
    for i in range(n):
        s = float(scores[i])
        r = float(returns[i])
        if abs(s) <= neutral_band or r == 0:
            continue
        total += 1
        if (s > 0) == (r > 0):
            hits += 1
    return hits / total if total else 0.0


def forward_returns(closes: Sequence[float], horizon_days: int) -> List[Optional[float]]:
    """由收盘价序列计算未来 horizon_days 的收益率。

    末尾 horizon_days 个样本未到期，返回 None（**绝不猜测**），调用方应跳过。
    """
    out: List[Optional[float]] = []
    n = len(closes)
    for i in range(n):
        j = i + horizon_days
        if j >= n:
            out.append(None)
            continue
        base = float(closes[i])
        out.append((float(closes[j]) - base) / base if base else None)
    return out


@dataclass
class ICResult:
    """单周期 IC 评估结果。"""

    horizon: str
    horizon_days: int
    samples: int = 0
    ic: float = 0.0
    icir: float = 0.0
    hit_rate: float = 0.0
    windows: int = 0
    available: bool = False
    reason: str = ""
    passed: bool = False
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "horizon_days": self.horizon_days,
            "samples": self.samples,
            "ic": round(self.ic, 4),
            "icir": round(self.icir, 4),
            "hit_rate": round(self.hit_rate, 4),
            "windows": self.windows,
            "available": self.available,
            "passed": self.passed,
            "reason": self.reason,
            "thresholds": self.thresholds,
        }


class ICCalculator:
    """按周期计算 IC / ICIR / 命中率并给出门禁判定。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 min_ic: float = DEFAULT_MIN_IC,
                 min_hit_rate: float = DEFAULT_MIN_HIT_RATE,
                 min_windows: int = DEFAULT_MIN_WINDOWS):
        cfg = ((config or {}).get("strategy_gate", {}) or {})
        self.min_samples = int(cfg.get("min_samples", min_samples))
        self.min_ic = float(cfg.get("min_ic", min_ic))
        self.min_hit_rate = float(cfg.get("min_hit_rate", min_hit_rate))
        self.min_windows = int(cfg.get("min_windows", min_windows))

    def evaluate(self, horizon: str, horizon_days: int,
                 scores: Sequence[float], returns: Sequence[Optional[float]],
                 window_size: Optional[int] = None) -> ICResult:
        """评估单周期。

        Args:
            scores     : 与 returns 对齐的信号分数（概率或综合得分）
            returns    : 未来收益（None = 未到期，自动跳过）
            window_size: 滚动窗口大小；给定时额外计算 ICIR（各窗口 IC 的均值/标准差）
        """
        pairs: List[Tuple[float, float]] = [
            (float(s), float(r))
            for s, r in zip(scores, returns)
            if r is not None and s is not None and not _isnan(s) and not _isnan(r)
        ]
        res = ICResult(
            horizon=horizon,
            horizon_days=int(horizon_days),
            samples=len(pairs),
            thresholds={
                "min_samples": self.min_samples,
                "min_ic": self.min_ic,
                "min_hit_rate": self.min_hit_rate,
                "min_windows": self.min_windows,
            },
        )
        if len(pairs) < self.min_samples:
            res.available = False
            res.reason = f"样本不足（{len(pairs)} < {self.min_samples}）"
            return res

        ss = [p[0] for p in pairs]
        rr = [p[1] for p in pairs]
        res.available = True
        res.ic = spearman_ic(ss, rr)
        res.hit_rate = hit_rate(ss, rr)

        if window_size and window_size >= self.min_windows and len(pairs) >= window_size * self.min_windows:
            ics = [
                spearman_ic(ss[i:i + window_size], rr[i:i + window_size])
                for i in range(0, len(pairs) - window_size + 1, window_size)
            ]
            res.windows = len(ics)
            mean_ic = sum(ics) / len(ics) if ics else 0.0
            var = sum((x - mean_ic) ** 2 for x in ics) / len(ics) if ics else 0.0
            std = var ** 0.5
            # ICIR 仅在 IC 存在真实波动时才有意义；窗口 IC 全等（std≈0）时
            # 分母趋近 0 会把 ICIR 放大到天文数字，故低于 1e-6 直接记 0
            res.icir = float(mean_ic / std) if std > 1e-6 else 0.0

        # 仅当调用方请求了窗口统计（window_size 给定）时，才校验窗口数；
        # 未请求窗口统计的场景（如快速抽查）不受 min_windows 约束
        window_ok = (not window_size) or res.windows >= self.min_windows
        res.passed = bool(
            res.available
            and abs(res.ic) >= self.min_ic
            and res.hit_rate >= self.min_hit_rate
            and window_ok
        )
        if not res.passed and not res.reason:
            fails = []
            if abs(res.ic) < self.min_ic:
                fails.append(f"|IC| {abs(res.ic):.3f} < {self.min_ic}")
            if res.hit_rate < self.min_hit_rate:
                fails.append(f"命中率 {res.hit_rate:.2%} < {self.min_hit_rate:.0%}")
            if not window_ok:
                fails.append(f"有效窗口 {res.windows} < {self.min_windows}")
            res.reason = "；".join(fails) if fails else "未过门禁"
        return res

    def evaluate_all(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估多周期。

        Args:
            data: {horizon_name: {"horizon_days": int, "scores": [...], "returns": [...], "window_size": int|None}}
        """
        results: Dict[str, Any] = {}
        for hname, payload in data.items():
            results[hname] = self.evaluate(
                hname,
                int(payload.get("horizon_days", 0)),
                payload.get("scores", []),
                payload.get("returns", []),
                window_size=payload.get("window_size"),
            ).to_dict()
        return {
            "horizons": results,
            "all_passed": bool(results) and all(r["passed"] for r in results.values()),
            "any_passed": any(r["passed"] for r in results.values()),
        }


def _isnan(value: Any) -> bool:
    try:
        return value != value  # NaN 特性
    except Exception:  # noqa: BLE001
        return True


def evaluate_series(closes: Iterable[float], scores: Iterable[float],
                    horizon_days: int) -> Tuple[List[float], List[Optional[float]]]:
    """便捷工具：由收盘价与信号序列直接产出 (scores, forward_returns) 对齐对。"""
    return list(scores), forward_returns(list(closes), horizon_days)
