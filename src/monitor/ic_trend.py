"""信号衰减监控（Q5）：把 IC 从「一个快照」变成「一条时间序列」。

## 为什么需要它

Q2 引入的策略门禁（`src/trading/gate.py`）回答的是**静态**问题：
*此刻*的 IC / 命中率是否达标？结论只有 `gated` / `readonly` 两种。

但量化信号真正的运维问题是**动态**的：IC 是在变好还是变坏？

- 门禁显示 `|IC| = 0.031`（刚过线），它可能是**稳定在 0.031**，
  也可能是**从 0.09 一路衰减到 0.031** —— 后者意味着下周就会掉出放行线；
- 反过来 `|IC| = 0.028`（差一点没过）也可能正从 0.01 回升。

两者在门禁看来**完全一样**，对使用者的含义却相反。

## 这个模块做什么

用**同一条 walk-forward 序列**（收盘价 + 信号分数）按交易日锚点滚动切片，
在每个锚点算一段 IC，得到 IC 时序，再回答四个问题：

1. **当前水位**：最近一段 IC 是多少（`latest_ic`）；
2. **趋势方向**：IC 对时间做线性回归的斜率（`slope_per_step`），并折算成
   「每个步长衰减多少」（`decay_per_step_pct`）；
3. **离门禁多远 / 还有多久掉出去**：按当前斜率外推，预估还有几个步长
   会跌破 `|IC|` 下限（`steps_to_breach`）；斜率不为负或已跌破时为 `None`
   或 `0`，**绝不外推出「永远安全」这种结论**；
4. **是否显著衰减**：斜率显著为负（且超过最小步数）→ `decaying`，否则 `stable`。

## 设计原则

- **只读纯计算**：不训练、不触网、不写文件（除调用方显式导出）；
- **样本不足即沉默**：锚点不足 → `status=unknown` + `reason`，不臆测趋势；
- **IC 用现有实现**：复用 `src/inference/ic.py` 的 `spearman_ic` / `hit_rate`，
  保证与门禁口径完全同源（不同实现会出现「监控说衰减、门禁说达标」的错位）；
- **不外推未来收益**：所有收益率由 `forward_returns` 产出，未到期样本为 `None` 并被跳过。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from src.inference.ic import forward_returns, hit_rate, spearman_ic

logger = logging.getLogger(__name__)

STATUS_DECAYING = "decaying"    # 显著衰减
STATUS_STABLE = "stable"        # 无明显趋势
STATUS_IMPROVING = "improving"  # 显著改善
STATUS_UNKNOWN = "unknown"      # 数据不足以判定

DEFAULT_WINDOW = 120        # 每个锚点的 IC 观察窗（交易日）
DEFAULT_STEP = 20           # 锚点之间的步长（交易日）
DEFAULT_MIN_ANCHORS = 5     # 判定趋势所需的最少锚点数
DEFAULT_MIN_OBS = 30        # 单锚点最少有效样本（与门禁 min_samples 对齐）
# 显著衰减的判定阈值：每步 |IC| 下滑超过该绝对值才算 "decaying"，
# 避免把 0.001 级别的噪声抖动当成趋势。
DEFAULT_MIN_DECAY = 0.002


def _linear_slope(ys: Sequence[float]) -> float:
    """最小二乘斜率（x 取 0..n-1）。点数 < 2 或 x 无方差时返回 0.0。"""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0:
        return 0.0
    return float(cov / var)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """皮尔逊相关系数（用于判定趋势方向的稳定性）；方差为 0 时返回 0.0。"""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx, my = sum(xs[:n]) / n, sum(ys[:n]) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    vx = sum((xs[i] - mx) ** 2 for i in range(n))
    vy = sum((ys[i] - my) ** 2 for i in range(n))
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(cov / (vx ** 0.5 * vy ** 0.5))


@dataclass
class ICTrendResult:
    """单周期 IC 趋势结果（可直接序列化进 API / 日报 / 监控）。"""

    horizon: str = ""
    horizon_days: int = 0
    status: str = STATUS_UNKNOWN
    available: bool = False
    reason: str = ""
    anchors: int = 0
    latest_ic: Optional[float] = None
    mean_ic: Optional[float] = None
    ic_series: List[float] = field(default_factory=list)
    slope_per_step: Optional[float] = None
    decay_per_step_pct: Optional[float] = None
    trend_r2: Optional[float] = None
    latest_hit_rate: Optional[float] = None
    steps_to_breach: Optional[int] = None
    min_ic: float = 0.0
    window: int = DEFAULT_WINDOW
    step: int = DEFAULT_STEP

    def to_dict(self) -> Dict[str, Any]:
        def r(v: Optional[float], nd: int = 4) -> Optional[float]:
            return None if v is None else round(float(v), nd)

        return {
            "horizon": self.horizon,
            "horizon_days": self.horizon_days,
            "status": self.status,
            "available": self.available,
            "reason": self.reason,
            "anchors": self.anchors,
            "latest_ic": r(self.latest_ic),
            "mean_ic": r(self.mean_ic),
            "ic_series": [r(x) for x in self.ic_series],
            "slope_per_step": r(self.slope_per_step, 6),
            "decay_per_step_pct": r(self.decay_per_step_pct, 2),
            "trend_r2": r(self.trend_r2, 4),
            "latest_hit_rate": r(self.latest_hit_rate),
            "steps_to_breach": self.steps_to_breach,
            "min_ic": self.min_ic,
            "window": self.window,
            "step": self.step,
        }


class ICTrendMonitor:
    """按 walk-forward 锚点滚动计算 IC 时序并判定衰减趋势（纯读）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 window: int = DEFAULT_WINDOW,
                 step: int = DEFAULT_STEP,
                 min_anchors: int = DEFAULT_MIN_ANCHORS,
                 min_obs: int = DEFAULT_MIN_OBS,
                 min_decay: float = DEFAULT_MIN_DECAY):
        cfg = ((config or {}).get("ic_trend", {}) or {})
        gate = ((config or {}).get("strategy_gate", {}) or {})
        self.window = int(cfg.get("window", window))
        self.step = int(cfg.get("step", step))
        self.min_anchors = int(cfg.get("min_anchors", min_anchors))
        self.min_obs = int(cfg.get("min_obs", min_obs))
        self.min_decay = float(cfg.get("min_decay", min_decay))
        # 「预计 N 步内跌破门禁线」的告警阈值（由配置决定，不写死在汇总逻辑里）
        self.near_breach_steps = int(cfg.get("near_breach_steps", 3))
        # 与门禁同源：离门禁多远 / 还有多久跌破，都按门禁的 min_ic 算
        self.min_ic = float(gate.get("min_ic", 0.03))
        self.min_hit_rate = float(gate.get("min_hit_rate", 0.52))
        self.neutral_band = float(cfg.get("neutral_band", 0.0))

    # ------------------------------------------------------------------
    def evaluate(self, horizon: str, horizon_days: int,
                 closes: Sequence[float], scores: Sequence[float]) -> ICTrendResult:
        """滚动切片计算 IC 时序。

        Args:
            closes : 与 scores 对齐的收盘价序列（时间升序）
            scores : 信号分数序列（概率或综合得分）
        """
        res = ICTrendResult(
            horizon=horizon,
            horizon_days=int(horizon_days),
            min_ic=self.min_ic,
            window=self.window,
            step=self.step,
        )
        n = min(len(closes), len(scores))
        if n == 0:
            res.reason = "无数据（收盘价与信号序列均为空）"
            return res
        closes = [float(c) for c in closes[:n]]
        scores = [float(s) for s in scores[:n]]
        rets = forward_returns(closes, int(horizon_days))

        # 滚动锚点：每个锚点取 [end - window, end) 一段，锚点间隔 step。
        # 「窗口内 IC」与「整段 IC」的区别正是本模块存在的理由。
        anchors_end = list(range(self.window, n + 1, self.step))
        if not anchors_end:
            # 数据不足以构成一个完整窗口 → 退化为单点（但必须标注，不让调用方误读成趋势）
            if n >= self.min_obs:
                anchors_end = [n]
            else:
                res.reason = (
                    f"数据不足：有效样本 {n} < 窗口 {self.window} "
                    f"且 < 最小样本 {self.min_obs}"
                )
                return res

        ics: List[float] = []
        hits: List[float] = []
        for end in anchors_end:
            start = max(0, end - self.window)
            seg_s = scores[start:end]
            seg_r = [r for r in rets[start:end]]
            pairs = [
                (s, r) for s, r in zip(seg_s, seg_r)
                if r is not None
            ]
            if len(pairs) < self.min_obs:
                continue
            ss = [p[0] for p in pairs]
            rr = [p[1] for p in pairs]
            ics.append(spearman_ic(ss, rr))
            hits.append(hit_rate(ss, rr, self.neutral_band))

        res.anchors = len(ics)
        res.ic_series = ics
        if res.anchors == 0:
            res.reason = f"无有效锚点（单锚点样本少于 {self.min_obs} 条）"
            return res

        res.available = True
        res.latest_ic = ics[-1]
        res.mean_ic = sum(ics) / len(ics)
        res.latest_hit_rate = hits[-1] if hits else None
        self._finalize_trend(res, ics)
        return res

    def _finalize_trend(self, res: ICTrendResult, ics: List[float]) -> None:
        """由 IC 时序填充斜率 / 趋势 / 预估跌破步数（evaluate 与 evaluate_pairs 共用）。"""
        if res.anchors < self.min_anchors:
            res.status = STATUS_UNKNOWN
            res.reason = f"锚点不足（{res.anchors} < {self.min_anchors}），不足以判定趋势"
            return

        slope = _linear_slope(ics)
        res.slope_per_step = slope
        xs = list(range(len(ics)))
        res.trend_r2 = _pearson(xs, ics) ** 2

        # 「每步衰减百分之多少」：以最近一段 IC 的绝对值为基准，
        # 基准为 0 时无意义（记 None），避免除零产出无穷大。
        base = abs(res.latest_ic or 0.0)
        res.decay_per_step_pct = (slope / base * 100.0) if base > 1e-9 else None

        # 外推跌破门禁线所需步数：只在斜率为负且当前仍未跌破时给出。
        # 已跌破 → 0（现在就是）；斜率非负 → None（不外推"永远安全"）。
        if abs(res.latest_ic or 0.0) < self.min_ic:
            res.steps_to_breach = 0
        elif slope < 0:
            gap = abs(res.latest_ic or 0.0) - self.min_ic
            res.steps_to_breach = int(gap / (-slope))
        else:
            res.steps_to_breach = None

        # 已跌破门禁线 → 状态优先报 `decaying`（含义是"已不可用且不在好转"）。
        # 否则会出现「status=stable 但 steps_to_breach=0（已跌破）」这种自相矛盾的报表。
        # 若此刻已跌破但在回升（斜率 ≥ min_decay），状态报 `improving`，如实反映回升中。
        if abs(res.latest_ic or 0.0) < self.min_ic and slope < self.min_decay:
            res.status = STATUS_DECAYING
            res.reason = (
                f"|IC| {abs(res.latest_ic or 0.0):.4f} 已低于门禁线 {self.min_ic}，"
                f"且每步变化 {slope:+.6f}（未回升）"
            )
        elif slope <= -self.min_decay:
            res.status = STATUS_DECAYING
            if res.decay_per_step_pct is not None:
                res.reason = (
                    f"IC 每步下滑 {abs(slope):.4f}（{abs(res.decay_per_step_pct):.1f}%/步）"
                )
            else:
                res.reason = f"IC 每步下滑 {abs(slope):.4f}"
        elif slope >= self.min_decay:
            res.status = STATUS_IMPROVING
            res.reason = f"IC 每步上升 {slope:.4f}"
        else:
            res.status = STATUS_STABLE
            res.reason = f"IC 无明显趋势（每步变化 {slope:+.4f}）"

    def _summarize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """汇总多周期结果 + 生成待处理事项（供监控/日报直接消费）。"""
        decaying = [h for h, r in results.items() if r["status"] == STATUS_DECAYING]
        near = [
            h for h, r in results.items()
            if r.get("steps_to_breach") is not None
            and 0 < r["steps_to_breach"] <= self.near_breach_steps
        ]
        issues: List[str] = []
        for h in decaying:
            issues.append(f"{h} IC 持续衰减（每步 {results[h].get('slope_per_step')}）")
        for h in near:
            issues.append(
                f"{h} 预计 {results[h]['steps_to_breach']} 步内跌破门禁线（|IC| < {self.min_ic}）"
            )
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "horizons": results,
            "decaying": sorted(decaying),
            "near_breach": sorted(near),
            "issues": issues,
            "min_ic": self.min_ic,
            "window": self.window,
            "step": self.step,
            "near_breach_steps": self.near_breach_steps,
        }

    # ------------------------------------------------------------------
    def evaluate_aligned(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """直接消费**已对齐**的 (scores, returns) 序列 —— 门禁/趋势同源入口。

        为什么不复用 `evaluate(closes, scores)`：walk-forward 的折与折之间
        **时间不连续**，用相邻两行收盘价算 forward return 会跨折错位。因此
        凡是分数来自 walk-forward 折的场景，都应走本方法（收益已由真实收盘价
        算好并按折切片），避免出现「看着像衰减、其实是折缝」的伪趋势。

        Args:
            data: {horizon_name: {"horizon_days": int, "scores": [...], "returns": [...]}}
        """
        results: Dict[str, Any] = {}
        for hname, payload in data.items():
            results[hname] = self.evaluate_pairs(
                hname,
                int(payload.get("horizon_days", 0)),
                payload.get("scores", []) or [],
                payload.get("returns", []) or [],
            ).to_dict()
        return self._summarize(results)

    def evaluate_pairs(self, horizon: str, horizon_days: int,
                       scores: Sequence[float],
                       returns: Sequence[Optional[float]]) -> ICTrendResult:
        """按滚动锚点计算 IC 时序（输入为已对齐的分数与未来收益）。"""
        res = ICTrendResult(
            horizon=horizon,
            horizon_days=int(horizon_days),
            min_ic=self.min_ic,
            window=self.window,
            step=self.step,
        )
        pairs = [
            (float(s), float(r))
            for s, r in zip(scores, returns)
            if r is not None
        ]
        n = len(pairs)
        if n == 0:
            res.reason = "无有效样本（未来收益全部未到期或为空）"
            return res

        anchors_end = list(range(self.window, n + 1, self.step))
        if not anchors_end:
            if n >= self.min_obs:
                anchors_end = [n]
            else:
                res.reason = (
                    f"数据不足：有效样本 {n} < 窗口 {self.window} 且 < 最小样本 {self.min_obs}"
                )
                return res

        ics: List[float] = []
        hits: List[float] = []
        for end in anchors_end:
            start = max(0, end - self.window)
            seg = pairs[start:end]
            if len(seg) < self.min_obs:
                continue
            ss = [p[0] for p in seg]
            rr = [p[1] for p in seg]
            ics.append(spearman_ic(ss, rr))
            hits.append(hit_rate(ss, rr, self.neutral_band))

        res.anchors = len(ics)
        res.ic_series = ics
        if res.anchors == 0:
            res.reason = f"无有效锚点（单锚点样本少于 {self.min_obs} 条）"
            return res

        res.available = True
        res.latest_ic = ics[-1]
        res.mean_ic = sum(ics) / len(ics)
        res.latest_hit_rate = hits[-1] if hits else None
        self._finalize_trend(res, ics)
        return res

    # ------------------------------------------------------------------
    def evaluate_all(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估多周期（输入为原始收盘价 + 分数）。

        Args:
            data: {horizon_name: {"horizon_days": int, "closes": [...], "scores": [...]}}
        """
        results: Dict[str, Any] = {}
        for hname, payload in data.items():
            results[hname] = self.evaluate(
                hname,
                int(payload.get("horizon_days", 0)),
                payload.get("closes", []) or [],
                payload.get("scores", []) or [],
            ).to_dict()
        return self._summarize(results)
