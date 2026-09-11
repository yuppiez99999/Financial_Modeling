"""三重障碍法标签（S12 / G2）：替代写死的 5/10/20 日固定窗口标签。

问题从哪来（Issue #29 集成方案 G2）：
  现有标签由 ``FeatureEngineer.create_target`` 构造：
  ``target = 未来 h 日收盘收益 > 0``，其中 h 是**写死的** 5/10/20 日。
  这个口径有两个硬伤：
    1. **与波动率无关**：高波动标的 5 日就能走出 10%，低波动标的 20 日
       也未必动 1%。同一个 h 把不同标的的"信息量"强行拉平，标签噪声大；
    2. **只看终点不看路径**：不区分"先跌 15% 再涨 2%"（早被止损打掉）
       和"一路上涨 2%"。终点符号相同，但真实可执行结果完全相反。
  门禁长期卡在 short/mid 命中率 52% —— 标签口径是首要嫌疑（见
  ``00_kickoff/eval_criteria.md``：「样本期收益」口径过粗）。

参照 stefan-jansen/machine-learning-for-trading（MIT）的 **triple-barrier
method**（López de Prado《Advances in Financial Machine Learning》），
用「止盈 + 止损 + 时间」三重障碍给样本打标签，并让止盈/止损阈值
**随波动率自适应**（volatility-scaled），取代固定窗口。

本模块做什么：
  1. **三重障碍打标**：对每个时点 i，从 i+1 起向前扫描，谁先触发
     - 上障碍（止盈）→ +1
     - 下障碍（止损）→ -1
     - 时间障碍（到期）→ 按终点符号给 ±1（0 归 0）
     三者**触及时点最早**者胜出；用于生成三分类 ``label``；
  2. **波动率自适应阈值**：上下障碍 = ``k × σ_t``，σ_t 由**截至 t 的**
     滚动波动率估计（EWMA 或滚动 std，不含未来信息），k 可配；
  3. **二分类视图**：三分类 ``label`` 可折叠为 ``label_binary``（+1 为 1，
     -1/0 为 0），与现有 ``target_{h}d`` 语义对齐，便于 A/B 直接对比。

本模块**不做什么**（边界比功能重要）：
  - **不打未来信息**：所有滚动统计只用 ``[.., t]``，决策点 t 之后的价格
    仅用于判定障碍是否触发（这是标签本身的定义，非特征泄漏）；
  - **不改门禁**：标签产出只喂给训练/评估，是否替换主线由人工检查点决定；
  - **不猜**：数据不足（长度 < 窗口+horizon）时返回 NaN，绝不凑数。

无前视要点（对照 ``00_kickoff/leakage_checklist.md``）：
  - 标签窗口起止与预测时点一致：判定区间恒为 (t, t+horizon]；
  - 波动率只用过去：``volatility`` 序列第 i 个元素仅依赖 closes[:i+1]；
  - 逐标的构造：多标的 concat 后统一扫描会跨标的取价（同 trainer 注释）。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# 默认参数（可被 config.labeling 覆盖；三分类标签的默认口径写死在此，
# 避免"参数遍地找"造成选择自由度回流）
DEFAULT_K_UP = 1.0          # 上障碍倍数：profit_target = k_up × σ
DEFAULT_K_DOWN = 1.0        # 下障碍倍数：stop_loss     = k_down × σ
DEFAULT_VOL_WINDOW = 20     # 波动率滚动窗口（日）
DEFAULT_HORIZON = 5         # 时间障碍（日）
MIN_VOL_SAMPLES = 5         # 波动率最少样本，不足则该点不可用


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (config or {}).get("labeling", {}) or {}


def rolling_volatility(
    closes: Sequence[float],
    window: int = DEFAULT_VOL_WINDOW,
    *,
    log_returns: bool = True,
) -> List[Optional[float]]:
    """滚动波动率（截至 t，不含未来信息）。

    第 i 个元素 = 由 ``closes[max(0, i-window+1) : i+1]`` 估计的标准差：
      - ``log_returns=True``：用对数收益差分的 std（更接近几何布朗运动假设）；
      - 否则：用简单收益差分。

    样本 < ``MIN_VOL_SAMPLES`` 时该点为 None（不足不猜）。
    """
    n = len(closes)
    out: List[Optional[float]] = [None] * n
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = closes[lo : i + 1]
        if len(seg) < MIN_VOL_SAMPLES:
            continue
        arr = np.asarray(seg, dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            continue
        if log_returns:
            rets = np.diff(np.log(arr))
        else:
            rets = np.diff(arr) / arr[:-1]
        if rets.size < 2:
            continue
        vol = float(np.std(rets, ddof=1))
        out[i] = vol if math.isfinite(vol) else None
    return out


def triple_barrier_labels(
    closes: Sequence[float],
    *,
    highs: Optional[Sequence[float]] = None,
    lows: Optional[Sequence[float]] = None,
    horizon: int = DEFAULT_HORIZON,
    k_up: float = DEFAULT_K_UP,
    k_down: float = DEFAULT_K_DOWN,
    vol_window: int = DEFAULT_VOL_WINDOW,
    volatility: Optional[Sequence[Optional[float]]] = None,
) -> List[Optional[int]]:
    """三重障碍法打标，返回与 closes 等长的三分类标签序列。

    对每个有效时点 i（有波动率估计且 i+horizon < n）：
      - 上障碍价 ``up  = close_i × (1 + k_up  × σ_i)``；
      - 下障碍价 ``down= close_i × (1 - k_down× σ_i)``；
      - 从 i+1 到 i+horizon 逐日扫描，**先触碰者胜**：
          触碰上障碍 → +1；触碰下障碍 → -1；
          同日同时触碰（罕见，用当日 bar 的 high/low 无法区分先后）→ 0（不猜）；
      - 都没触碰 → 时间障碍：终点收益 > 0 → +1，< 0 → -1，= 0 → 0。

    参数：
      - ``highs`` / ``lows``：若提供，用日内高低价判定触碰（更接近真实触发，
        且不会被收盘价抹平路径）；缺省用收盘价近似（保守，会低估触发）。
      - ``volatility``：外部传入的 σ 序列（便于测试/复用）；缺省内部按
        ``rolling_volatility(closes, vol_window)`` 计算。

    返回：长度 = len(closes)；无效/未到期位置为 None（绝不猜）。
    """
    n = len(closes)
    if n == 0:
        return []
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon 必须为正整数")

    vol = list(volatility) if volatility is not None else rolling_volatility(
        closes, vol_window
    )
    hi = list(highs) if highs is not None else None
    lo = list(lows) if lows is not None else None

    labels: List[Optional[int]] = [None] * n
    for i in range(n):
        sigma = vol[i] if i < len(vol) else None
        if sigma is None:
            continue
        j_end = i + horizon
        if j_end >= n:
            continue  # 未到期，不猜
        base = float(closes[i])
        if not math.isfinite(base) or base <= 0:
            continue
        up = base * (1.0 + k_up * sigma)
        down = base * (1.0 - k_down * sigma)

        label: Optional[int] = None
        for j in range(i + 1, j_end + 1):
            if hi is not None or lo is not None:
                # 有高低价：分别判定上下触碰
                h_j = float(hi[j]) if (hi is not None and j < len(hi)) else float(closes[j])
                l_j = float(lo[j]) if (lo is not None and j < len(lo)) else float(closes[j])
                touch_up = h_j >= up
                touch_down = l_j <= down
            else:
                c_j = float(closes[j])
                touch_up = c_j >= up
                touch_down = c_j <= down
            if touch_up and touch_down:
                label = 0  # 同日双向触发：顺序不可知，如实给中性
                break
            if touch_up:
                label = 1
                break
            if touch_down:
                label = -1
                break
        if label is None:
            # 时间障碍：按终点符号
            end_ret = (float(closes[j_end]) - base) / base
            if end_ret > 0:
                label = 1
            elif end_ret < 0:
                label = -1
            else:
                label = 0
        labels[i] = label
    return labels


def to_binary(labels: Sequence[Optional[int]]) -> List[Optional[float]]:
    """三分类 → 二分类（与现有 ``target_{h}d`` 语义对齐）。

    +1 → 1.0；-1 / 0 → 0.0；None → None（未到期保留缺失）。
    用于在同一训练配置下做「旧标签 vs 新标签」的 A/B 直接对比。
    """
    out: List[Optional[float]] = []
    for lb in labels:
        if lb is None:
            out.append(None)
        else:
            out.append(1.0 if lb > 0 else 0.0)
    return out


def label_summary(labels: Sequence[Optional[int]]) -> Dict[str, Any]:
    """标签分布摘要（诊断用，report_only）。

    返回可用样本数、三类占比、未到期数；样本为 0 时 ``available=False``。
    """
    valid = [x for x in labels if x is not None]
    n = len(labels)
    if not valid:
        return {
            "available": False,
            "reason": "no_valid_labels",
            "total": n,
            "valid": 0,
        }
    nv = len(valid)
    pos = sum(1 for x in valid if x > 0)
    neg = sum(1 for x in valid if x < 0)
    neu = nv - pos - neg
    return {
        "available": True,
        "total": n,
        "valid": nv,
        "pending": n - nv,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "pos_ratio": round(pos / nv, 4),
        "neg_ratio": round(neg / nv, 4),
        "neutral_ratio": round(neu / nv, 4),
    }


class TripleBarrierLabeler:
    """三重障碍法标签器（面向项目接线：逐标的构造，避免跨标的污染）。

    用法：
        labeler = TripleBarrierLabeler(config)
        labels = labeler.label_series(df, horizon=5)   # df 含 close/high/low
        df = labeler.attach(df, horizon=5)             # 追加 label_tb_{h}d 列
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = _cfg(config)
        self.k_up = float(cfg.get("k_up", DEFAULT_K_UP))
        self.k_down = float(cfg.get("k_down", DEFAULT_K_DOWN))
        self.vol_window = int(cfg.get("vol_window", DEFAULT_VOL_WINDOW))
        self.horizon = int(cfg.get("horizon", DEFAULT_HORIZON))

    def label_series(self, df, horizon: Optional[int] = None) -> List[Optional[int]]:
        """对单标的 DataFrame 打标（需含 close；有 high/low 则用之）。"""
        import pandas as pd  # noqa: F401  仅用于类型判定

        if df is None or len(df) == 0:
            return []
        h = int(horizon if horizon is not None else self.horizon)
        closes = df["close"].tolist()
        highs = df["high"].tolist() if "high" in df.columns else None
        lows = df["low"].tolist() if "low" in df.columns else None
        cols = set(df.columns)
        return triple_barrier_labels(
            closes,
            highs=highs if highs is not None and "high" in cols else None,
            lows=lows if lows is not None and "low" in cols else None,
            horizon=h,
            k_up=self.k_up,
            k_down=self.k_down,
            vol_window=self.vol_window,
        )

    def attach(self, df, horizon: Optional[int] = None, *, binary: bool = False):
        """在 df 上追加 ``label_tb_{h}d``（及可选 ``label_tb_{h}d_bin``）列。"""
        h = int(horizon if horizon is not None else self.horizon)
        labels = self.label_series(df, h)
        out = df.copy()
        out[f"label_tb_{h}d"] = labels
        if binary:
            out[f"label_tb_{h}d_bin"] = to_binary(labels)
        return out
