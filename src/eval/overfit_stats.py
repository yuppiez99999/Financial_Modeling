"""组合级过拟合审计（S25 / I5：PSR / DSR / MinTRL / PBO，零新依赖）。

把 S17 的过拟合口径从 IC 层升到**组合净值层**（Issue #54）：I1/I2 的组合
回测引入新的选择自由度（加权臂 × 成本档），不审计则重蹈 S11/S12 的
「事后补记」覆辙。

## 公式口径（Bailey & López de Prado，直译实现；pypbo 为 AGPL 仅公式参考）

- **PSR(SR\\*)** = Φ( (SR − SR\\*)·√(n−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )
  —— 观测 Sharpe 高于基准 SR\\* 的概率；γ₃ 偏度、γ₄ 峰度（非超额，正态=3）。
- **DSR** = PSR(SR₀)，SR₀ 为 N 次试验下期望最大 Sharpe：
  SR₀ = √V[SR]·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]，γ=欧拉常数 0.5772…，
  V[SR] = (1 − γ₃·SR + (γ₄−1)/4·SR²)/(n−1)（单试验实用变体）。
- **MinTRL** = PSR 达到目标概率（缺省 0.95）所需最小样本数：
  n\\* = 1 + (1 − γ₃·SR + (γ₄−1)/4·SR²)·(Φ⁻¹(1−α)/(SR−SR\\*))²。
- **PBO**（Probability of Backtest Overfitting，CSCV）：N 个试验的收益矩阵
  (T × N)，枚举 C(N, N/2) 个对称切分；IS 表现最优的试验在 OS 的秩 logit
  ≤ 0 的比例 —— 「IS 选出的冠军在 OS 拖后腿」的概率。

## 纪律

- 每次审计读数必须声明 N（本次扫描的试验数），与 S17 统一试验预算联动；
- 样本不足 / 常数序列 / 非正方差 → ``available=False``，不猜；
- 只读报表，`affects_gate` 恒 false；是否引入「DSR 校正后仍为正才采信」
  的采信标准属 T25.4 人工检查点。
"""
from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649015329
E = math.e
MIN_SAMPLE = 30


def _phi(x: float) -> float:
    """标准正态 CDF。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """标准正态分位数（Acklam 近似足够 1e-9 精度，零依赖）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在 (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _sr_moments(returns: np.ndarray) -> Optional[Dict[str, float]]:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = len(r)
    if n < MIN_SAMPLE:
        return None
    std = r.std(ddof=1)
    if std <= 1e-12:
        return None
    sr = float(r.mean() / std)
    centered = r - r.mean()
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    gamma3 = m3 / std ** 3
    gamma4 = m4 / std ** 4          # 非超额峰度（正态 = 3）
    return {"n": n, "sr": sr, "gamma3": gamma3, "gamma4": gamma4}


def _psr_with(sr: float, n: int, gamma3: float, gamma4: float,
              benchmark_sr: float) -> Optional[float]:
    denom = 1.0 - gamma3 * sr + (gamma4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return None
    return float(_phi((sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)))


def psr(returns: Sequence[float], benchmark_sr: float = 0.0) -> Optional[float]:
    """Probabilistic Sharpe Ratio（对基准 SR\\* 的优势概率）。"""
    m = _sr_moments(returns)
    if m is None:
        return None
    return _psr_with(m["sr"], m["n"], m["gamma3"], m["gamma4"], benchmark_sr)


def dsr(returns: Sequence[float], n_trials: int) -> Optional[float]:
    """Deflated Sharpe Ratio：PSR 对「N 次试验的期望最大 SR」校正。"""
    if n_trials < 1:
        raise ValueError("n_trials 必须 ≥ 1")
    if n_trials == 1:
        # 单试验无选择偏差 → DSR 退化为 PSR(SR*=0)
        return psr(returns, benchmark_sr=0.0)
    m = _sr_moments(returns)
    if m is None:
        return None
    denom = 1.0 - m["gamma3"] * m["sr"] + (m["gamma4"] - 1.0) / 4.0 * m["sr"] ** 2
    if denom <= 0:
        return None
    v_sr = denom / (m["n"] - 1)
    sr0 = math.sqrt(max(v_sr, 0.0)) * (
        (1 - EULER_GAMMA) * _phi_inv(1 - 1.0 / n_trials)
        + EULER_GAMMA * _phi_inv(1 - 1.0 / (n_trials * E)))
    return _psr_with(m["sr"], m["n"], m["gamma3"], m["gamma4"], sr0)


def min_track_length(returns: Sequence[float], target: float = 0.95,
                     benchmark_sr: float = 0.0) -> Optional[int]:
    """PSR 达到 target 所需最小样本数（对基准 SR\\*）。"""
    if not 0 < target < 1:
        raise ValueError("target 必须在 (0,1)")
    m = _sr_moments(returns)
    if m is None:
        return None
    denom = 1.0 - m["gamma3"] * m["sr"] + (m["gamma4"] - 1.0) / 4.0 * m["sr"] ** 2
    if denom <= 0 or m["sr"] <= benchmark_sr:
        return None      # SR 不高于基准 → 无限长跟踪期也到不了，如实返回
    z = _phi_inv(target)
    n_star = 1.0 + denom * (z / (m["sr"] - benchmark_sr)) ** 2
    return int(math.ceil(n_star))


def pbo_cscv(returns_matrix: np.ndarray, n_blocks: int = 16) -> Optional[float]:
    """PBO（CSCV）：returns_matrix (T × N)，N 个试验的逐期收益。

    标准 CSCV（Bailey et al.）：把 **时间维** 切成 ``n_blocks`` 个连续块，
    枚举 C(n_blocks, n_blocks/2) 种「训练块组合」（测试块 = 补集）；
    每种组合下按训练段 Sharpe 选冠军，看冠军在测试段的相对秩，
    λ = log(r/(N+1−r))，PBO = P(λ ≤ 0)（冠军在 OS 掉到后一半的概率）。

    性能：块级 (n, Σx, Σx²) 预聚合，组合内 mean/std 可加重组，不重扫原始行。
    """
    r = np.asarray(returns_matrix, dtype="float64")
    if r.ndim != 2 or r.shape[0] < MIN_SAMPLE or r.shape[1] < 2:
        return None
    t_total = r.shape[0]
    nb = min(int(n_blocks), t_total // 15)
    if nb < 2:
        return None
    if nb % 2 != 0:
        nb -= 1
    if nb < 2:
        return None
    t_use = (t_total // nb) * nb
    r = r[:t_use]
    block_len = t_use // nb

    # 块级预聚合：count / Σx / Σx²（逐块逐策略）
    counts, s1, s2 = [], [], []
    for b in range(nb):
        seg = r[b * block_len:(b + 1) * block_len]
        counts.append(len(seg))
        s1.append(seg.sum(axis=0))
        s2.append((seg ** 2).sum(axis=0))
    counts = np.asarray(counts, dtype="float64")
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)

    def subset_sharpe(blocks: Sequence[int]) -> np.ndarray:
        idx = list(blocks)
        n = counts[idx].sum()
        mean = s1[idx].sum(axis=0) / n
        var = s2[idx].sum(axis=0) / n - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return np.where(std > 1e-12, mean / np.where(std > 1e-12, std, 1.0), 0.0)

    half = nb // 2
    overfit_flags: List[float] = []
    for is_blocks in combinations(range(nb), half):
        os_blocks = [b for b in range(nb) if b not in is_blocks]
        is_sr = subset_sharpe(is_blocks)
        os_sr = subset_sharpe(os_blocks)
        champ = int(np.argmax(is_sr))
        # PBO 判据（Bailey et al. 的操作化定义）：
        # IS 冠军在测试段能否跑赢过半对手 —— 跑不赢（落在后一半）即过拟合迹象。
        beaten = float((os_sr < os_sr[champ]).sum())
        rel = beaten / (r.shape[1] - 1)      # 冠军击败的对手占比 ∈ [0,1]
        overfit_flags.append(1.0 if rel < 0.5 else 0.0)
    if not overfit_flags:
        return None
    pbo = float(np.mean(overfit_flags))
    return round(pbo, 6)


def audit_series(returns: Sequence[float], n_trials: int,
                 name: str = "series") -> Dict[str, Any]:
    """单条净值收益序列的审计读数（PSR/DSR/MinTRL + 矩），供报表消费。"""
    m = _sr_moments(returns)
    if m is None:
        return {"name": name, "available": False,
                "reason": f"有效样本不足（< {MIN_SAMPLE}）或方差为 0",
                "affects_gate": False}
    p = psr(returns)
    d = dsr(returns, n_trials)
    mtl = min_track_length(returns)
    return {
        "name": name,
        "available": True,
        "affects_gate": False,
        "n": m["n"],
        "sr_per_period": round(m["sr"], 6),
        "skew": round(m["gamma3"], 6),
        "kurtosis": round(m["gamma4"], 6),
        "psr": round(p, 6) if p is not None else None,
        "dsr": round(d, 6) if d is not None else None,
        "min_track_length_days": mtl,
        "n_trials": int(n_trials),
        "note": "DSR < PSR 恒成立（N>1）；采信标准属 T25.4 人工检查点",
    }
