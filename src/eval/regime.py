"""市场状态识别与状态内分层评估（S18 / H3，T18.1 + T18.2 + T18.3）。

问题从哪来（S15 / S17 的共同读数）：
  S15 收敛搜索后三周期 **IC 均为正、OOT 命中率却全部 < 52%**；
  S17 的 CPCV 收缩指标显示「选择偏差没有吃掉全部 IC」——
  两轮都指向同一句话：**IC 为正但命中率卡线**。
  S9 已从**资产维度**（个股 / ETF 分池）回答「谁的池」，
  但从未从**时间维度**回答「什么时候」：模型是不是只在某些市场状态下有效？

参照 hmmlearn（HMM 市场状态识别）的做法，本模块提供：

  1. **T18.1 状态识别 `fit_regime_model` / `regime_labels`**：
     用高斯隐马尔可夫模型（GaussianHMM）对**市场收益 + 波动率**序列做状态
     识别，把每个交易日打上 `bull / bear / range` 三态标签之一。
     状态标签严格**只用当日及之前信息**（无前视，见下）；
  2. **T18.2 状态内分层评估 `stratified_by_regime`**：
     在**同一批样本**上按状态分组，逐组给 IC / 命中率 / 样本数，
     回答「模型是否只在特定状态有效」；
  3. **T18.3 状态作为特征的增量验证 `compare_regime_feature`**：
     同数据 / 同折 / 同模型、**只加一个变量**（状态 one-hot）的 A/B，
     保守口径判定（IC 与命中率**同向**变好才算改善迹象）。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：``affects_gate`` 恒为 False；状态是否进入门禁 / 风控
    ``withheld`` 语义属 T18.4 人工检查点（本模块只产证据）；
  - **不自动落地**：``model.regime.enabled`` 缺省 false，状态不进生产特征集；
  - **不猜**：hmmlearn 缺失 / 样本不足 / 某状态样本过少时如实
    ``available=false`` + 原因，不降级成假标签、不硬凑三段；
  - **不虚报因果**：状态是**统计归纳**（拟合出来的隐状态），不是
    「牛市/熊市」的客观标签；报告里对 ``bull/bear/range`` 的命名只按
    **拟合出的状态均值收益排序**做可读化映射，并如实标注该映射口径。

无前视说明（这是本模块最容易被写错的地方）：
  - HMM 一次 `fit` 会看到**整段观测**——若直接对全样本 decode，则第 t 天
    的状态用到了 t 之后的收益信息，属**前视**。因此本模块提供两种口径：
      * ``expanding``（缺省，严格无前视）：第 t 天的状态由**只用
        [0, t] 观测**重新 fit 出来的模型给出（`--refit-every` 控制重训步长，
        步长内沿用最近一次模型，仍是只用过去信息）；
      * ``full_sample``（**仅供对照，明确标注有前视**）：对全样本一次 fit
        再 decode，报告里 ``lookahead_prefixed=True``，不得用于达标证据；
  - 状态只由**市场层收益 / 波动**决定（不消费标签、不消费未来收益），
    回归用例：篡改 t 之后的观测 → t 的状态标签必须不变；
  - NaN / 样本不足处如实返回 ``None``（无标签），不填默认状态。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 状态数：**固定 3**（bull / bear / range）。定成常量而不是配置项，
# 是为了防止「扫几个 n_states 挑好看的那个」这种选择自由度回流——
# 那正是 S11/S17 反复要防的事。要改口径必须改这里并重跑全部证据。
N_STATES = 3
STATE_NAMES = ("bear", "range", "bull")   # 按拟合状态均值收益**升序**映射
DEFAULT_REFIT_EVERY = 20   # expanding 口径的重训步长（交易日）
DEFAULT_WINDOW = 20        # 状态观测的回看窗口（收益/波动）
MIN_FIT_SAMPLES = 60       # 低于此样本数不 fit（不猜）
MIN_STATE_SAMPLES = 30     # 分层评估时单状态最少样本数（不足不发指标）
MIN_STRATIFIED_STATES = 2  # 至少两个状态可评估才算分层成立

_REGIME_ORDER = {name: i for i, name in enumerate(STATE_NAMES)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# 可用性探测
# ----------------------------------------------------------------------
def hmmlearn_available() -> bool:
    """hmmlearn 是否可用（缺省不静默降级：没装就明确失败）。"""
    try:
        import hmmlearn  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def hmmlearn_version() -> Optional[str]:
    try:
        import hmmlearn
        return getattr(hmmlearn, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# 观测构造（只用价格历史，无前视）
# ----------------------------------------------------------------------
def build_observations(close: Sequence[float], volume: Optional[Sequence[float]] = None,
                       window: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    """把收盘价序列转成 HMM 观测矩阵（**只用当日及之前**的信息）。

    观测两列：
      - ``ret``：日对数收益（当期值，不含未来）；
      - ``vol``：``window`` 日滚动收益波动率（rolling std，只回看不前视）。

    返回 ``{"X": ndarray[n,2], "valid": bool_ndarray[n], "reason": str|None}``。
    样本不足（< window + 2）时 ``valid`` 全 False 并给原因，不猜。
    """
    c = np.asarray(close, dtype=float)
    n = int(len(c))
    out = {"X": np.zeros((n, 2), dtype=float),
           "valid": np.zeros(n, dtype=bool),
           "reason": None}
    if n < int(window) + 2:
        out["reason"] = f"insufficient_samples({n}<{int(window) + 2})"
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(c[1:] / c[:-1])
    ret = np.concatenate([[0.0], ret])
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.full(n, np.nan, dtype=float)
    for i in range(int(window) - 1, n):
        seg = ret[i - int(window) + 1: i + 1]
        vol[i] = float(np.std(seg))
    valid = np.isfinite(vol)
    out["X"][:, 0] = ret
    out["X"][:, 1] = np.nan_to_num(vol, nan=0.0)
    out["valid"] = valid
    if not valid.any():
        out["reason"] = "no_valid_observations"
    return out


# ----------------------------------------------------------------------
# 状态拟合（HMM，可复现）
# ----------------------------------------------------------------------
def fit_hmm(X: np.ndarray, n_states: int = N_STATES, seed: int = 42):
    """在观测矩阵上 fit 一个 GaussianHMM（固定种子，可复现）。

    返回 ``(model, means)``；hmmlearn 缺失 / 样本不足时抛 ``ValueError``
    （调用方负责转成 ``available=false``，不静默给随机标签）。
    """
    import hmmlearn.hmm as hm

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < MIN_FIT_SAMPLES:
        raise ValueError(f"insufficient_fit_samples({X.shape[0] if X.ndim == 2 else 0})")
    if not np.isfinite(X).all():
        raise ValueError("non_finite_observations")
    model = hm.GaussianHMM(n_components=int(n_states), covariance_type="diag",
                           n_iter=100, random_state=int(seed), tol=1e-4)
    model.fit(X)
    means = np.asarray(model.means_, dtype=float)[:, 0]
    return model, means


def name_states(means: Sequence[float]) -> Dict[int, str]:
    """把拟合出的隐状态按**状态均值收益升序**映射为 bear / range / bull。

    这只是**可读化映射**，不代表客观牛熊判定；报告里如实标注口径。
    状态数与 ``STATE_NAMES`` 不一致时退化为 ``state_<i>``（不硬套牛熊）。
    """
    order = np.argsort(np.asarray(means, dtype=float))
    names: Dict[int, str] = {}
    if len(order) == len(STATE_NAMES):
        for rank, idx in enumerate(order):
            names[int(idx)] = STATE_NAMES[rank]
    else:
        for rank, idx in enumerate(order):
            names[int(idx)] = f"state_{rank}"
    return names


def regime_labels(X: np.ndarray, valid: np.ndarray, n_states: int = N_STATES,
                  refit_every: int = DEFAULT_REFIT_EVERY,
                  seed: int = 42) -> Dict[str, Any]:
    """**expanding 无前视**状态标签：第 t 天只用 ``[0, t]`` 的观测。

    做法：从 ``MIN_FIT_SAMPLES`` 起步，每 ``refit_every`` 个有效样本重新
    fit 一次模型；两次重训之间，新样本用**最近一次**模型（该模型只见过
    更早的数据）做 ``predict``——因此任一 t 的标签都只依赖 ≤ t 的信息。

    返回 ``{"labels": list[str|None], "states": list[int|None], "meta": {...}}``。
    """
    X = np.asarray(X, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    n = int(len(X))
    labels: List[Optional[str]] = [None] * n
    states: List[Optional[int]] = [None] * n
    idx_valid = np.flatnonzero(valid)
    if len(idx_valid) < MIN_FIT_SAMPLES:
        return {"labels": labels, "states": states,
                "meta": {"mode": "expanding", "available": False,
                         "reason": f"insufficient_valid_samples({len(idx_valid)}<{MIN_FIT_SAMPLES})"}}

    model = None
    names: Dict[int, str] = {}
    next_refit = 0
    refits = 0
    for k, i in enumerate(idx_valid):
        if model is None or k >= next_refit:
            hist = X[idx_valid[: k + 1]]
            try:
                model, means = fit_hmm(hist, n_states=n_states, seed=seed)
                names = name_states(means)
                refits += 1
            except ValueError:
                continue
            next_refit = (k // max(1, int(refit_every)) + 1) * max(1, int(refit_every))
        if model is None:
            continue
        try:
            s = int(model.predict(X[i: i + 1])[0])
        except Exception:  # noqa: BLE001 - 单个点预测失败标 None，不猜
            continue
        states[i] = s
        labels[i] = names.get(s, f"state_{s}")
    return {"labels": labels, "states": states,
            "meta": {"mode": "expanding", "available": any(l is not None for l in labels),
                     "n_states": int(n_states), "refit_every": int(refit_every),
                     "refits": int(refits), "lookahead_prefixed": False,
                     "mapping": "state_mean_return_ascending", "seed": int(seed)}}


def regime_labels_full_sample(X: np.ndarray, valid: np.ndarray,
                              n_states: int = N_STATES,
                              seed: int = 42) -> Dict[str, Any]:
    """**仅供对照**的全样本标签（**有前视**，明确标注，不得作达标证据）。

    对全部有效观测一次 fit 再 decode —— 第 t 天的状态用到了 t 之后的收益。
    保留它的唯一理由是：让「expanding 口径 vs 全样本口径」的差异**可测量**，
    而不是让人以为标签本来就没前视。
    """
    X = np.asarray(X, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    n = int(len(X))
    labels: List[Optional[str]] = [None] * n
    states: List[Optional[int]] = [None] * n
    idx_valid = np.flatnonzero(valid)
    try:
        model, means = fit_hmm(X[idx_valid], n_states=n_states, seed=seed)
    except ValueError as e:
        return {"labels": labels, "states": states,
                "meta": {"mode": "full_sample", "available": False, "reason": str(e),
                         "lookahead_prefixed": True}}
    names = name_states(means)
    pred = model.predict(X[idx_valid])
    for i, s in zip(idx_valid, pred):
        si = int(s)
        states[i] = si
        labels[i] = names.get(si, f"state_{si}")
    return {"labels": labels, "states": states,
            "meta": {"mode": "full_sample", "available": True,
                     "n_states": int(n_states), "lookahead_prefixed": True,
                     "mapping": "state_mean_return_ascending", "seed": int(seed)}}


# ----------------------------------------------------------------------
# 状态作为特征（one-hot）
# ----------------------------------------------------------------------
def regime_feature_columns() -> List[str]:
    """状态 one-hot 特征列名（与既有特征命名空间隔离）。"""
    return [f"factor_regime_{name}" for name in STATE_NAMES]


def regime_features(labels: Sequence[Optional[str]]) -> np.ndarray:
    """状态标签 → one-hot 特征矩阵；``None``（未知状态）→ 全 0（如实中性）。"""
    cols = regime_feature_columns()
    pos = {c: i for i, c in enumerate(cols)}
    out = np.zeros((len(labels), len(cols)), dtype=float)
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        j = pos.get(f"factor_regime_{lab}")
        if j is not None:
            out[i, j] = 1.0
    return out


# ----------------------------------------------------------------------
# T18.2 状态内分层评估
# ----------------------------------------------------------------------
def stratified_by_regime(labels: Sequence[Optional[str]],
                         proba: Sequence[float],
                         fwd_ret: Sequence[float],
                         y_true: Optional[Sequence[int]] = None,
                         group_key: str = "regime") -> Dict[str, Any]:
    """按状态分组给 IC / 命中率 / 样本数（保守口径，样本不足不发指标）。

    ``hit_rate`` 用与门禁同源的方向命中率：``sign(proba-0.5) == sign(fwd_ret)``；
    同时给出基于标签的 ``accuracy``（y_true 可用时）。群组样本 < 
    ``MIN_STATE_SAMPLES`` 时标 ``insufficient`` 并不给指标（不猜）。
    """
    from src.inference.ic import spearman_ic

    labels = list(labels)
    proba = np.asarray(proba, dtype=float)
    fwd = np.asarray(fwd_ret, dtype=float)
    yt = None if y_true is None else np.asarray(y_true, dtype=float)

    groups: Dict[str, List[int]] = {}
    unknown = 0
    for i, lab in enumerate(labels):
        if lab is None:
            unknown += 1
            continue
        groups.setdefault(str(lab), []).append(i)

    rows: Dict[str, Any] = {}
    for name in sorted(groups, key=lambda x: _REGIME_ORDER.get(x, 99)):
        idx = np.asarray(groups[name], dtype=int)
        n_grp = int(len(idx))
        if n_grp < MIN_STATE_SAMPLES:
            rows[name] = {"available": False, "samples": n_grp,
                          "reason": f"insufficient_samples({n_grp}<{MIN_STATE_SAMPLES})"}
            continue
        p = proba[idx]
        r = fwd[idx]
        pn = np.nan_to_num(p, nan=0.5)
        rn = np.nan_to_num(r, nan=0.0)
        hit = float(np.mean(np.sign(pn - 0.5) == np.sign(rn)))
        row: Dict[str, Any] = {
            "available": True,
            "samples": n_grp,
            "ic": round(float(spearman_ic(p, r)), 6),
            "hit_rate": round(hit, 6),
        }
        if yt is not None and len(yt) == len(labels):
            row["accuracy"] = round(float(np.mean((pn > 0.5) == (yt[idx] > 0.5))), 6)
        rows[name] = row

    usable = sum(1 for v in rows.values() if v.get("available"))
    covered = int(sum(v.get("samples", 0) for v in rows.values() if v.get("available")))
    return {
        "available": usable >= MIN_STRATIFIED_STATES,
        "reason": None if usable >= MIN_STRATIFIED_STATES
                  else f"insufficient_usable_states({usable}<{MIN_STRATIFIED_STATES})",
        "group_key": group_key,
        "states": rows,
        "usable_states": usable,
        "unknown_samples": int(unknown),
        "covered_samples": covered,
        "evaluated_samples": int(len(labels)),
    }


def summarize_regime_spread(stratified: Dict[str, Any]) -> Dict[str, Any]:
    """把分层结果压成一句话：状态间命中率是否**真的分得开**。

    判定口径（保守，不看绝对水平只看差异）：
      - 可用状态 < 2 → ``insufficient``；
      - 命中率极差 < 2pt → ``uniform``（状态分不开，时间维度解释力弱）；
      - 极差 ≥ 2pt 且最优状态样本占比 ≥ 10% → ``differentiated``；
      - 其余 → ``marginal``。
    **不判定「模型只在某状态有效」的因果结论**——那属 T18.4 人判。
    """
    states = {k: v for k, v in (stratified.get("states") or {}).items()
              if v.get("available")}
    if len(states) < 2:
        return {"verdict": "insufficient", "spread": None,
                "best_state": None, "worst_state": None}
    hits = {k: float(v["hit_rate"]) for k, v in states.items()}
    best = max(hits, key=hits.get)
    worst = min(hits, key=hits.get)
    spread = hits[best] - hits[worst]
    covered = max(1, int(stratified.get("covered_samples") or 1))
    best_share = float(states[best]["samples"]) / covered
    if spread < 0.02:
        verdict = "uniform"
    elif best_share >= 0.10:
        verdict = "differentiated"
    else:
        verdict = "marginal"
    return {"verdict": verdict, "spread": round(spread, 6),
            "best_state": best, "best_hit_rate": round(hits[best], 6),
            "worst_state": worst, "worst_hit_rate": round(hits[worst], 6),
            "best_state_sample_share": round(best_share, 6)}


# ----------------------------------------------------------------------
# T18.3 状态作为特征的增量验证（同口径 A/B）
# ----------------------------------------------------------------------
def compare_regime_feature(base_ic: float, base_hit: float,
                           aug_ic: float, aug_hit: float,
                           min_samples_ok: bool = True) -> Dict[str, Any]:
    """只加「状态 one-hot」一个变量的 A/B 判定（保守口径，**不择优**）。

    与 S12 label-ab / S13 qlib-ab 同一判定语言：
      - IC 与命中率**同向变好** → ``improved``；
      - 同向变差 → ``degraded``；
      - 一升一降 → ``mixed``（不挑对自己有利的那条引用）；
      - 样本不足 → ``insufficient_samples``。
    """
    if not min_samples_ok:
        return {"verdict": "insufficient_samples",
                "delta_ic": None, "delta_hit": None}
    dic = float(aug_ic) - float(base_ic)
    dhit = float(aug_hit) - float(base_hit)
    ic_up = dic > 0
    hit_up = dhit > 0
    if abs(dic) < 1e-9 and abs(dhit) < 1e-9:
        verdict = "unchanged"
    elif ic_up and hit_up:
        verdict = "improved"
    elif (not ic_up) and (not hit_up):
        verdict = "degraded"
    else:
        verdict = "mixed"
    return {"verdict": verdict,
            "delta_ic": round(dic, 6), "delta_hit": round(dhit, 6),
            "base_ic": round(float(base_ic), 6), "aug_ic": round(float(aug_ic), 6),
            "base_hit": round(float(base_hit), 6), "aug_hit": round(float(aug_hit), 6)}


# ----------------------------------------------------------------------
# 报告装配
# ----------------------------------------------------------------------
def build_regime_report(horizons: Dict[str, Any], meta: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """装配状态分层报告（落盘 ``reports/regime/regime_stratification.json``）。"""
    return {
        "generated_at": _now(),
        "module": "regime",
        "stage": "S18/H3",
        "tasks": ["T18.1", "T18.2", "T18.3"],
        "affects_gate": False,
        "freeze_structure": True,
        "lookahead_policy": {
            "primary": "expanding（第 t 天只用 [0,t] 观测，严格无前视）",
            "reference": "full_sample（有前视，仅作差异对照，不得作达标证据）",
        },
        "n_states": N_STATES,
        "state_names": list(STATE_NAMES),
        "min_state_samples": MIN_STATE_SAMPLES,
        "horizons": horizons,
        "meta": dict(meta or {}),
    }


def build_ab_report(horizons: Dict[str, Any], meta: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """装配状态特征 A/B 报告（落盘 ``reports/regime/regime_feature_ab.json``）。"""
    verdicts = {h: (v.get("verdict") if isinstance(v, dict) else None)
                for h, v in horizons.items()}
    return {
        "generated_at": _now(),
        "module": "regime_feature_ab",
        "stage": "S18/H3",
        "task": "T18.3",
        "affects_gate": False,
        "single_variable": "factor_regime_*（状态 one-hot，仅此一项）",
        "verdicts": verdicts,
        "horizons": horizons,
        "meta": dict(meta or {}),
    }


def regime_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    """报告目录（缺省 ``reports/regime``）。"""
    base = ((config or {}).get("report", {}) or {}).get("output_dir", "reports")
    return Path(base) / "regime"


def regime_report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    return regime_dir(config) / "regime_stratification.json"


def ab_report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    return regime_dir(config) / "regime_feature_ab.json"
