"""保形预测区间（S16 / H1，T16.1 + T16.2）：给概率预测配上**覆盖率保证**的区间，
并把区间宽度经既有 `confidence_from_interval` 换算成置信分，与现行
`|p − 0.5| × 2` 口径做**同数据、同折、同模型**对照。

问题从哪来（T15.3 遗留，Issue #29 / #40 的 H1 定义）：
  1. 现行置信度口径 `|p − 0.5| × 2` 建立在**未校准**的概率上：模型说 0.9 就
     真的对 90% 吗？仓库里既无 Brier/ECE，也无覆盖率保证；
  2. S15 把 `confidence_from_interval`（区间宽度 → 置信分）抽象就绪，但
     **从未喂过真实区间** —— 缺的正是「带覆盖率保证的区间」这一步。

本模块做什么（两件事，一件一个函数）：
  1. **T16.1 `run_conformal_interval`**：用 MAPIE 的 split conformal
     （LAC：Least Ambiguous set-valued Classifier）在现行 LightGBM 上产出
     带覆盖率保证的预测集合 → 折叠成 [0,1] 区间 → 宽松度 `width`
     → 置信分 `confidence_from_interval(0, width, 0.5)`；
     同一批样本上跑**覆盖率审计**（目标覆盖率 vs 实测覆盖率）、
     **区间宽度校准**（留一/保形区间宽 vs 实际误差）与
     **概率校准曲线**（可靠性图 + Brier + ECE，含 isotonic 参考臂，
     为 S19/H4 留证据），落盘 `reports/calibration/`。
  2. **T16.2 `compare_interval_vs_proba`**：把【区间置信分】与
     【现行 `|p − 0.5| × 2`】放进**同一条** `sweep_confidence` 阈值链路，
     同折、同样本、同模型，输出逐阈值并排对照与判定
     （保守口径：IC 与命中率**同向**变好才算改善迹象）。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：`affects_gate` 恒为 False；`strategy_gate` 逐字段不动；
    区间口径与概率距离口径的取舍属 T16.4 人工检查点（本模块只产证据）；
  - **不自动落地**：`model.conformal.enabled` 缺省 false，区间口径不进生产链路；
  - **不用保留期做任何拟合**：proper-train / calibration / holdout 三段严格
    按时间切分，保形分位数只用 calibration 段算，holdout 段只被评估一次；
  - **不猜**：样本不足 / MAPIE 缺失 / 覆盖率不可得时如实
    `available=false` 并给原因，不降级成假数字；
  - **不虚报覆盖率**：CONFIDENCE_LEVELS 给的是**目标**覆盖率，
    实测覆盖率单独算并列出偏差（偏保守/偏激）。

无前视说明：
  - 三段切分按时间顺序（proper-train < calibration < holdout），标签不跨段；
  - 保形分位数是 calibration 段的经验分位数，holdout 段标签对模型不可见
    （回归用例：篡改 holdout 标签 → 区间与置信分必须逐位不变）；
  - `include_last_label` 采用 MAPIE 默认的 "randomized" 语义
    （保形集合可为空集，覆盖保证才严格成立）—— 空集 = 全部标签都不确定，
    映射为**最宽区间**（置信分最低），与 `withheld` 语义一致。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.eval.confidence_curve import confidence_from_interval, sweep_confidence

logger = logging.getLogger(__name__)

# 目标覆盖率档位（1 − alpha）。80% 为主线档：既有 S15 材料里就爱用「两成宽度」
# 做「有把握」的门槛方言，直接对齐能少一次口径换算；90% 作敏感性对照。
DEFAULT_CONFIDENCE_LEVELS: Tuple[float, ...] = (0.8, 0.9)
N_BOOTSTRAP = 200          # 覆盖率置信区间的 bootstrap 次数（固定种子，可复现）
BOOTSTRAP_SEED = 42
MIN_EVAL_SAMPLES = 30      # 低于此样本数不给指标（不猜）
ECE_BINS = 10              # 可靠性图分箱数
CALIBRATION_DIR = "calibration"


# ----------------------------------------------------------------------
# 三段切分（纯函数，时间序，无前视）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ThreeWaySplit:
    """proper-train / calibration / holdout 三段索引（全局坐标，严格时间序）。"""
    proper_train: np.ndarray
    calibration: np.ndarray
    holdout: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(len(self.proper_train) and len(self.calibration) and len(self.holdout))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "proper_train": int(len(self.proper_train)),
            "calibration": int(len(self.calibration)),
            "holdout": int(len(self.holdout)),
            "available": self.available,
            **dict(self.meta),
        }


def three_way_split(n: int, holdout_ratio: float = 0.3,
                    calib_ratio_of_train_pool: Optional[float] = None) -> ThreeWaySplit:
    """把 n 个**按时间排序**的样本切成 proper-train / calibration / holdout。

    切法（与 T16.3 `holdout_split` 同源口径）：
      - 先切出后 `holdout_ratio` 作 holdout（与 T16.3 的保留期完全对齐，
        保证两种证据在同一段上可比）；
      - 前段（训练池）再切：后 `calib_ratio_of_train_pool` 作 calibration
        （保形分位数来源），其余作 proper-train（模型拟合）；
        缺省比例下整体约为 **60% / 10% / 30%**（目标比例，实际以整除切点为准）：
        保留期与 T16.3 的 `holdout_split` **完全同一段**，且 calibration
        占训练池 1/6，保形分位数样本量够用。

    样本不足时如实给空段（`available=False`），不做打乱切分、不退化复用。
    """
    n = int(max(n, 0))
    if n <= 0:
        return ThreeWaySplit(np.array([], dtype=int), np.array([], dtype=int),
                             np.array([], dtype=int))
    # holdout_ratio = **保留期占比**（后段）：切点与 T16.3 `holdout_split`
    # 完全一致（`int(n × train_ratio)`，clamp 到 [0.1, 0.9]），
    # 保证「保形区间」与「置信度阈值曲线」两条证据落在**同一段**保留期上。
    ratio = min(max(float(1.0 - holdout_ratio), 0.1), 0.9)
    cut = max(int(n * ratio), 1)
    if cut >= n:
        return ThreeWaySplit(np.arange(n), np.array([], dtype=int),
                             np.array([], dtype=int),
                             {"reason": "holdout_empty"})
    train_pool = np.arange(0, cut)                # 前段 = 训练池
    holdout = np.arange(cut, n)                   # 后段 = 保留期

    # 缺省 calibration 占训练池 1/6（0.1 / 0.6 ≈ 0.166667），
    # 即整体 60% / 10% / 30%
    calib_ratio = (1.0 / 6.0 if calib_ratio_of_train_pool is None
                   else float(calib_ratio_of_train_pool))
    calib_ratio = min(max(calib_ratio, 0.05), 0.9)
    calib_n = max(int(len(train_pool) * calib_ratio), 1)
    if calib_n >= len(train_pool):
        return ThreeWaySplit(np.array([], dtype=int), np.array([], dtype=int),
                             holdout, {"reason": "proper_train_empty"})
    proper = train_pool[: len(train_pool) - calib_n]
    calib = train_pool[len(train_pool) - calib_n:]
    # 三段严格时间序：proper_train < calibration < holdout（无重叠、无遗漏）
    assert proper.max() < calib.min() and calib.max() < holdout.min()
    return ThreeWaySplit(proper, calib, holdout,
                         {"proper_train_ratio_of_all": round(len(proper) / n, 6),
                          "calibration_ratio_of_all": round(len(calib) / n, 6),
                          "holdout_ratio_of_all": round(len(holdout) / n, 6)})


# ----------------------------------------------------------------------
# 区间换算与覆盖率审计（纯函数）
# ----------------------------------------------------------------------
def interval_from_prediction_sets(sets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """保形预测集合 (n, 2) → 闭区间 [lo, hi]，宽松度 = (hi − lo) ∈ [0, 1]。

    口径（单周期二分类归一化到 [0,1]）：
      - 集合 == {0, 1}（两个标签都在）→ 区间 [0, 1]，最宽，置信分 0（完全不确定）；
      - 集合为空集（LAC randomized 的合法输出）→ 同样映射为最宽区间 [0, 1]：
        空集在「该给什么方向」上没有信息，与「两个方向都不敢排除」在
        决策语义上等价，一律按最低置信处理（宁可漏信号，不假自信）；
      - 集合 == {1} → [0.5 + 0.5·lo_class, 1]，集合 == {0} → [0, 0.5 − 0.5·hi_class]；
        单标签集合的**绝对位置**在这里不重要，重要的是它的宽度（=0.5）——
        宽度进入置信分，方向由模型点预测负责（与现行链路分工一致）。
    """
    arr = np.asarray(sets)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"prediction sets 形状须为 (n, 2)，实际 {arr.shape}")
    has0 = arr[:, 0].astype(bool)
    has1 = arr[:, 1].astype(bool)
    # 缺省最宽：双标签集合与**空集**都落在这里（空集 = 哪个方向都不敢说，
    # 与「两个方向都不敢排除」在决策语义上等价，一律按最低置信处理）
    lo = np.zeros(len(arr), dtype=float)
    hi = np.ones(len(arr), dtype=float)
    # 只含 {1}：区间宽 0.5，落在上半区
    only1 = has1 & ~has0
    lo[only1] = 0.5
    # 只含 {0}：区间宽 0.5，落在下半区
    only0 = has0 & ~has1
    hi[only0] = 0.5
    return lo, hi


def interval_confidence(sets: np.ndarray) -> np.ndarray:
    """保形预测集合 → 置信分（复用 S15 既有 `confidence_from_interval`，不另立口径）。

    归一化基准取满宽 1.0（单周期二分类的区间取值域就是 [0,1]），
    于是置信分 = 1 − 宽度，语义与 `withheld` 完全一致：

      - 双标签集合 / 空集 → 宽度 1.0 → 置信分 **0**（完全不确定，不给信号）；
      - 单标签集合     → 宽度 0.5 → 置信分 **0.5**（排除了一个方向）；
      - （未来若引入更窄的值域区间，宽度更小则置信分更高，机制无需改动）。
    """
    lo, hi = interval_from_prediction_sets(sets)
    mid = np.ones(len(lo), dtype=float)
    return confidence_from_interval(lo, hi, mid)


def coverage_report(y_true: Sequence[int], sets: np.ndarray,
                    confidence_level: float) -> Dict[str, Any]:
    """覆盖率审计：目标覆盖率 vs 实测覆盖率（含 bootstrap 置信区间）。

    保形预测的**唯一硬承诺**是覆盖：`P(y ∈ C) ≥ 1 − α`。这里只算实测值，
    不给「应该更好/更差」的叙事 —— 偏保守（实测 > 目标）与偏激（实测 < 目标）
    都单独标注。
    """
    y = np.asarray(y_true, dtype=int)
    arr = np.asarray(sets)
    n = len(y)
    if n < MIN_EVAL_SAMPLES or arr.shape[0] != n:
        return {"available": False, "reason": "insufficient_samples",
                "samples": int(n)}
    covered = arr[np.arange(n), y].astype(bool)
    empirical = float(covered.mean())
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    idx = rng.randint(0, n, size=(N_BOOTSTRAP, n))
    boots = covered[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    target = float(confidence_level)
    gap = round(empirical - target, 6)
    if empirical + 1e-12 >= target:
        verdict = "conservative" if gap > 0.05 else "on_target"
    else:
        verdict = "anti_conservative"    # 低于目标：覆盖承诺未兑现，如实标注
    return {
        "available": True,
        "confidence_level": target,
        "empirical_coverage": round(empirical, 6),
        "coverage_gap": gap,
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "verdict": verdict,
        "samples": int(n),
        "set_size_mean": round(float(arr.sum(axis=1).mean()), 6),
        "empty_set_ratio": round(float((~arr.any(axis=1)).mean()), 6),
        "note": "覆盖率是保形预测的唯一硬承诺；偏激（低于目标）不得引用为达标证据",
    }


def reliability_curve(proba: Sequence[float], y_true: Sequence[int],
                      n_bins: int = ECE_BINS,
                      isotonic: bool = False) -> Dict[str, Any]:
    """概率校准曲线（可靠性图 + Brier + ECE）。

    与覆盖率审计互补：覆盖率管「区间有没有兜住」，
    可靠性管「模型说的概率能不能当概率用」——T16.2 对照两条置信度口径时，
    这是「谁的置信分更可信」的直接依据（不做任何达标声明）。
    """
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n = min(len(p), len(y))
    p, y = p[:n], y[:n]
    if n < MIN_EVAL_SAMPLES:
        return {"available": False, "reason": "insufficient_samples",
                "samples": int(n)}

    source = "raw"
    fitted_note = None
    if isotonic:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        p = np.clip(iso.fit_transform(p, y), 0.0, 1.0)
        source = "isotonic_in_sample_reference"
        fitted_note = ("isotonic 为**同批数据参考臂**（非独立保形），只用来标注可靠性"
                       "上限，其 Brier/ECE 不得与 raw 做「谁更好」的结论 —— 那属 S19/H4")

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, int(n_bins) - 1)
    rows: List[Dict[str, Any]] = []
    ece = 0.0
    for b in range(int(n_bins)):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin": b, "range": [round(float(bins[b]), 4),
                                             round(float(bins[b + 1]), 4)],
                         "samples": 0, "available": False,
                         "reason": "empty_bin"})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        gap = acc - conf
        ece += (cnt / n) * abs(gap)
        rows.append({"bin": b, "range": [round(float(bins[b]), 4),
                                         round(float(bins[b + 1]), 4)],
                     "samples": cnt, "available": True,
                     "mean_predicted": round(conf, 6),
                     "observed_freq": round(acc, 6),
                     "gap": round(gap, 6)})
    brier = float(np.mean((p - y) ** 2))
    return {
        "available": True,
        "source": source,
        "samples": int(n),
        "brier": round(brier, 6),
        "ece": round(float(ece), 6),
        "bins": rows,
        "note": fitted_note or ("原始概率的可靠性（未做任何再校准）："
                                "ECE 越大代表「说的概率」越不能当概率用"),
    }


def coverage_width_curve(errors: Sequence[float], widths: Sequence[float],
                         n_bins: int = ECE_BINS) -> Dict[str, Any]:
    """区间宽度校准：把 |误差| 分箱，看每箱实际误差分位 vs 声明宽度能否兜住。

    这是「区间宽度是否可信」的直接检验：声明宽度应当是误差分位的**上界**，
    而不是「平均误差」。每箱给出能否覆盖、以及覆盖/超界计数。
    """
    e = np.abs(np.asarray(errors, dtype=float))
    w = np.asarray(widths, dtype=float)
    n = min(len(e), len(w))
    e, w = e[:n], w[:n]
    if n < MIN_EVAL_SAMPLES:
        return {"available": False, "reason": "insufficient_samples", "samples": int(n)}
    edges = np.unique(np.quantile(w, np.linspace(0, 1, int(n_bins) + 1)))
    if len(edges) < 2:
        # 宽度全等（单点分布）：没有分箱意义，但仍要给出这一档的读数（不猜、不空转）
        edges = np.array([float(w.min()) - 1e-12, float(w.max()) + 1e-12])
    rows: List[Dict[str, Any]] = []
    for b in range(len(edges) - 1):
        m = (w >= edges[b]) & (w < edges[b + 1] if b < len(edges) - 2 else w <= edges[b + 1])
        cnt = int(m.sum())
        if cnt == 0:
            continue
        q90 = float(np.quantile(e[m], 0.9))
        declared = float(w[m].mean())
        rows.append({
            "width_range": [round(float(edges[b]), 6), round(float(edges[b + 1]), 6)],
            "samples": cnt,
            "declared_width_mean": round(declared, 6),
            "abs_error_p90": round(q90, 6),
            "covers_p90": bool(declared >= q90),
        })
    viol = [r for r in rows if not r["covers_p90"]]
    return {
        "available": True,
        "samples": int(n),
        "bins": rows,
        "violation_ratio": round(len(viol) / len(rows), 6) if rows else None,
        "note": ("声明宽度应兜住同箱误差 90 分位；violation_ratio > 0 说明"
                 "区间对误差的刻画偏窄，不得当作「有把握」的证据"),
    }


# ----------------------------------------------------------------------
# 保形预测装配（MAPIE 可选依赖，缺失时明确报错不静默）
# ----------------------------------------------------------------------
def mapie_available() -> bool:
    try:
        import mapie  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def build_conformal_sets(estimator: Any, X_proper: np.ndarray, y_proper: np.ndarray,
                         X_calib: np.ndarray, y_calib: np.ndarray,
                         X_eval: np.ndarray,
                         confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
                         conformity_score: str = "lac",
                         ) -> Dict[float, np.ndarray]:
    """训练 LightGBM + split conformal（LAC）→ 逐覆盖率档的预测集合。

    为什么用 LAC：单周期二分类只有两个标签，LAC（Least Ambiguous set-valued
    Classifier）在给定覆盖率下**最不模糊**（集合期望规模最小）——正是我们想要的
    「区间尽量窄、但承诺不破」。APS/RAPS 针对多标签排序，本场景无增量。

    `prefit=True`：estimator 由本函数在 proper-train 段拟合，避免 MAPIE
    内部再切一刀导致保形分位数样本量减半。
    """
    from mapie.classification import SplitConformalClassifier

    levels = [float(c) for c in confidence_levels]
    clf = SplitConformalClassifier(estimator=estimator, confidence_level=levels,
                                   conformity_score=conformity_score, prefit=True)
    clf.conformalize(X_calib, y_calib)
    _y_pred, y_sets = clf.predict_set(X_eval)
    # MAPIE 返回 (n_eval, n_classes, n_confidence_levels)，与 levels 同序
    return {lv: np.asarray(y_sets)[:, :, i] for i, lv in enumerate(levels)}


def train_and_conformal(combined: Any, cols: Sequence[str], target_col: str,
                        split: ThreeWaySplit, config: Optional[Dict[str, Any]] = None,
                        confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
                        lgb_module: Any = None, scaler_cls: Any = None,
                        ) -> Dict[str, Any]:
    """在给定三段切分上训练 + 保形化，返回区间与置信分（与 T16.3 同口径模型装配）。

    与 `confidence_holdout.collect_predictions` 保持**同一套** StandardScaler +
    LightGBM 超参 + random_state，保证两条证据链（覆盖率 vs 阈值曲线）可同源对照。
    """
    if lgb_module is None:
        import lightgbm as lgb_module  # type: ignore
    if scaler_cls is None:
        from sklearn.preprocessing import StandardScaler as Scaler  # type: ignore
        scaler_cls = Scaler
    from src.eval.hyperopt_tuner import current_lightgbm_params

    cfg = config or {}

    X = combined[list(cols)].to_numpy(dtype=float)
    y = combined[target_col].to_numpy(dtype=int)
    ret = combined["_fwd_ret"].to_numpy(dtype=float)

    scaler = scaler_cls()
    X_proper = scaler.fit_transform(X[split.proper_train])
    X_calib = scaler.transform(X[split.calibration])
    X_eval = scaler.transform(X[split.holdout])

    params = {k: (int(v) if k in ("num_leaves", "max_depth", "n_estimators") else float(v))
              for k, v in current_lightgbm_params(cfg).items()}
    clf = lgb_module.LGBMClassifier(verbose=-1, random_state=42, **params)
    clf.fit(X_proper, y[split.proper_train])

    proba_eval = clf.predict_proba(X_eval)[:, 1]
    sets = build_conformal_sets(clf, X_proper, y[split.proper_train],
                                X_calib, y[split.calibration], X_eval,
                                confidence_levels=confidence_levels)
    return {
        "estimator": clf,
        "proba": proba_eval,
        "returns": ret[split.holdout],
        "y_true": y[split.holdout],
        "sets": sets,
    }


# ----------------------------------------------------------------------
# 对照（T16.2）
# ----------------------------------------------------------------------
def compare_interval_vs_proba(proba: Sequence[float], returns: Sequence[float],
                              conf_interval: Sequence[float],
                              conf_proba: Optional[Sequence[float]] = None,
                              grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                              ) -> Dict[str, Any]:
    """同折同样本对照：区间置信分 vs 现行 `|p − 0.5| × 2`。

    判定口径（保守，与 S12 label-ab / S13 qlib-ab 同款）：
      - **同向变好**（IC 与命中率同时 ≥ 对方）才算「改善迹象」；
      - 一升一降 → `mixed`（如实记录，不择优、不裁成结论）；
      - 可用阈值行不足 → `insufficient_samples`（不猜）。
    """
    p = np.asarray(proba, dtype=float)
    r = np.asarray(returns, dtype=float)
    ci = np.asarray(conf_interval, dtype=float)
    if conf_proba is None:
        from src.eval.confidence_curve import confidence_from_proba
        conf_proba = confidence_from_proba(p)
    cprob = np.asarray(conf_proba, dtype=float)

    curve_interval = sweep_confidence(p, r, confidence=ci, grid=grid)
    curve_proba = sweep_confidence(p, r, confidence=cprob, grid=grid)

    by_thr = {float(row["threshold"]): row for row in curve_interval["rows"]}
    pairs: List[Dict[str, Any]] = []
    for row in curve_proba["rows"]:
        thr = float(row["threshold"])
        other = by_thr.get(thr)
        if other is None or not row.get("available") or not other.get("available"):
            pairs.append({"threshold": thr, "available": False,
                          "reason": "insufficient_samples"})
            continue
        d_ic = round(float(other["ic"]) - float(row["ic"]), 6)
        d_hit = round(float(other["hit_rate"]) - float(row["hit_rate"]), 6)
        pairs.append({
            "threshold": thr,
            "available": True,
            "interval": {"hit_rate": other["hit_rate"], "ic": other["ic"],
                         "coverage": other["coverage"], "samples": other["samples"]},
            "proba_distance": {"hit_rate": row["hit_rate"], "ic": row["ic"],
                               "coverage": row["coverage"], "samples": row["samples"]},
            "delta_interval_minus_proba": {"hit_rate": d_hit, "ic": d_ic},
        })

    usable = [x for x in pairs if x.get("available")]
    verdict, reason = "insufficient_samples", "无同时可用的阈值行，不构成结论"
    if usable:
        better_hit = sum(1 for x in usable
                         if x["delta_interval_minus_proba"]["hit_rate"] > 1e-9)
        better_ic = sum(1 for x in usable
                        if x["delta_interval_minus_proba"]["ic"] > 1e-9)
        worse_hit = sum(1 for x in usable
                        if x["delta_interval_minus_proba"]["hit_rate"] < -1e-9)
        worse_ic = sum(1 for x in usable
                       if x["delta_interval_minus_proba"]["ic"] < -1e-9)
        if better_hit == len(usable) and better_ic == len(usable):
            verdict, reason = "improved", "区间口径在全部可用阈值行上命中率与 IC 同向更高"
        elif worse_hit == len(usable) and worse_ic == len(usable):
            verdict, reason = "degraded", "区间口径在全部可用阈值行上命中率与 IC 同向更低"
        else:
            verdict, reason = "mixed", ("存在阈值行一升一降（或方向不一致），"
                                        "不构成改善结论，如实记录不择优")

    return {
        "kind": "conformal_vs_proba_distance",
        "verdict": verdict,
        "reason": reason,
        "usable_thresholds": len(usable),
        "grid": [float(g) for g in grid],
        "pairs": pairs,
        "interval_curve": curve_interval,
        "proba_distance_curve": curve_proba,
        "affects_gate": False,
        "note": ("保留期对照：同数据、同折、同模型，只变置信度口径。"
                 "口径取舍属 T16.4 人工检查点，本对照不自动落地任何一侧。"),
    }


# ----------------------------------------------------------------------
# 总装（main.py conformal-* 命令调用）
# ----------------------------------------------------------------------
def calibration_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    """校准产物目录（尊重配置里的 report_dir）。"""
    gate = ((config or {}).get("strategy_gate", {}) or {})
    seg = (gate.get("confidence_gate", {}) or {}) if isinstance(gate, dict) else {}
    base = Path(str(seg.get("report_dir", "reports") or "reports"))
    return base / CALIBRATION_DIR


def build_interval_report(horizons: Dict[str, Any],
                          meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """区间报告（T16.1 产物）：覆盖率审计 + 可靠性 + 宽度校准。"""
    return {
        "kind": "conformal_interval_report",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_source": "holdout_conformal",
        "method": "mapie_split_conformal_lac",
        "confidence_levels": list((meta or {}).get("confidence_levels",
                                                   DEFAULT_CONFIDENCE_LEVELS)),
        "note": ("保形预测区间：覆盖率为硬承诺，实测覆盖率与目标覆盖率的偏差"
                 "如实记录；区间口径是否纳入生产属 T16.4 人工检查点"),
        "horizons": horizons,
        "meta": dict(meta or {}),
        "affects_gate": False,
    }


def build_comparison_report(horizons: Dict[str, Any],
                            meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """对照报告（T16.2 产物）：区间置信分 vs `|p − 0.5| × 2`。"""
    return {
        "kind": "conformal_vs_proba_report",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_source": "holdout_conformal",
        "note": ("同一保留期上两条置信度口径的并排读数；任何单阈值读数仍属"
                 "选择自由度，未经多重比较校正不得引用为达标证据"),
        "horizons": horizons,
        "meta": dict(meta or {}),
        "affects_gate": False,
    }


def interval_report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    return calibration_dir(config) / "conformal_interval.json"


def comparison_report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    return calibration_dir(config) / "conformal_vs_proba.json"
