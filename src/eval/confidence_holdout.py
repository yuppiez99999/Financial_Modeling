"""置信度保留期复验（T16.3 / S16）：独立保留期 + 多时段滚动证据链。

问题从哪来（T15.3 遗留的两个证据缺口，见 `confidence_gate.py` 的
`single_period_warning` 与 Issue #29 评论区）：
  1. `confidence-gate` 决策单消费的保留期报告
     `reports/confidence_holdout_verify.json` 一直**没有可复现的生成命令**
     ——上一轮是临时脚本手工构造，报告在 `.gitignore`（reports/）里，
     换机器/过期后证据链断掉，签字流程走不通；
  2. 保留期只有**单一时段**（最近 ~30% 行情），「thr↑ → 子集命中率↑」
     是否跨时段稳定无从回答——这是上一轮决策单里显式标注的最大软肋。

本模块做什么：
  1. **独立保留期复验**（`run_holdout_verification`）：
     汇集与 walk-forward 曲线同口径的样本，但训练**只允许用前
     `train_ratio`（默认 70%）**，保留期 = 后 30%，
     从未参与任何训练、阈值扫描、超参搜索——与 S15 T15.2 的切分语义一致；
     保留期上沿阈值网格产出「覆盖率 × 命中率 × IC」曲线，
     落盘 `reports/confidence_holdout_verify.json`（决策单唯一证据源）。
  2. **多时段滚动复验**（`run_rolling_verification`）：
     把保留期再按时间切成 `n_periods` 个**互不重叠**的时段，
     每个时段独立评估「thr∈[0.2,0.3] 的子集命中率是否 ≥ 同时段全样本
     命中率」——检验置信度优势是**跨时段稳定**还是单一时段巧合；
     落盘 `reports/confidence_rolling_verify.json`（补充证据）。
  3. **判定纯函数**（`evaluate_period_stability` / `summarize_stability`）：
     时段稳定 / 不稳定 / 样本不足三态，如实输出，不猜。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：`affects_gate` 恒为 False；`strategy_gate` 逐字段不变；
  - **不选阈值**：候选区间是范围不是单点；挑阈值 + 签字属 T16.4/T15.3
    人工检查点，本模块只产证据；
  - **不用保留期做任何训练侧决策**：模型/特征/超参全部沿用现行配置，
    保留期只被「评估」一次；
  - **不猜**：样本不足的时段 / 阈值行 `available=false` 并写明原因。

无前视说明：
  - 训练索引严格在前 70%，保留期索引在后 30%，按时间排序后切分；
  - 滚动时段只在保留期内部切分，模型对整段保留期一次性预测
    （预测不依赖保留期标签），随后才按时段分组算指标——
    时段边界对模型不可见，不存在用后段信息预测前段。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.eval.confidence_curve import sweep_confidence

logger = logging.getLogger(__name__)

HOLDOUT_REPORT_NAME = "confidence_holdout_verify.json"
ROLLING_REPORT_NAME = "confidence_rolling_verify.json"

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
STABLE_TAG = "stable"
UNSTABLE_TAG = "unstable"
INSUFFICIENT_TAG = "insufficient_samples"


# ----------------------------------------------------------------------
# 索引切分（纯函数）
# ----------------------------------------------------------------------
def holdout_split(n: int, train_ratio: float = DEFAULT_TRAIN_RATIO
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """前 `train_ratio` 训练 / 后段保留（时间序，无前视）。

    数据不足时如实返回空保留期（不猜、不退化成打乱切分）。
    """
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    ratio = min(max(float(train_ratio), 0.1), 0.9)
    cut = max(int(n * ratio), 1)
    if cut >= n:
        return np.arange(0, n), np.array([], dtype=int)
    return np.arange(0, cut), np.arange(cut, n)


def rolling_periods(holdout_idx: np.ndarray, n_periods: int = 3
                    ) -> List[Tuple[int, int]]:
    """把保留期索引切成 n 个**互不重叠**的连续时段（按时间顺序）。

    返回 [(start, end)]（相对于保留期起点的局部坐标）。
    样本不足时返回能切的最多时段；一切不了返回 []（不猜）。
    """
    n = int(len(holdout_idx))
    if n_periods <= 1:
        return [(0, n)] if n > 0 else []
    k = min(int(n_periods), n)
    if k <= 1:
        return [(0, n)] if n > 0 else []
    size = n // k
    if size <= 0:
        return []
    out: List[Tuple[int, int]] = []
    for i in range(k):
        start = i * size
        end = n if i == k - 1 else (i + 1) * size
        if end > start:
            out.append((start, end))
    return out


# ----------------------------------------------------------------------
# 单时段稳定性判定（纯函数）
# ----------------------------------------------------------------------
def evaluate_period_stability(rows: Sequence[Dict[str, Any]],
                              thr_min: float = 0.2,
                              thr_max: float = 0.3,
                              min_samples: int = 50) -> Dict[str, Any]:
    """单时段：thr∈[thr_min,thr_max] 子集命中率是否优于同时段全样本命中率。

    判定口径（保守）：
      - 候选行 = 网格上落在 [thr_min, thr_max] 内且 `available` 的行；
      - 稳定   = **每一行**候选行的子集命中率都 ≥ 全样本命中率（thr=0 行）；
      - 只要有一行候选命中率 < 全样本命中率 → unstable（如实，不择优）；
      - 无全样本行或候选行样本不足 → insufficient_samples（不猜）。
    """
    valid_rows = [r for r in rows if isinstance(r, dict)]
    full = next((r for r in valid_rows
                 if abs(float(r.get("threshold", -1)) - 0.0) < 1e-9
                 and r.get("available")), None)
    if full is None:
        return {"verdict": INSUFFICIENT_TAG,
                "reason": "缺少可用的全样本（thr=0）基准行",
                "full_hit_rate": None,
                "candidate_rows": []}

    full_hit = float(full.get("hit_rate") or 0.0)
    cand = [r for r in valid_rows
            if float(thr_min) - 1e-9 <= float(r.get("threshold", -1)) <= float(thr_max) + 1e-9
            and r.get("available")
            and int(r.get("samples") or 0) >= int(min_samples)]
    evaluated = [{
        "threshold": float(r["threshold"]),
        "hit_rate": float(r.get("hit_rate") or 0.0),
        "coverage": float(r.get("coverage") or 0.0),
        "samples": int(r.get("samples") or 0),
        "delta_vs_full": round(float(r.get("hit_rate") or 0.0) - full_hit, 6),
    } for r in cand]

    if not evaluated:
        return {"verdict": INSUFFICIENT_TAG,
                "reason": f"thr ∈ [{thr_min},{thr_max}] 内无样本足够的候选行",
                "full_hit_rate": round(full_hit, 6),
                "candidate_rows": []}

    all_ge = all(e["hit_rate"] >= full_hit for e in evaluated)
    verdict = STABLE_TAG if all_ge else UNSTABLE_TAG
    reason = ("候选区间内全部阈值命中率 ≥ 全样本基准"
              if all_ge else
              "存在候选阈值命中率低于全样本基准（优势非跨时段稳定信号）")
    return {"verdict": verdict,
            "reason": reason,
            "full_hit_rate": round(full_hit, 6),
            "candidate_rows": evaluated}


def summarize_stability(periods: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """跨时段汇总：多数时段 stable 才算 stable；一切不猜。"""
    verdicts = [p.get("verdict") for p in periods if isinstance(p, dict)]
    usable = [v for v in verdicts if v in (STABLE_TAG, UNSTABLE_TAG)]
    if not usable:
        return {"verdict": INSUFFICIENT_TAG,
                "stable_periods": 0, "unstable_periods": 0,
                "insufficient_periods": len(verdicts),
                "reason": "无可用时段（样本不足），不构成任何结论"}
    stable = sum(1 for v in usable if v == STABLE_TAG)
    unstable = len(usable) - stable
    verdict = STABLE_TAG if stable > unstable else (
        UNSTABLE_TAG if stable < unstable else INSUFFICIENT_TAG)
    note = {STABLE_TAG: "置信度优势跨时段稳定",
            UNSTABLE_TAG: "置信度优势跨时段不稳定",
            INSUFFICIENT_TAG: "稳定/不稳定时段各半，证据不构成结论"}[verdict]
    return {"verdict": verdict,
            "stable_periods": stable,
            "unstable_periods": unstable,
            "insufficient_periods": len(verdicts) - len(usable),
            "reason": note}


# ----------------------------------------------------------------------
# 模型装配（与 confidence / ic 同口径）
# ----------------------------------------------------------------------
def collect_predictions(combined: "Any", cols: Sequence[str],
                        target_col: str, train_idx: np.ndarray,
                        test_idx: np.ndarray, config: Optional[Dict[str, Any]] = None,
                        lgb_module: Any = None, scaler_cls: Any = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """在给定训练/测试索引上训练 LightGBM 并返回保留期概率与未来收益。

    与 `main.py run_confidence` 完全同口径（StandardScaler + 现行超参 +
    random_state=42），保证保留期报告与 walk-forward 曲线**可同源对照**。
    """
    import numpy as _np  # noqa: F401
    if lgb_module is None or scaler_cls is None:
        try:
            import lightgbm as lgb_module  # type: ignore
            from sklearn.preprocessing import StandardScaler as Scaler  # type: ignore
            lgb_module, Scaler = lgb_module, Scaler  # noqa
        except ImportError as e:  # pragma: no cover
            raise ImportError("需要 lightgbm/scikit-learn（可选依赖未安装）") from e
    from src.eval.hyperopt_tuner import current_lightgbm_params

    X = combined[list(cols)].to_numpy(dtype=float)
    y = combined[target_col].to_numpy(dtype=int)
    ret = combined["_fwd_ret"].to_numpy(dtype=float)

    scaler = scaler_cls()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])
    clf = lgb_module.LGBMClassifier(
        verbose=-1, random_state=42,
        **{k: (int(v) if k in ("num_leaves", "max_depth", "n_estimators") else float(v))
           for k, v in current_lightgbm_params(config).items()})
    clf.fit(X_tr, y[train_idx])
    return clf.predict_proba(X_te)[:, 1], ret[test_idx]


# ----------------------------------------------------------------------
# 报告生成
# ----------------------------------------------------------------------
def build_holdout_report(curves: Dict[str, Dict[str, Any]],
                         meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """组装保留期报告（决策单 `confidence_gate` 的唯一证据源格式）。

    结构与 `_extract_holdout_curves` 期望兼容：
    ``{"kind": ..., "generated_at": ..., "horizons": {"5d": {rows...}}}``。
    """
    return {
        "kind": "confidence_holdout_verify",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_source": "holdout",
        "train_ratio": float((meta or {}).get("train_ratio", DEFAULT_TRAIN_RATIO)),
        "note": ("独立保留期复验：训练只用前段，保留期从未参与训练/扫描/调参；"
                 "thr ∈ [0.2,0.3] 候选读数仍属选择自由度，签字前不得引用为达标证据"),
        "horizons": curves,
        "meta": dict(meta or {}),
        "affects_gate": False,
    }


def build_rolling_report(period_rows: Dict[str, Dict[str, Any]],
                         summaries: Dict[str, Dict[str, Any]],
                         meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """组装多时段滚动复验报告（补充证据，不得替代保留期报告）。"""
    return {
        "kind": "confidence_rolling_verify",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_source": "holdout_rolling",
        "n_periods": int((meta or {}).get("n_periods", 0)),
        "note": ("多时段滚动复验（补充证据）：检验 thr∈[0.2,0.3] 命中率优势是否"
                 "跨时段稳定；不稳定不代表机制无效，但不得作为签字依据"),
        "horizons": {h: {"periods": rows, "stability": summaries.get(h, {})}
                     for h, rows in period_rows.items()},
        "meta": dict(meta or {}),
        "affects_gate": False,
    }


def holdout_report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """保留期报告路径（尊重 strategy_gate.confidence_gate.report_dir）。"""
    gate = ((config or {}).get("strategy_gate", {}) or {})
    seg = (gate.get("confidence_gate", {}) or {}) if isinstance(gate, dict) else {}
    report_dir = str(seg.get("report_dir", "reports") or "reports")
    return Path(report_dir) / HOLDOUT_REPORT_NAME


def rolling_report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    gate = ((config or {}).get("strategy_gate", {}) or {})
    seg = (gate.get("confidence_gate", {}) or {}) if isinstance(gate, dict) else {}
    report_dir = str(seg.get("report_dir", "reports") or "reports")
    return Path(report_dir) / ROLLING_REPORT_NAME
