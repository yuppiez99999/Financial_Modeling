"""optuna 超参搜索（S15 / G5，T15.1）：包裹 LightGBM（LSTM 可选）训练。

为什么需要（Issue #29 集成方案 G5 验收口径）：
  此前所有 A/B（label-ab / qlib-ab / feature-experiment）都用**同一份写死的
  超参**对比。这保证"只差一个变量"，但也意味着**超参从未被系统搜索过** ——
  门禁指标可能部分卡在次优超参上。optuna 包裹训练把"超参到底卡不卡门禁"
  变成可复算的数字。

本模块做什么：
  - 与 qlib-ab / label-ab **完全同口径**的数据与切分：`scripts/evaluate_models`
    的 `load_market_data` + `build_supervised` + `walk_forward_splits`
    （严格时序、无前视、测试折在训练折之后）；
  - 每个候选超参在全部 walk-forward 折上取 **测试折 IC 均值** 作为目标
    （不是训练集指标 —— 用训练集指标选超参 = 泄漏）；
  - 用 **时间外样本（OOT：最后一折之前从未参与搜索的数据）** 复评最优试验，
    给出"搜索收益 vs 过拟合风险"对照：search IC（搜索中均值）与 OOT IC；
  - 搜索记录写入 `reports/hyperopt_<horizon>_<model>.json`（trials + best +
    OOT 复评），optuna study 以 SQLite 持久化到 `reports/optuna_studies/`
    （可断点续跑，便于审计）。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：`affects_gate` 恒为 False，只产出证据；
  - **不自动替换主线超参**：最优超参是否落地由人工检查点（T15.3）决定，
    本轮一个配置字段都不改；
  - **不挑折**：所有候选在**同一组折**上评估，禁止"换折换出好结果"；
  - **不计入推断统计**：超参搜索属选择自由度，结果须登记试验
    （`python main.py trials` 可见），不得直接当"门禁达标"证据引用。

统计口径备注：
  - IC 用 Spearman（与门禁同源 `src.inference.ic.spearman_ic`）；
  - OOT 复评用 **最后一折的测试段**（该段在搜索目标中以均值参与过 ——
    因此 OOT IC 只是**保守参考**，报告里如实标注该局限，不作独立证据）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 搜索空间（与 LightGBMModel 现行配置同族，只动已被证明敏感的树结构/正则项）
SEARCH_SPACE: Dict[str, Tuple[Any, Any, Any]] = {
    "num_leaves": (16, 128),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.2),
    "n_estimators": (100, 400),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (1e-3, 10.0),
    "reg_lambda": (1e-3, 10.0),
}

BEST_TRIALS_TOP = 5


def _suggest(optuna_trial, space: Dict[str, Tuple[Any, Any, Any]]) -> Dict[str, Any]:
    """把搜索空间映射为 optuna suggest 调用（int/log 分布分开处理）。"""
    params: Dict[str, Any] = {}
    for name, (lo, hi) in space.items():
        if name in ("num_leaves", "max_depth", "n_estimators"):
            params[name] = optuna_trial.suggest_int(name, int(lo), int(hi))
        elif name in ("learning_rate", "reg_alpha", "reg_lambda"):
            params[name] = optuna_trial.suggest_float(name, float(lo), float(hi), log=True)
        else:
            params[name] = optuna_trial.suggest_float(name, float(lo), float(hi))
    return params


def _fold_mean_ic(
    X: np.ndarray,
    y: np.ndarray,
    fwd_ret: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: Dict[str, Any],
    seed: int = 42,
) -> float:
    """给定超参，在全部 walk-forward 折上训练并返回测试折 IC 均值。

    严格时序：每折只用 train 段训练、test 段评估；fwd_ret 对齐 test 段用于 IC。
    """
    from sklearn.preprocessing import StandardScaler

    from src.train.models import lightgbm_model as lgbm_module

    from src.inference.ic import spearman_ic

    try:
        lgb = lgbm_module.lgb
    except AttributeError:  # pragma: no cover - 未安装 lightgbm 的环境
        raise ImportError("超参搜索需要 lightgbm（pip install lightgbm）")

    ics: List[float] = []
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            num_leaves=int(params["num_leaves"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            verbose=-1,
            random_state=seed,
        )
        clf.fit(X_tr, y[train_idx])
        proba = clf.predict_proba(X_te)[:, 1]
        ic = spearman_ic(proba, fwd_ret[test_idx])
        if ic is not None and np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        raise ValueError("无有效折（样本不足或 IC 不可计算）")
    return float(np.mean(ics))


def _oot_ic(
    X: np.ndarray,
    y: np.ndarray,
    fwd_ret: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: Dict[str, Any],
    seed: int = 42,
) -> Optional[dict]:
    """用最后一折的测试段复评最优超参（保守参考，见模块 docstring 局限说明）。"""
    if not splits:
        return None
    train_idx, test_idx = splits[-1]
    from sklearn.preprocessing import StandardScaler

    from src.train.models import lightgbm_model as lgbm_module

    from src.inference.ic import hit_rate, spearman_ic

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])
    clf = lgbm_module.lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        num_leaves=int(params["num_leaves"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        verbose=-1,
        random_state=seed,
    )
    clf.fit(X_tr, y[train_idx])
    proba = clf.predict_proba(X_te)[:, 1]
    return {
        "ic": spearman_ic(proba, fwd_ret[test_idx]),
        "hit_rate": hit_rate(proba, fwd_ret[test_idx]),
        "samples": int(len(test_idx)),
    }


def tune_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    fwd_ret: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 20,
    horizon_days: int = 5,
    study_dir: str = "reports/optuna_studies",
    seed: int = 42,
) -> Dict[str, Any]:
    """optuna 包裹 LightGBM 训练并返回可序列化的搜索报告。"""
    try:
        import optuna
    except ImportError as e:  # pragma: no cover - 未安装 optuna
        raise ImportError("超参搜索需要 optuna（pip install optuna）") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = f"lgbm_h{horizon_days}_{seed}"
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir}/{study_name}.db"

    def objective(optuna_trial) -> float:
        params = _suggest(optuna_trial, SEARCH_SPACE)
        return _fold_mean_ic(X, y, fwd_ret, splits, params, seed=seed)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    best = study.best_trial
    oot = _oot_ic(X, y, fwd_ret, splits, best.params, seed=seed)

    top = sorted(study.trials, key=lambda t: (t.value if t.value is not None else -np.inf),
                 reverse=True)[:BEST_TRIALS_TOP]
    trials_out = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state).split(".")[-1],
        }
        for t in top
    ]
    baseline = None  # 由调用方补（同折同数据的现行超参对照）

    report: Dict[str, Any] = {
        "kind": "hyperopt_lightgbm",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": int(horizon_days),
        "n_trials_requested": int(n_trials),
        "n_trials_total": len(study.trials),
        "folds": len(splits),
        "metric": "mean_test_fold_spearman_ic",
        "search_space": {k: [float(v[0]), float(v[1])] for k, v in SEARCH_SPACE.items()},
        "best_trial": {
            "number": best.number,
            "search_ic": best.value,
            "params": best.params,
        },
        "oot_reeval_last_fold": oot,
        "top_trials": trials_out,
        "baseline_current_params": baseline,
        "affects_gate": False,
        "note": (
            "搜索目标为全部 walk-forward 测试折 IC 均值；OOT 为最后一折复评（该折"
            "在搜索目标中参与过，仅作保守参考）。最优超参不自动落地（T15.3 人工"
            "检查点），本次结果属选择自由度，已登记试验次数。"
        ),
    }
    return report


def baseline_fold_ic(
    X: np.ndarray,
    y: np.ndarray,
    fwd_ret: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: Dict[str, Any],
    seed: int = 42,
) -> Optional[float]:
    """现行超参在同一组折上的对照 IC（供报告并列展示，不参与搜索）。"""
    try:
        return _fold_mean_ic(X, y, fwd_ret, splits, params, seed=seed)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[tune] 基线复评失败（不影响搜索）: {e}")
        return None


def current_lightgbm_params(config: dict) -> Dict[str, Any]:
    """从配置提取现行 LightGBM 超参（与训练路径同源）。"""
    m = (config.get("model", {}) or {}).get("lightgbm", {}) or {}
    return {
        "num_leaves": int(m.get("num_leaves", 31)),
        "max_depth": int(m.get("max_depth", 6)),
        "learning_rate": float(m.get("learning_rate", 0.05)),
        "n_estimators": int(m.get("n_estimators", 200)),
        "subsample": float(m.get("subsample", 1.0)),
        "colsample_bytree": float(m.get("colsample_bytree", 1.0)),
        "reg_alpha": float(m.get("reg_alpha", 0.0)),
        "reg_lambda": float(m.get("reg_lambda", 0.0)),
    }
