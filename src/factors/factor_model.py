"""因子加权组合预测器（FactorModel）。

把因子库输出的 `factor_*` 列按 **因子族权重 → 族内因子权重** 两级加权
合成为单一得分，再经 Platt 风格的单调校准映射为上涨概率。

为什么需要两级权重：
- 族级：不同族的预测力差异大（趋势/量能通常强于情绪），族级权重让整体
  结构可控、可解释；
- 因子级：族内各因子离散度不同，用 |IC| 归一化让信息量大的因子主导。

权重的来源（config → 显式 → 统计估计）：
1. `model.factors.family_weights` / `factor_weights` 显式配置（最高优先级）；
2. IC 统计：训练集上各因子与未来收益的 Spearman 秩相关，取 |IC| 归一化
   （**只用训练集**，评估/测试集不参与，天然无泄漏）。

设计约束：
- 与 LightGBM / LSTM 同构：实现 `train / predict / predict_proba / save / load`，
  因此可直接被 `ModelTrainer` 与 `PredictionEngine` 复用；
- 纯 numpy/pandas 实现，无重型依赖，CI 离线可跑；
- `scale_`（概率校准）只依赖训练集得分分布，推理时可直接套用。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.factors.factor_library import FACTOR_FAMILIES

logger = logging.getLogger(__name__)


def _as_frame(X: Any, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """把输入统一成带列名的 DataFrame（ndarray 时套用 feature_cols）。"""
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = list(feature_cols or [])
    if len(cols) != arr.shape[1]:
        cols = [f"col_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关；样本不足或无波动时返回 0。"""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    try:
        import warnings

        from scipy.stats import spearmanr  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic, _ = spearmanr(a, b)
        return float(ic) if ic is not None and not np.isnan(ic) else 0.0
    except ImportError:
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])


class FactorModel:
    """多因子加权组合预测模型（二分类：未来 h 日上涨 → 1）。"""

    NAME = "factor_model"
    VERSION = "1.0"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = ((config or {}).get("model", {}) or {}).get("factors", {}) or {}
        self.config = config
        self.save_dir = Path((config.get("training", {}) or {}).get("save_dir", "models"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_n = int(cfg.get("top_n", 8))
        self.min_abs_ic = float(cfg.get("min_abs_ic", 0.005))
        self.shrink = float(cfg.get("level2_shrink", 0.35))
        self.calibration_center = float(cfg.get("score_center", 0.0))
        self._explicit_family_weights: dict[str, float] = dict(cfg.get("family_weights", {}) or {})
        self._explicit_factor_weights: dict[str, float] = dict(cfg.get("factor_weights", {}) or {})

        self.feature_cols_: list[str] = []
        self.family_weights_: dict[str, float] = {}
        self.factor_weights_: dict[str, float] = {}
        self.factor_ic_: dict[str, float] = {}
        self.scale_: float = 4.0      # 得分 → logit 的放大系数（仅由训练集估计）
        self.center_: float = 0.0     # 训练集得分均值（去偏）
        self.n_factors_: int = 0

    # ------------------------------------------------------------------
    @staticmethod
    def is_factor_column(col: Any) -> bool:
        return str(col).startswith("factor_")

    def _family_of(self, col: str) -> str:
        for family, cols in FACTOR_FAMILIES.items():
            if col in cols:
                return family
        return "other"

    # ------------------------------------------------------------------
    def _estimate_ic(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """训练集 IC：各因子与标签的 Spearman 秩相关。"""
        ic: dict[str, float] = {}
        for col in X.columns:
            values = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ic[col] = _spearman(values, y)
        return ic

    def _resolve_weights(self, ic: dict[str, float]) -> None:
        """族级 + 因子级权重解析（显式配置优先，否则用 |IC| 归一化）。"""
        families: dict[str, list[str]] = {}
        for col in self.feature_cols_:
            families.setdefault(self._family_of(col), []).append(col)

        # ---- 因子级权重（族内归一化；按 |IC| 排序取 top_n）
        factor_weights: dict[str, float] = {}
        for _family, cols in families.items():
            scored = [(c, abs(ic.get(c, 0.0))) for c in cols]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            scored = [kv for kv in scored if kv[1] >= self.min_abs_ic] or scored[:1]
            scored = scored[: max(self.top_n, 1)]
            if any(c in self._explicit_factor_weights for c, _ in scored):
                raw = {c: float(self._explicit_factor_weights.get(c, 0.0)) for c, _ in scored}
            else:
                raw = {c: v for c, v in scored if v > 0}
            total = sum(raw.values())
            if total <= 0:
                raw = {c: 1.0 for c, _ in scored}
                total = sum(raw.values())
            # 1 - shrink 部分等权分配，防止单一因子过度集中
            equal = 1.0 / len(raw)
            for c, v in raw.items():
                factor_weights[c] = (1 - self.shrink) * (v / total) + self.shrink * equal

        # ---- 族级权重（族的**信息量** → 族重要度；显式配置优先）
        # 用族内最强 |IC| 而非"族内因子权重和"：后者受 top_n / shrink 影响，
        # 会让"有 1 个强因子"与"有 5 个弱因子"的族拿到相同权重，且无信号的族
        # （如数据缺失的 sentiment/macro，IC=0）会与有效族等权，稀释整体信号。
        family_strength: dict[str, float] = {}
        for family, cols in families.items():
            ics = [abs(ic.get(c, 0.0)) for c in cols]
            family_strength[family] = max(ics) if ics else 0.0
        if self._explicit_family_weights:
            raw_family = {
                f: float(self._explicit_family_weights.get(f, 0.0)) for f in families
            }
        else:
            raw_family = family_strength
        total_family = sum(raw_family.values())
        if total_family <= 0:
            raw_family = {f: 1.0 for f in families}
            total_family = float(len(families))
        self.family_weights_ = {f: v / total_family for f, v in raw_family.items()}

        # ---- 合成最终因子权重：family_weight × 族内归一化因子权重
        final: dict[str, float] = {}
        for family, cols in families.items():
            within = sum(factor_weights.get(c, 0.0) for c in cols) or 1.0
            for c in cols:
                final[c] = self.family_weights_[family] * (factor_weights.get(c, 0.0) / within)
        total_final = sum(final.values()) or 1.0
        self.factor_weights_ = {c: v / total_final for c, v in final.items()}

    # ------------------------------------------------------------------
    def _raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """按权重合成原始得分（∈ 约 [-1, 1]，各因子已裁剪）。"""
        score = np.zeros(len(X), dtype=float)
        for col, w in self.factor_weights_.items():
            if col not in X.columns:
                continue
            values = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            score += w * np.clip(values, -1.0, 1.0)
        return score

    def _calibrate(self, score: np.ndarray) -> np.ndarray:
        """得分 → 概率：logit 空间单调映射（sigmoid(scale * (score - center))）。"""
        logit = self.scale_ * (score - self.center_)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))

    # ------------------------------------------------------------------
    def train(self, X_train: Any, y_train: np.ndarray,
              X_val: Any = None, y_val: np.ndarray | None = None,
              feature_cols: list[str] | None = None,
              **_ignored: Any) -> dict[str, list[float]]:
        """估计因子权重 + 概率校准（不使用验证集拟合权重，仅用于日志）。

        Args:
            X_train/X_val: 因子矩阵（DataFrame 带列名，或 ndarray + feature_cols）
            feature_cols : 当 X 为 ndarray 时必须提供，用于标识各列因子名
        """
        # 保留完整特征列名用于 ndarray → DataFrame 映射（因子列是其中子集）
        all_cols = list(feature_cols) if feature_cols else None
        X_train = _as_frame(X_train, all_cols)
        if all_cols is None:
            feature_cols = [c for c in X_train.columns if self.is_factor_column(c)]
        else:
            feature_cols = [c for c in all_cols if self.is_factor_column(c)]
        if not feature_cols:
            raise ValueError("FactorModel 需要 factor_* 特征列，当前输入不含任何因子列")

        self.feature_cols_ = feature_cols
        y = np.asarray(y_train, dtype=float)
        ic = self._estimate_ic(X_train[feature_cols], y)
        self.factor_ic_ = ic
        self._resolve_weights(ic)

        score = self._raw_score(X_train[feature_cols])
        self.center_ = float(np.mean(score)) if len(score) else 0.0
        std = float(np.std(score))
        # 训练集得分分布 → 概率标定；std 过小（因子近乎恒定）时退回保守默认
        self.scale_ = float(1.0 / (2.0 * std)) if std > 1e-6 else 4.0
        self.n_factors_ = len(self.factor_weights_)
        logger.info(
            "[FactorModel] 训练完成: 因子 %d 个 / 家族 %d 个 | 权重 top: %s",
            self.n_factors_, len(self.family_weights_),
            ", ".join(
                f"{c}={w:.3f}" for c, w in sorted(
                    self.factor_weights_.items(), key=lambda kv: kv[1], reverse=True
                )[:5]
            ),
        )

        history = {
            "train_score": [float(np.mean(score))],
            "train_prob": [float(np.mean(self._calibrate(score)))],
        }
        if X_val is not None and y_val is not None and len(np.asarray(y_val)) > 0:
            val_score = self._raw_score(_as_frame(X_val, all_cols)[feature_cols])
            history["val_score"] = [float(np.mean(val_score))]
        return history

    def score(self, X: Any) -> np.ndarray:
        """输出原始因子得分（未校准，∈ 约 [-1, 1]）。"""
        if not self.factor_weights_:
            raise RuntimeError("模型未训练")
        X = pd.DataFrame(X)
        return self._raw_score(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """输出 (n, 2) 概率矩阵，与 sklearn 接口一致。"""
        proba = self._calibrate(self.score(X))
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: Any) -> np.ndarray:
        """输出 0/1 类别。"""
        return (self._calibrate(self.score(X)) > 0.5).astype(int)

    # ------------------------------------------------------------------
    def save(self, path: str | None = None) -> Path:
        save_path = Path(path) if path else self.save_dir / "factor_model.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_cols": self.feature_cols_,
                "family_weights": self.family_weights_,
                "factor_weights": self.factor_weights_,
                "factor_ic": self.factor_ic_,
                "scale": self.scale_,
                "center": self.center_,
                "n_factors": self.n_factors_,
                "version": self.VERSION,
            },
            save_path,
        )
        logger.info("[FactorModel] 模型已保存: %s", save_path)
        return save_path

    def load(self, path: str | None = None) -> None:
        load_path = Path(path) if path else self.save_dir / "factor_model.pkl"
        data = joblib.load(load_path)
        self.feature_cols_ = list(data.get("feature_cols", []))
        self.family_weights_ = dict(data.get("family_weights", {}))
        self.factor_weights_ = dict(data.get("factor_weights", {}))
        self.factor_ic_ = dict(data.get("factor_ic", {}))
        self.scale_ = float(data.get("scale", 4.0))
        self.center_ = float(data.get("center", 0.0))
        self.n_factors_ = int(data.get("n_factors", len(self.factor_weights_)))
        logger.info("[FactorModel] 模型已加载: %s（%d 因子）", load_path, self.n_factors_)

    # ------------------------------------------------------------------
    def explain(self, top_n: int = 10) -> dict[str, Any]:
        """因子解释：权重与 IC 排名（供报告 / 监控 / 审计消费）。"""
        rows = [
            {
                "factor": c,
                "family": self._family_of(c),
                "weight": round(w, 4),
                "ic": round(float(self.factor_ic_.get(c, 0.0)), 4),
            }
            for c, w in self.factor_weights_.items()
        ]
        rows.sort(key=lambda r: (r["weight"], abs(r["ic"])), reverse=True)
        return {
            "n_factors": self.n_factors_,
            "family_weights": {k: round(float(v), 4) for k, v in self.family_weights_.items()},
            "top_factors": rows[:top_n],
            "scale": round(self.scale_, 6),
            "center": round(self.center_, 6),
        }
