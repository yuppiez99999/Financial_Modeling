"""因子组合预测器：把因子得分与机器学习模型概率做**类级加权**融合。

与 `src/inference/predictor.py` 的 `ensemble`（LightGBM × TimesFM 两分量）不同，
本模块面向 Q2 的"多因子模型集成"目标：

    最终得分 = Σ_class w_class × class_得分     （Σw = 1）

类（class）与权重来自 `model.factors.ensemble.weights`：

| 类 | 来源 | 得分含义 |
|----|------|---------|
| factor   | FactorModel | 因子加权组合得分（含 IC 权重） |
| tree     | LightGBM    | 梯度提升树概率 |
| sequence | LSTM        | 时序深度模型概率 |

每类得分先映射到 [-1, 1] 的**方向分**（概率 0.5 为中性）再融合，
避免不同量纲直接相加。缺失类自动剔除并重新归一化（fail-soft），
因此**只装了 lightgbm 没装 torch** 的环境依然可用。

配套 `align_features` 处理"因子列缺失"（例如模型在启用扩展指标的环境训练、
推理环境未启用）：缺失因子视为中性 0，绝不因列不齐而中断推理。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 类默认权重（可被 config.model.factors.ensemble.weights 覆盖）
DEFAULT_CLASS_WEIGHTS: dict[str, float] = {
    "factor": 0.40,
    "tree": 0.35,
    "sequence": 0.25,
}


def direction_score(proba: float) -> float:
    """概率 → 方向分：0.5 中性映射为 0，两端线性放大到 ±1。"""
    return float(np.clip((float(proba) - 0.5) * 2.0, -1.0, 1.0))


class FactorPredictor:
    """多因子组合预测器（特征 DataFrame 输入 → 概率）。"""

    NAME = "factor"

    def __init__(self, config: dict[str, Any], factor_model: Any,
                 tree_entry: Any = None, sequence_model: Any = None) -> None:
        """
        Args:
            config        : 全局配置
            factor_model  : 已训练/已加载的 FactorModel
            tree_entry    : LightGBM 模型条目 {model, scaler}（可选）
            sequence_model: LSTM 模型条目（可选，需含 model + feature_cols）
        """
        self.config = config
        self.factor_model = factor_model
        self.tree_entry = tree_entry
        self.sequence_model = sequence_model
        self.last_components: dict[str, float] = {}

        cfg = ((config.get("model", {}) or {}).get("factors", {}) or {})
        ens_cfg = cfg.get("ensemble", {}) or {}
        weights = dict(DEFAULT_CLASS_WEIGHTS)
        weights.update({k: float(v) for k, v in (ens_cfg.get("weights", {}) or {}).items()})
        self.class_weights = weights
        self.enabled_classes = list(ens_cfg.get("classes", list(DEFAULT_CLASS_WEIGHTS)))
        self.proba_clip = tuple(ens_cfg.get("proba_clip", (0.02, 0.98)))

        # 因子列集合（用于推理特征对齐）
        self.factor_cols: list[str] = list(getattr(factor_model, "feature_cols_", []) or [])

    # ------------------------------------------------------------------
    def align_features(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
        """补齐缺失因子列（中性 0）；多余列按需裁剪，列序与训练一致。"""
        factor_cols = feature_cols if feature_cols is not None else self.factor_cols
        if not factor_cols:
            return df
        out = df.copy()
        missing = [c for c in factor_cols if c not in out.columns]
        if missing:
            logger.warning("[FactorPredictor] 缺失因子列 %d 个，按中性 0 补齐: %s",
                           len(missing), ", ".join(missing[:6]))
            for col in missing:
                out[col] = 0.0
        return out

    # ------------------------------------------------------------------
    def _prepare_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保因子列存在：缺失时用因子库按原始行情现算（推理链路的常见形态）。

        - 已带全部所需因子列 → 原样返回（避免重复计算）；
        - 只有原始 OHLCV / 部分特征 → 用 FactorLibrary 现算，**并把结果合并回
          原表**（保留 tree / sequence 类依赖的其他特征列）；
        - 缺 close/high/low 且因子列不全 → 原样返回（由各分量自行降级）。
        """
        if not self.factor_cols or all(c in df.columns for c in self.factor_cols):
            return df
        if not {"close", "high", "low"}.issubset(df.columns):
            return df
        try:
            from src.factors.factor_library import FactorLibrary

            computed = FactorLibrary(self.config).compute(df)
            add_cols = [c for c in self.factor_cols if c in computed.columns and c not in df.columns]
            if not add_cols:
                return computed
            merged = df.copy()
            for col in add_cols:
                merged[col] = computed[col].to_numpy()
            return merged
        except Exception as e:  # noqa: BLE001
            logger.warning("[FactorPredictor] 因子库现算失败: %s", e)
            return df

    def _factor_proba(self, df: pd.DataFrame) -> float | None:
        if self.factor_model is None:
            return None
        try:
            prepared = self._prepare_factors(df)
            cols = self.factor_cols or [c for c in prepared.columns if str(c).startswith("factor_")]
            if not cols:
                return None
            X = self.align_features(prepared[cols].iloc[-1:], cols)
            return float(self.factor_model.predict_proba(X)[0][1])
        except Exception as e:  # noqa: BLE001
            logger.warning("[FactorPredictor] factor 类预测失败: %s", e)
            return None

    def _tree_proba(self, df: pd.DataFrame) -> float | None:
        if not isinstance(self.tree_entry, dict) or "model" not in self.tree_entry:
            return None
        model = self.tree_entry["model"]
        scaler = self.tree_entry.get("scaler")
        try:
            cols = [c for c in df.columns if not str(c).startswith("factor_")]
            X = df[cols].iloc[-1:]
            latest = X.to_numpy(dtype=float)
            expected = getattr(model, "n_features_in_", None)
            if expected is not None and latest.shape[1] != expected:
                latest = latest[:, :expected] if latest.shape[1] > expected else np.hstack(
                    [latest, np.zeros((1, expected - latest.shape[1]))]
                )
            X_scaled = scaler.transform(latest) if scaler is not None else latest
            return float(model.predict_proba(X_scaled)[0][1])
        except Exception as e:  # noqa: BLE001
            logger.warning("[FactorPredictor] tree 类预测失败: %s", e)
            return None

    def _sequence_proba(self, df: pd.DataFrame) -> float | None:
        if self.sequence_model is None:
            return None
        try:
            import torch

            model = self.sequence_model.get("model") if isinstance(self.sequence_model, dict) \
                else self.sequence_model
            feature_cols = (
                self.sequence_model.get("feature_cols") if isinstance(self.sequence_model, dict) else None
            ) or [c for c in df.columns if not str(c).startswith("factor_")]
            seq_len = int(
                (self.config.get("model", {}).get("pytorch_lstm", {}) or {}).get("sequence_length", 20)
            )
            if len(df) < seq_len:
                return None
            seq = df[feature_cols].iloc[-seq_len:].to_numpy(dtype=float)
            expected = getattr(model, "expected_features", None) or getattr(model, "input_size", None)
            expected = getattr(model, "lstm", None) and model.lstm.input_size or expected
            if expected is not None and seq.shape[1] != expected:
                seq = seq[:, :expected] if seq.shape[1] > expected else np.hstack(
                    [seq, np.zeros((seq.shape[0], expected - seq.shape[1]))]
                )
            tensor = torch.FloatTensor(seq).unsqueeze(0)
            with torch.no_grad():
                return float(model(tensor).item())
        except Exception as e:  # noqa: BLE001
            logger.warning("[FactorPredictor] sequence 类预测失败: %s", e)
            return None

    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> float:
        """融合各类得分，返回上涨概率（0~1）。"""
        prepared = self._prepare_factors(df)
        handlers = {
            "factor": self._factor_proba,
            "tree": self._tree_proba,
            "sequence": self._sequence_proba,
        }
        low, high = self.proba_clip
        components: dict[str, float] = {}
        weighted = 0.0
        weight_sum = 0.0

        for name in self.enabled_classes:
            handler = handlers.get(name)
            if handler is None:
                continue
            proba = handler(prepared)
            if proba is None:
                continue
            proba = float(np.clip(proba, low, high))
            components[name] = round(proba, 4)
            w = float(self.class_weights.get(name, 0.0))
            weighted += direction_score(proba) * w
            weight_sum += w

        if weight_sum <= 0:
            logger.warning("[FactorPredictor] 无可用预测类，返回中性概率 0.5")
            self.last_components = {}
            return 0.5

        fused_dir = weighted / weight_sum
        proba = float(np.clip(0.5 + fused_dir / 2.0, low, high))
        self.last_components = components
        logger.info("[FactorPredictor] 分量=%s 融合概率=%.4f", components, proba)
        return proba

    def build_entry(self, df: pd.DataFrame) -> dict[str, Any]:
        """产出与 LightGBM 条目同构的推理条目，便于 PredictionEngine 统一消费。"""
        proba = self.predict_proba(df)
        return {
            "multifactor": True,
            "probability": proba,
            "components": dict(self.last_components),
            "class_weights": {k: v for k, v in self.class_weights.items() if k in self.last_components},
            "explain": self.factor_model.explain(5) if self.factor_model is not None else {},
        }
