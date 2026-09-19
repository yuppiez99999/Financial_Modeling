"""金融市场预测模型 - LightGBM 基线模型"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LightGBMModel:
    """LightGBM 二分类基线模型"""

    def __init__(self, config: dict[str, Any]):
        self.config = config["model"]["lightgbm"]
        self.save_dir = Path(config["training"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model: lgb.LGBMClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.feature_importance_: np.ndarray | None = None
        # 训练期特征列名（与 X 的列序逐位置对应）：推理/回填按名对齐的依据
        self.feature_cols: list[str] | None = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
              feature_cols: list[str] | None = None) -> dict[str, list[float]]:
        """训练模型（X_val/y_val 可选；缺省时不做早停验证）。

        ``feature_cols`` 必须传：LightGBM 从纯 ndarray 训练时只会记下
        ``Column_0..N`` 这类**位置占位名**，推理侧拿到的却是语义列名
        （``ma_5`` / ``rsi`` ...）——两边对不上时按名对齐会判**全部缺列**，
        最终静默走 0 填充产出一条与任何标的无关的常数概率
        （2026-09-16 实测：全池 3409 个锚点置信度恒为 0.0303756）。
        把训练期列名写进模型与产物，是让"按名对齐"真正可用的前提。
        """
        logger.info("开始训练 LightGBM 模型...")
        if feature_cols:
            self.feature_cols = list(feature_cols)

        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = lgb.LGBMClassifier(
            objective=self.config["objective"],
            n_estimators=self.config["n_estimators"],
            learning_rate=self.config["learning_rate"],
            max_depth=self.config["max_depth"],
            num_leaves=self.config["num_leaves"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            verbose=self.config["verbose"],
            random_state=42,
        )

        callbacks = [lgb.log_evaluation(self.config.get("verbose", -1) + 50)]
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            fit_kwargs["eval_set"] = [(X_val_scaled, y_val)]
            callbacks.insert(0, lgb.early_stopping(self.config["early_stopping_rounds"]))

        self.model.fit(X_train_scaled, y_train, callbacks=callbacks, **fit_kwargs)

        # 训练期列名 → 模型原生特征名（按名对齐的唯一依据）
        if feature_cols:
            self._sync_feature_names()

        self.feature_importance_ = self.model.feature_importances_
        logger.info(f"训练完成，最佳迭代: {self.model.best_iteration_}")

        # 训练历史
        eval_result = self.model.evals_result_
        return {
            "val_logloss": eval_result.get("valid_0", {}).get("binary_logloss", []),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str | None = None) -> Path:
        """保存模型"""
        save_path = Path(path) if path else self.save_dir / "lightgbm_model.pkl"
        payload = {"model": self.model, "scaler": self.scaler}
        if self.feature_cols:
            payload["feature_cols"] = list(self.feature_cols)
        joblib.dump(payload, save_path)
        logger.info(f"模型已保存到 {save_path}")
        return save_path

    def load(self, path: str | None = None) -> None:
        """加载模型"""
        load_path = Path(path) if path else self.save_dir / "lightgbm_model.pkl"
        data = joblib.load(load_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_cols = list(data.get("feature_cols") or []) or None
        if self.feature_cols:
            # 既有产物（先于本次修复训练）只带 feature_cols、模型内仍是 Column_N：
            # 在这里补写回模型，让按名对齐对旧产物同样生效。
            self._sync_feature_names()

    def _sync_feature_names(self) -> bool:
        """把 `feature_cols` 同步为**模型原生特征名**；失败返回 False（回落位置对齐）。

        两条约束（均为实测结论，不是猜测）：
          1. `LGBMClassifier.feature_name_` 是**只读 property**（无 setter），
             且 `fit` 会用 `Column_N` 覆盖构造期传入的名字 —— 所以只能在
             `fit` 之后、经 `booster_.feature_name` 写；
          2. sklearn 包装的 booster 会**缓存** `feature_name()`，因此必须把
             `boosters` 属性一并清掉，否则改动不生效（写了等于没写）。
        两条都漏掉的后果不是报错，而是"以为按名对齐生效了、实际仍是占位名"。
        """
        cols = list(self.feature_cols or [])
        if not cols or self.model is None:
            return False
        # 复用推理侧的同步实现：两边必须**同一套**写回逻辑，
        # 否则训练写进去的名字与加载读出来的名字会不是一回事。
        from src.inference.predictor import _sync_model_feature_names

        return _sync_model_feature_names(self.model, cols)
        logger.info(f"模型已从 {load_path} 加载")
