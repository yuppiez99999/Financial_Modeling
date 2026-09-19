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
        # 训练时特征列契约（名称 + 顺序）：持久化进 pkl，供推理按名对齐，
        # 防止特征管线变更时静默错位（2026-09-15 退化诊断遗留隐患修复）
        self.feature_cols_: list[str] | None = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
              feature_cols: list[str] | None = None) -> dict[str, list[float]]:
        """训练模型（X_val/y_val 可选；feature_cols 记录训练特征契约）。"""
        logger.info("开始训练 LightGBM 模型...")
        self.feature_cols_ = [str(c) for c in feature_cols] if feature_cols else None
        if self.feature_cols_ is None:
            logger.warning("训练未提供 feature_cols：pkl 将缺少特征契约，推理只能按数量对齐（存在静默错位风险）")

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
        """保存模型（含 feature_cols 特征契约）"""
        save_path = Path(path) if path else self.save_dir / "lightgbm_model.pkl"
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "feature_cols": self.feature_cols_}, save_path)
        logger.info(f"模型已保存到 {save_path}（feature_cols={'有' if self.feature_cols_ else '缺失'}）")
        return save_path

    def load(self, path: str | None = None) -> None:
        """加载模型"""
        load_path = Path(path) if path else self.save_dir / "lightgbm_model.pkl"
        data = joblib.load(load_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_cols_ = list(data.get("feature_cols") or []) or None
        logger.info(f"模型已从 {load_path} 加载")
