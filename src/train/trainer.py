"""金融市场预测模型 - 模型训练器"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """统一模型训练器 - 支持多市场、多模型"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_type = config["model"]["type"]
        self.horizons = config["data"]["prediction_horizons"]
        self.save_dir = Path(config["training"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.collector = DataCollector(config)
        self.preprocessor = DataPreprocessor(config)
        self.models: dict[str, dict[str, Any]] = {}

    VALID_MODEL_TYPES = frozenset({"lightgbm", "pytorch_lstm"})

    def _create_model(self, model_type: str):
        """创建模型实例"""
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {sorted(self.VALID_MODEL_TYPES)}")
        if model_type == "lightgbm":
            from src.train.models.lightgbm_model import LightGBMModel
            return LightGBMModel(self.config)
        elif model_type == "pytorch_lstm":
            from src.train.models.lstm_model import PyTorchLSTMTrainer
            return PyTorchLSTMTrainer(self.config)

    def train_pipeline(self) -> dict[str, Any]:
        """完整训练流水线: 数据采集 → 预处理 → 训练 → 评估"""
        results: dict[str, Any] = {}

        # Step 1: 数据采集
        logger.info("=" * 60)
        logger.info("Step 1/4: 数据采集")
        logger.info("=" * 60)
        all_data = self.collector.collect_all()
        if not all_data:
            logger.error("数据采集失败，程序退出")
            return {"status": "error", "message": "数据采集失败"}

        # Step 2: 特征工程与预处理
        logger.info("=" * 60)
        logger.info("Step 2/4: 数据预处理与特征工程")
        logger.info("=" * 60)
        processed_data = self.preprocessor.process(all_data)

        original_horizon = self.config["data"]["forecast_horizon"]

        # Step 3: 对每个预测周期训练模型
        for horizon_name, horizon_days in self.horizons.items():
            logger.info("=" * 60)
            logger.info(f"Step 3/4: 训练 {horizon_name} 模型 (horizon={horizon_days}d)")
            logger.info("=" * 60)

            # 临时覆盖预测周期（不污染原始 config）
            self.config["data"]["forecast_horizon"] = horizon_days

            datasets, feature_cols = self._split_data(processed_data, horizon_days)

            model_key = f"{horizon_name}_{horizon_days}d"
            self.models[model_key] = {}

            # 训练选定模型
            model = self._create_model(self.model_type)
            train_history = model.train(
                datasets["X_train"], datasets["y_train"],
                datasets["X_val"], datasets["y_val"],
            )

            # 保存模型
            model_path = model.save(str(self.save_dir / f"{self.model_type}_{model_key}.pkl"))
            self.models[model_key] = {
                "model": model,
                "feature_cols": feature_cols,
                "train_history": train_history,
                "model_path": str(model_path),
            }
            logger.info(f"{model_key} 模型训练完成，已保存到 {model_path}")

            # Step 4: 评估
            logger.info(f"评估 {model_key} 模型...")
            from src.eval.evaluator import ModelEvaluator
            evaluator = ModelEvaluator(self.config)
            eval_result = evaluator.evaluate(
                model, datasets["X_test"], datasets["y_test"],
                horizon_name=horizon_name, horizon_days=horizon_days,
            )
            results[model_key] = {
                "train_history": train_history,
                "evaluation": eval_result,
                "feature_count": len(feature_cols),
                "test_samples": len(datasets["y_test"]),
            }

        # 恢复原始配置
        self.config["data"]["forecast_horizon"] = original_horizon

        logger.info("=" * 60)
        logger.info("所有模型训练完成!")
        logger.info("=" * 60)
        return results

    def _split_data(
        self, processed_data: dict[str, pd.DataFrame], horizon: int
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """分割数据集"""
        combined = pd.concat(list(processed_data.values()), ignore_index=True)
        sort_col = "date" if "date" in combined.columns else combined.columns[0]
        combined = combined.sort_values(sort_col)

        feature_cols = self.preprocessor.feature_engineer.get_feature_columns(combined, horizon)
        target_col = f"target_{horizon}d"

        if target_col not in combined.columns:
            combined = self.preprocessor.feature_engineer.create_target(combined, horizon)
            combined = combined.dropna()

        X = combined[feature_cols].values
        y = combined[target_col].values

        split_ratio = self.config["data"]["split_ratio"]
        n = len(X)
        train_end = int(n * split_ratio["train"])
        val_end = int(n * (split_ratio["train"] + split_ratio["val"]))

        datasets = {
            "X_train": X[:train_end], "y_train": y[:train_end],
            "X_val": X[train_end:val_end], "y_val": y[train_end:val_end],
            "X_test": X[val_end:], "y_test": y[val_end:],
        }
        logger.info(f"数据集分割 (horizon={horizon}d): 训练={train_end}, 验证={val_end-train_end}, 测试={n-val_end}")
        return datasets, feature_cols


if __name__ == "__main__":
    import logging
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trainer = ModelTrainer(cfg)
    results = trainer.train_pipeline()
    print(f"\n训练结果: {list(results.keys())}")
