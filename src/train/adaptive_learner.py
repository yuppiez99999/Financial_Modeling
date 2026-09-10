"""金融市场预测模型 - 自适应学习模块

实现模型自我学习能力，根据市场变化不断优化模型。
核心功能：
  - 模型性能监控
  - 自动漂移检测
  - 增量学习
  - 模型选择与集成优化
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """模型性能监控器"""

    def __init__(self, config: dict[str, Any] | str, monitor_dir: str | None = None):
        """config 可为完整配置字典，也可直接传监控目录路径（字符串）。

        传入字符串时等价于指定 monitor_dir，便于独立使用与单测。
        """
        if isinstance(config, (str, Path)):
            monitor_dir = monitor_dir or str(config)
            config = {}
        self.config = config
        self.performance_history: list[dict] = []
        base = monitor_dir or str(
            Path(config.get("training", {}).get("save_dir", "models")) / "monitor"
        )
        self.monitor_dir = Path(base)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)

    def record_performance(self, horizon: str, metrics: Any, timestamp: str | None = None):
        """记录模型性能

        metrics 既支持结构化字典（{"accuracy": 0.7, ...}），也支持直接传入
        准确率数值（0.7），后者会自动包装为 {"accuracy": 0.7}。
        """
        if not isinstance(metrics, dict):
            metrics = {"accuracy": float(metrics)}
        record = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "horizon": horizon,
            "metrics": metrics,
        }
        self.performance_history.append(record)
        self._save_history()
        logger.info(f"记录性能: {horizon} -> accuracy={metrics.get('accuracy', 0):.4f}")

    def record(self, horizon: str, accuracy: float, timestamp: str | None = None) -> None:
        """记录单周期准确率（record_performance 的简化别名）"""
        self.record_performance(horizon, {"accuracy": float(accuracy)}, timestamp)

    def _save_history(self):
        """保存性能历史"""
        path = self.monitor_dir / "performance_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.performance_history, f, ensure_ascii=False, indent=2)

    def _load_history(self):
        """加载性能历史"""
        path = self.monitor_dir / "performance_history.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.performance_history = json.load(f)

    def get_recent_performance(self, days: int = 30) -> list[dict]:
        """获取近期性能"""
        self._load_history()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        return [r for r in self.performance_history if r["timestamp"] >= cutoff]

    def detect_drift(self, horizon: str, threshold: float = 0.05) -> bool:
        """检测模型漂移"""
        recent = self.get_recent_performance(30)
        if len(recent) < 5:
            return False

        horizon_records = [r for r in recent if r["horizon"] == horizon]
        if len(horizon_records) < 3:
            return False

        recent_scores = [r["metrics"].get("accuracy", 0) for r in horizon_records[-3:]]
        earlier_scores = [r["metrics"].get("accuracy", 0) for r in horizon_records[:-3]]

        if not earlier_scores or not recent_scores:
            return False

        recent_avg = np.mean(recent_scores)
        earlier_avg = np.mean(earlier_scores)

        if earlier_avg - recent_avg > threshold:
            logger.warning(
                f"检测到模型漂移: {horizon} - 近期准确率={recent_avg:.4f}, "
                f"早期准确率={earlier_avg:.4f}, 下降={earlier_avg - recent_avg:.4f}"
            )
            return True
        return False


class IncrementalLearner:
    """增量学习器"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.save_dir = Path(config["training"]["save_dir"])

    def update_model(self, model_path: str, new_X: np.ndarray, new_y: np.ndarray):
        """增量更新模型"""
        try:
            data = joblib.load(model_path)
            model = data["model"]
            scaler = data["scaler"]

            logger.info(f"开始增量更新模型: {model_path}")

            X_scaled = scaler.transform(new_X)
            model.n_estimators += 100
            model.fit(X_scaled, new_y, init_model=model)

            joblib.dump({"model": model, "scaler": scaler}, model_path)
            logger.info(f"增量更新完成: {model_path}")
            return True
        except Exception as e:
            logger.error(f"增量更新失败: {e}")
            return False

    def retrain_with_new_data(self, new_data: dict[str, pd.DataFrame], horizon: int = 5):
        """使用新数据重新训练"""
        try:
            from src.data.preprocessor import DataPreprocessor
            from src.train.trainer import ModelTrainer

            preprocessor = DataPreprocessor(self.config)
            processed = preprocessor.process(new_data)

            self.config["data"]["forecast_horizon"] = horizon
            trainer = ModelTrainer(self.config)
            results = trainer.train_pipeline()
            logger.info(f"使用新数据重训练完成: {len(results)} 个模型")
            return results
        except Exception as e:
            logger.error(f"重训练失败: {e}")
            return {}


class AdaptiveLearningEngine:
    """自适应学习引擎"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.monitor = ModelPerformanceMonitor(config)
        self.learner = IncrementalLearner(config)
        self.drift_threshold = config.get("training", {}).get("drift_threshold", 0.05)
        self.auto_retrain = config.get("training", {}).get("auto_retrain", True)

    def evaluate_and_adapt(self, horizon: str, metrics: dict):
        """评估并自适应调整"""
        self.monitor.record_performance(horizon, metrics)

        if self.monitor.detect_drift(horizon, self.drift_threshold):
            if self.auto_retrain:
                logger.info(f"触发自适应重训练: {horizon}")
                self._trigger_retrain(horizon)
            else:
                logger.warning(f"检测到漂移但自动重训练已禁用: {horizon}")

    def _trigger_retrain(self, horizon: str):
        """触发重训练"""
        try:
            from src.data.collector import DataCollector
            from src.train.trainer import ModelTrainer

            collector = DataCollector(self.config)
            new_data = collector.collect_all()

            if new_data:
                trainer = ModelTrainer(self.config)
                results = trainer.train_pipeline()
                logger.info(f"自适应重训练完成: {len(results)} 个模型")
        except Exception as e:
            logger.error(f"自适应重训练失败: {e}")

    def get_adaptive_report(self) -> str:
        """生成自适应学习报告"""
        recent = self.monitor.get_recent_performance(30)
        lines = [
            "=" * 60,
            "自适应学习报告",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"近期记录数: {len(recent)}",
            "",
        ]

        if recent:
            horizons = set(r["horizon"] for r in recent)
            for horizon in sorted(horizons):
                h_records = [r for r in recent if r["horizon"] == horizon]
                accuracies = [r["metrics"].get("accuracy", 0) for r in h_records]
                lines.extend([
                    f"[{horizon}]",
                    f"  记录数: {len(h_records)}",
                    f"  平均准确率: {np.mean(accuracies):.4f}",
                    f"  最高准确率: {max(accuracies):.4f}",
                    f"  最低准确率: {min(accuracies):.4f}",
                    f"  漂移状态: {'⚠️ 检测到漂移' if self.monitor.detect_drift(horizon) else '✅ 正常'}",
                    "",
                ])
        else:
            lines.append("暂无性能记录")

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = {
        "training": {"save_dir": "models", "drift_threshold": 0.05, "auto_retrain": True},
        "data": {"raw_dir": "data/raw", "processed_dir": "data/processed"},
    }

    engine = AdaptiveLearningEngine(config)
    print("自适应学习引擎初始化完成")
    print(engine.get_adaptive_report())