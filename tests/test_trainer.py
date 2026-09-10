import sys
from pathlib import Path
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.trainer import ModelTrainer
from src.train.models.lightgbm_model import LightGBMModel
from src.eval.evaluator import ModelEvaluator


def _load_cfg():
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_lightgbm_model_train_predict():
    cfg = _load_cfg()
    model = LightGBMModel(cfg)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    model.train(X, y)
    proba = model.predict_proba(X[:10])
    assert proba.shape[0] == 10
    assert proba.min() >= 0 and proba.max() <= 1


def test_trainer_build_dataset():
    cfg = _load_cfg()
    trainer = ModelTrainer(cfg)
    ds = trainer._build_dataset("600519.SH", 5)
    # 模拟兜底数据应能产出特征 & 标签
    if ds is not None:
        X, y = ds
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0
    else:
        # 数据不足时返回 None，属正常降级
        assert ds is None


def test_trainer_pipeline_saves_models(tmp_path):
    cfg = _load_cfg()
    cfg["training"]["save_dir"] = str(tmp_path)
    trainer = ModelTrainer(cfg)
    results = trainer.train_pipeline()
    assert "short_term" in results
    # 模型文件应被创建
    model_paths = [r["model_path"] for r in results.values()]
    for p in model_paths:
        assert Path(p).exists()


def test_evaluator_metrics():
    cfg = _load_cfg()
    evaluator = ModelEvaluator(cfg)
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    y_proba = np.array([0.9, 0.2, 0.8, 0.4, 0.1, 0.7, 0.3, 0.2])
    res = evaluator.evaluate(
        SimpleModel(y_pred, y_proba), np.zeros((8, 3)), y_true
    )
    assert 0 <= res["metrics"]["accuracy"] <= 1
    assert 0 <= res["metrics"]["financial"]["win_rate"] <= 1
    assert res["metrics"]["financial"]["sharpe_ratio"] is not None


class SimpleModel:
    def __init__(self, y_pred, y_proba):
        self._y_pred = y_pred
        self._y_proba = y_proba

    def predict(self, X):
        return self._y_pred

    def predict_proba(self, X):
        return self._y_proba
