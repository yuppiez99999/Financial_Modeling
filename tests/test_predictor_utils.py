import joblib
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predictor_utils as pu


class DummyModel:
    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.3, 0.7]]


class DummyScaler:
    def transform(self, X):
        return X


def test_wrap_pickle_model_with_plain_object():
    obj = DummyModel()
    wrapped = pu._wrap_pickle_model(obj)
    assert isinstance(wrapped, dict)
    assert "model" in wrapped and wrapped["model"] is obj
    assert "scaler" in wrapped and wrapped["scaler"] is None


def test_wrap_pickle_model_with_dict():
    d = {"model": DummyModel(), "scaler": DummyScaler()}
    wrapped = pu._wrap_pickle_model(d)
    assert wrapped is d


def test_load_placeholder_and_wrap(tmp_path):
    p = tmp_path / "plain_model.pkl"
    joblib.dump(DummyModel(), p)
    res = pu.load_placeholder(p)
    assert isinstance(res, dict)
    assert "model" in res


def test_load_placeholder_missing(tmp_path):
    p = tmp_path / "no_exist.pkl"
    res = pu.load_placeholder(p)
    assert res is None


def test_load_ensemble_entry_normalizes_lightgbm_and_timesfm_placeholder(tmp_path):
    # create lightgbm file as plain object
    lgb_file = tmp_path / "lgb_plain.pkl"
    joblib.dump(DummyModel(), lgb_file)

    # create timesfm placeholder file
    tfm_file = tmp_path / "timesfm_short_term.pkl"
    joblib.dump({"model": DummyModel(), "scaler": DummyScaler()}, tfm_file)

    def instantiate_timesfm():
        # should not be called because placeholder exists
        raise RuntimeError("should not instantiate")

    entry = pu.load_ensemble_entry(tmp_path, "short_term", lgb_file, instantiate_timesfm)
    assert isinstance(entry, dict)
    assert "lightgbm" in entry and entry["lightgbm"] is not None
    assert isinstance(entry["lightgbm"], dict)
    assert "model" in entry["lightgbm"]
    assert "timesfm" in entry and entry["timesfm"] is not None
    assert isinstance(entry["timesfm"], dict)
from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predictor_utils as utils


class DummyInst:
    def __init__(self, name="dummy"):
        self.name = name


def test_load_placeholder_existing(tmp_path):
    """占位模型目录被 .gitignore 忽略，干净克隆下不存在 —— 测试自行生成到 tmp。"""
    from scripts.create_timesfm_placeholders import create_placeholders

    create_placeholders(tmp_path)
    p = tmp_path / "timesfm_short_term_5d.pkl"
    res = utils.load_placeholder(p)
    assert res is not None
    assert isinstance(res, dict)
    assert "model" in res
    assert "scaler" in res


def test_try_load_timesfm_prefers_placeholder(monkeypatch, tmp_path):
    from scripts.create_timesfm_placeholders import create_placeholders

    create_placeholders(tmp_path)
    # provide instantiate fn that would raise if called
    called = {"inst": False}

    def inst_fn():
        called["inst"] = True
        return DummyInst()

    res = utils.try_load_timesfm_or_instance(tmp_path, "short_term_5d", inst_fn)
    # because placeholder exists, instantiate_fn should not be called
    assert called["inst"] is False
    assert isinstance(res, dict) and "model" in res


def test_try_load_timesfm_instantiates_when_no_placeholder(tmp_path):
    # use a temp dir with no placeholder
    instantiated = DummyInst("instanced")

    def inst_fn():
        return instantiated

    res = utils.try_load_timesfm_or_instance(tmp_path, "no_such_key", inst_fn)
    assert res is instantiated


def test_load_ensemble_entry_with_placeholder():
    save_dir = PROJECT_ROOT / "models"
    model_key = "short_term_5d"
    lightgbm_file = PROJECT_ROOT / "models" / "lightgbm_short_term_5d.pkl"

    def inst_fn():
        return DummyInst()

    entry = utils.load_ensemble_entry(save_dir, model_key, lightgbm_file, inst_fn)
    assert "lightgbm" in entry and "timesfm" in entry
    # lightgbm part may be dict or model but should not raise
    # timesfm part should be present (placeholder dict)
    assert entry["timesfm"] is not None
