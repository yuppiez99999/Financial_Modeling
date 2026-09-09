from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predictor_utils as utils


class DummyInst:
    def __init__(self, name="dummy"):
        self.name = name


def test_load_placeholder_existing():
    p = PROJECT_ROOT / "models" / "timesfm_short_term_5d.pkl"
    res = utils.load_placeholder(p)
    assert res is not None
    assert isinstance(res, dict)
    assert "model" in res
    assert "scaler" in res


def test_try_load_timesfm_prefers_placeholder(monkeypatch):
    # provide instantiate fn that would raise if called
    called = {"inst": False}

    def inst_fn():
        called["inst"] = True
        return DummyInst()

    p = PROJECT_ROOT / "models"
    res = utils.try_load_timesfm_or_instance(p, "short_term_5d", inst_fn)
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
