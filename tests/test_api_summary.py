"""测试 /api/v1/portfolio/summary 端点与 build_portfolio_summary 归一化逻辑。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

try:
    from src.api import server
except Exception:  # noqa: BLE001  # FastAPI/依赖缺失时跳过
    server = None

try:
    from fastapi.testclient import TestClient
except Exception:  # noqa: BLE001
    TestClient = None


# ---------- 测试夹具 ----------
class _FakeEngine:
    def __init__(self, models, response_map):
        self.models = models
        self._response_map = response_map

    def predict_all_horizons(self, symbol):
        return self._response_map.get(symbol, {"symbol": symbol, "predictions": {}})


def _make_config():
    return {
        "model": {"type": "lightgbm"},
        "data": {
            "prediction_horizons": {
                "short_term": 5,
                "mid_term": 10,
                "long_term": 20,
            }
        },
    }


def _good_response(symbol):
    return {
        "symbol": symbol,
        "predictions": {
            "short_term": {"direction": "看涨", "probability": 0.62, "horizon_days": 5},
            "mid_term": {"direction": "看涨", "probability": 0.71, "horizon_days": 10},
            "long_term": {"direction": "看跌", "probability": 0.55, "horizon_days": 20},
        },
    }


def _all_error_response(symbol):
    return {
        "symbol": symbol,
        "predictions": {
            "short_term": {"error": "无数据"},
            "mid_term": {"error": "无数据"},
            "long_term": {"error": "无数据"},
        },
    }


# ---------- 归一化逻辑测试 ----------
def test_build_summary_normal():
    if server is None:
        pytest.skip("src.api.server 不可用")
    engine = _FakeEngine(models={"short_term_5d": 1, "mid_term_10d": 1, "long_term_20d": 1},
                          response_map={"600519.SH": _good_response("600519.SH")})
    cfg = _make_config()
    out = server.build_portfolio_summary(engine, cfg, ["600519.SH"])
    assert out["model_type"] == "lightgbm"
    assert out["meta"]["models_loaded"] == 3
    assert out["meta"]["symbol_count"] == 1
    assert out["meta"]["error_symbols"] == []
    pred = out["predictions"][0]
    assert pred["symbol"] == "600519.SH"
    assert pred["horizons"]["short_term"]["direction"] == "看涨"
    assert pred["horizons"]["short_term"]["probability"] == 0.62
    assert pred["horizons"]["short_term"]["model"] == "lightgbm_short_term_5d"
    assert pred["horizons"]["long_term"]["direction"] == "看跌"
    assert pred["horizons"]["long_term"]["model"] == "lightgbm_long_term_20d"


def test_build_summary_error_symbols():
    if server is None:
        pytest.skip("src.api.server 不可用")
    engine = _FakeEngine(models={"short_term_5d": 1},
                          response_map={"ERR.SH": _all_error_response("ERR.SH")})
    out = server.build_portfolio_summary(engine, _make_config(), ["ERR.SH"])
    assert out["meta"]["error_symbols"] == ["ERR.SH"]
    assert out["predictions"][0]["error"] == "预测失败"
    assert out["predictions"][0]["horizons"] == {}


def test_build_summary_empty():
    if server is None:
        pytest.skip("src.api.server 不可用")
    engine = _FakeEngine(models={}, response_map={})
    out = server.build_portfolio_summary(engine, _make_config(), [])
    assert out["predictions"] == []
    assert out["meta"]["symbol_count"] == 0
    assert out["meta"]["error_symbols"] == []


# ---------- 路由冒烟（FastAPI 可用时） ----------
def test_route_summary_exists():
    if server is None or TestClient is None:
        pytest.skip("FastAPI TestClient 不可用")
    client = TestClient(server.app)
    resp = client.get("/api/v1/portfolio/summary?symbols=600519.SH")
    # 无模型→503；已加载→200。两者都接受，仅校验契约键存在。
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "predictions" in body and "meta" in body
