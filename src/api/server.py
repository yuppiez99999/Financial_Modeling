"""专业版 - Web API 服务 (FastAPI)

提供 REST API 接口，支持：
  - 单标的/批量预测
  - 模型信息查询
  - 历史预测审计
  - 健康检查
  - 信号推送触发
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 全局引擎实例（线程安全初始化）
_engine = None
_config = None
_notifier = None
_audit = None
_init_lock = threading.Lock()
_initialized = False


def _init_engine():
    """初始化推理引擎（幂等，线程安全）"""
    global _engine, _config, _notifier, _audit, _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        config_path = Path(__file__).parent.parent.parent / "configs" / "config_pro.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)

        if _engine is None:
            from src.inference.predictor import PredictionEngine
            model_type = _config["model"]["type"]
            _engine = PredictionEngine(_config)
            try:
                _engine.load_models(model_type)
                logger.info(f"API 引擎已加载模型: {model_type}")
            except Exception as e:
                logger.warning(f"模型未加载（可能未训练）: {e}")

        if _notifier is None:
            from src.notification.notifier import SignalNotifier
            _notifier = SignalNotifier(_config)

        if _audit is None:
            from src.audit.prediction_audit import PredictionAudit
            _audit = PredictionAudit()

        _initialized = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_engine()
    yield


app = FastAPI(
    title="TrendCast Pro API",
    description="金融市场预测模型 - 专业版 Web API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================

class PredictRequest(BaseModel):
    symbol: str
    horizon: str = "short_term"


class BatchPredictRequest(BaseModel):
    symbols: list[str]
    horizon: str = "all"


class NotifyRequest(BaseModel):
    predictions: list[dict[str, Any]]


# ==================== API 路由 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "edition": "professional",
        "models_loaded": len(_engine.models) if _engine else 0,
    }


@app.get("/api/v1/models")
async def get_models():
    """查询已加载模型信息"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(404, "无已加载模型，请先训练")
    models_info = {}
    for key, data in _engine.models.items():
        if isinstance(data, dict) and "model" in data:
            models_info[key] = {
                "type": "lightgbm",
                "feature_count": getattr(data["model"], "n_features_in_", None),
            }
        else:
            models_info[key] = {"type": "pytorch_lstm"}
    return {"models": models_info}


@app.get("/api/v1/predict/{symbol}")
async def predict_symbol(
    symbol: str,
    horizon: str = Query("short_term", description="预测周期: short_term/mid_term/long_term/all"),
):
    """单标的预测"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")

    try:
        if horizon == "all":
            result = _engine.predict_all_horizons(symbol)
        else:
            result = _engine.predict(symbol, horizon)

        # 审计记录
        if _audit and "predictions" in result:
            for h, pred in result["predictions"].items():
                if "error" not in pred:
                    _audit.record_prediction(pred)
        elif _audit and "error" not in result:
            _audit.record_prediction(result)

        return result
    except Exception as e:
        raise HTTPException(500, f"预测失败: {e}")


@app.post("/api/v1/predict/batch")
async def batch_predict(req: BatchPredictRequest):
    """批量预测（专业版支持 50+ 标的）"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载")

    if len(req.symbols) > 100:
        raise HTTPException(400, "单次最多 100 个标的")

    results = []
    for symbol in req.symbols:
        try:
            if req.horizon == "all":
                result = _engine.predict_all_horizons(symbol)
            else:
                result = _engine.predict(symbol, req.horizon)
            results.append(result)

            # 审计记录
            if _audit and "predictions" in result:
                for h, pred in result["predictions"].items():
                    if "error" not in pred:
                        _audit.record_prediction(pred)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})

    return {"count": len(results), "results": results}


@app.post("/api/v1/notify")
async def send_notifications(req: NotifyRequest):
    """手动触发信号推送"""
    _init_engine()
    if not _notifier:
        raise HTTPException(500, "通知器未初始化")
    result = _notifier.notify(req.predictions)
    return {"status": "sent", "channels": result}


@app.get("/api/v1/audit/report")
async def get_audit_report():
    """获取预测审计报告"""
    _init_engine()
    if not _audit:
        raise HTTPException(500, "审计器未初始化")
    report = _audit.generate_report()
    return {"report": report}


@app.get("/api/v1/audit/stats")
async def get_audit_stats():
    """获取审计统计摘要"""
    _init_engine()
    if not _audit:
        raise HTTPException(500, "审计器未初始化")
    records = _audit._load_records()
    verified = [r for r in records if r.get("verified")]
    hits = sum(1 for r in verified if r.get("hit"))
    return {
        "total_predictions": len(records),
        "verified": len(verified),
        "hits": hits,
        "hit_rate": hits / len(verified) if verified else 0,
    }


@app.get("/api/v1/config/markets")
async def get_markets():
    """查询支持的标的列表"""
    _init_engine()
    markets = _config.get("data", {}).get("markets", {})
    result = {}
    for name, cfg in markets.items():
        if cfg.get("enabled"):
            result[name] = cfg.get("symbols", [])
    return {"markets": result}


def run_server(host: str = "0.0.0.0", port: int = 8800):
    """启动 API 服务"""
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info(f"启动 TrendCast Pro API: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
