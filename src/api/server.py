"""专业版 - Web API 服务 (FastAPI)

提供 REST API 接口，支持：
  - 单标的/批量预测
  - 模型信息查询
  - 历史预测审计
  - 健康检查
  - 信号推送触发
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import yaml

try:  # FastAPI 为可选依赖：未安装时本模块仍可导入，仅无法启动服务
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - 取决于运行环境
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment,misc]
    CORSMiddleware = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment,misc]

    def Query(*args, **kwargs):  # type: ignore[misc]
        """FastAPI 缺失时的 Query 占位，保证模块可导入。"""
        return None

    _HAS_FASTAPI = False

logger = logging.getLogger(__name__)

# 全局引擎实例（线程安全初始化）
_engine = None
_config = None
_notifier = None
_audit = None
_init_lock = threading.Lock()
_initialized = False
# create_app() 指定的配置文件路径（None = 默认查找顺序）
_config_override: str | None = None


def _init_engine():
    """初始化推理引擎（幂等，线程安全）"""
    global _engine, _config, _notifier, _audit, _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        if _config_override:
            config_path = Path(_config_override)
        else:
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


def create_app(config_path: str | None = None):
    """创建 FastAPI 应用（供 CLI / 测试 / 部署复用）。

    config_path 为 None 时沿用默认配置查找顺序（config_pro.yaml → config.yaml）。
    """
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "未安装 FastAPI，无法创建 API 应用。请执行: pip install fastapi uvicorn pydantic"
        )
    global _config_override
    _config_override = config_path
    application = FastAPI(
        title="TrendCast Pro API",
        description="金融市场预测模型 - 专业版 Web API",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 复用同一组路由定义（装饰器注册在模块级 app 上，这里整体挂载）
    application.router.routes = list(app.router.routes)
    return application


if _HAS_FASTAPI:
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
else:  # pragma: no cover - 未安装 FastAPI 的降级分支
    app = None


# ==================== 请求/响应模型 ====================

class PredictRequest(BaseModel):
    symbol: str
    horizon: str = "short_term"


class BatchPredictRequest(BaseModel):
    symbols: List[str]
    horizon: str = "all"


class NotifyRequest(BaseModel):
    predictions: List[Dict[str, Any]]


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


@app.get("/api/v1/monitor/report")
async def get_monitor_report():
    """模型监控报表（审计命中率 / 自适应漂移 / 数据源健康 / 模型产物）"""
    _init_engine()
    try:
        from src.monitor.health_report import ModelMonitor

        return ModelMonitor(_config or {}).collect().to_dict()
    except Exception as e:
        raise HTTPException(500, f"监控报表生成失败: {e}")


# ==================== Q2 路线：门禁与多因子 ====================

@app.get("/api/v1/strategy/gate")
async def get_strategy_gate():
    """策略门禁状态：IC / 命中率判定结果（只读，不触发训练）。

    读取 ``reports/strategy_gate.json``（由 ``python main.py gate`` 产出）；
    文件缺失时返回 fail-close 的 readonly 判定，绝不默认放行。
    """
    from src.trading.gate import StrategyGate

    gate_dir = (_config or {}).get("strategy_gate", {}).get("report_dir", "reports")
    path = Path(gate_dir) / "strategy_gate.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gate] 读取门禁结果失败: {e}")
    return StrategyGate(_config or {}).decide({}).to_dict()


@app.get("/api/v1/factors/{symbol}")
async def get_factors(symbol: str):
    """多因子加权组合预测（单标的，逐周期输出因子得分与贡献）"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")
    try:
        from src.data.collector import DataCollector
        from src.inference.factor_combiner import FactorCombiner, combine_horizon

        pred = _engine.predict_all_horizons(symbol)
        df = DataCollector(_config).load_cached(symbol)
        combiner = FactorCombiner(_config)
        horizons_out = {
            hname: combine_horizon(combiner, symbol, df, hp).to_dict()
            for hname, hp in (pred.get("predictions") or {}).items()
        }
        return {"symbol": symbol, "weighter": combiner.weighter, "horizons": horizons_out}
    except Exception as e:
        raise HTTPException(500, f"多因子组合失败: {e}")


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


# ==================== 交易适配层接口（MR#6 并入） ====================

@app.get("/api/v1/signal/{symbol}")
async def get_signal(symbol: str):
    """返回单标的综合交易信号（仅信号，不含订单/风控/执行）"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")
    try:
        from src.trading.signal import SignalEngine

        pred = _engine.predict_all_horizons(symbol)
        sig = SignalEngine(_config).build_signal(symbol, pred)
        return {"symbol": symbol, "signal": sig.to_dict()}
    except Exception as e:
        raise HTTPException(500, f"信号生成失败: {e}")


@app.get("/api/v1/trade/{symbol}")
async def get_trade(symbol: str):
    """返回完整交易适配结果：信号 + 风控 + 订单"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")
    try:
        from src.trading.adapter import TradingAdapter

        adapter = TradingAdapter(_config, predictor=_engine)
        return adapter.process_symbol(symbol)
    except Exception as e:
        raise HTTPException(500, f"交易适配失败: {e}")


def build_portfolio_summary(engine: Any, config: dict, symbol_list: list[str]) -> dict[str, Any]:
    """组合级多周期方向预测摘要（纯逻辑，可单测；不触网）。

    返回固定契约：
    {
      "generated_at": ISO,
      "model_type": str,
      "predictions": [{"symbol","sector","horizons":{h:{direction,probability,model}}}],
      "meta": {"models_loaded","symbol_count","error_symbols"}
    }
    """
    model_type = config.get("model", {}).get("type", "lightgbm")
    horizons_cfg = config.get("data", {}).get("prediction_horizons", {})
    predictions: list[dict[str, Any]] = []
    error_symbols: list[str] = []
    for symbol in symbol_list:
        try:
            result = engine.predict_all_horizons(symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[portfolio/summary] {symbol} 预测异常: {e}")
            error_symbols.append(symbol)
            continue
        per = result.get("predictions", {})
        if not per or all("error" in p for p in per.values()):
            error_symbols.append(symbol)
            predictions.append({"symbol": symbol, "sector": "", "horizons": {}, "error": "预测失败"})
            continue
        horizons_out: dict[str, Any] = {}
        for hname, hdays in horizons_cfg.items():
            pred = per.get(hname, {})
            if not pred or "error" in pred:
                continue
            horizons_out[hname] = {
                "direction": pred.get("direction", "未知"),
                "probability": pred.get("probability", 0.0),
                "model": f"{model_type}_{hname}_{hdays}d",
            }
        predictions.append({"symbol": symbol, "sector": "", "horizons": horizons_out})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_type,
        "predictions": predictions,
        "meta": {
            "models_loaded": len(engine.models) if getattr(engine, "models", None) is not None else 0,
            "symbol_count": len(symbol_list),
            "error_symbols": error_symbols,
        },
    }


@app.get("/api/v1/portfolio/summary")
async def get_portfolio_summary(
    symbols: str = Query(None, description="逗号分隔的标的列表，缺省用 config 启用的 markets 标的"),
):
    """组合级多周期方向预测摘要（供 28 量化系统消费；16_ 只出方向与概率，不输出仓位建议）"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        markets = _config.get("data", {}).get("markets", {})
        symbol_list = []
        for name, cfg in markets.items():
            if cfg.get("enabled"):
                symbol_list.extend(cfg.get("symbols", []))
    symbol_list = list(dict.fromkeys(symbol_list))  # 去重保序

    return build_portfolio_summary(_engine, _config, symbol_list)


def run_server(host: str = "0.0.0.0", port: int = 8800):
    """启动 API 服务"""
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info(f"启动 TrendCast Pro API: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
