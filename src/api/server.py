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

from src.factors import FACTOR_FAMILIES
from src.inference.probability_calibrator import enrich_prediction

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


# 默认允许的跨域来源（本地开发）。生产通过 api.cors_origins 显式配置。
DEFAULT_CORS_ORIGINS = [
    "http://localhost", "http://localhost:3000", "http://localhost:8000",
    "http://127.0.0.1", "http://127.0.0.1:3000", "http://127.0.0.1:8000",
]


def _cors_settings() -> tuple[list[str], bool]:
    """解析 CORS 配置。

    真实缺陷（安全）：原实现硬编码 ``allow_origins=["*"]``。该 API 提供
    ``/api/v1/predict``、``/api/v1/strategy/gate`` 等**无鉴权**的读写端点，
    配合通配来源，任意网页都能静默读取本机服务返回的持仓/信号数据。
    这里改为**默认白名单**，需要放开时用 ``api.cors_origins`` 显式声明；
    显式通配 ``["*"]`` 仍允许（保持向后兼容），但会记 WARNING 留痕，
    且按规范**不使用** ``allow_credentials``（通配来源 + 携带凭证是被浏览器
    拒绝、且属于危险组合的配置）。
    """
    origins = (_config or {}).get("api", {}).get("cors_origins")
    if not origins:
        return list(DEFAULT_CORS_ORIGINS), False
    if isinstance(origins, str):
        origins = [origins]
    origins = [str(o) for o in origins]
    if "*" in origins:
        logger.warning(
            "[api] CORS allow_origins 配置为通配 '*'，任意站点均可读取本服务响应"
            "（api.cors_origins）：生产环境请改为显式来源白名单"
        )
        return ["*"], False
    return origins, False


def _internal_error(message: str, exc: Exception) -> Exception:
    """构造对外可读、对内可诊断的 500 错误（**不回显内部异常文本**）。

    真实缺陷：原实现一律 ``HTTPException(500, f"...: {e}")``。异常文本里
    可能含**绝对路径、模型/配置内容、上游数据源返回体**（乃至凭据片段），
    等于把内部实现细节交给任意调用方。这里改为：对外只回一句稳定文案 +
    由 ``logger.exception`` 把完整堆栈写进服务端日志，两者用同一句
    message 关联，排障不受影响。
    """
    logger.exception("[api] %s: %s", message, exc)
    return HTTPException(500, message)


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
    _cors_origins, _cors_credentials = _cors_settings()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=_cors_credentials,
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

    _cors_origins, _cors_credentials = _cors_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=_cors_credentials,
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
        raise _internal_error("预测失败", e)


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
        raise _internal_error("监控报表生成失败", e)


# ==================== Q5 路线：信号衰减监控 ====================

@app.get("/api/v1/monitor/ic-trend")
async def get_ic_trend():
    """IC 趋势 / 信号衰减状态（只读，不触发训练）。

    读取 ``reports/ic_trend.json``（由 ``python main.py ic-trend`` 产出）；
    文件缺失时如实返回 ``available=false`` + 生成提示，**不臆测趋势**。
    """
    report_dir = (_config or {}).get("ic_trend", {}).get("report_dir", "reports")
    path = Path(report_dir) / "ic_trend.json"
    if not path.exists():
        return {
            "available": False,
            "reason": "no_ic_trend_report",
            "hint": "先运行 `python main.py ic-trend` 生成 reports/ic_trend.json",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise _internal_error("IC 趋势报告读取失败", e)
    payload["available"] = True
    payload["source"] = str(path)
    return payload


# ==================== S9：按资产类别分池 ====================

@app.get("/api/v1/strategy/pool-gate")
async def get_pool_gate():
    """分池门禁状态（只读，不触发训练）。

    读取 ``reports/stratified_gate.json``（由 ``python main.py ic-pool`` 产出）；
    文件缺失时如实返回 ``available=false`` + 生成提示，**不臆测分池结论**。
    """
    from src.eval.stratified import StratifiedEvaluator

    try:
        return StratifiedEvaluator(_config or {}).load()
    except Exception as e:  # noqa: BLE001
        raise _internal_error("分池门禁报告读取失败", e)


@app.get("/api/v1/strategy/pool-train")
async def get_pool_train():
    """分池训练产物状态（只读）。

    读取 ``models/pools/pool_manifest.json``（由 ``python main.py pool-train`` 产出）。
    """
    from src.train.stratified_train import StratifiedTrainer, summarize_manifest

    try:
        manifest = StratifiedTrainer(_config or {}).load_manifest()
    except Exception as e:  # noqa: BLE001
        raise _internal_error("分池模型清单读取失败", e)
    if manifest.get("available"):
        manifest["rows"] = summarize_manifest(manifest)
    return manifest


@app.get("/api/v1/strategy/horizon-scan")
async def get_horizon_scan():
    """多周期口径探索扫描（只读，不触发训练）。

    读取 ``reports/horizon_scan.json``（由 ``python main.py horizon-scan`` 产出）；
    文件缺失时如实返回 ``available=false`` + 生成提示，**不臆测周期结论**。

    ⚠️ 结果只作口径敏感性证据（``affects_gate=false``），
    不改变现行 ``strategy_gate`` 放行结论。
    """
    from src.eval.horizon_scan import HorizonScanner, compare_with_current

    try:
        scanner = HorizonScanner(_config or {})
        payload = scanner.load()
    except Exception as e:  # noqa: BLE001
        raise _internal_error("多周期扫描报告读取失败", e)
    if payload.get("available"):
        payload["rows"] = scanner.summarize_rows(payload)
        payload["vs_current"] = compare_with_current(payload)
    return payload


@app.get("/api/v1/strategy/horizon-decision")
async def get_horizon_decision():
    """预测周期切换决策单（只读，不触发训练/不重跑扫描）。

    读取 ``reports/horizon_decision.json``（由 ``python main.py horizon-decision`` 产出）；
    文件缺失时如实返回 ``available=false`` + 生成提示，**不臆测结论**。

    返回内容包括：多重比较校正后的逐候选显著性、verdict（approve/reject/defer）、
    status（pending/confirmed/stale）与阻塞项。

    ⚠️ 决策单不改变门禁口径（``affects_gate`` 恒为 ``False``），
    也不代表可直接切换 —— ``approve`` 仍需人工修改配置并重做泄漏/偏差审查。
    """
    try:
        from src.eval import horizon_decision as hd

        record = hd.load(_config or {})
    except Exception as e:  # noqa: BLE001
        raise _internal_error("周期切换决策单读取失败", e)
    if not record:
        return {
            "available": False,
            "reason": "no_horizon_decision",
            "hint": "先运行 `python main.py horizon-scan` 再跑 `python main.py horizon-decision`",
            "affects_gate": False,
        }
    record["available"] = True
    return record


@app.get("/api/v1/eval/feature-experiment")
async def get_feature_experiment():
    """特征扩充正交对照实验结果（只读，不重跑实验）。

    读取 ``reports/feature_experiment.json``（由 ``python main.py feature-experiment`` 产出）；
    文件缺失时如实返回 ``available=false`` + 生成提示，**不臆测结论**。

    ⚠️ 实验结论**不改变生产特征集**（``affects_features`` 恒为 ``False``），
    也不是放行依据：``adopt`` 仍需人工确认并重跑全量门禁。
    """
    try:
        from src.eval.feature_experiment import load

        payload = load(_config or {})
    except Exception as e:  # noqa: BLE001
        raise _internal_error("特征扩充实验报告读取失败", e)
    if not payload:
        return {
            "available": False,
            "reason": "no_feature_experiment",
            "hint": "先运行 `python main.py feature-experiment`",
            "affects_features": False,
        }
    payload["available"] = True
    return payload


@app.get("/api/v1/strategy/asset-class/{symbol}")
async def get_asset_class(symbol: str):
    """单标的资产类别识别（只读、零网络）。

    用于解释「为什么这只标的被分到 ETF 分池」——分池口径必须可解释。
    """
    from src.eval.asset_class import classify, label

    res = classify(symbol)
    res["label"] = label(res["asset_class"])
    return res


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
        raise _internal_error("多因子组合失败", e)


@app.get("/api/v1/factor-model")
async def get_factor_model():
    """可训练多因子模型信息：因子/族权重与 IC 排名（Q2 多因子模型）。

    与 `/api/v1/factors/{symbol}`（推理期特征因子组合）互补：本端点读取
    `python main.py train --model-type factor_model` 训练出的持久化因子模型。
    """
    _init_engine()
    try:
        save_dir = Path((_config or {}).get("training", {}).get("save_dir", "models"))
        path = save_dir / "factor_model_short_term_5d.pkl"
        if not path.exists():
            raise HTTPException(
                404, "多因子模型未训练，请先执行 python main.py train --model-type factor_model"
            )
        from src.factors.factor_model import FactorModel

        model = FactorModel(_config or {})
        model.load(str(path))
        explain = model.explain(10)
        explain.update({
            "available": True,
            "model_path": str(path),
            "families": dict(FACTOR_FAMILIES),
        })
        return explain
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error("多因子模型查询失败", e)


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
        # S19 / H4（T19.3）：信号接口同样追加校准字段（既有结构不变）
        cal = {
            hname: {
                "calibrated_probability": enrich_prediction(p, hname).get("calibrated_probability"),
                "uncertainty": enrich_prediction(p, hname).get("uncertainty"),
            }
            for hname, p in (pred.get("predictions", {}) or {}).items()
            if isinstance(p, dict) and "error" not in p
        }
        return {"symbol": symbol, "signal": sig.to_dict(), "calibration": cal}
    except Exception as e:
        raise _internal_error("信号生成失败", e)


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
        raise _internal_error("交易适配失败", e)


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
            # S19 / H4（T19.3）：**追加**校准字段（calibrated_probability /
            # uncertainty），既有 direction / probability / model 逐字段不变，
            # 保证 28 侧可渐进消费。校准参数缺失时如实标 calibration_applied=false。
            enriched = enrich_prediction(pred, hname)
            horizons_out[hname] = {
                "direction": pred.get("direction", "未知"),
                "probability": pred.get("probability", 0.0),
                "model": f"{model_type}_{hname}_{hdays}d",
                "calibrated_probability": enriched.get("calibrated_probability"),
                "uncertainty": enriched.get("uncertainty"),
                "calibration_applied": enriched.get("calibration_applied", False),
                "calibration_method": enriched.get("calibration_method"),
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


# ==================== Q3 实时流 与 一致性接口 ====================

@app.get("/api/v1/stream/status")
async def get_stream_status():
    """实时流状态（只读）：今日快照覆盖 / 新鲜度 / 是否交易时段。"""
    _init_engine()
    try:
        from src.monitor.health_report import ModelMonitor

        return ModelMonitor(_config)._collect_streaming()
    except Exception as e:
        raise _internal_error("实时流状态获取失败", e)


@app.get("/api/v1/stream/{symbol}")
async def get_stream_signal(symbol: str):
    """单标的盘中信号更新（T-1 baseline vs 盘中快照 → intact/stale）。

    需要 `streaming.enabled: true`；未启用或快照不可用时返回 `available: false`，
    不做任何估算（观测路径 fail-open，不阻断调用方主流程）。
    """
    _init_engine()
    try:
        from src.inference.intraday import IntradayPredictor
        from src.data.collector import DataCollector

        predictor = IntradayPredictor(_config)
        if not predictor.enabled:
            return {"symbol": symbol, "available": False,
                    "reason": "实时流未启用（configs: streaming.enabled=false）"}
        daily_df = DataCollector(_config).load_cached(symbol)
        snapshot = predictor.quote_client.fetch_quotes([symbol]).get(symbol)
        return predictor.build(symbol, daily_df, snapshot).to_dict()
    except Exception as e:
        raise _internal_error("盘中信号生成失败", e)


@app.get("/api/v1/consistency/{symbol}")
async def get_consistency(symbol: str):
    """信号一致性校验：跨周期 / 跨模型 / 跨口径是否自相矛盾（观测项，不参与门禁）。"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")
    try:
        from src.monitor.signal_consistency import SignalConsistencyChecker

        predicted = _engine.predict_all_horizons(symbol)
        probs = [
            float(p.get("probability", 0.5))
            for p in (predicted.get("predictions") or {}).values()
            if isinstance(p, dict) and "error" not in p
        ]
        components = predicted.get("components") or {}
        factor_score = None
        for key, value in components.items():
            if str(key).startswith("factor"):
                factor_score = float(value)
                break
        return SignalConsistencyChecker(_config).check(
            symbol,
            predictions=predicted,
            components=components,
            model_probability=(sum(probs) / len(probs)) if probs else None,
            factor_score=factor_score,
        ).to_dict()
    except Exception as e:
        raise _internal_error("一致性校验失败", e)


# ==================== Q4 智能风控建议接口 ====================

from src.trading.signal import SignalEngine  # noqa: E402  (Q4 风控建议用信号聚合)


@app.get("/api/v1/risk/advice/{symbol}")
async def get_risk_advice(symbol: str):
    """单标的智能风控建议（止损/止盈）。

    契约要点：
      - **建议，不是下单指令**（`not_trade_instruction: true`）；
      - 门禁未放行（`readonly`/`unknown`）时返回 `status=withheld` 且价位为 `null`，
        fail-close —— 未过门禁的信号不配"可直接使用的止损价"；
      - 数据不足返回 `status=unavailable` 与原因，绝不用默认比例兜底。
    """
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")
    try:
        from src.data.collector import DataCollector
        from src.trading.risk import RiskManager
        from src.trading.risk_advice import RiskAdvisor

        payload = _engine.predict_all_horizons(symbol)
        advisor = RiskAdvisor(_config)
        df = DataCollector(_config).load_cached(symbol)
        plan = advisor.advise_payload(symbol, payload, df=df)
        if plan.action != "HOLD" and plan.available:
            sig = SignalEngine(_config).build_signal(symbol, payload)
            price = next((b.get("latest_close") for b in (payload.get("predictions") or {}).values()
                          if isinstance(b, dict) and b.get("latest_close")), None)
            budget = RiskManager(_config).budget(sig, price=price)
            plan.position_pct = budget.position_pct
            plan.position_amount = budget.position_amount
            plan.risk_amount = budget.risk_per_trade
            plan.suggested_qty = budget.suggested_qty
        return plan.to_dict()
    except Exception as e:
        raise _internal_error("风控建议生成失败", e)


@app.get("/api/v1/risk/advice")
async def get_risk_advice_batch(
    symbols: str = Query(None, description="逗号分隔的标的列表，缺省用 config 启用的 markets 标的"),
):
    """标的池批量风控建议（只读；门禁未放行时统一 withheld）。"""
    _init_engine()
    if not _engine or not _engine.models:
        raise HTTPException(503, "模型未加载，请先训练模型")
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        symbol_list = []
        for name, cfg in (_config.get("data", {}).get("markets", {}) or {}).items():
            if cfg.get("enabled"):
                symbol_list.extend(cfg.get("symbols", []))
    symbol_list = list(dict.fromkeys(symbol_list))
    try:
        from src.data.collector import DataCollector
        from src.trading.risk_advice import RiskAdvisor

        advisor = RiskAdvisor(_config)
        collector = DataCollector(_config)
        items = []
        for sym in symbol_list:
            try:
                payload = _engine.predict_all_horizons(sym)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[risk/advice] {sym} 预测失败: {e}")
                payload = {}
            items.append({
                "symbol": sym,
                "signal": SignalEngine(_config).build_signal(sym, payload),
                "df": collector.load_cached(sym),
                "price": next((b.get("latest_close")
                               for b in (payload.get("predictions") or {}).values()
                               if isinstance(b, dict) and b.get("latest_close")), None),
            })
        return advisor.advise_portfolio(items)
    except Exception as e:
        raise _internal_error("风控建议批量生成失败", e)


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
