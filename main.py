"""
金融市场预测模型
================
基于机器学习和时间序列分析，预测股市、期货、外汇等金融市场的短期和长期走势。

功能：
  - 多市场数据采集（A股 / 期货 / 外汇）
  - 特征工程（技术指标 / 收益率 / 波动率 / 成交量）
  - 多模型训练（LightGBM / PyTorch LSTM / 集成）
  - 多周期预测（短期5日 / 中期10日 / 长期20日）
  - 模型评估（ML指标 + 金融专用指标）
  - 模型导出（ONNX / TorchScript）
  - 在线推理预测

用法:
  python main.py train          # 完整训练流水线
  python main.py evaluate       # 评估已训练模型
  python main.py predict 600519.SH  # 预测指定标的
  python main.py export         # 导出模型
  python main.py all            # 执行全部流程

作者: yuppiez99999
日期: 2026-06-19
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

import pandas as pd  # noqa: E402（qlib-ab 序列对齐用）

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("financial_prediction")


def load_config(config_path: str | None = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """配置日志"""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log_file = log_cfg.get("file")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def run_train(config: dict) -> dict:
    """执行训练流水线"""
    from src.train.trainer import ModelTrainer

    logger.info("=" * 60)
    logger.info("启动训练流水线")
    logger.info("=" * 60)

    trainer = ModelTrainer(config)
    results = trainer.train_pipeline()

    # 保存评估报告
    from src.eval.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(config)
    evaluator.results = {
        k: v["evaluation"] for k, v in results.items()
        if "evaluation" in v
    }
    report = evaluator.generate_report("logs/evaluation_report.txt")
    print("\n" + report)

    return results


def run_predict(config: dict, symbol: str, horizon: str = "short_term") -> dict:
    """执行预测"""
    from src.inference.predictor import PredictionEngine

    logger.info(f"预测 {symbol} [{horizon}]")

    engine = PredictionEngine(config)
    model_type = config["model"]["type"]
    engine.load_models(model_type)

    if horizon == "all":
        result = engine.predict_all_horizons(symbol)
    else:
        result = engine.predict(symbol, horizon)

    print("\n预测结果:")
    print("-" * 40)
    if "predictions" in result:
        for h, pred in result["predictions"].items():
            if "error" in pred:
                print(f"  {h}: 错误 - {pred['error']}")
            else:
                print(f"  {h} ({pred['horizon_days']}日): {pred['direction']} "
                      f"(概率={pred['probability']:.2%}, 置信度={pred['confidence']:.2%})")
    else:
        if "error" in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  标的: {result['symbol']}")
            print(f"  周期: {result['horizon']} ({result['horizon_days']}日)")
            print(f"  方向: {result['direction']}")
            print(f"  概率: {result['probability']:.2%}")
            print(f"  置信度: {result['confidence']:.2%}")

    return result


def run_export(config: dict) -> list:
    """导出模型"""
    from src.export.exporter import ModelExporter

    logger.info("导出模型...")
    exporter = ModelExporter(config)
    model_type = config["model"]["type"]
    exported = exporter.export_all(model_type)

    print(f"\n导出完成: {len(exported)} 个模型")
    for path in exported:
        print(f"  - {path}")

    return exported


def run_all(config: dict) -> None:
    """执行全部流程: 训练 → 评估 → 导出"""
    logger.info("执行完整流程: 训练 → 评估 → 导出")

    # 训练 + 评估
    results = run_train(config)

    # 导出
    run_export(config)

    logger.info("全部流程完成!")


# ==================== 专业版功能 ====================

def run_batch_predict(config: dict, symbols: list[str], horizon: str = "all") -> list:
    """批量预测多个标的（专业版支持 50+ 标的）"""
    from src.inference.predictor import PredictionEngine
    from src.audit.prediction_audit import PredictionAudit

    logger.info(f"批量预测 {len(symbols)} 个标的 [{horizon}]")

    engine = PredictionEngine(config)
    engine.load_models(config["model"]["type"])
    audit = PredictionAudit()

    results = engine.batch_predict(symbols)

    # 审计记录
    for r in results:
        if "predictions" in r:
            for h, pred in r["predictions"].items():
                if "error" not in pred:
                    audit.record_prediction(pred)

    print(f"\n批量预测完成: {len(results)} 个标的")
    for r in results:
        symbol = r.get("symbol", "?")
        if "predictions" in r:
            for h, pred in r["predictions"].items():
                if "error" in pred:
                    print(f"  {symbol} [{h}]: 错误 - {pred['error']}")
                else:
                    print(f"  {symbol} [{h}]: {pred['direction']} (概率={pred['probability']:.2%})")
        elif "error" in r:
            print(f"  {symbol}: 错误 - {r['error']}")

    return results


def run_serve(config: dict, host: str | None = None, port: int | None = None) -> None:
    """启动 Web API 服务"""
    from src.api.server import run_server

    api_cfg = config.get("api", {})
    h = host or api_cfg.get("host", "0.0.0.0")
    p = port or api_cfg.get("port", 8800)
    logger.info(f"启动 TrendCast Pro API 服务: http://{h}:{p}")
    run_server(h, p)


def run_schedule(config: dict) -> None:
    """启动自动重训练调度器"""
    from src.scheduler.retrain_scheduler import RetrainScheduler

    scheduler = RetrainScheduler(config)
    scheduler.start()

    logger.info("调度器已在后台运行，按 Ctrl+C 退出")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()
        logger.info("已退出")


def run_audit(config: dict) -> str:
    """生成预测审计报告"""
    from src.audit.prediction_audit import PredictionAudit
    from src.data.collector import DataCollector

    audit = PredictionAudit()
    collector = DataCollector(config)

    # 回溯验证
    verified = audit.verify_predictions(
        lambda symbol, start, end: collector.load_cached(symbol)
    )
    report = audit.generate_report()
    print(f"\n审计验证: 新验证 {verified} 条预测")
    print("=" * 60)
    print(report)
    return report


def run_notify(config: dict, symbol: str, horizon: str = "all") -> dict:
    """预测并推送信号"""
    from src.inference.predictor import PredictionEngine
    from src.notification.notifier import SignalNotifier
    from src.audit.prediction_audit import PredictionAudit

    logger.info(f"预测 {symbol} 并推送信号")

    engine = PredictionEngine(config)
    engine.load_models(config["model"]["type"])

    predictions = []
    if horizon == "all":
        result = engine.predict_all_horizons(symbol)
    else:
        result = engine.predict(symbol, horizon)
        result = {"symbol": symbol, "predictions": {horizon: result}}
    predictions.append(result)

    # 审计记录
    audit = PredictionAudit()
    for r in predictions:
        if "predictions" in r:
            for h, pred in r["predictions"].items():
                if "error" not in pred:
                    audit.record_prediction(pred)

    # 推送
    notifier = SignalNotifier(config)
    result = notifier.notify(predictions)
    print(f"\n信号推送结果: {result}")
    if all(v == False for v in result.values()):
        print("提示: 未配置 Webhook/邮件，请在 config_pro.yaml 的 notification 段填写配置")
    return result


def run_daily_report(config: dict) -> str:
    """生成每日预测报告"""
    from src.inference.predictor import PredictionEngine
    from src.report.daily_report import DailyReportGenerator

    logger.info("生成每日预测报告")

    engine = PredictionEngine(config)
    engine.load_models(config["model"]["type"])

    symbols = config["data"]["markets"]["stock"]["symbols"]
    predictions = engine.batch_predict(symbols)

    generator = DailyReportGenerator(config)
    report = generator.generate_report(predictions)

    print("\n" + "=" * 60)
    print("每日预测报告已生成")
    print("=" * 60)
    print(report[:3000] + "..." if len(report) > 3000 else report)
    return report


def run_weekly_report(config: dict) -> str:
    """生成周度预测报告"""
    from src.inference.predictor import PredictionEngine
    from src.report.daily_report import WeeklyReportGenerator

    logger.info("生成周度预测报告")

    engine = PredictionEngine(config)
    engine.load_models(config["model"]["type"])

    symbols = config["data"]["markets"]["stock"]["symbols"]
    predictions = engine.batch_predict(symbols)

    generator = WeeklyReportGenerator(config)
    report = generator.generate_report(predictions)

    print("\n" + "=" * 60)
    print("周度预测报告已生成")
    print("=" * 60)
    print(report[:3000] + "..." if len(report) > 3000 else report)
    return report


def run_adaptive(config: dict) -> str:
    """运行自适应学习引擎"""
    from src.train.adaptive_learner import AdaptiveLearningEngine

    logger.info("运行自适应学习引擎")

    engine = AdaptiveLearningEngine(config)
    report = engine.get_adaptive_report()

    print("\n" + report)
    return report


def run_monitor(config: dict, output: str | None = None) -> dict:
    """生成模型监控报表（审计命中率 / 自适应漂移 / 数据源 / 模型产物）"""
    from src.monitor.health_report import ModelMonitor

    logger.info("生成模型监控报表")
    report = ModelMonitor(config).collect()
    print("\n" + report.to_markdown())

    path = output or (Path(config.get("report", {}).get("output_dir", "reports")) / "monitor_report.md")
    saved = report.save(path)
    print(f"\n监控报表已保存: {saved}")
    return report.to_dict()


def run_macro(config: dict) -> dict:
    """查看/刷新宏观指标（CPI/PMI/GDP/M2/LPR）数据可用性"""
    from src.data.macro_client import MacroClient

    logger.info("检查宏观指标数据源")
    client = MacroClient(config)
    health = client.health()

    print("\n" + "=" * 60)
    print("宏观指标数据源状态")
    print("=" * 60)
    for indicator, info in health.items():
        mark = "✅" if info["status"] == "ok" else "⚠️"
        latest = f"{info['latest_value']:.2f} @ {info['latest_date']}" if info["points"] else "无数据"
        print(f"  {mark} {indicator:<6} {info['name']:<26} {info['points']:>4} 期  {latest}")

    ok = sum(1 for v in health.values() if v["status"] == "ok")
    print(f"\n可用指标: {ok}/{len(health)}")
    if ok < len(health):
        print("提示: 安装 akshare（pip install akshare）可启用免费宏观数据源；")
        print("      或把历史数据放入 data/macro/macro_<indicator>.csv（列: date,value）")

    if config.get("features", {}).get("macro_enabled"):
        feats = client.get_features()
        print("\n当前宏观特征（供模型特征注入）:")
        for k, v in sorted(feats.items()):
            print(f"  {k} = {v:.4f}")
    return health


# ==================== 量化交易适配层 ====================

def run_signal(config: dict, symbol: str) -> dict:
    """对单个标的输出综合交易信号（仅信号，不下单）"""
    from src.inference.predictor import PredictionEngine
    from src.trading.signal import SignalEngine

    logger.info(f"生成交易信号 {symbol}")
    engine = PredictionEngine(config)
    engine.load_models(config["model"].get("type", "lightgbm"))
    pred = engine.predict_all_horizons(symbol)
    sig = SignalEngine(config).build_signal(symbol, pred)
    payload = {"symbol": symbol, "signal": sig.to_dict()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def run_orders(config: dict, symbols: list[str]) -> list:
    """输出下单明细（预测 → 信号 → 风控 → 订单，不落盘）"""
    from src.trading.adapter import TradingAdapter

    logger.info(f"生成订单明细: {len(symbols)} 个标的")
    results = TradingAdapter(config).run(symbols)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def run_trade(config: dict, symbols: list[str]) -> list:
    """执行完整交易适配流程并导出数据流（JSON + CSV）"""
    from src.trading.adapter import TradingAdapter
    from datetime import datetime

    logger.info(f"执行交易适配流程: {len(symbols)} 个标的")
    out_dir = Path(config.get("trading", {}).get("output_dir", "output/trading"))
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = TradingAdapter(config)
    results = adapter.run(symbols)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = adapter.export_feed(results, out_dir / f"trading_feed_{ts}.json")
    csv_path = adapter.export_csv(results, out_dir / f"trading_feed_{ts}.csv")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n已导出数据流:\n  {json_path}\n  {csv_path}")
    return results


def run_backtest(config: dict, symbol: str) -> None:
    """对指定标的做信号假设成交回测（基于历史行情与最近信号）"""
    from src.data.collector import DataCollector
    from src.trading.adapter import TradingAdapter
    from src.trading.backtest import Backtester

    logger.info(f"回测 {symbol}")
    collector = DataCollector(config)
    df = collector.load_cached(symbol) or collector._fetch_with_fallback(symbol)
    if df is None or len(df) < 30:
        print(f"数据不足，无法回测 {symbol}")
        return

    horizon_days = config.get("data", {}).get("prediction_horizons", {}).get("short_term", 5)
    adapter = TradingAdapter(config)
    bt = Backtester(config)
    closes = df["close"].tolist()
    signals = [{"action": "HOLD", "horizon": 1} for _ in closes]
    # 每 horizon_days 根用一次最近预测触发方向判断，模拟低频信号（接口演示）
    demo_pred = {"predictions": {
        "short_term": {"prediction": 1, "probability": 0.6},
        "mid_term": {"prediction": 1, "probability": 0.55},
        "long_term": {"prediction": 0, "probability": 0.5},
    }}
    for i in range(0, len(closes) - horizon_days, horizon_days):
        sig = adapter.signal_engine.build_signal(symbol, demo_pred)
        signals[i] = {"action": sig.action, "horizon": horizon_days}

    result = bt.run(closes, signals)
    print("回测结果:")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


# ==================== Q2 路线：门禁与多因子 ====================

def _ic_scores_for_horizon(config: dict, combined, horizon_days: int, folds: int,
                           derive_band: bool) -> tuple[list, list, float]:
    """对给定监督数据集做 walk-forward 训练，产出 (scores, returns, neutral_band)。

    - ``scores`` 用「概率 - 0.5」去中性化（0 = 无观点），与因子/信号口径一致；
    - ``derive_band=True`` 时，逐折**只用训练折**推导中性带，再施加到该折测试样本；
      训练折学到的常数用在测试折上不构成前视（见 ``derive_neutral_band``）；
    - 未 ``derive_band`` 时 neutral_band 返回 0.0，逐字节保持既有门禁口径。
    """
    import numpy as np
    import scripts.evaluate_models as ev
    from src.inference.ic import derive_neutral_band

    from src.data.preprocessor import FeatureEngineer

    fe = FeatureEngineer(config)
    cols = [c for c in fe.get_feature_columns(combined, int(horizon_days))
            if not str(c).startswith("_")]
    X = combined[cols].to_numpy(dtype=float)
    y = combined[f"target_{int(horizon_days)}d"].to_numpy(dtype=float)
    fwd = combined["_fwd_ret"].to_numpy(dtype=float)

    scores: list = []
    returns: list = []
    bands: list = []
    for train_idx, test_idx in ev.walk_forward_splits(len(X), folds):
        if len(np.unique(y[train_idx])) < 2:
            continue
        try:
            from src.train.models.lightgbm_model import LightGBMModel

            model = LightGBMModel(config)
            model.train(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            train_proba = model.predict_proba(X[train_idx])[:, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ic] horizon={horizon_days}d 折训练失败: {e}")
            continue
        fold_scores = [float(p) - 0.5 for p in proba]
        fold_returns = [float(r) for r in fwd[test_idx]]
        if derive_band:
            band_cfg = (config.get("strategy_gate", {}) or {}).get("neutral_band", {}) or {}
            band = derive_neutral_band(
                [float(p) - 0.5 for p in train_proba],
                [float(r) for r in fwd[train_idx]],
                min_keep_ratio=float(band_cfg.get("min_keep_ratio", 0.5)),
            )
            bands.append(band)
        scores.extend(fold_scores)
        returns.extend(fold_returns)
    # 多折取中位数：单折异常不污染整体口径（且仍全部来自训练折）
    band_out = float(np.median(bands)) if bands else 0.0
    return scores, returns, band_out


def run_ic(config: dict, symbols: list[str] | None = None, folds: int = 3,
           stratify: bool = False, derive_band: bool | None = None,
           detail: bool = False) -> dict:
    """IC / 命中率门禁评估（walk-forward 时序回测，口径与 scripts/evaluate_models.py 一致）。

    复用评估脚本的 ``build_supervised`` / ``walk_forward_splits`` / ``compute_metrics``，
    保证「门禁判定」与「评估报告」同源同口径，避免两套数字互相打架：
      - 逐标的特征工程 + 逐标的构造目标（防跨标的 shift 污染）；
      - 测试折严格在训练折之后（无未来函数）；
      - forward return 取真实收盘价，未到期样本自动跳过。

    S7 门禁解锁攻坚新增（均默认关闭，向后兼容）：
      - ``stratify=True``    ：额外按标的分解结果，定位拖后腿的标的；
      - ``derive_band``      ：逐折用训练折推导中性带，过滤 ≈0.5 的噪音样本。
        ``True``/``False`` = 显式开关；``None`` = 跟随
        ``strategy_gate.neutral_band.enabled``（缺省 false）。

    S11（G1）新增：
      - ``detail=True``      ：追加统一评估量尺报告（src/eval/factor_metrics.py）：
        IC 置信区间 / 分位数分层收益 / 信号换手率 / 多周期衰减曲线 /
        成交成本敏感性三档扫描。全部 report_only，不影响门禁判定，
        落盘 ``reports/factor_metrics.json``。
    """
    import scripts.evaluate_models as ev
    from src.inference.ic import ICCalculator

    logger.info("执行 IC / 命中率门禁评估")
    band_cfg = (config.get("strategy_gate", {}) or {}).get("neutral_band", {}) or {}
    if derive_band is None:
        derive_band = bool(band_cfg.get("enabled", False))
    horizons_cfg = config.get("data", {}).get("prediction_horizons", {})
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据（请先准备 data/raw/<symbol>.csv 或联网采集）",
                   "horizons": {}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    calc = ICCalculator(config)
    per_horizon: dict = {}
    # 分标的：{symbol: {hname: {scores, returns}}}
    per_symbol: dict = {} if stratify else None

    for hname, days in horizons_cfg.items():
        combined = ev.build_supervised(data, config, int(days))
        if combined.empty:
            per_horizon[hname] = {"horizon_days": int(days), "scores": [], "returns": [],
                                  "window_size": 0}
            continue
        scores, returns, band = _ic_scores_for_horizon(
            config, combined, int(days), folds, derive_band)
        per_horizon[hname] = {
            "horizon_days": int(days),
            "scores": scores,
            "returns": returns,
            # 门禁需 IC 稳定性：按 20 样本滚动窗口统计 ICIR
            "window_size": 20 if len(scores) >= 60 else None,
            "neutral_band": band,
        }
        if stratify:
            _fill_per_symbol(per_symbol, config, data, hname, int(days), folds, derive_band)

    result = calc.evaluate_all(per_horizon)
    result["symbols_evaluated"] = len(data)
    result["folds"] = folds
    result["neutral_band_enabled"] = bool(derive_band)
    # S13 试验登记：每次门禁评估都是一次"比较"，登记后校正才有真实的分母
    _record_trial(config, "ic", {
        "symbols": len(data), "folds": folds,
        "passed": bool(result.get("all_passed")),
        "horizons": {k: v.get("passed") for k, v in (result.get("horizons") or {}).items()},
    })
    if stratify:
        result["per_symbol"] = _summarize_per_symbol(per_symbol or {})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if detail:
        result["factor_metrics"] = _run_factor_metrics(
            config, data, horizons_cfg, per_horizon)
    return result


def _run_factor_metrics(config: dict, data: dict, horizons_cfg: dict,
                        per_horizon: dict) -> dict:
    """S11（G1）统一评估量尺：对同一批 walk-forward 序列出因子健康度报告。

    与 `ic` 门禁**同一条序列**（per_horizon 里已含 scores/returns），
    额外由收盘价构造多周期前视收益做衰减曲线。report_only，不改门禁。
    """
    import scripts.evaluate_models as ev
    from src.eval.factor_metrics import FactorMetricsCalculator
    from src.inference.ic import forward_returns

    calculator = FactorMetricsCalculator(config)
    sequences: dict = {}
    # 整池收盘价拼接：按时间拼接去重，仅用于构造前视收益（衰减曲线）
    for hname, payload in per_horizon.items():
        returns_by_days: dict = {}
        try:
            combined = ev.build_supervised(data, config, int(payload["horizon_days"]))
        except Exception as e:  # noqa: BLE001 - 诊断失败不得影响门禁输出
            logger.warning(f"[factor-metrics] {hname} 数据集构造失败: {e}")
            combined = None
        if combined is not None and not combined.empty and "close" in combined.columns:
            closes = combined["close"].astype(float).tolist()
            for cand_days in sorted({int(d) for d in horizons_cfg.values()}):
                returns_by_days[cand_days] = forward_returns(closes, cand_days)
        sequences[hname] = {
            "horizon_days": int(payload["horizon_days"]),
            "scores": payload.get("scores", []),
            "returns": payload.get("returns", []),
            "returns_by_days": returns_by_days or None,
        }
    result = calculator.evaluate_all(sequences)
    try:
        saved = calculator.save(result)
        print(f"\n因子健康度报告已保存: {saved}")
        result["report_path"] = str(saved)
    except Exception as e:  # noqa: BLE001 - 落盘失败不影响诊断结果返回
        logger.warning(f"[factor-metrics] 报告落盘失败: {e}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _fill_per_symbol(per_symbol: dict, config: dict, data: dict,
                     hname: str, days: int, folds: int, derive_band: bool) -> None:
    """逐标的独立跑一遍 walk-forward，把结果挂到 per_symbol[symbol][hname]。"""
    import scripts.evaluate_models as ev

    for symbol in data:
        one = ev.build_supervised({symbol: data[symbol]}, config, days)
        if one.empty:
            continue
        scores, returns, band = _ic_scores_for_horizon(
            config, one, days, folds, derive_band)
        if not scores:
            continue
        per_symbol.setdefault(symbol, {})[hname] = {
            "horizon_days": days,
            "scores": scores,
            "returns": returns,
            "neutral_band": band,
        }


def _summarize_per_symbol(per_symbol: dict) -> dict:
    """把逐标的原始序列压成精简指标（避免把大数组塞进报告 JSON）。"""
    from src.inference.ic import ICCalculator, hit_rate

    out: dict = {}
    for symbol, horizons in per_symbol.items():
        entry: dict = {}
        calc = ICCalculator()
        for hname, payload in horizons.items():
            res = calc.evaluate(hname, int(payload["horizon_days"]),
                                payload["scores"], payload["returns"],
                                neutral_band=payload.get("neutral_band", 0.0))
            entry[hname] = {
                "samples": res.samples,
                "ic": round(res.ic, 4),
                "hit_rate": round(res.hit_rate, 4),
                "hit_rate_raw": round(res.hit_rate_raw or hit_rate(
                    payload["scores"], payload["returns"]), 4),
                "neutral_band": round(res.neutral_band, 6),
                "available": res.available,
                "passed": res.passed,
            }
        if entry:
            out[symbol] = entry
    return out


def build_walkforward_sequences(
    data: dict, config: dict, horizon_days: int, folds: int = 3
) -> dict:
    """构造 walk-forward 序列（供 `ic` 门禁与 `ic-trend` 衰减监控共用）。

    两个命令的口径必须**完全同源**，否则会出现「监控说衰减、门禁说达标」的错位。
    返回 {scores, returns, folds_used, dates}：
      - scores  ：去中性化分数（概率 - 0.5，0 = 无观点），与因子/信号口径一致；
      - returns ：对应的真实未来收益（含 `_fwd_ret`，未到期样本已被剔除）；
      - dates   ：每个样本对应的交易日（透传给趋势监控做时间轴标注）。
    """
    import numpy as np
    import scripts.evaluate_models as ev
    from src.data.preprocessor import FeatureEngineer
    from src.train.models.lightgbm_model import LightGBMModel

    scores: list = []
    returns: list = []
    dates: list = []
    folds_used = 0

    combined = ev.build_supervised(data, config, int(horizon_days))
    if combined.empty:
        return {"scores": [], "returns": [], "dates": [], "folds_used": 0}

    fe = FeatureEngineer(config)
    cols = [c for c in fe.get_feature_columns(combined, int(horizon_days))
            if not str(c).startswith("_")]
    X = combined[cols].to_numpy(dtype=float)
    y = combined[f"target_{int(horizon_days)}d"].to_numpy(dtype=float)
    fwd = combined["_fwd_ret"].to_numpy(dtype=float)
    if "date" in combined.columns:
        dates_all = combined["date"].astype(str).tolist()
    else:  # pragma: no cover - 数据管道异常时的兜底
        dates_all = ["" for _ in range(len(combined))]

    splits = ev.walk_forward_splits(len(X), folds)
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2:
            continue
        try:
            model = LightGBMModel(config)
            model.train(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ic] {horizon_days}d 折训练失败: {e}")
            continue
        scores.extend([float(p) - 0.5 for p in proba])
        returns.extend([float(r) for r in fwd[test_idx]])
        dates.extend(dates_all[i] for i in test_idx)
        folds_used += 1

    return {"scores": scores, "returns": returns, "dates": dates, "folds_used": folds_used}


def _config_symbols(config: dict) -> list[str]:
    """取配置中启用的标的（去重保序）。"""
    out: list[str] = []
    for _name, cfg in (config.get("data", {}).get("markets", {}) or {}).items():
        if isinstance(cfg, dict) and cfg.get("enabled"):
            out.extend(cfg.get("symbols", []))
    return list(dict.fromkeys(out))


def _record_trial(config: dict, command: str, summary_payload: dict) -> None:
    """登记一次评估试验（S13，fail-soft）。

    登记失败**不得**影响评估本身：告警后继续。登记的价值在于让
    「试了多少次」在多次会话之间保持可见 —— 这正是多重比较校正拿不到的分母。
    """
    try:
        from src.eval.trial_registry import record

        entry = record(command, config, summary=summary_payload)
        if not entry.get("_persisted", True):
            logger.warning("[trial] 试验登记未落盘（评估结果不受影响）")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[trial] 试验登记失败（评估结果不受影响）: {e}")


def _cli_symbols(args) -> list[str] | None:
    """解析 CLI 的标的来源：位置参数优先，其次 `--symbols a,b`（逗号分隔）。

    返回 None = 未指定，由调用方回落到配置启用的标的池。
    """
    positional = [a for a in (getattr(args, "args", None) or []) if a]
    if positional:
        return positional
    raw = getattr(args, "symbols", None)
    if raw:
        return [s.strip() for s in str(raw).split(",") if s.strip()]
    return None


def run_gate(config: dict, ic_path: str | None = None) -> dict:
    """策略门禁判定：IC 门禁（+ 可选审计命中率）→ readonly / gated。"""
    from src.trading.gate import StrategyGate, recent_audit_stats

    logger.info("执行策略门禁判定")
    ic_payload = None
    if ic_path and Path(ic_path).exists():
        ic_payload = json.loads(Path(ic_path).read_text(encoding="utf-8"))
    else:
        ic_payload = run_ic(config)

    audit_stats = None
    if (config.get("strategy_gate", {}) or {}).get("use_audit"):
        audit_dir = (config.get("audit", {}) or {}).get("dir", "logs/audit")
        audit_stats = recent_audit_stats(audit_dir)

    decision = StrategyGate(config).decide(ic_payload, audit_stats).to_dict()
    out = Path(config.get("strategy_gate", {}).get("report_dir", "reports")) / "strategy_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"\n门禁判定已保存: {out}")
    return decision


def run_gate_diagnose(config: dict, ic_path: str | None = None) -> dict:
    """门禁阻塞诊断：把 `readonly` 拆成「还差多少 / 哪条腿卡住 / 该不该动」。

    纯读既有产物（IC 评估 + 门禁判定），不训练、不触网、不改门禁结论。
    """
    from src.inference.gate_diagnosis import diagnose

    logger.info("执行门禁阻塞诊断")
    ic_payload = None
    if ic_path and Path(ic_path).exists():
        ic_payload = json.loads(Path(ic_path).read_text(encoding="utf-8"))
    else:
        # 不重跑评估（可能很贵）：优先复用落盘产物
        cached = Path(config.get("strategy_gate", {}).get("report_dir", "reports")) / "ic_report.json"
        if cached.exists():
            ic_payload = json.loads(cached.read_text(encoding="utf-8"))
            logger.info(f"复用 IC 评估产物: {cached}")
        else:
            ic_payload = run_ic(config)

    gate_decision = None
    gate_cached = Path(config.get("strategy_gate", {}).get("report_dir", "reports")) / "strategy_gate.json"
    if gate_cached.exists():
        try:
            gate_decision = json.loads(gate_cached.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gate-diagnose] 门禁判定读取失败: {e}")

    result = diagnose(ic_payload, config.get("strategy_gate", {}), gate_decision)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_pool_ic(config: dict, symbols: list[str] | None = None, folds: int = 3,
                derive_band: bool | None = None) -> dict:
    """按资产类别分池门禁评估（S9）：给每个分池独立出一份门禁判定。

    与 `ic`（整池口径）共用同一套 walk-forward 序列构造，保证同源；
    差别只在于「哪些标的进同一条 IC 序列」——这正是 S7 实测指出的症结：
    个股与宽基 ETF 混在一条序列里，方向性互相抵消，木桶短板把整池拖死在门槛线上。

    落盘 `reports/stratified_gate.json`，并打印与整池口径的对照结论。
    ⚠️ 默认**不改变** `strategy_gate` 的放行结论（`pool_gate` 为 report_only）。
    """
    import scripts.evaluate_models as ev
    from src.eval.stratified import (
        StratifiedEvaluator,
        build_sequences_for_pools,
        compare_with_pooled,
    )

    logger.info("执行按资产类别分池门禁评估")
    band_cfg = (config.get("strategy_gate", {}) or {}).get("neutral_band", {}) or {}
    if derive_band is None:
        derive_band = bool(band_cfg.get("enabled", False))
    horizons_cfg = config.get("data", {}).get("prediction_horizons", {}) or {}
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据（请先准备 data/raw/<symbol>.csv 或联网采集）",
                   "pools": {}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    evaluator = StratifiedEvaluator(config)
    splits = evaluator.split(data)

    def _build_pool_sequences(pool_data: dict) -> dict:
        """单池 → {horizon: {scores, returns}}（与整池门禁完全同源）。"""
        out: dict = {}
        for hname, days in horizons_cfg.items():
            combined = ev.build_supervised(pool_data, config, int(days))
            if combined.empty:
                out[hname] = {"horizon_days": int(days), "scores": [], "returns": [],
                              "window_size": 0}
                continue
            scores, returns, band = _ic_scores_for_horizon(
                config, combined, int(days), folds, derive_band)
            out[hname] = {
                "horizon_days": int(days),
                "scores": scores,
                "returns": returns,
                "window_size": 20 if len(scores) >= 60 else None,
                "neutral_band": band,
            }
        return out

    sequences = build_sequences_for_pools(data, splits, _build_pool_sequences)
    result = evaluator.decide(data, sequences)
    result["folds"] = folds
    result["symbols_evaluated"] = len(data)
    result["neutral_band_enabled"] = bool(derive_band)

    # 与整池口径对照：回答「是不是被木桶短板拖死的」
    pooled = None
    pooled_path = Path((config.get("strategy_gate", {}) or {}).get("report_dir", "reports")) / "ic_report.json"
    if pooled_path.exists():
        try:
            pooled = json.loads(pooled_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[pool-ic] 整池 IC 产物读取失败: {e}")
    result["vs_pooled"] = compare_with_pooled(result, pooled)

    saved = evaluator.save(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n分池门禁报告已保存: {saved}")
    if result["vs_pooled"].get("narrative"):
        print(f"对照结论: {result['vs_pooled']['narrative']}")
    return result


def run_pool_train(config: dict, symbols: list[str] | None = None) -> dict:
    """按资产类别分层训练（S9）：每个分池训一套独立模型。

    产出 `models/pools/pool_<class>_<model_type>_<horizon>_<days>d.pkl`
    + `models/pools/pool_manifest.json`（推理侧据此路由，绝不跨池串用）。
    """
    import scripts.evaluate_models as ev
    from src.train.stratified_train import StratifiedTrainer, summarize_manifest

    logger.info("执行按资产类别分层训练")
    horizons_cfg = config.get("data", {}).get("prediction_horizons", {}) or {}
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据", "pools": {}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    trainer = StratifiedTrainer(config)
    manifest = trainer.train_pools(data, horizons_cfg)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    rows = summarize_manifest(manifest)
    if rows:
        print("\n分池训练结果:")
        for r in rows:
            print(f"  {r['label']:<14} status={r['status']:<8} "
                  f"标的={r['symbol_count']} 产物={r['files']} {r['reason']}")
    if manifest.get("manifest_path"):
        print(f"\n分池模型清单已保存: {manifest['manifest_path']}")
    return manifest


def run_ic_trend(config: dict, symbols: list[str] | None = None, folds: int = 3) -> dict:
    """IC 趋势 / 信号衰减评估（Q5）：把门禁的静态 IC 快照变成 IC 时序。

    与 `ic`（门禁口径）共用 `build_walkforward_sequences`，保证同源：
    同一个 (分数, 未来收益) 序列，`ic` 回答「现在达不达标」，
    `ic-trend` 回答「在变好还是变坏、还有多久跌破放行线」。

    落盘 `reports/ic_trend.json`（监控报表与日报只读消费）。
    """
    import scripts.evaluate_models as ev
    from src.monitor.ic_trend import ICTrendMonitor

    logger.info("执行 IC 趋势 / 信号衰减评估")
    horizons_cfg = config.get("data", {}).get("prediction_horizons", {})
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据（请先准备 data/raw/<symbol>.csv 或联网采集）",
                   "horizons": {}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    monitor = ICTrendMonitor(config)
    per_horizon: dict = {}
    for hname, days in horizons_cfg.items():
        seq = build_walkforward_sequences(data, config, int(days), folds)
        per_horizon[hname] = {
            "horizon_days": int(days),
            "scores": seq["scores"],
            "returns": seq["returns"],
        }

    result = monitor.evaluate_aligned(per_horizon)
    result["symbols_evaluated"] = len(data)
    result["folds"] = folds

    report_dir = (config.get("ic_trend", {}) or {}).get("report_dir", "reports")
    out = Path(report_dir) / "ic_trend.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nIC 趋势报告已保存: {out}")
    return result


def run_horizon_scan(config: dict, symbols: list[str] | None = None, folds: int = 3,
                     candidates: list[int] | None = None, include_pools: bool = True) -> dict:
    """多周期口径探索（S10）：把「门禁卡在周期选择上」变成可复算的证据。

    S9 实测发现：现行口径 5/10/20 日中长期卡线，而 40/60 日整池与分池全部达标
    —— 信号真实存在，只是 5/10/20 日这个尺度上模型没有优势。
    S9 把这条结论留作「产品口径变更，须人工决策」，本命令提供**该决策所需的证据**：

      - 对候选周期各自独立跑同一套 walk-forward 口径（与 `ic` 门禁同源）；
      - 输出每周期 / 每分池的 IC、命中率、样本、达标情况；
      - 给出「换周期有没有用」的对照叙述。

    ⚠️ **不改变现行门禁口径**，`affects_gate` 恒为 False：
    `data.prediction_horizons` 一个字不动，`strategy_gate` 放行结论逐字段不变。
    切换周期属于产品口径变更，须人工决策并重做泄漏与偏差审查。

    落盘 `reports/horizon_scan.json`（监控报表 / API 只读消费）。
    """
    import scripts.evaluate_models as ev
    from src.eval.horizon_scan import HorizonScanner, compare_with_current
    from src.eval.stratified import StratifiedEvaluator

    logger.info("执行多周期口径探索扫描")
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据（请先准备 data/raw/<symbol>.csv 或联网采集）",
                   "pooled": {}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    scanner = HorizonScanner(config)
    # CLI 显式给了候选周期 → 以 CLI 为准（覆盖配置，而非追加）；
    # 避免"用户想只扫 3 个周期，实际扫了配置里的 5 个"这种不可解释行为。
    if candidates:
        scanner.candidates = scanner._normalize_candidates(list(candidates))
    days_list = [int(d) for d in scanner.candidates]

    # 覆盖度前置检查：样本不够的周期如实标注
    coverage = scanner.assess_coverage(data)

    def _build_one(one_data: dict, days: int) -> dict:
        """单周期 → walk-forward 序列（与 `ic` 门禁完全同源）。"""
        combined = ev.build_supervised(one_data, config, int(days))
        if combined.empty:
            return {"horizon_days": int(days), "scores": [], "returns": [], "window_size": 0}
        scores, returns, band = _ic_scores_for_horizon(
            config, combined, int(days), folds, None)
        return {
            "horizon_days": int(days),
            "scores": scores,
            "returns": returns,
            "window_size": 20 if len(scores) >= 60 else None,
            "neutral_band": band,
        }

    pooled_seqs: dict = {}
    for days in days_list:
        cov = (coverage.get("per_horizon") or {}).get(str(days), {})
        if cov and not cov.get("available", True):
            # 样本覆盖不住：不跑、不猜，留空序列让 decide 如实标注不可用
            logger.warning(f"[horizon-scan] {days}d 覆盖度不足，跳过：{cov.get('reason')}")
            pooled_seqs[str(days)] = {"horizon_days": int(days), "scores": [], "returns": []}
            continue
        try:
            pooled_seqs[str(days)] = _build_one(data, days)
        except Exception as e:  # noqa: BLE001 - 单周期失败不得拖垮整次扫描
            logger.warning(f"[horizon-scan] {days}d 序列构造失败: {e}")
            pooled_seqs[str(days)] = {"horizon_days": int(days), "scores": [], "returns": []}

    # 分池口径（可选，与 `ic-pool` 同源）
    pools_seqs: dict = {}
    if include_pools:
        evaluator = StratifiedEvaluator(config)
        splits = evaluator.split(data)
        for cls, pool_data in splits.items():
            if not pool_data:
                continue
            rows: dict = {}
            for days in days_list:
                cov = (coverage.get("per_horizon") or {}).get(str(days), {})
                if cov and not cov.get("available", True):
                    rows[str(days)] = {"horizon_days": int(days), "scores": [], "returns": []}
                    continue
                try:
                    rows[str(days)] = _build_one(pool_data, days)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[horizon-scan] 分池 {cls} {days}d 构造失败: {e}")
                    rows[str(days)] = {"horizon_days": int(days), "scores": [], "returns": []}
            if rows:
                pools_seqs[cls] = rows

    result = scanner.decide(pooled_seqs, coverage=coverage, pools=pools_seqs,
                            symbol_count=len(data))
    result["folds"] = folds
    result["symbols_evaluated"] = len(data)
    result["vs_current"] = compare_with_current(result)

    saved = scanner.save(result)
    _record_trial(config, "horizon-scan", {
        "symbols": len(data), "folds": folds,
        "candidates": result.get("candidates"),
        "passed_horizons": (result.get("summary") or {}).get("passed_horizons"),
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n多周期扫描报告已保存: {saved}")
    print(f"\n对照结论: {result['vs_current'].get('narrative')}")
    return result


def run_horizon_decision(config: dict, days: list[int] | None = None,
                         decided_by: str = "", reason: str = "") -> dict:
    """预测周期切换的决策前置评估（S11）：多重比较校正 + 决策单。

    为什么需要它（S10 的遗留）：
      S10 用一条命令复算出「5/10/20 日未过、40/60 日达标」的表象，但候选周期是
      **逐个试出来的** —— 在 5 个候选里挑最好看的那个当结论，是典型的多重比较。
      本轮用当前缓存行情复算时，S10 报告里 40/60 日的「达标」**没有复现**
      （40 日 IC +0.0035、60 日 IC −0.0279，均未过线），正说明未校正的探索性结果
      不能直接当产品变更依据。

    本命令做的事：
      1. 对扫描报告里每个候选周期做 **IC 显著性检验**（IC / (σ_IC/√n)）；
      2. 用 **Bonferroni + Holm** 把「挑最好看」的选择偏差折算成族错误率；
      3. 输出**决策单**（verdict + status + blockers），结论为 approve 时
         仍需人工签字（`--decided-by`）才算 confirmed。

    ⚠️ **不改门禁口径**：`data.prediction_horizons` 一个字不动，
    `affects_gate` 恒为 False，`strategy_gate` 放行结论逐字段不变。
    结论为 reject / defer 时如实写出，绝不因为某个候选数字好看而放宽口径。

    落盘 `reports/horizon_decision.json`。
    """
    from src.eval import horizon_decision as hd

    logger.info("执行预测周期切换决策前置评估")
    scan = hd.load_json(hd.scan_path(config))
    if scan is None:
        logger.warning("[horizon-decision] 未找到多周期扫描报告，先跑 `python main.py horizon-scan`")

    evidence = hd.evaluate_scan(scan, config, proposed_days=days or None)
    record = hd.build_decision_record(
        scan, config, proposed_days=days or None,
        decided_by=decided_by, reason=reason)
    out = hd.save(record, config)

    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\n决策单已保存: {out}")
    print(f"\n结论: {record['verdict']} / 状态: {record['status']}")
    print(f"说明: {evidence.get('narrative')}")
    if record.get("blockers"):
        print("阻塞项:")
        for b in record["blockers"]:
            print(f"  - {b}")
    return record


def run_feature_experiment(config: dict, symbols: list[str] | None = None,
                           arms: list[str] | None = None, folds: int = 3) -> dict:
    """特征扩充正交对照实验（S12）：先证明"该不该扩"，再扩。

    背景：
      README §17.3 / SALES_PLAN §8.2 写着「特征扩充（横截面/宏观/情感）可把
      short/mid 命中率推过门禁线」 —— 这是一条**因果断言**，但一直没有对照实验支撑。
      特征越多越容易过拟合，"某次跑出好看数字"无法区分「真带来正交信息」与
      「噪声被拟合进去」。

    本命令做的事（每臂只改特征集，其余全不动）：
      1. 基准臂 = 生产现行特征（技术 + 扩展指标 [+ 因子列]）；
      2. 对照臂 = 逐族加入横截面 / 宏观 / 情感特征；
      3. 与基准**同折配对**对比 IC / 命中率增量；
      4. 对增量做近似 t 检验 + **Bonferroni 多重比较校正**（与 S11 同一纪律）；
      5. 结论三态：adopt / reject / defer。

    ⚠️ **不改生产特征集**：`affects_features` 恒为 False，只产出证据；
    是否接入需人工确认并重跑全量门禁。样本不足 / 特征缺失一律如实标注。

    落盘 `reports/feature_experiment.json`。
    """
    import scripts.evaluate_models as ev
    from src.eval.feature_experiment import DEFAULT_ARMS, run_experiment, save

    logger.info("执行特征扩充正交对照实验")
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据（请先准备 data/raw/<symbol>.csv 或联网采集）",
                   "horizons": {}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    result = run_experiment(data, config, arms=arms or DEFAULT_ARMS, folds=folds)
    result["symbols_evaluated"] = len(data)
    result["folds"] = folds
    _record_trial(config, "feature-experiment", {
        "symbols": len(data), "folds": folds,
        "arms": result.get("arms"), "verdict": result.get("verdict"),
    })
    out = save(result, config)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n特征扩充实验报告已保存: {out}")
    print(f"\n结论: {result['verdict']}")
    print(f"说明: {result['narrative']}")
    return result


def run_trials(config: dict, command: str | None = None, asof: str | None = None,
               note: str = "") -> dict:
    """评估试验登记（S13）：把「试了多少次」变成不可篡改的事实。

    背景（S11 + S12 的共同结论）：
      两次独立排查都撞上同一堵墙 —— 校正只能惩罚**本次**比较过的次数，
      它不知道上周改过口径、上上周换过特征开关。
      每个未被登记的"再试一次"都在悄悄稀释 p 值的有效性。

    本命令做的事：
      - `python main.py trials`                 查看累计试验次数与口径指纹；
      - `python main.py trials --note "..."`     手动追加一条登记（说明这次试了什么）；
      - `--asof` 查询某时点的累计次数（保证"当时不知道未来"）。

    ⚠️ 登记为 **append-only**：不删除、不改写历史记录。
    登记文件缺失 = 0 次；文件损坏 = 如实报 `available=false`，**绝不当作 0**。
    """
    from src.eval.trial_registry import (count_trials, make_entry, record,
                                         registry_path, summary)

    logger.info("查看评估试验登记")
    if note:
        entry = record(command or "manual", config, note=note)
        print(f"已登记：{json.dumps(entry, ensure_ascii=False)}")
        print(f"登记文件：{registry_path(config)}")

    stats = summary(config, asof=asof)
    stats["filtered_command"] = str(command or "")
    filtered = count_trials(config, command=command, asof=asof)
    stats["filtered"] = filtered

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n登记文件: {registry_path(config)}")
    if stats.get("available"):
        print(f"累计试验次数（多重比较校正应使用）：{stats.get('count')}"
              f"（其中命令 {command or '全部'}：{filtered.get('count')}）")
        print(f"口径指纹: {stats.get('fingerprint_hash')}")
    else:
        print(f"⚠️ 登记不可用：{stats.get('reason')}（未按 0 次处理）")
    return stats


def run_release_check(config: dict, notify: bool = False, as_json: bool = False) -> dict:
    """发布态健康检查（S14）：把散落的运维信号收敛成一条可执行判断。

    为什么需要它：
      仓库里已有门禁 / IC 趋势 / 分池 / 多周期扫描 / 决策单 / 特征实验 / 试验登记 ——
      但它们是**各自独立的报告**。运维上要回答的只有一个问题：
      「现在这套东西能不能继续按当前口径往下跑？要不要先做某件事？」

      过去靠人翻五份报告，于是最常见的失败不是"模型坏了"，
      而是**没人发现某份报告过期了**（门禁判定用的是两周前的 IC）。
      本命令把「产物可用性」放在「指标好坏」之前：过期的判定比不达标更危险。

    产出三类结论：
      - `blocking`：必须处理，否则当前结论不可信（扫描过期、登记损坏、模型缺失…）
      - `action`  ：建议动作（IC 衰减 → 提前重训练；有显著特征臂待确认…）
      - `info`    ：只是状态（门禁仍 readonly）

    ⚠️ 只汇总与排序，**不重算指标、不自动修复**、不改变门禁结论。
    加 `--notify` 时按去重规则给出告警路由决定（blocking 必发；action 仅在
    待处理项变化时发 —— 每天播同一句会把人训练成忽略告警）。

    落盘 `reports/release_check.json`。
    """
    from src.monitor.release_check import collect_and_check, route_alerts

    logger.info("执行发布态健康检查")
    result = collect_and_check(config)
    routing = None
    if notify:
        routing = route_alerts(result, config)
        result["alert_routing"] = routing

    out = Path((config.get("release_check", {}) or {}).get("report_dir", "reports")) / "release_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"发布态: {result['status']}")
        for group in ("blocking", "actions", "info"):
            for it in result.get(group) or []:
                icon = {"blocking": "⛔", "action": "⚠️", "info": "ℹ️"}.get(it["severity"], "-")
                print(f"  {icon} [{it['code']}] {it['message']}")
                if it.get("next_step"):
                    print(f"        → {it['next_step']}")
        if routing:
            print(f"\n告警路由: send={routing['should_send']} 原因={routing['reason']}")
    print(f"\n发布态检查已保存: {out}")
    return result


def run_label_ab(config: dict, symbols: list[str] | None = None,
                 horizons: list[int] | None = None, folds: int = 3) -> dict:
    """标签口径 A/B 对比（S12 / G2）：旧固定窗口 vs 三重障碍法。

    为什么需要（Issue #29 集成方案 G2 验收口径）：
      标签重构是本轮唯一可能直接改变门禁判定的一步，必须用**同数据、同折、
      同模型配置**的对照实验把「换了标签是否变好」变成可复算数字。

    本命令做的事：
      1. 对每个周期分别构造两份标签：
         · 旧：`target_{h}d`（未来 h 日收益 > 0，写死窗口）；
         · 新：三重障碍法（止盈/止损/时间三重障碍，阈值随波动率自适应）
               折叠为二分类；
      2. 在同一批样本、同一组 walk-forward 折上分别训练 LightGBM 并评估
         IC / 命中率；
      3. 输出增量对照（新 − 旧）与命中率二项 z 值提示。

    ⚠️ **不改门禁**：`affects_gate` 恒为 False，只产出证据；
    是否把新标签纳入主线由人工检查点（T12.3）决定。结论**不管好坏都如实入库**。

    落盘 `reports/label_ab.json`。
    """
    import scripts.evaluate_models as ev
    from src.data.preprocessor import FeatureEngineer
    from src.eval.label_ab import LabelABExperiment, build_ab_report

    logger.info("执行标签口径 A/B 对比（三重障碍法 vs 固定窗口）")
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据（请先准备 data/raw/<symbol>.csv 或联网采集）",
                   "affects_gate": False}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    horizons_cfg = (config.get("data", {}) or {}).get("prediction_horizons", {}) or {}
    horizon_map = horizons or [int(v.get("days", 5)) for v in horizons_cfg.values()
                               if isinstance(v, dict)]
    if not horizon_map:
        horizon_map = [5, 10, 20]

    experiment = LabelABExperiment(config)
    results: dict = {}
    for days in horizon_map:
        combined = ev.build_supervised(data, config, int(days))
        if combined.empty:
            results[f"{days}d"] = {"available": False, "reason": "no_supervised_data",
                                   "horizon_days": int(days)}
            continue
        fe = FeatureEngineer(config)
        cols = [c for c in fe.get_feature_columns(combined, int(days))
                if not str(c).startswith("_")]
        splits = ev.walk_forward_splits(len(combined), folds)
        try:
            results[f"{days}d"] = experiment.compare(combined, int(days), cols, splits)
        except Exception as e:  # noqa: BLE001 - 单周期失败不得拖垮整次对比
            logger.warning(f"[label-ab] {days}d 对比失败: {e}")
            results[f"{days}d"] = {"available": False, "reason": f"error: {e}",
                                   "horizon_days": int(days)}

    report = build_ab_report(results)
    report["folds"] = folds
    report["symbols_evaluated"] = len(data)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "label_ab.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _record_trial(config, "label-ab", {
        "symbols": len(data), "folds": folds, "horizons": list(results.keys()),
        "any_improved": report["summary"]["any_improved"],
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n标签 A/B 报告已保存: {out_path}")
    print(f"\n结论: {report['summary']['conclusion']}")
    return report


def run_qlib_ab(config: dict, symbols: list[str] | None = None,
                folds: int = 3, dump_bin: bool = True) -> dict:
    """Alpha158 因子增量验证（S13 / G3）：qlib 因子 vs 现有 15 因子特征集。

    为什么需要（Issue #29 集成方案 G3 验收口径）：
      「Alpha158 有没有增量」必须变成同数据、同折、同模型配置的可复算数字。
      本命令做的事：
        1. （可选）把行情缓存转 qlib .bin 列存（``integrations/qlib/data_layer``）；
        2. 逐标的计算 Alpha158 并压缩为 ``factor_a158_*`` 族因子
           （``src/factors/qlib_factor_provider.py``，纯 pandas，无 qlib 运行时依赖）；
        3. 同一批样本、同一组折、同一 LightGBM 配置下 A/B：
           基准臂 = 现行特征集；对照臂 = 现行特征 + factor_a158_*。

    ⚠️ **不改门禁**：``affects_gate`` 恒为 False；是否纳入主线由 T13.4
    人工检查点决定。结论不管好坏写入 ``00_kickoff/qlib_alpha158_conclusion.md``。

    落盘 ``reports/qlib_ab.json``。
    """
    import scripts.evaluate_models as ev
    from src.data.preprocessor import FeatureEngineer
    from src.eval.qlib_ab import QlibABExperiment, build_ab_report
    from src.factors.qlib_factor_provider import PROVIDER_FACTOR_COLUMNS, QlibFactorProvider

    logger.info("执行 Alpha158 因子增量验证（qlib vs 现有特征集）")
    symbols = symbols or _config_symbols(config)
    data = ev.load_market_data(config, symbols)
    if not data:
        payload = {"error": "无可用行情数据（请先准备 data/raw/<symbol>.csv 或联网采集）",
                   "affects_gate": False}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    dump_payload: dict = {}
    if dump_bin:
        try:
            from integrations.qlib.data_layer import dump_all as qlib_dump_all

            dump_payload = qlib_dump_all("data/qlib_bin", data)
            logger.info("[qlib-ab] .bin 转换: %s 标的 / %s 行",
                        dump_payload.get("symbols_written"), dump_payload.get("total_rows"))
        except Exception as e:  # noqa: BLE001 - 数据层失败不影响 A/B 本身
            logger.warning(f"[qlib-ab] .bin 转换失败（不影响 A/B）: {e}")
            dump_payload = {"error": str(e)}

    horizons_cfg = (config.get("data", {}) or {}).get("prediction_horizons", {}) or {}
    horizon_map = [int(v.get("days", 5)) for v in horizons_cfg.values()
                   if isinstance(v, dict)]
    if not horizon_map:
        horizon_map = [5, 10, 20]

    provider = QlibFactorProvider(config)
    experiment = QlibABExperiment(config)
    results: dict = {}
    for days in horizon_map:
        combined = ev.build_supervised(data, config, int(days))
        if combined.empty:
            results[f"{days}d"] = {"available": False, "reason": "no_supervised_data",
                                   "horizon_days": int(days)}
            continue
        # 逐标的追加 Alpha158 族因子（provider 内部 fail-soft）
        enriched_parts = []
        for symbol, df in data.items():
            try:
                part = provider.compute(df.copy())
                part["_symbol"] = symbol
                enriched_parts.append(part)
            except Exception as e:  # noqa: BLE001 - 单标的失败跳过
                logger.warning(f"[qlib-ab] {symbol} Alpha158 供给失败: {e}")
        if not enriched_parts:
            results[f"{days}d"] = {"available": False, "reason": "a158_enrichment_failed",
                                   "horizon_days": int(days)}
            continue
        enriched = pd.concat(enriched_parts, ignore_index=True)
        # 重建监督集：与 combined 同一套特征工程 + target，再按 (_symbol, date) 对齐因子
        fe = FeatureEngineer(config)
        base_cols = [c for c in fe.get_feature_columns(combined, int(days))
                     if not str(c).startswith("_")]
        # 对齐：enriched 只取因子列与键
        key_cols = ["date"]
        a158_cols = [c for c in PROVIDER_FACTOR_COLUMNS if c in enriched.columns]
        if "_symbol" in combined.columns:
            enriched = enriched.set_index(["_symbol", "date"])
            combined_k = combined.set_index(["_symbol", "date"])
        else:
            # combined 无 _symbol（旧版）：按 date 对齐（降级口径，样本可能少）
            enriched = enriched.set_index("date")
            combined_k = combined.set_index("date")
        factor_frame = enriched[a158_cols].reindex(combined_k.index)
        combined_k = pd.concat([combined_k, factor_frame], axis=1).reset_index()
        splits = ev.walk_forward_splits(len(combined_k), folds)
        try:
            results[f"{days}d"] = experiment.compare(
                combined_k, int(days), base_cols, a158_cols, splits)
        except Exception as e:  # noqa: BLE001 - 单周期失败不拖垮整次对比
            logger.warning(f"[qlib-ab] {days}d 对比失败: {e}")
            results[f"{days}d"] = {"available": False, "reason": f"error: {e}",
                                   "horizon_days": int(days)}

    report = build_ab_report(results)
    report["folds"] = folds
    report["symbols_evaluated"] = len(data)
    report["qlib_bin_dump"] = dump_payload
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "qlib_ab.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _record_trial(config, "qlib-ab", {
        "symbols": len(data), "folds": folds, "horizons": list(results.keys()),
        "any_improved": report["summary"]["any_improved"],
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAlpha158 A/B 报告已保存: {out_path}")
    print(f"\n结论: {report['summary']['conclusion']}")
    return report


def run_factor_model(config: dict, symbol: str | None = None, top_n: int = 10) -> dict:
    """多因子模型诊断：因子权重 / 族权重 / IC 排名 / 当前因子值。

    与 `factors`（推理期特征因子组合）互补：
      - `factors`      ：不需要训练，直接用技术特征因子做组合打分；
      - `factor-model` ：读取 `ModelTrainer --model-type factor_model` 训练出的
        **IC 加权因子模型**，输出可解释的权重与 IC 排名。
    """
    from src.data.collector import DataCollector
    from src.factors import FactorLibrary
    from src.factors.factor_model import FactorModel

    logger.info("多因子模型诊断")
    save_dir = Path(config.get("training", {}).get("save_dir", "models"))
    model_path = save_dir / "factor_model_short_term_5d.pkl"

    result: dict = {
        "factors_enabled": bool(
            (config.get("model", {}).get("factors", {}) or {}).get("enabled", False)
        ),
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
    }

    model: FactorModel | None = None
    if model_path.exists():
        model = FactorModel(config)
        model.load(str(model_path))
        result["explain"] = model.explain(top_n)
        print("\n" + "=" * 60)
        print("多因子模型 · 因子权重与 IC")
        print("=" * 60)
        for row in result["explain"]["top_factors"]:
            print(f"  {row['factor']:<24} family={row['family']:<10} "
                  f"weight={row['weight']:.4f} ic={row['ic']:+.4f}")
        print(f"\n族权重: {result['explain']['family_weights']}")
    else:
        print(f"未找到因子模型：{model_path}")
        print("提示：先运行 `python main.py train --model-type factor_model`")

    if symbol:
        collector = DataCollector(config)
        df = collector.load_cached(symbol) or collector._fetch_with_fallback(symbol)
        if df is None:
            print(f"无法获取 {symbol} 的数据")
            return result
        lib = FactorLibrary(config)
        feat = lib.compute(df)
        cols = lib.factor_columns(feat)
        latest = {c: round(float(feat[c].iloc[-1]), 4) for c in cols}
        result["symbol"] = symbol
        result["latest_factors"] = latest
        if model is not None:
            proba = float(model.predict_proba(feat[cols].iloc[-1:])[0][1])
            result["probability"] = round(proba, 4)
            print(f"\n{symbol} 当前多因子概率: {proba:.4f} "
                  f"({'看涨' if proba > 0.5 else '看跌'})")
        print(f"\n最新因子值（{symbol}）:")
        for k, v in latest.items():
            print(f"  {k:<24} {v:+.4f}")
    return result


def run_factors(config: dict, symbol: str) -> dict:
    """多因子加权组合预测：模型因子 + 技术特征因子 → 单周期综合得分。"""
    from src.data.collector import DataCollector
    from src.inference.factor_combiner import FactorCombiner, combine_horizon
    from src.inference.predictor import PredictionEngine

    logger.info(f"多因子组合预测 {symbol}")
    engine = PredictionEngine(config)
    engine.load_models(config["model"].get("type", "lightgbm"))
    pred = engine.predict_all_horizons(symbol)

    df = DataCollector(config).load_cached(symbol)
    combiner = FactorCombiner(config)
    horizons_out = {}
    for hname, hp in (pred.get("predictions") or {}).items():
        horizons_out[hname] = combine_horizon(combiner, symbol, df, hp).to_dict()
    payload = {"symbol": symbol, "combiner": combiner.weighter, "horizons": horizons_out}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# ==================== Q3 路线：实时数据流 与 信号一致性 ====================

def run_stream(config: dict, symbols: list[str] | None = None,
               once: bool = False, persist: bool = True) -> dict:
    """盘中实时流：拉快照 → 逐只生成分钟级信号更新 → 输出盘中观点失真预警。

    只使用真实数据源；快照失败一律 fail-open（返回 available=False），绝不编造行情。
    """
    from src.inference.intraday import IntradayPredictor

    symbols = symbols or _config_symbols(config)
    predictor = IntradayPredictor(config)
    logger.info(f"实时流盘中更新: {len(symbols)} 个标的 (once={once})")

    def _cycle() -> dict:
        payload = predictor.poll_once(symbols, persist=persist)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    if once:
        payload = _cycle()
        stale = payload.get("stale") or []
        if stale:
            print(f"\n⚠️ 盘中观点失真预警（{len(stale)} 只）: {', '.join(stale)}")
        return payload

    # 常驻轮询：Ctrl+C 退出（观测路径，异常不终止循环）
    poll = max(int(predictor.poll_seconds), 5)
    print(f"盘中轮询已启动（间隔 {poll}s，Ctrl+C 退出）")
    last = {}
    try:
        while True:
            try:
                last = _cycle()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[stream] 本轮更新失败（继续下一轮）: {e}")
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\n已停止盘中轮询")
    return last


def run_intraday(config: dict, symbol: str) -> dict:
    """单只标的的盘中信号更新（一次性，不常驻）。"""
    from src.inference.intraday import IntradayPredictor

    predictor = IntradayPredictor(config)
    payload = predictor.build(
        symbol,
        _load_daily(config, symbol),
        predictor.quote_client.fetch_quotes([symbol]).get(symbol),
    ).to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def run_consistency(config: dict, symbol: str) -> dict:
    """信号一致性校验：跨周期 / 跨模型 / 跨口径是否自相矛盾。"""
    from src.inference.predictor import PredictionEngine
    from src.monitor.signal_consistency import SignalConsistencyChecker

    logger.info(f"信号一致性校验 {symbol}")
    engine = PredictionEngine(config)
    engine.load_models(config["model"].get("type", "lightgbm"))
    predicted = engine.predict_all_horizons(symbol)

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

    checker = SignalConsistencyChecker(config)
    report = checker.check(
        symbol,
        predictions=predicted,
        components=components,
        model_probability=(sum(probs) / len(probs)) if probs else None,
        factor_score=factor_score,
    ).to_dict()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


# ==================== Q4 路线：智能风控建议（止损止盈） ====================

def run_risk_advice(config: dict, symbols: list[str] | None = None,
                    as_json: bool = False) -> dict:
    """智能风控建议：为标的池产出止损/止盈建议（Q4 路线）。

    **定位**：建议，不是下单指令。门禁未放行（`readonly`/`unknown`）时**不产出可用价位**，
    只保留波动结构与仓位侧信息，避免未过门禁的信号被误当作可交易。
    """
    from src.data.collector import DataCollector
    from src.inference.predictor import PredictionEngine
    from src.trading.risk import RiskManager
    from src.trading.risk_advice import RiskAdvisor
    from src.trading.signal import SignalEngine

    symbols = symbols or _config_symbols(config)
    logger.info(f"生成智能风控建议: {len(symbols)} 个标的")

    advisor = RiskAdvisor(config)
    signal_engine = SignalEngine(config)
    risk_manager = RiskManager(config)
    collector = DataCollector(config)

    engine = None
    if not _offline(config):
        try:
            engine = PredictionEngine(config)
            engine.load_models(config["model"].get("type", "lightgbm"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[risk-advice] 模型加载失败，退化为纯波动结构建议（无信号）：{e}")
            engine = None

    # 无模型时按 HOLD 处理 —— 不臆造方向，此时建议必然 withheld（诚实标注）
    from src.trading.signal import Signal

    items: list[dict] = []
    for symbol in symbols:
        try:
            df = collector.load_cached(symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[risk-advice] {symbol} 行情读取失败: {e}")
            df = None

        if engine is not None:
            try:
                payload = engine.predict_all_horizons(symbol)
                sig = signal_engine.build_signal(symbol, payload)
                price = _latest_close(payload)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[risk-advice] {symbol} 预测失败: {e}")
                payload, sig, price = {}, None, None
        else:
            payload, sig, price = {}, None, _last_close(df)

        if sig is None:
            sig = Signal(symbol=symbol, action="HOLD", score=0.0, strength=0.0,
                         confidence=0.0, direction_consensus="未知")

        budget = None
        if getattr(sig, "action", "HOLD") != "HOLD":
            try:
                budget = risk_manager.budget(sig, price=price)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[risk-advice] {symbol} 仓位预算失败: {e}")

        items.append({
            "symbol": symbol, "signal": sig, "df": df, "price": price,
            "risk_budget": budget, "horizon": "", "confidence": None,
        })

    result = advisor.advise_portfolio(items)

    # 落盘：供监控报表（reports/risk_advice.json）与下游只读消费
    out_dir = Path(config.get("report", {}).get("output_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "risk_advice.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(advisor.render_markdown(result))
        print(f"\n风控建议已保存: {report_path}")
    return result


def _latest_close(payload: dict) -> float | None:
    """从预测结果中取最新收盘价（多周期任一携带即可）。"""
    for block in (payload.get("predictions") or {}).values():
        if isinstance(block, dict) and block.get("latest_close"):
            return float(block["latest_close"])
    return None


def _last_close(df) -> float | None:
    try:
        if df is not None and len(df) and "close" in df.columns:
            return float(df["close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return None
    return None


def _offline(config: dict) -> bool:
    """是否处于离线模式（离线时不做任何模型推理，只读本地缓存）。"""
    return bool((config.get("data", {}) or {}).get("offline", False))


def _load_daily(config: dict, symbol: str):
    """读取日K缓存（只读，不触发刷新；供盘中/一致性等观测路径复用）。"""
    from src.data.collector import DataCollector

    try:
        return DataCollector(config).load_cached(symbol)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取 {symbol} 行情缓存失败: {e}")
        return None


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（供 main() 与测试复用）"""
    parser = argparse.ArgumentParser(
        description="TrendCast Pro - 金融市场预测模型（专业版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py train                    # 训练模型
  python main.py predict 600519.SH        # 预测贵州茅台短期走势
  python main.py predict 600519.SH --horizon all  # 预测所有周期
  python main.py predict RB.SHF --horizon long_term  # 预测螺纹钢长期走势
  python main.py export                   # 导出模型为 ONNX
  python main.py all                      # 执行全部流程
  python main.py serve                    # 启动 Web API 服务
  python main.py schedule                 # 启动自动重训练调度
  python main.py audit                    # 生成预测审计报告
  python main.py notify 600519.SH         # 推送预测信号
  python main.py batch 600519.SH 000858.SZ  # 批量预测
  python main.py daily-report             # 生成每日预测报告
  python main.py weekly-report            # 生成周度预测报告
  python main.py adaptive                 # 运行自适应学习引擎
  python main.py signal 600519.SH         # 输出综合交易信号（BUY/SELL/HOLD）
  python main.py orders 600519.SH 000858.SZ  # 输出下单明细
  python main.py trade 600519.SH          # 执行交易适配流程并导出数据流
  python main.py backtest 600519.SH       # 信号假设成交回测
  python main.py macro                    # 查看宏观指标数据源状态
  python main.py monitor                  # 生成模型监控报表
  python main.py ic                       # IC / 命中率门禁评估（全标的池）
  python main.py ic --stratify            # 额外按标的分解（定位拖后腿的标的）
  python main.py ic --derive-band         # 逐折用训练折推导中性带（过滤 ≈0.5 噪音）
  python main.py ic-trend                 # IC 时序 / 信号衰减监控（Q5）
  python main.py ic-pool                  # 按资产类别分池门禁（个股 / ETF 分开判定，S9）
  python main.py pool-train               # 按资产类别分层训练（每池一套模型，S9）
  python main.py horizon-scan             # 多周期口径探索扫描（换周期有没有用，S10）
  python main.py horizon-scan --days 5,20,40   # 自定义候选周期
  python main.py horizon-decision          # 周期切换决策前置评估（多重比较校正 + 决策单，S11）
  python main.py horizon-decision --days 40 --decided-by 安然 --reason "业务可接受 40 日延迟"  # 人工签字
  python main.py feature-experiment        # 特征扩充正交对照实验（横截面/宏观/情感，S12）
  python main.py label-ab                  # 标签口径 A/B 对比（三重障碍法 vs 固定窗口，S12/G2）
  python main.py qlib-ab                   # Alpha158 因子增量验证（qlib vs 现有特征集，S13/G3）
  python main.py trials                    # 查看评估试验登记（累计比较次数，S13）
  python main.py release-check             # 发布态健康检查（收敛阻塞项与建议动作，S14）
  python main.py release-check --notify    # 附带告警路由决定（含去重）
  python main.py trials --note "试了 40 日周期"  # 手动登记一次探索
  python main.py feature-experiment --arms cross_sectional,macro  # 指定对照臂
  python main.py gate                     # 策略门禁判定（IC + 审计命中率）
  python main.py gate-diagnose            # 门禁阻塞诊断（还差多少 / 哪条腿卡住）
  python main.py factors 600519.SH        # 多因子加权组合预测（推理期特征因子）
  python main.py factor-model 600519.SH   # 多因子模型权重 / IC 诊断（可训练模型）
  python main.py stream --once            # 盘中实时流一次性更新（Q3）
  python main.py stream                   # 盘中轮询常驻（Ctrl+C 退出）
  python main.py intraday 600519.SH       # 单只标的盘中信号更新
  python main.py consistency 600519.SH    # 信号一致性校验（跨周期/跨模型/跨口径）
  python main.py risk-advice 600519.SH    # 智能风控建议：止损/止盈（Q4）
  python main.py risk-advice --all        # 全标的池风控建议（Markdown）
        """,
    )
    parser.add_argument("command", choices=[
        "train", "evaluate", "predict", "export", "all",
        "serve", "schedule", "audit", "notify", "batch",
        "daily-report", "weekly-report", "adaptive",
        "signal", "orders", "trade", "backtest", "macro", "monitor",
        "ic", "ic-trend", "ic-pool", "pool-train", "horizon-scan", "horizon-decision",
        "feature-experiment", "label-ab", "qlib-ab", "trials", "release-check",
        "gate", "gate-diagnose", "factors", "factor-model",
        "stream", "intraday", "consistency", "risk-advice",
    ], help="执行命令")
    parser.add_argument("args", nargs="*", help="附加参数")
    parser.add_argument("--horizon", default="short_term",
                        choices=["short_term", "mid_term", "long_term", "all"],
                        help="预测周期")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--model-type", default=None,
                        choices=["lightgbm", "pytorch_lstm", "timesfm", "ensemble",
                                 "factor_model", "multifactor"],
                        help="模型类型 (覆盖配置文件), 支持: lightgbm, pytorch_lstm, timesfm, ensemble")
    parser.add_argument("--once", action="store_true",
                        help="stream 命令：只执行一轮盘中更新（不常驻轮询）")
    parser.add_argument("--symbols", default=None,
                        help="stream / ic 命令：逗号分隔的标的列表（缺省用配置启用标的）")
    parser.add_argument("--all", dest="all_symbols", action="store_true",
                        help="risk-advice 命令：对配置内全部启用标的产出建议")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="risk-advice 命令：输出 JSON（缺省输出 Markdown）")
    parser.add_argument("--detail", action="store_true",
                        help="ic 评估追加统一评估量尺报告（因子健康度，report_only）")
    parser.add_argument("--stratify", action="store_true",
                        help="ic 命令：额外按标的分层评估，定位拖后腿的标的")
    parser.add_argument("--derive-band", dest="derive_band", action="store_true", default=None,
                        help="ic 命令：逐折用训练折推导中性带，过滤 ≈0.5 噪音样本（缺省跟随配置）")
    parser.add_argument("--no-pools", dest="no_pools", action="store_true",
                        help="horizon-scan 命令：只扫整池口径，跳过分池（更快）")
    parser.add_argument("--days", default=None,
                        help="horizon-scan 命令：逗号分隔的候选周期（交易日），如 5,20,40；"
                             "缺省用配置 horizon_scan.candidates。位置参数仍是标的列表")
    parser.add_argument("--notify", action="store_true",
                        help="release-check 命令：附带告警路由决定（含与上次的去重比较）")
    parser.add_argument("--note", default="",
                        help="trials 命令：追加一条试验登记说明（append-only）")
    parser.add_argument("--asof", default=None,
                        help="trials 命令：只看该时点之前的累计次数（ISO 时间字符串）")
    parser.add_argument("--command-filter", dest="command_filter", default=None,
                        help="trials 命令：只统计指定命令的试验次数")
    parser.add_argument("--arms", default=None,
                        help="feature-experiment 命令：逗号分隔的对照臂 "
                             "(cross_sectional/macro/sentiment)，缺省为全部")
    parser.add_argument("--horizons", default=None,
                        help="label-ab 命令：逗号分隔的预测周期（交易日），如 5,10,20；"
                             "缺省用配置 prediction_horizons")
    parser.add_argument("--decided-by", dest="decided_by", default="",
                        help="horizon-decision 命令：人工确认人（非空才可能把决策单置为 confirmed）")
    parser.add_argument("--reason", dest="reason", default="",
                        help="horizon-decision 命令：人工确认/驳回的理由（写入决策单审计字段）")
    parser.add_argument("--host", default=None, help="API 服务地址")
    parser.add_argument("--port", type=int, default=None, help="API 服务端口")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 加载配置（优先使用专业版配置）
    if args.config is None:
        pro_config = PROJECT_ROOT / "configs" / "config_pro.yaml"
        args.config = str(pro_config) if pro_config.exists() else str(PROJECT_ROOT / "configs" / "config.yaml")
    config = load_config(args.config)
    if args.model_type:
        config["model"]["type"] = args.model_type

    setup_logging(config)

    edition = config.get("project", {}).get("edition", "standard")
    logger.info(f"项目: {config['project']['name']} v{config['project']['version']} [{edition}]")
    logger.info(f"模型类型: {config['model']['type']}")
    logger.info(f"数据源: {config['data']['source']}")

    # 验证模型类型
    # 支持的模型类型：LightGBM（默认）、PyTorch LSTM、TimesFM（外部可选模块）、以及 Ensemble（混合）
    valid_types = frozenset({
        "lightgbm", "pytorch_lstm", "timesfm", "ensemble", "factor_model", "multifactor",
    })
    if config["model"]["type"] not in valid_types:
        logger.error(f"不支持的模型类型: {config['model']['type']}，支持的类型: {sorted(valid_types)}")
        sys.exit(1)

    # 执行命令
    if args.command == "train":
        run_train(config)
    elif args.command == "evaluate":
        run_train(config)
    elif args.command == "predict":
        symbol = args.args[0] if args.args else None
        if not symbol:
            print("错误: predict 命令需要指定标的代码")
            print("示例: python main.py predict 600519.SH")
            sys.exit(1)
        run_predict(config, symbol, args.horizon)
    elif args.command == "batch":
        symbols = args.args
        if not symbols:
            print("错误: batch 命令需要至少一个标的代码")
            sys.exit(1)
        run_batch_predict(config, symbols, args.horizon)
    elif args.command == "export":
        run_export(config)
    elif args.command == "all":
        run_all(config)
    elif args.command == "serve":
        run_serve(config, args.host, args.port)
    elif args.command == "schedule":
        run_schedule(config)
    elif args.command == "audit":
        run_audit(config)
    elif args.command == "notify":
        symbol = args.args[0] if args.args else None
        if not symbol:
            print("错误: notify 命令需要指定标的代码")
            sys.exit(1)
        run_notify(config, symbol, args.horizon)
    elif args.command == "daily-report":
        run_daily_report(config)
    elif args.command == "weekly-report":
        run_weekly_report(config)
    elif args.command == "adaptive":
        run_adaptive(config)
    elif args.command == "signal":
        symbol = args.args[0] if args.args else None
        if not symbol:
            print("错误: signal 命令需要指定标的代码")
            sys.exit(1)
        run_signal(config, symbol)
    elif args.command == "orders":
        symbols = args.args
        if not symbols:
            print("错误: orders 命令需要至少一个标的代码")
            sys.exit(1)
        run_orders(config, symbols)
    elif args.command == "trade":
        symbols = args.args
        if not symbols:
            print("错误: trade 命令需要至少一个标的代码")
            sys.exit(1)
        run_trade(config, symbols)
    elif args.command == "backtest":
        symbol = args.args[0] if args.args else None
        if not symbol:
            print("错误: backtest 命令需要指定标的代码")
            sys.exit(1)
        run_backtest(config, symbol)
    elif args.command == "macro":
        run_macro(config)
    elif args.command == "monitor":
        output = args.args[0] if args.args else None
        run_monitor(config, output)
    elif args.command == "ic":
        run_ic(config, args.args or None,
               stratify=bool(getattr(args, "stratify", False)),
               derive_band=getattr(args, "derive_band", None),
               detail=bool(getattr(args, "detail", False)))
    elif args.command == "ic-pool":
        run_pool_ic(config, args.args or _cli_symbols(args),
                    derive_band=getattr(args, "derive_band", None))
    elif args.command == "pool-train":
        run_pool_train(config, args.args or _cli_symbols(args))
    elif args.command == "ic-trend":
        run_ic_trend(config, args.args or None)
    elif args.command == "horizon-scan":
        _cand = None
        _raw_days = getattr(args, "days", None)
        if _raw_days:
            try:
                _cand = [int(x) for x in str(_raw_days).replace(" ", "").split(",") if x]
            except ValueError:
                logger.warning(f"--days 解析失败，改用配置候选周期: {_raw_days}")
                _cand = None
        run_horizon_scan(config, symbols=_cli_symbols(args),
                         candidates=_cand or None,
                         include_pools=not getattr(args, "no_pools", False))
    elif args.command == "horizon-decision":
        _proposed = None
        _raw_prop = getattr(args, "days", None)
        if _raw_prop:
            try:
                _proposed = [int(x) for x in str(_raw_prop).replace(" ", "").split(",") if x]
            except ValueError:
                logger.warning(f"--days 解析失败，改用扫描中最有力的候选: {_raw_prop}")
                _proposed = None
        run_horizon_decision(config, _proposed,
                             decided_by=str(getattr(args, "decided_by", "") or ""),
                             reason=str(getattr(args, "reason", "") or ""))
    elif args.command == "feature-experiment":
        _arms = None
        _raw_arms = getattr(args, "arms", None)
        if _raw_arms:
            _arms = [a.strip() for a in str(_raw_arms).split(",") if a.strip()]
        run_feature_experiment(config, symbols=_cli_symbols(args), arms=_arms)
    elif args.command == "label-ab":
        _horizons = None
        _raw_h = getattr(args, "horizons", None)
        if _raw_h:
            try:
                _horizons = [int(x.strip()) for x in str(_raw_h).split(",") if x.strip()]
            except ValueError:
                logger.warning(f"--horizons 解析失败，改用配置 prediction_horizons: {_raw_h}")
        run_label_ab(config, symbols=_cli_symbols(args), horizons=_horizons)
    elif args.command == "qlib-ab":
        run_qlib_ab(config, symbols=_cli_symbols(args))
    elif args.command == "trials":
        run_trials(config, command=getattr(args, "command_filter", None),
                   asof=getattr(args, "asof", None),
                   note=str(getattr(args, "note", "") or ""))
    elif args.command == "release-check":
        run_release_check(config, notify=bool(getattr(args, "notify", False)),
                          as_json=bool(getattr(args, "as_json", False)))
    elif args.command == "gate":
        run_gate(config, args.args[0] if args.args else None)
    elif args.command == "gate-diagnose":
        run_gate_diagnose(config, args.args[0] if args.args else None)
    elif args.command == "factor-model":
        symbol = args.args[0] if args.args else None
        run_factor_model(config, symbol)
    elif args.command == "factors":
        symbol = args.args[0] if args.args else None
        if not symbol:
            print("错误: factors 命令需要指定标的代码")
            sys.exit(1)
        run_factors(config, symbol)
    elif args.command == "stream":
        symbols = args.args or (
            [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
        )
        run_stream(config, symbols, once=args.once)
    elif args.command == "intraday":
        symbol = args.args[0] if args.args else None
        if not symbol:
            print("错误: intraday 命令需要指定标的代码")
            sys.exit(1)
        run_intraday(config, symbol)
    elif args.command == "consistency":
        symbol = args.args[0] if args.args else None
        if not symbol:
            print("错误: consistency 命令需要指定标的代码")
            sys.exit(1)
        run_consistency(config, symbol)
    elif args.command == "risk-advice":
        symbols = args.args or None
        if not symbols and not args.all_symbols:
            print("错误: risk-advice 命令需要指定标的代码，或使用 --all 对全池产出建议")
            sys.exit(1)
        run_risk_advice(config, symbols, as_json=args.as_json)


if __name__ == "__main__":
    main()
