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
from pathlib import Path

import yaml

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
        """,
    )
    parser.add_argument("command", choices=[
        "train", "evaluate", "predict", "export", "all",
        "serve", "schedule", "audit", "notify", "batch",
        "daily-report", "weekly-report", "adaptive",
        "signal", "orders", "trade", "backtest", "macro",
    ], help="执行命令")
    parser.add_argument("args", nargs="*", help="附加参数")
    parser.add_argument("--horizon", default="short_term",
                        choices=["short_term", "mid_term", "long_term", "all"],
                        help="预测周期")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--model-type", default=None,
                        choices=["lightgbm", "pytorch_lstm", "timesfm", "ensemble"],
                        help="模型类型 (覆盖配置文件), 支持: lightgbm, pytorch_lstm, timesfm, ensemble")
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
    valid_types = frozenset({"lightgbm", "pytorch_lstm", "timesfm", "ensemble"})
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


if __name__ == "__main__":
    main()
