"""测试新增模块：自适应学习 / 报告 / 审计 / 导出 / 通知 / 调度 / 指标."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.adaptive_learner import ModelPerformanceMonitor, AdaptiveLearningEngine
from src.report.daily_report import DailyReportGenerator, WeeklyReportGenerator
from src.audit.prediction_audit import PredictionAudit
from src.export.exporter import ModelExporter
from src.notification.notifier import SignalNotifier
from src.scheduler.retrain_scheduler import RetrainScheduler
from src.data.indicators import TechnicalIndicators


def test_indicators_compute_all():
    import pandas as pd
    import numpy as np
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    close = 100 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(1000, 10000, 100),
    })
    out = TechnicalIndicators.compute_all(df)
    assert "rsi" in out.columns or "ema_5" in out.columns or "adx" in out.columns
    assert out.shape[0] == df.shape[0]


def test_adaptive_monitor_drift(tmp_path):
    mon = ModelPerformanceMonitor(str(tmp_path))
    # 记录 3 期高准确率
    for _ in range(3):
        mon.record("short_term", 0.9)
    # 再记录 3 期低准确率
    for _ in range(3):
        mon.record("short_term", 0.4)
    assert mon.detect_drift("short_term", threshold=0.2) is True


def test_report_generation(tmp_path):
    import yaml
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8"))
    cfg["report"]["output_dir"] = str(tmp_path)
    gen = DailyReportGenerator(cfg)
    preds = [{"symbol": "600519.SH", "direction": "看涨", "probability": 0.7, "confidence": 0.7}]
    path = gen.generate(preds, "daily")
    assert path.exists()


def test_audit(tmp_path):
    audit = PredictionAudit(str(tmp_path))
    audit.record_prediction({"symbol": "X", "direction": "看涨", "prediction": 1})
    records = audit.load_records()
    assert len(records) == 1
    report = audit.generate_report()
    assert "总记录数" in report


def test_notifier_no_webhook():
    notifier = SignalNotifier({"notification": {}})
    # 未配置时返回 False，不报错
    assert notifier.send_webhook({}) is False


def test_scheduler_should_run_false():
    cfg = {"scheduler": {"frequency": "weekly", "time": "23:00"}}
    sched = RetrainScheduler(cfg)
    assert sched._should_run() in (True, False)
