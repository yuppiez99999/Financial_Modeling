"""Q1 排期：已知缺陷修复的回归测试。

覆盖 README「九、已知限制」中遗留的 7 例失败所对应的缺失接口：
  main.build_parser / server._HAS_FASTAPI+create_app /
  DailyReportGenerator.generate / PredictionAudit.load_records /
  SignalNotifier.send_webhook / RetrainScheduler._should_run
以及 pandas 3.0 的 fillna(method=) 兼容性。
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as main_cli
from src.api import server
from src.audit.prediction_audit import PredictionAudit
from src.notification.notifier import SignalNotifier
from src.report.daily_report import DailyReportGenerator, WeeklyReportGenerator
from src.scheduler.retrain_scheduler import RetrainScheduler
from src.train.adaptive_learner import ModelPerformanceMonitor


def _cfg():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- CLI ----------

def test_build_parser_commands_and_defaults():
    parser = main_cli.build_parser()
    ns = parser.parse_args(["train"])
    assert ns.command == "train"
    assert ns.horizon == "short_term"
    # 全部子命令均可解析
    for cmd in ["predict", "batch", "serve", "signal", "trade", "backtest"]:
        assert parser.parse_args([cmd]).command == cmd


def test_build_parser_rejects_unknown_command():
    parser = main_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["not-a-command"])


# ---------- API 降级 ----------

def test_server_exposes_has_fastapi_flag():
    assert isinstance(server._HAS_FASTAPI, bool)


def test_create_app_returns_application():
    if not server._HAS_FASTAPI:
        pytest.skip("未安装 FastAPI")
    app = server.create_app(str(PROJECT_ROOT / "configs" / "config.yaml"))
    assert app is not None
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/health" in paths
    assert "/api/v1/portfolio/summary" in paths


# ---------- 报告 ----------

def test_daily_report_generate_returns_path(tmp_path):
    cfg = _cfg()
    cfg["report"]["output_dir"] = str(tmp_path)
    gen = DailyReportGenerator(cfg)
    preds = [{
        "symbol": "600519.SH",
        "predictions": {"short_term": {"direction": "看涨", "probability": 0.7,
                                       "confidence": 0.7, "horizon_days": 5}},
    }]
    path = gen.generate(preds, "daily")
    assert path.exists()
    assert "600519.SH" in path.read_text(encoding="utf-8")


def test_weekly_report_generate_returns_path(tmp_path):
    cfg = _cfg()
    cfg["report"]["output_dir"] = str(tmp_path)
    gen = DailyReportGenerator(cfg)
    path = gen.generate([], "weekly")
    assert path.exists()
    assert path.name.startswith("weekly_report_")
    assert isinstance(WeeklyReportGenerator(cfg), DailyReportGenerator)


# ---------- 审计 ----------

def test_audit_load_records_public_api(tmp_path):
    audit = PredictionAudit(str(tmp_path))
    audit.record_prediction({"symbol": "600519.SH", "horizon": "short_term",
                             "horizon_days": 5, "prediction": 1, "direction": "看涨"})
    records = audit.load_records()
    assert len(records) == 1
    assert records[0]["verified"] is False


def test_audit_report_always_has_overview_fields(tmp_path):
    audit = PredictionAudit(str(tmp_path))
    report = audit.generate_report()
    assert "总记录数" in report and "已验证数" in report
    audit.record_prediction({"symbol": "X", "prediction": 1, "direction": "看涨"})
    assert "总记录数" in audit.generate_report()


# ---------- 通知 ----------

def test_notifier_send_webhook_without_config_is_false():
    assert SignalNotifier({"notification": {}}).send_webhook({}) is False


def test_notifier_send_webhook_posts(monkeypatch):
    sent = {}

    def fake_send(self, predictions, summary):
        sent["summary"] = summary
        return True

    monkeypatch.setattr(SignalNotifier, "_send_webhook", fake_send)
    n = SignalNotifier({"notification": {"webhook_url": "http://example.invalid/hook"}})
    assert n.send_webhook({"hello": "world"}) is True
    assert "hello" in sent["summary"]
    assert n.send_webhook([{"symbol": "A", "predictions": {}}]) is True


# ---------- 调度 ----------

def test_scheduler_should_run_logic():
    cfg = {"scheduler": {"frequency": "weekly", "time": "23:00", "day": "sun"}}
    sched = RetrainScheduler(cfg)

    monday_noon = datetime(2026, 9, 7, 12, 0)      # 周一，未到 23:00
    assert sched._should_run(monday_noon) is False

    monday_night = datetime(2026, 9, 7, 23, 30)
    assert sched._should_run(monday_night) is True   # 从未运行过

    sched.last_run = monday_night
    assert sched._should_run(datetime(2026, 9, 8, 23, 30)) is False   # 同一 ISO 周
    assert sched._should_run(datetime(2026, 9, 14, 23, 30)) is True   # 下一周


def test_scheduler_daily_should_run():
    sched = RetrainScheduler({"scheduler": {"frequency": "daily", "time": "08:00"}})
    assert sched._should_run(datetime(2026, 9, 7, 7, 0)) is False
    assert sched._should_run(datetime(2026, 9, 7, 9, 0)) is True
    sched.last_run = datetime(2026, 9, 7, 9, 0)
    assert sched._should_run(datetime(2026, 9, 7, 10, 0)) is False
    assert sched._should_run(datetime(2026, 9, 8, 9, 0)) is True


def test_scheduler_bad_time_does_not_raise():
    sched = RetrainScheduler({"scheduler": {"frequency": "daily", "time": "not-a-time"}})
    assert sched._should_run() is False


# ---------- 性能监控 ----------

def test_monitor_accepts_dir_and_plain_accuracy(tmp_path):
    mon = ModelPerformanceMonitor(str(tmp_path))
    for _ in range(3):
        mon.record("short_term", 0.9)
    for _ in range(3):
        mon.record("short_term", 0.4)
    assert mon.detect_drift("short_term", threshold=0.2) is True
    assert (tmp_path / "performance_history.json").exists()


# ---------- pandas 3.0 兼容 ----------

def test_no_legacy_fillna_method_in_sources():
    """仓库源码与测试不应再出现 pandas 3.0 已移除的 fillna(method=)。"""
    offenders = []
    targets = [
        p for p in list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "tests").glob("*.py"))
        if p.name != Path(__file__).name
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        if "fillna(method" in text or ".backfill(" in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []
