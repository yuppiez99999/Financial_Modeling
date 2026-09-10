"""Q1 排期：模型监控报表（ModelMonitor / HealthReport）测试。

全部离线运行；重点验证 fail-soft、漂移判定与序列化稳定性。
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.prediction_audit import PredictionAudit
from src.monitor.health_report import HealthReport, ModelMonitor, render_markdown
from src.train.adaptive_learner import ModelPerformanceMonitor


@pytest.fixture()
def cfg(tmp_path):
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["audit"] = {"dir": str(tmp_path / "audit")}
    c["training"]["save_dir"] = str(tmp_path / "models")
    c["data"]["raw_dir"] = str(tmp_path / "raw")
    c["data"]["macro"] = {"source": ["local"], "dir": str(tmp_path / "macro")}
    # 隔离到 tmp：IC 趋势章节读 reports/ 下的报告，
    # 否则本地残留的 reports/ic_trend.json 会让「全健康」前提失效
    c["ic_trend"] = {"report_dir": str(tmp_path / "reports")}
    c["strategy_gate"] = {"report_dir": str(tmp_path / "reports")}
    c["report"] = {"output_dir": str(tmp_path / "reports")}
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return c


def _seed_models(cfg, horizon_keys=("short_term_5d", "mid_term_10d", "long_term_20d")):
    save_dir = Path(cfg["training"]["save_dir"])
    for key in horizon_keys:
        (save_dir / f"lightgbm_{key}.pkl").write_bytes(b"x" * 100)


def _seed_audit(cfg, n=20, hits=15, horizon="short_term", age_days=0):
    """直接写入审计 JSONL（绕开 record_prediction 的时间戳 ID 冲突）。"""
    audit = PredictionAudit(cfg["audit"]["dir"])
    ts = (datetime.now() - timedelta(days=age_days)).isoformat()
    records = []
    for i in range(n):
        records.append({
            "id": f"{horizon}_{age_days}_{i}",
            "timestamp": ts,
            "symbol": f"S{i}.SH",
            "horizon": horizon,
            "horizon_days": 5,
            "prediction": 1,
            "direction": "看涨",
            "probability": 0.6,
            "confidence": 0.6,
            "verified": True,
            "hit": i < hits,
        })
    existing = audit.record_file.read_text(encoding="utf-8") if audit.record_file.exists() else ""
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    audit.record_file.write_text(existing + body + "\n", encoding="utf-8")


# ---------- 基础结构 ----------

def test_collect_returns_health_report(cfg):
    report = ModelMonitor(cfg).collect()
    assert isinstance(report, HealthReport)
    assert report.generated_at
    assert report.status in ("ok", "warning", "critical")
    payload = report.to_dict()
    for key in ("audit", "adaptive", "data_sources", "models", "issues"):
        assert key in payload


def test_fail_soft_when_subsystems_missing(cfg):
    """全部子系统不可用时仍能产出报表，不抛异常。"""
    payload = ModelMonitor(cfg).collect().to_dict()
    assert payload["audit"]["available"] is True          # 空审计目录也是可用状态
    assert payload["audit"]["total_records"] == 0
    assert payload["adaptive"]["available"] is False
    assert payload["models"]["count"] == 0
    assert payload["status"] == "critical"                # 缺模型 + 无宏观 + 无行情


def test_status_ok_when_all_healthy(cfg):
    _seed_models(cfg)
    _seed_audit(cfg, n=20, hits=15)
    # 宏观可用
    from src.data.macro_client import MacroClient

    client = MacroClient(cfg)
    for ind in ("cpi", "pmi", "gdp", "m2", "lpr"):
        client.seed_from_dict(ind, {"2024-01-01": 1.0, "2024-02-01": 1.1})
    # 行情缓存
    raw = Path(cfg["data"]["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2024-01-01"], "close": [1.0]}).to_csv(raw / "X.SH.csv", index=False)

    payload = ModelMonitor(cfg).collect().to_dict()
    assert payload["status"] == "ok"
    assert payload["issues"] == []
    assert payload["audit"]["hit_rate"] == pytest.approx(0.75)
    assert payload["models"]["missing"] == []


# ---------- 审计章节 ----------

def test_audit_hit_rate_and_by_horizon(cfg):
    _seed_audit(cfg, n=10, hits=4, horizon="short_term")
    _seed_audit(cfg, n=10, hits=8, horizon="mid_term")
    audit = ModelMonitor(cfg).collect().to_dict()["audit"]
    assert audit["verified"] == 20
    assert audit["hits"] == 12
    assert audit["hit_rate"] == pytest.approx(0.6)
    assert audit["by_horizon"]["short_term"]["hit_rate"] == pytest.approx(0.4)
    assert audit["by_horizon"]["mid_term"]["hit_rate"] == pytest.approx(0.8)


def test_audit_recent_30d_window(cfg):
    _seed_audit(cfg, n=10, hits=10, age_days=0)
    _seed_audit(cfg, n=10, hits=0, age_days=60)
    audit = ModelMonitor(cfg).collect().to_dict()["audit"]
    assert audit["verified"] == 20
    assert audit["recent_30d_verified"] == 10
    assert audit["recent_30d_hit_rate"] == pytest.approx(1.0)


def test_drift_requires_min_verified(cfg):
    _seed_audit(cfg, n=5, hits=0)          # 命中率 0 但样本 < 10
    audit = ModelMonitor(cfg, min_verified=10).collect().to_dict()["audit"]
    assert audit["drift"] is False

    _seed_audit(cfg, n=20, hits=2)         # 命中率 10%，样本 >= 10
    audit = ModelMonitor(cfg, min_verified=10).collect().to_dict()["audit"]
    assert audit["drift"] is True
    issues = ModelMonitor(cfg, min_verified=10).collect().to_dict()["issues"]
    assert any("命中率" in i for i in issues)


def test_drift_threshold_configurable(cfg):
    _seed_audit(cfg, n=20, hits=8)          # 40%
    assert ModelMonitor(cfg, drift_threshold=0.5).collect().to_dict()["audit"]["drift"] is True
    assert ModelMonitor(cfg, drift_threshold=0.3).collect().to_dict()["audit"]["drift"] is False


# ---------- 自适应章节 ----------

def test_adaptive_section_reads_history(cfg):
    mon = ModelPerformanceMonitor(str(Path(cfg["training"]["save_dir"]) / "monitor"))
    mon.record_performance("short_term", {"accuracy": 0.6})
    mon.record_performance("short_term", {"accuracy": 0.7})
    mon.record_performance("mid_term", {"accuracy": 0.5})

    adaptive = ModelMonitor(cfg).collect().to_dict()["adaptive"]
    assert adaptive["available"] is True
    assert adaptive["total_records"] == 3
    assert adaptive["by_horizon"]["short_term"]["records"] == 2
    assert adaptive["by_horizon"]["short_term"]["avg_accuracy"] == pytest.approx(0.65)


def test_adaptive_empty_history(cfg):
    adaptive = ModelMonitor(cfg).collect().to_dict()["adaptive"]
    assert adaptive["available"] is False
    assert adaptive["reason"] == "no_history"


def test_adaptive_corrupt_history_is_soft(cfg):
    path = Path(cfg["training"]["save_dir"]) / "monitor"
    path.mkdir(parents=True, exist_ok=True)
    (path / "performance_history.json").write_text("{not json", encoding="utf-8")
    adaptive = ModelMonitor(cfg).collect().to_dict()["adaptive"]
    assert adaptive["available"] is False
    assert "error" in adaptive


# ---------- 数据源 / 模型章节 ----------

def test_market_cache_counted(cfg):
    raw = Path(cfg["data"]["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        pd.DataFrame({"date": ["2024-01-01"], "close": [1.0]}).to_csv(raw / f"S{i}.SH.csv", index=False)
    cache = ModelMonitor(cfg).collect().to_dict()["data_sources"]["market_cache"]
    assert cache["cached_symbols"] == 3
    assert cache["latest_update"]


def test_models_missing_detected(cfg):
    _seed_models(cfg, horizon_keys=("short_term_5d",))
    models = ModelMonitor(cfg).collect().to_dict()["models"]
    assert models["count"] == 1
    assert set(models["missing"]) == {"lightgbm_mid_term_10d.pkl", "lightgbm_long_term_20d.pkl"}
    assert models["files"][0]["size_kb"] > 0


# ---------- 渲染 / 序列化 ----------

def test_to_json_roundtrip(cfg):
    report = ModelMonitor(cfg).collect()
    parsed = json.loads(report.to_json())
    assert parsed["generated_at"] == report.generated_at
    assert parsed["status"] == report.status


def test_markdown_has_all_sections(cfg):
    _seed_models(cfg)
    _seed_audit(cfg, n=20, hits=15)
    md = ModelMonitor(cfg).collect().to_markdown()
    for section in ("# TrendCast Pro · 模型监控报表", "## 预测审计", "## 自适应学习",
                    "## 数据源健康", "## 模型产物"):
        assert section in md
    assert "不构成投资建议" in md


def test_markdown_renders_issues(cfg):
    payload = ModelMonitor(cfg).collect().to_dict()
    assert payload["issues"]
    md = render_markdown(payload)
    assert "## 待处理事项" in md


def test_save_markdown_and_json(tmp_path, cfg):
    report = ModelMonitor(cfg).collect()
    md_path = report.save(tmp_path / "r.md")
    json_path = report.save(tmp_path / "r.json")
    assert md_path.exists() and json_path.exists()
    assert md_path.read_text(encoding="utf-8").startswith("# TrendCast Pro")
    json.loads(json_path.read_text(encoding="utf-8"))


# ---------- CLI / API ----------

def test_cli_monitor_command(capsys, tmp_path, monkeypatch):
    import main as main_cli

    out = tmp_path / "monitor.md"
    cfg_path = tmp_path / "cfg.yaml"
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["report"]["output_dir"] = str(tmp_path / "reports")
    cfg_path.write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")

    ns = main_cli.build_parser().parse_args(["monitor", "--config", str(cfg_path)])
    assert ns.command == "monitor"

    result = main_cli.run_monitor(c, str(out))
    assert out.exists()
    assert result["status"] in ("ok", "warning", "critical")


def test_api_monitor_route_registered():
    from src.api import server

    if not server._HAS_FASTAPI:
        pytest.skip("未安装 FastAPI")
    app = server.create_app()
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/api/v1/monitor/report" in paths
