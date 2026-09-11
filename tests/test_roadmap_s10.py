"""S10 多周期口径探索扫描测试：算得对 / 不给假结论 / 不动门禁。

全部离线运行（不触网、不依赖已训练模型、不依赖 config 全局状态）。
覆盖点：
  - 候选周期归一化：去重 / 升序 / 非法值丢弃（不静默修正、不整条崩掉）；
  - 现行口径标注：``is_current`` 只对 ``data.prediction_horizons`` 的周期为真；
  - **不改门禁**：``affects_gate`` 恒为 False，``strategy_gate`` 结论逐字段不变；
  - fail-close：样本不足 → ``available=false`` + reason，**绝不给结论**；
  - 覆盖度：周期长于数据长度 → 明确不可用（不硬凑半截窗口）；
  - 结论文案：现行未过 + 候选过关 → 明说"不是自动解锁"；都不行 → 明说换周期无用；
  - 消费口：落盘 / 读取 / 缺失不臆测；监控报表两分支；API 两分支；
  - 配置段默认值与 CLI 契约。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.horizon_scan import (  # noqa: E402
    DEFAULT_CANDIDATES,
    REPORT_NAME,
    HorizonScanner,
    compare_with_current,
)


def _cfg(reports: Path | None = None, **scan_overrides) -> dict:
    scan = {"candidates": [5, 10, 20, 40, 60], "min_samples": 30}
    scan.update(scan_overrides)
    if reports is not None:
        scan["report_dir"] = str(reports)
    return {
        "strategy_gate": {"enabled": True, "min_ic": 0.03, "min_hit_rate": 0.52,
                          "min_samples": 30, "min_windows": 3,
                          "report_dir": str(reports) if reports else "reports"},
        "data": {"prediction_horizons": {"short_term": 5, "mid_term": 10, "long_term": 20}},
        "horizon_scan": scan,
    }


def _seq(n: int, strength: float, seed: int) -> dict:
    """合成序列：strength=0 → 纯噪声；越大 → 信号与收益越单调相关。"""
    rnd = random.Random(seed)
    scores, rets = [], []
    for _ in range(n):
        x = rnd.gauss(0, 1)
        scores.append(x)
        rets.append(x * strength + rnd.gauss(0, 1))
    return {"scores": scores, "returns": rets, "window_size": 20}


def _long_df(rows: int = 300):
    return list(range(rows))


# ---------------------------------------------------------------- 候选周期
def test_candidates_normalized_dedup_sorted():
    sc = HorizonScanner(_cfg(candidates=[60, 5, "20", 5, 40]))
    assert sc.candidates == [5, 20, 40, 60]


@pytest.mark.parametrize("bad", [0, -5, "abc", None, [], {}])
def test_candidates_drop_invalid_without_crash(bad):
    """非法候选值丢弃而不是抛错：诊断工具不能因一个配置笔误整条挂掉。"""
    sc = HorizonScanner(_cfg(candidates=[5, bad, 20]))
    assert 5 in sc.candidates and 20 in sc.candidates
    assert all(isinstance(d, int) and d > 0 for d in sc.candidates)


def test_candidates_fall_back_to_default_when_unset():
    cfg = _cfg()
    cfg["horizon_scan"].pop("candidates")
    assert HorizonScanner(cfg).candidates == sorted(DEFAULT_CANDIDATES)


def test_current_horizons_read_from_config():
    sc = HorizonScanner(_cfg())
    assert sc.current_days == {"short_term": 5, "mid_term": 10, "long_term": 20}


# ---------------------------------------------------------------- 判定
def test_is_current_flag_only_for_config_horizons():
    sc = HorizonScanner(_cfg())
    seqs = {str(d): _seq(400, 0.5, d) for d in (5, 20, 40, 60)}
    res = sc.decide(seqs, symbol_count=3)
    assert res["pooled"]["5"]["is_current"] is True
    assert res["pooled"]["20"]["is_current"] is True
    assert res["pooled"]["40"]["is_current"] is False
    assert res["pooled"]["60"]["is_current"] is False


def test_insufficient_samples_fail_close():
    """样本不足必须 fail-close：available=false + reason，绝不给结论。"""
    sc = HorizonScanner(_cfg())
    res = sc.decide({"5": {"scores": [0.1] * 3, "returns": [0.1] * 3}}, symbol_count=1)
    entry = res["pooled"]["5"]
    assert entry["available"] is False
    assert entry["passed"] is False
    assert entry["reason"], "不可用必须给出原因，不得沉默"


def test_noise_never_passes():
    """纯噪声周期不得被判达标（防止扫描变成'总能挑出一条好看曲线'）。"""
    sc = HorizonScanner(_cfg())
    seqs = {str(d): _seq(600, 0.0, d) for d in (5, 20, 40, 60)}
    res = sc.decide(seqs, symbol_count=3)
    assert res["summary"]["passed_horizons"] == []


def test_strong_signal_passes_on_long_horizons():
    sc = HorizonScanner(_cfg())
    seqs = {"5": _seq(600, 0.0, 1), "40": _seq(600, 0.6, 2), "60": _seq(600, 0.7, 3)}
    res = sc.decide(seqs, symbol_count=3)
    passed = res["summary"]["passed_horizons"]
    assert 40 in passed and 60 in passed
    assert 5 not in passed


# ---------------------------------------------------------------- 不动门禁
def test_affects_gate_always_false():
    """扫描是证据不是解锁手段：任何输入下 affects_gate 都必须为 False。"""
    sc = HorizonScanner(_cfg())
    for seqs in (
        {},
        {"5": _seq(600, 0.0, 1)},
        {"40": _seq(600, 0.9, 2), "60": _seq(600, 0.9, 3)},
    ):
        res = sc.decide(seqs, symbol_count=3)
        assert res["affects_gate"] is False
        assert "不改变现行门禁口径" in res["gate_note"]


def test_compare_with_current_declares_not_unlock():
    """现行未过 + 候选过关：文案必须明确"不是自动解锁"。"""
    sc = HorizonScanner(_cfg())
    seqs = {"5": _seq(600, 0.0, 1), "10": _seq(600, 0.0, 2),
            "20": _seq(600, 0.0, 3), "40": _seq(600, 0.8, 4)}
    res = sc.decide(seqs, symbol_count=3)
    vs = compare_with_current(res)
    assert vs["available"] is True
    assert vs["current_passed"] is False
    assert 40 in vs["candidate_passed"]
    assert vs["affects_gate"] is False
    assert "放行依据" in vs["narrative"] or "人工决策" in vs["narrative"]


def test_compare_with_current_no_switch_when_current_passes():
    sc = HorizonScanner(_cfg())
    seqs = {"5": _seq(600, 0.7, 1), "40": _seq(600, 0.7, 2)}
    vs = compare_with_current(sc.decide(seqs, symbol_count=3))
    assert vs["current_passed"] is True
    assert "不敏感" in vs["narrative"] or "无切换必要" in vs["narrative"]


def test_compare_narrative_when_all_fail():
    sc = HorizonScanner(_cfg())
    seqs = {str(d): _seq(600, 0.0, d) for d in (5, 20, 40)}
    vs = compare_with_current(sc.decide(seqs, symbol_count=3))
    assert "换周期无用" in vs["narrative"] or "不在周期选择" in vs["narrative"]


def test_compare_unavailable_when_empty():
    vs = compare_with_current({"pooled": {}})
    assert vs["available"] is False
    assert vs["narrative"]


# ---------------------------------------------------------------- 覆盖度
def test_coverage_flags_horizon_longer_than_data():
    """60 日周期在 30 行数据上必然覆盖不住 → 明确不可用，不硬凑。"""
    sc = HorizonScanner(_cfg(candidates=[5, 60]))
    cov = sc.assess_coverage({"600519.SH": _long_df(30)})
    assert cov["per_horizon"]["5"]["available"] is True
    assert cov["per_horizon"]["60"]["available"] is False
    assert cov["per_horizon"]["60"]["reason"]


def test_coverage_handles_empty_and_bad_input():
    sc = HorizonScanner(_cfg())
    cov = sc.assess_coverage({})
    assert cov["max_symbol_length"] == 0
    assert all(not v["available"] for v in cov["per_horizon"].values())
    cov2 = sc.assess_coverage({"X": object()})  # 非序列输入不得中断扫描
    assert cov2["max_symbol_length"] == 0


# ---------------------------------------------------------------- 分池
def test_pools_reported_independently():
    sc = HorizonScanner(_cfg())
    # 大样本：纯噪声的 IC 会收敛到 0，稳定不过门禁（小样本下噪声可能偶然过线）
    pools = {
        "stock": {str(d): _seq(2000, 0.8, d) for d in (5, 40)},
        "etf": {str(d): _seq(2000, 0.0, d) for d in (5, 40)},
    }
    res = sc.decide({"5": _seq(2000, 0.0, 9)}, pools=pools, symbol_count=4)
    assert set(res["pools"]) == {"stock", "etf"}
    assert 40 in res["pools"]["stock"]["passed_horizons"]
    assert res["pools"]["etf"]["passed_horizons"] == []
    # 分池不得改变整池结论 / 门禁
    assert res["affects_gate"] is False


# ---------------------------------------------------------------- 落盘 / 读取
def test_save_and_load_roundtrip(tmp_path):
    sc = HorizonScanner(_cfg(tmp_path))
    res = sc.decide({"40": _seq(400, 0.8, 1)}, symbol_count=3)
    path = sc.save(res)
    assert path.name == REPORT_NAME and path.exists()
    loaded = HorizonScanner(_cfg(tmp_path)).load()
    assert loaded["available"] is True
    assert loaded["report_path"] == str(path)
    assert loaded["affects_gate"] is False


def test_load_missing_reports_unavailable(tmp_path):
    payload = HorizonScanner(_cfg(tmp_path)).load()
    assert payload["available"] is False
    assert payload["reason"] == "no_horizon_scan_report"
    assert "horizon-scan" in payload["hint"]


def test_load_corrupt_file_reports_error(tmp_path):
    (tmp_path / REPORT_NAME).write_text("{not json", encoding="utf-8")
    payload = HorizonScanner(_cfg(tmp_path)).load()
    assert payload["available"] is False
    assert payload.get("error")


def test_summarize_rows_shape(tmp_path):
    sc = HorizonScanner(_cfg(tmp_path))
    res = sc.decide({"5": _seq(400, 0.0, 1), "40": _seq(400, 0.8, 2)}, symbol_count=3)
    rows = sc.summarize_rows(res)
    assert [r["horizon_days"] for r in rows] == [5, 40]
    assert rows[0]["is_current"] is True
    assert rows[1]["is_current"] is False
    # 报表行不得塞原始序列
    assert "scores" not in rows[0] and "returns" not in rows[0]


# ---------------------------------------------------------------- 监控报表
def _monitor_cfg(tmp_path, *, write_report=True):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(reports)
    cfg["training"] = {"save_dir": str(tmp_path / "models")}
    cfg["data"]["raw_dir"] = str(tmp_path / "raw")
    cfg["report"] = {"output_dir": str(tmp_path)}
    cfg["ic_trend"] = {"report_dir": str(reports)}
    cfg["audit"] = {"dir": str(tmp_path / "audit")}
    if write_report:
        sc = HorizonScanner(cfg)
        res = sc.decide({"5": _seq(600, 0.0, 1), "40": _seq(600, 0.8, 2)}, symbol_count=3)
        sc.save(res)
    return cfg


def test_monitor_includes_horizon_scan_section(tmp_path):
    from src.monitor.health_report import ModelMonitor

    md = ModelMonitor(_monitor_cfg(tmp_path)).collect().to_markdown()
    assert "多周期口径探索（S10" in md
    assert "affects_gate=false" in md
    assert "40 日" in md


def test_monitor_missing_horizon_scan_is_graceful(tmp_path):
    from src.monitor.health_report import ModelMonitor

    report = ModelMonitor(_monitor_cfg(tmp_path, write_report=False)).collect()
    md = report.to_markdown()
    assert "多周期口径探索（S10" in md
    assert "不可用" in md
    assert report.to_dict()["horizon_scan"]["available"] is False


# ---------------------------------------------------------------- API
def test_api_horizon_scan_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from src.api import server

    server._config = _monitor_cfg(tmp_path)
    client = TestClient(server.app)
    resp = client.get("/api/v1/strategy/horizon-scan")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["affects_gate"] is False
    assert body["rows"]


def test_api_horizon_scan_missing_report(tmp_path):
    from fastapi.testclient import TestClient

    from src.api import server

    server._config = _monitor_cfg(tmp_path, write_report=False)
    client = TestClient(server.app)
    resp = client.get("/api/v1/strategy/horizon-scan")
    assert resp.status_code == 200
    assert resp.json()["available"] is False


# ---------------------------------------------------------------- 配置 / CLI
def test_config_has_horizon_scan_section():
    for name in ("config.yaml", "config_pro.yaml"):
        cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / name, encoding="utf-8"))
        assert "horizon_scan" in cfg, name
        scan = cfg["horizon_scan"]
        assert scan["candidates"] == list(DEFAULT_CANDIDATES), name
        # 扫描不得改动现行门禁口径
        horizons = cfg["data"]["prediction_horizons"]
        assert set(horizons.values()) == {5, 10, 20}, (
            f"{name}: 现行门禁周期不得被扫描改动"
        )


def test_cli_exposes_horizon_scan():
    import main as main_cli

    parser = main_cli.build_parser()
    args = parser.parse_args(["horizon-scan", "600519.SH", "510300.SH"])
    assert args.command == "horizon-scan"
    assert main_cli._cli_symbols(args) == ["600519.SH", "510300.SH"]
    args = parser.parse_args(["horizon-scan", "--no-pools"])
    assert args.no_pools is True


def test_cli_horizon_scan_days_flag():
    """候选周期走 --days，位置参数仍是标的列表（与仓库既有 CLI 约定一致）。"""
    import main as main_cli

    parser = main_cli.build_parser()
    args = parser.parse_args(["horizon-scan", "--days", "5,20,40"])
    assert args.days == "5,20,40"
    args = parser.parse_args(["horizon-scan", "600519.SH", "--days", "5,40"])
    assert main_cli._cli_symbols(args) == ["600519.SH"]
    assert args.days == "5,40"


def test_run_horizon_scan_no_data_reports_error(monkeypatch, capsys):
    """无行情数据时如实报错，不编造结论。"""
    import main as main_cli

    monkeypatch.setattr(
        "scripts.evaluate_models.load_market_data", lambda *a, **k: {}
    )
    out = main_cli.run_horizon_scan(_cfg(), symbols=["600519.SH"])
    assert "error" in out
    assert out["pooled"] == {}

# ---------------------------------------------------------------- 日报 / 周报
def test_daily_report_includes_horizon_scan_section(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    cfg = _monitor_cfg(tmp_path)
    gen = DailyReportGenerator(cfg)
    lines = gen._generate_horizon_scan_section()
    text = "\n".join(lines)
    assert "多周期口径探索（S10）" in text
    assert "40 日" in text
    assert "是否影响放行结论：否" in text
    assert "放行结论不变" in text


def test_daily_report_missing_horizon_scan_is_graceful(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    cfg = _monitor_cfg(tmp_path, write_report=False)
    text = "\n".join(DailyReportGenerator(cfg)._generate_horizon_scan_section())
    assert "多周期口径探索（S10）" in text
    assert "未扫描" in text

def test_cli_candidates_override_config(monkeypatch):
    """CLI 显式候选应覆盖配置（而非追加），否则"只扫 3 个周期"不可解释。"""
    import main as main_cli
    from src.eval.horizon_scan import HorizonScanner

    captured = {}

    class _FakeCollector:
        pass

    monkeypatch.setattr(
        "scripts.evaluate_models.load_market_data",
        lambda *a, **k: {"600519.SH": list(range(300))},
    )
    monkeypatch.setattr(
        "scripts.evaluate_models.build_supervised",
        lambda *a, **k: _EmptyDF(),
    )

    def _spy(self, *a, **k):
        captured["candidates"] = list(self.candidates)
        return {}

    monkeypatch.setattr(HorizonScanner, "decide", _spy)
    monkeypatch.setattr(HorizonScanner, "save", lambda self, p, path=None: path or "x")
    main_cli.run_horizon_scan(_cfg(), symbols=["600519.SH"], candidates=[5, 40],
                              include_pools=False)
    assert set(captured["candidates"]) == {5, 40}, captured


class _EmptyDF:
    """最小空 DataFrame 替身：只需 .empty 为 True。"""
    empty = True
