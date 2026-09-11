"""S9 按资产类别分池测试：分类器 / 分池门禁 / 分池训练。

全部离线运行（不触网、不依赖已训练模型、不依赖 config 全局状态）。
覆盖点：
  - 分类器：ETF / 个股 / 期货 / 外汇 / 可转债 / 未识别 六类规则，可解释（返回命中规则）；
  - **不猜**：未知代码归 unknown，绝不硬塞进 stock；分池筛选空池不回退全池；
  - 分池门禁：fail-close（任一分池任一周期未过 → 该池 readonly）；
    样本不足 → available=false（不判定）；标的数不足 → 只记指标不判定；
  - **不改变整池结论**：默认 report_only，`affects_gate` 必须为 False；
    显式 per_class 时也不"取最好看的池子"，存在不可判定分池 → fail-close；
  - 对照结论：整池未过 + 分池过关 → 明确说"被拖后腿的分池稀释"，且声明不改门禁；
  - 分池训练：样本不足如实拒答（不静默回退整池），显式开启回退才 fallback；
  - 消费口：监控报表 / 日报 / 周报 / API 的缺失与正常分支；
  - 配置段与 CLI 命令存在性。
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

from src.eval.asset_class import (  # noqa: E402
    CLASS_CONVERTIBLE,
    CLASS_ETF,
    CLASS_FOREX,
    CLASS_FUTURES,
    CLASS_STOCK,
    CLASS_UNKNOWN,
    classify,
    classify_symbols,
    describe,
    group_symbols,
    resolve_pool,
    summarize_classification,
)
from src.eval.stratified import (  # noqa: E402
    REPORT_NAME,
    StratifiedEvaluator,
    classify_unknown_symbols,
    compare_with_pooled,
)


# ---------------------------------------------------------------- helpers
def _pairs(n: int, strength: float, seed: int):
    """构造 (score, return) 序列：strength 越大方向性越强。"""
    rnd = random.Random(seed)
    scores, returns = [], []
    for _ in range(n):
        s = rnd.gauss(0, 1)
        r = strength * s + rnd.gauss(0, 1)
        scores.append(s)
        returns.append(r)
    return scores, returns


def _sequences(passed: bool, n: int = 400, hdays: int = 5):
    """按「是否应过门禁」造一组序列。

    强方向性 → IC 与命中率都高（过线）；弱/反向 → 不过线。
    """
    strength = 0.35 if passed else -0.02
    scores, returns = _pairs(n, strength, seed=7 if passed else 13)
    return {
        "short_term": {"horizon_days": hdays, "scores": scores, "returns": returns,
                       "window_size": 20},
    }


def _data(symbols):
    return {s: {"placeholder": True} for s in symbols}


def _cfg(tmp_path: Path, **pool_gate):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    base = {
        "strategy_gate": {"enabled": True, "min_ic": 0.03, "min_hit_rate": 0.52,
                          "min_samples": 30, "min_windows": 3, "report_dir": str(reports)},
        "pool_gate": {"min_symbols": 2, "min_samples": 30, "report_dir": str(reports)},
    }
    base["pool_gate"].update(pool_gate)
    return base


# ---------------------------------------------------------------- 分类器
@pytest.mark.parametrize("symbol,expected", [
    ("600519.SH", CLASS_STOCK),
    ("000858.SZ", CLASS_STOCK),
    ("300308.SZ", CLASS_STOCK),
    ("688041.SH", CLASS_STOCK),
    ("510300.SH", CLASS_ETF),
    ("588080.SH", CLASS_ETF),
    ("512800.SH", CLASS_ETF),
    ("159915.SZ", CLASS_ETF),
    ("RB.SHF", CLASS_FUTURES),
    ("AU.SHF", CLASS_FUTURES),
    ("EURUSD.FXCM", CLASS_FOREX),
    ("123456.SZ", CLASS_CONVERTIBLE),
    ("ZZZ", CLASS_UNKNOWN),
    ("", CLASS_UNKNOWN),
])
def test_classify_known_rules(symbol, expected):
    assert classify(symbol)["asset_class"] == expected


def test_classify_is_case_and_space_insensitive():
    assert classify(" 600519.sh ")["asset_class"] == CLASS_STOCK
    assert classify("600519.sh")["asset_class"] == classify("600519.SH")["asset_class"]


def test_classify_returns_human_readable_rule():
    res = classify("510300.SH")
    assert res["rule"], "必须返回命中规则，否则分池口径不可解释"
    assert "51" in res["rule"] or "基金" in res["rule"]
    assert describe("510300.SH").startswith("510300.SH →")


def test_unknown_symbol_is_not_forced_into_stock():
    """未识别标的必须归 unknown —— 硬塞进 stock 会让 stock 分池的指标失去含义。"""
    assert classify("ZZZ")["asset_class"] == CLASS_UNKNOWN
    groups = group_symbols(["600519.SH", "ZZZ"])
    assert groups[CLASS_UNKNOWN] == ["ZZZ"]
    assert "ZZZ" not in groups[CLASS_STOCK]


def test_unknown_symbol_listed_for_manual_mapping():
    unknown = classify_unknown_symbols(["600519.SH", "510300.SH", "ZZZ", "yyy"])
    assert unknown == ["ZZZ", "YYY"]


def test_group_symbols_keeps_stable_order_and_drops_empty():
    groups = group_symbols(["510300.SH", "600519.SH", "159915.SZ"])
    assert list(groups.keys()) == [CLASS_STOCK, CLASS_ETF]
    assert groups[CLASS_STOCK] == ["600519.SH"]
    assert groups[CLASS_ETF] == ["510300.SH", "159915.SZ"]


def test_summarize_classification_shape():
    out = summarize_classification(["600519.SH", "510300.SH", "ZZZ"])
    assert out["total"] == 3
    assert out["counts"] == {CLASS_STOCK: 1, CLASS_ETF: 1, CLASS_UNKNOWN: 1}
    assert out["unknown"] == ["ZZZ"]
    assert len(out["items"]) == 3


def test_resolve_pool_empty_result_does_not_fall_back_to_all():
    data = {"600519.SH": 1, "000858.SZ": 2}
    assert resolve_pool(data, CLASS_ETF) == {}, "没有 ETF 就必须返回空，不能回退成全池"
    assert set(resolve_pool(data, CLASS_STOCK)) == {"600519.SH", "000858.SZ"}


def test_resolve_pool_respects_manual_override():
    data = {"MY.CODE": 1, "600519.SH": 2}
    out = resolve_pool(data, CLASS_STOCK, symbol_map={"MY.CODE": CLASS_STOCK})
    assert set(out) == {"MY.CODE", "600519.SH"}


# ---------------------------------------------------------------- 分池拆分
def test_split_groups_data_by_asset_class():
    ev = StratifiedEvaluator({"strategy_gate": {"report_dir": "reports"}})
    split = ev.split(_data(["600519.SH", "510300.SH", "159915.SZ"]))
    assert set(split) == {CLASS_STOCK, CLASS_ETF}
    assert split[CLASS_STOCK] == {"600519.SH": {"placeholder": True}}
    assert set(split[CLASS_ETF]) == {"510300.SH", "159915.SZ"}


# ---------------------------------------------------------------- 分池门禁
def test_pool_gate_passes_when_all_horizons_pass(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    res = ev.decide(_data(["600519.SH", "000858.SZ"]),
                    {CLASS_STOCK: _sequences(passed=True)})
    pool = res["pools"][CLASS_STOCK]
    assert pool["available"] is True
    assert pool["passed"] is True
    assert pool["state"] == "gated"
    assert res["passed_pools"] == [CLASS_STOCK]


def test_pool_gate_is_fail_close_on_any_failing_horizon(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    seq = _sequences(passed=True)
    seq["mid_term"] = {"horizon_days": 10, "scores": [-1.0] * 100,
                       "returns": [0.01] * 100, "window_size": 20}
    res = ev.decide(_data(["600519.SH", "000858.SZ"]), {CLASS_STOCK: seq})
    pool = res["pools"][CLASS_STOCK]
    assert pool["passed"] is False
    assert pool["state"] == "readonly"
    assert any("mid_term" in b for b in pool["blocked_by"])


def test_pool_gate_insufficient_samples_is_not_passed(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    seq = {"short_term": {"horizon_days": 5, "scores": [0.1, 0.2],
                          "returns": [0.01, -0.01], "window_size": None}}
    res = ev.decide(_data(["600519.SH", "000858.SZ"]), {CLASS_STOCK: seq})
    pool = res["pools"][CLASS_STOCK]
    assert pool["available"] is False
    assert pool["passed"] is False
    assert res["unavailable_pools"] == [CLASS_STOCK]
    assert "样本不足" in pool["reason"] or "无可用标的" in pool["reason"]


def test_pool_gate_min_symbols_only_records_metrics(tmp_path):
    """单标的池化 IC 不具备跨标的代表性 → 只记指标，不判定放行。"""
    ev = StratifiedEvaluator(_cfg(tmp_path, min_symbols=3))
    res = ev.decide(_data(["600519.SH"]), {CLASS_STOCK: _sequences(passed=True)})
    pool = res["pools"][CLASS_STOCK]
    assert pool["available"] is True
    assert pool["passed"] is False
    assert any("标的数" in b for b in pool["blocked_by"])


def test_pool_gate_empty_pool_never_passes(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    res = ev.decide({"600519.SH": {}}, {})
    pool = res["pools"][CLASS_STOCK]
    assert pool["passed"] is False
    assert pool["available"] is False


def test_pools_are_evaluated_independently(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    data = _data(["600519.SH", "000858.SZ", "510300.SH", "159915.SZ"])
    res = ev.decide(data, {
        CLASS_STOCK: _sequences(passed=True),
        CLASS_ETF: _sequences(passed=False),
    })
    assert res["pools"][CLASS_STOCK]["passed"] is True
    assert res["pools"][CLASS_ETF]["passed"] is False
    assert res["passed_pools"] == [CLASS_STOCK]
    assert res["failed_pools"] == [CLASS_ETF]


# ---------------------------------------------------------------- 不改门禁
def test_default_pool_scope_does_not_affect_gate(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    res = ev.decide(_data(["600519.SH", "000858.SZ"]), {CLASS_STOCK: _sequences(True)})
    assert res["affects_gate"] is False
    assert res["pool_scope"]["affects_gate"] is False
    assert res["pool_scope"]["state"] is None


def test_per_class_scope_only_allows_when_every_pool_passes(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["strategy_gate"]["pool_scope"] = {"enabled": True, "mode": "per_class"}
    ev = StratifiedEvaluator(cfg)
    data = _data(["600519.SH", "000858.SZ", "510300.SH", "159915.SZ"])
    all_pass = ev.decide(data, {CLASS_STOCK: _sequences(True), CLASS_ETF: _sequences(True)})
    assert all_pass["pool_scope"]["state"] == "gated"
    mixed = ev.decide(data, {CLASS_STOCK: _sequences(True), CLASS_ETF: _sequences(False)})
    assert mixed["pool_scope"]["state"] == "readonly"


def test_per_class_scope_fails_close_on_undecidable_pool(tmp_path):
    """存在不可判定分池 → fail-close，绝不"忽略它然后放行"。"""
    cfg = _cfg(tmp_path)
    cfg["strategy_gate"]["pool_scope"] = {"enabled": True, "mode": "per_class"}
    ev = StratifiedEvaluator(cfg)
    data = _data(["600519.SH", "000858.SZ", "510300.SH", "159915.SZ"])
    res = ev.decide(data, {CLASS_STOCK: _sequences(True), CLASS_ETF: {}})
    assert CLASS_ETF in res["unavailable_pools"]
    assert res["pool_scope"]["state"] == "readonly"


# ---------------------------------------------------------------- 对照
def test_compare_narrative_when_pooled_fails_but_pool_passes(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    res = ev.decide(_data(["600519.SH", "000858.SZ"]), {CLASS_STOCK: _sequences(True)})
    cmp = compare_with_pooled(res, {"all_passed": False, "horizons": {}})
    assert cmp["available"] is True
    assert cmp["pooled_passed"] is False
    assert "稀释" in cmp["narrative"]
    assert "不改变整池门禁判定" in cmp["narrative"]


def test_compare_narrative_when_nothing_passes(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    res = ev.decide(_data(["600519.SH", "000858.SZ"]), {CLASS_STOCK: _sequences(False)})
    cmp = compare_with_pooled(res, {"all_passed": False})
    assert "不是分池造成的" in cmp["narrative"]


def test_compare_handles_empty_pools():
    cmp = compare_with_pooled({"pools": {}}, None)
    assert cmp["available"] is False


# ---------------------------------------------------------------- 落盘 / 读取
def test_report_roundtrip(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    res = ev.decide(_data(["600519.SH", "000858.SZ"]), {CLASS_STOCK: _sequences(True)})
    path = ev.save(res)
    assert path.name == REPORT_NAME
    loaded = ev.load()
    assert loaded["available"] is True
    assert loaded["pools"][CLASS_STOCK]["passed"] is True


def test_load_missing_report_is_not_guessed(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    loaded = ev.load()
    assert loaded["available"] is False
    assert loaded["reason"] == "no_stratified_report"
    assert loaded["hint"]


def test_summarize_pools_rows_are_compact(tmp_path):
    ev = StratifiedEvaluator(_cfg(tmp_path))
    res = ev.decide(_data(["600519.SH", "000858.SZ"]), {CLASS_STOCK: _sequences(True)})
    rows = ev.summarize_pools(res)
    assert rows and rows[0]["asset_class"] == CLASS_STOCK
    assert "scores" not in json.dumps(rows), "报表行不得塞原始序列"


# ---------------------------------------------------------------- 分池训练
def _df(n: int, seed: int = 3):
    import pandas as pd

    rnd = random.Random(seed)
    price = 100.0
    rows = []
    for i in range(n):
        price *= 1 + rnd.gauss(0, 0.01)
        rows.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "open": price, "high": price * 1.01, "low": price * 0.99,
            "close": price, "volume": 1000 + i,
        })
    return pd.DataFrame(rows)


def _train_cfg(tmp_path, **over):
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8"))
    cfg["training"]["save_dir"] = str(tmp_path / "models")
    cfg["data"]["prediction_horizons"] = {"short_term": 5}
    cfg.setdefault("pool_train", {})
    cfg["pool_train"].update({"enabled": True, "min_samples": 50, **over})
    return cfg


def test_pool_train_plan_flags_insufficient_pools(tmp_path):
    from src.train.stratified_train import StratifiedTrainer

    trainer = StratifiedTrainer(_train_cfg(tmp_path, min_samples=200))
    plan = trainer.plan({"600519.SH": _df(30), "510300.SH": _df(100)})
    assert plan["pools"][CLASS_STOCK]["trainable"] is False
    assert plan["pools"][CLASS_ETF]["trainable"] is False
    assert all(p["reason"] for p in plan["pools"].values())


def test_pool_train_writes_manifest_and_models(tmp_path):
    from src.train.stratified_train import MANIFEST_NAME, StratifiedTrainer

    trainer = StratifiedTrainer(_train_cfg(tmp_path))
    manifest = trainer.train_pools(
        {"600519.SH": _df(300, 1), "000858.SZ": _df(300, 2), "510300.SH": _df(300, 3)},
        {"short_term": 5},
    )
    assert manifest["pools"][CLASS_STOCK]["status"] == "trained"
    assert manifest["pools"][CLASS_STOCK]["files"]
    assert (tmp_path / "models" / "pools" / MANIFEST_NAME).exists()


def test_pool_train_does_not_silently_fallback(tmp_path):
    """样本不足必须如实拒答 —— 静默回退会让"分层"变成空话。"""
    from src.train.stratified_train import StratifiedTrainer

    trainer = StratifiedTrainer(_train_cfg(tmp_path, min_samples=10_000))
    manifest = trainer.train_pools({"600519.SH": _df(60)}, {"short_term": 5})
    pool = manifest["pools"][CLASS_STOCK]
    assert pool["status"] == "skipped"
    assert pool.get("files") in (None, [])
    assert "insufficient" in pool["reason"] or "样本" in pool["reason"]


def test_pool_train_fallback_is_explicit_and_labelled(tmp_path):
    from src.train.stratified_train import StratifiedTrainer

    trainer = StratifiedTrainer(
        _train_cfg(tmp_path, min_samples=10_000, fallback_to_pooled=True)
    )
    manifest = trainer.train_pools({"600519.SH": _df(60)}, {"short_term": 5})
    pool = manifest["pools"][CLASS_STOCK]
    assert pool["status"] == "fallback"
    assert "fallback_to_pooled=true" in pool["reason"]


def test_resolve_pool_model_routes_by_class_and_marks_unknown(tmp_path):
    from src.train.stratified_train import StratifiedTrainer

    trainer = StratifiedTrainer(_train_cfg(tmp_path))
    trainer.train_pools(
        {"600519.SH": _df(300, 1), "510300.SH": _df(300, 2)}, {"short_term": 5}
    )
    stock = trainer.resolve_pool_model("600519.SH")
    etf = trainer.resolve_pool_model("510300.SH")
    assert stock["asset_class"] == CLASS_STOCK and "pool_stock" in stock["file"]
    assert etf["asset_class"] == CLASS_ETF and "pool_etf" in etf["file"]
    unknown = trainer.resolve_pool_model("ZZZ")
    assert unknown["asset_class"] == CLASS_UNKNOWN
    assert unknown["file"] is None and unknown["fallback"] is False


# ---------------------------------------------------------------- 消费口
def _monitor_cfg(tmp_path, *, write_report: bool = True, with_pools=True):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    cfg = {
        "strategy_gate": {"enabled": True, "min_ic": 0.03, "min_hit_rate": 0.52,
                          "report_dir": str(reports)},
        "pool_gate": {"min_symbols": 2, "min_samples": 30, "report_dir": str(reports)},
        "pool_train": {"enabled": True, "min_samples": 50,
                       "save_dir": str(tmp_path / "models")},
        "training": {"save_dir": str(tmp_path / "models")},
        "data": {"raw_dir": str(tmp_path / "raw")},
        "report": {"output_dir": str(tmp_path)},
        "ic_trend": {"report_dir": str(reports)},
        "audit": {"dir": str(tmp_path / "audit")},
    }
    if write_report:
        (reports / REPORT_NAME).write_text(json.dumps({
            "generated_at": "2026-01-01T00:00:00", "min_symbols": 2, "min_samples": 30,
            "affects_gate": False, "passed_pools": [CLASS_STOCK],
            "failed_pools": [CLASS_ETF], "unavailable_pools": [],
            "pools": {
                CLASS_STOCK: {"asset_class": CLASS_STOCK, "label": "个股",
                              "symbol_count": 2, "samples": 400, "available": True,
                              "passed": True, "state": "gated",
                              "horizons": {"short_term": {"passed": True}},
                              "ic": {"short_term": 0.05}, "hit_rate": {"short_term": 0.55}},
                CLASS_ETF: {"asset_class": CLASS_ETF, "label": "ETF / 场内基金",
                            "symbol_count": 2, "samples": 400, "available": True,
                            "passed": False, "state": "readonly",
                            "horizons": {"short_term": {"passed": False}},
                            "ic": {"short_term": -0.01}, "hit_rate": {"short_term": 0.49}},
            },
        }, ensure_ascii=False), encoding="utf-8")
        if with_pools:
            pools_dir = tmp_path / "models" / "pools"
            pools_dir.mkdir(parents=True, exist_ok=True)
            (pools_dir / "pool_manifest.json").write_text(json.dumps({
                "model_type": "lightgbm", "pools": {
                    CLASS_STOCK: {"asset_class": CLASS_STOCK, "label": "个股",
                                  "symbols": ["600519.SH"], "status": "trained",
                                  "files": ["a.pkl"], "horizons": {
                                      "short_term": {"status": "ok", "samples": 400}}},
                },
            }, ensure_ascii=False), encoding="utf-8")
    return cfg


def test_monitor_report_renders_pool_section(tmp_path):
    from src.monitor.health_report import ModelMonitor

    payload = ModelMonitor(_monitor_cfg(tmp_path)).collect().to_dict()
    md = ModelMonitor(_monitor_cfg(tmp_path)).collect().to_markdown()
    assert payload["pool_gate"]["available"] is True
    assert "分池评估" in md
    assert "个股" in md and "ETF" in md
    assert "report_only" in md or "仅作补充证据" in md


def test_monitor_report_flags_dilution_when_pooled_not_passed(tmp_path):
    from src.monitor.health_report import ModelMonitor

    cfg = _monitor_cfg(tmp_path)
    reports = Path(cfg["strategy_gate"]["report_dir"])
    (reports / "strategy_gate.json").write_text(
        json.dumps({"state": "readonly", "passed": False}), encoding="utf-8"
    )
    payload = ModelMonitor(cfg).collect().to_dict()
    assert any("分池已过关" in i for i in payload["issues"])


def test_monitor_report_pool_gate_missing_branch(tmp_path):
    from src.monitor.health_report import ModelMonitor

    cfg = _monitor_cfg(tmp_path, write_report=False)
    payload = ModelMonitor(cfg).collect().to_dict()
    md = ModelMonitor(cfg).collect().to_markdown()
    assert payload["pool_gate"]["available"] is False
    assert "不可用" in md


def test_daily_and_weekly_report_include_pool_section(tmp_path):
    from src.report.daily_report import DailyReportGenerator, WeeklyReportGenerator

    cfg = _monitor_cfg(tmp_path)
    cfg["features"] = {"sentiment_enabled": False}
    daily = DailyReportGenerator(cfg).generate_report(
        [{"symbol": "600519.SH", "predictions": {}}]
    )
    weekly = WeeklyReportGenerator(cfg).generate_report(
        [{"symbol": "600519.SH", "predictions": {}}]
    )
    for report in (daily, weekly):
        assert "分池评估（S9" in report
        assert "个股" in report


def test_daily_report_pool_section_handles_missing(tmp_path):
    from src.report.daily_report import DailyReportGenerator

    cfg = _monitor_cfg(tmp_path, write_report=False)
    cfg["features"] = {"sentiment_enabled": False}
    report = DailyReportGenerator(cfg).generate_report(
        [{"symbol": "600519.SH", "predictions": {}}]
    )
    assert "分池评估（S9" in report
    assert "未评估" in report


def test_api_pool_gate_and_asset_class(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import server

    cfg = _monitor_cfg(tmp_path)
    server._config = cfg
    client = TestClient(server.app)

    resp = client.get("/api/v1/strategy/pool-gate")
    assert resp.status_code == 200
    assert resp.json()["available"] is True

    resp = client.get("/api/v1/strategy/asset-class/510300.SH")
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset_class"] == CLASS_ETF
    assert body["label"]


def test_api_pool_gate_missing_report(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import server

    server._config = _monitor_cfg(tmp_path, write_report=False)
    client = TestClient(server.app)
    resp = client.get("/api/v1/strategy/pool-gate")
    assert resp.status_code == 200
    assert resp.json()["available"] is False


def test_api_pool_train_missing_manifest(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import server

    server._config = _monitor_cfg(tmp_path, with_pools=False)
    client = TestClient(server.app)
    resp = client.get("/api/v1/strategy/pool-train")
    assert resp.status_code == 200
    assert resp.json()["available"] is False


# ---------------------------------------------------------------- 配置 / CLI
def test_config_has_pool_sections_disabled_by_default():
    for name in ("config.yaml", "config_pro.yaml"):
        cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / name, encoding="utf-8"))
        assert "pool_gate" in cfg, name
        assert "pool_train" in cfg, name
        assert cfg["strategy_gate"]["pool_scope"]["enabled"] is False, (
            f"{name}: 分池默认不得改变放行结论"
        )
        assert cfg["strategy_gate"]["pool_scope"]["mode"] == "report_only"
        assert cfg["pool_train"]["fallback_to_pooled"] is False, (
            f"{name}: 默认不得静默回退整池模型"
        )


def test_cli_exposes_ic_pool_and_pool_train():
    import main as main_cli

    parser = main_cli.build_parser()
    args = parser.parse_args(["ic-pool", "600519.SH", "510300.SH"])
    assert args.command == "ic-pool"
    assert main_cli._cli_symbols(args) == ["600519.SH", "510300.SH"]
    args = parser.parse_args(["pool-train", "--symbols", "600519.SH,510300.SH"])
    assert args.command == "pool-train"
    assert main_cli._cli_symbols(args) == ["600519.SH", "510300.SH"]


def test_cli_symbols_helper_falls_back_to_none():
    import main as main_cli

    parser = main_cli.build_parser()
    args = parser.parse_args(["ic-pool"])
    assert main_cli._cli_symbols(args) is None


def test_sequences_builder_skips_empty_pools(tmp_path):
    from src.eval.stratified import build_sequences_for_pools

    calls = []

    def _builder(pool_data):
        calls.append(sorted(pool_data))
        return {"short_term": {"horizon_days": 5, "scores": [], "returns": []}}

    out = build_sequences_for_pools(
        {"600519.SH": {}}, {CLASS_STOCK: {"600519.SH": {}}, CLASS_ETF: {}}, _builder
    )
    assert calls == [["600519.SH"]], "空分池不得触发构造（也不得伪造序列）"
    assert CLASS_ETF not in out
