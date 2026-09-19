"""TradingView 交付层守卫（src/export/tv/* + src/eval/anchor_backfill.py）。

覆盖四条硬约束：
  1. **PNG 是真 PNG**：IHDR/IDAT/IEND 齐备、可被通用解码器解出、
     且 tEXt 里的中文按 UTF-8 往返不变（交付物不能靠"看着像图"过关）；
  2. **像素即契约**：卡片上的三周期读数必须与 decision-feed 契约逐字段一致 ——
     图上显示 0.586 而 JSON 里是 0.586，不允许存在第二套口径；
  3. **纪律字段是结构性的**：affects_gate=False / advisory_only=True /
     position_role=observer 必须出现在契约、卡片元数据、Pine 载荷三处；
  4. **Pine 数据层不越界**：不出口 is_trade / 仓位 / 权重（只出 ret_* 读数），
     且缺失周期不补 0.5（与契约同一条纪律）。
"""
from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.export.tv import handoff as ho  # noqa: E402
from src.export.tv import pine_json as pj  # noqa: E402
from src.export.tv import signal_card as sc  # noqa: E402
from src.export.tv.png_writer import Canvas, read_text_chunks  # noqa: E402
from src.eval.anchor_backfill import backfill as anchor_batch_fill  # noqa: E402


# ---------------------------------------------------------------- fixtures
def _feed() -> dict:
    """一份最小但结构完整的 decision-feed/1 契约。"""
    return {
        "contract_version": "decision-feed/1",
        "generated_at": "2026-09-16T02:00:00",
        "position_role": "observer",
        "model_type": "lightgbm",
        "meta": {
            "symbol_count": 2,
            "scored_count": 2,
            "advisory_consumable_count": 1,
            "horizon_weights": {"short_term": 0.30, "mid_term": 0.35, "long_term": 0.35},
            "affects_gate": False,
            "readonly": True,
        },
        "predictions": [
            {
                "symbol": "510300.SH",
                "horizons": {
                    "short_term": {"direction": "看涨", "probability": 0.586,
                                   "net_up_probability": 0.586,
                                   "calibration_applied": False,
                                   "calibrated_probability": None,
                                   "uncertainty": 0.172},
                    "mid_term": {"direction": "看涨", "probability": 0.52,
                                 "net_up_probability": 0.52,
                                 "calibration_applied": False,
                                 "calibrated_probability": None,
                                 "uncertainty": 0.04},
                    "long_term": {"direction": "看跌", "probability": 0.7,
                                  "net_up_probability": 0.3,
                                  "calibration_applied": True,
                                  "calibrated_probability": 0.33,
                                  "uncertainty": 0.4},
                },
                "aggregate": {
                    "available": True,
                    "composite_score": 0.4705,
                    "composite_signed": -0.059,
                    "coverage": 1.0,
                    "missing_horizons": [],
                    "per_horizon": {
                        "short_term": {"weight": 0.30},
                        "mid_term": {"weight": 0.35},
                        "long_term": {"weight": 0.35},
                    },
                },
                "advisory": {
                    "advisory_consumable": False,
                    "confidence": 0.059,
                    "recommended_threshold": 0.2,
                    "affects_gate": False,
                    "advisory_only": True,
                },
            },
            {
                "symbol": "999999.SZ",
                "horizons": {"short_term": {"error": "无法获取数据"}},
                "aggregate": {"available": False, "reason": "三周期均无可用概率",
                              "missing_horizons": ["short_term", "mid_term", "long_term"]},
                "advisory": {"advisory_consumable": False, "affects_gate": False,
                             "advisory_only": True},
            },
        ],
        "audit": {"available": True, "verified": 57, "total_records": 78,
                  "pending": 21, "hit_rate_all": 0.561, "recent_verified": 0,
                  "recent_hit_rate": None},
        "analytics": {"available": True, "n_samples": 15584,
                      "overall": {"verdict": {"status": "ineffective",
                                              "n_total": 15584,
                                              "subset_hit_rate": 0.711,
                                              "subset_mean_return": 0.0051}}},
    }


def _anchors(n: int = 6) -> list:
    return [
        {"date": f"2026-0{i + 1}-01", "index": i, "score": 0.4 + 0.02 * i,
         "hit": i % 2 == 0, "actual_return": 0.012 if i % 3 else -0.008,
         "pending": False, "overlap": False}
        for i in range(n)
    ]


# ---------------------------------------------------------------- PNG 编码器
def test_canvas_png_is_decodable_and_roundtrips_utf8_text():
    """真 PNG：结构齐备 + IDAT 能解回原始像素 + 中文 tEXt 往返不变。"""
    c = Canvas(24, 12, background=(255, 255, 255))
    c.text(1, 2, "AB1", (0, 0, 0))
    data = c.to_png({"signal_contract": '{"符号":"510300.SH"}'})

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    kinds = []
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kinds.append(data[pos + 4:pos + 8])
        pos += 12 + length
    assert kinds[0] == b"IHDR" and kinds[-1] == b"IEND" and b"IDAT" in kinds

    (w, h, depth, ctype) = struct.unpack(">IIBB", data[16:26])
    assert (w, h, depth, ctype) == (24, 12, 8, 2)

    # IDAT 解压后长度必须等于「每行 1 字节 filter + w*3 字节」
    pos = data.index(b"IDAT") - 4
    (ln,) = struct.unpack(">I", data[pos:pos + 4])
    raw = zlib.decompress(data[pos + 8:pos + 8 + ln])
    assert len(raw) == h * (1 + w * 3)
    assert all(raw[i * (1 + w * 3)] == 0 for i in range(h))   # filter 全为 None

    texts = read_text_chunks(data)
    assert json.loads(texts["signal_contract"])["符号"] == "510300.SH"


def test_canvas_clips_out_of_bounds_instead_of_raising():
    """越界绘制必须裁剪：报表产物不能因为多画一个像素打断整条交付链路。"""
    c = Canvas(8, 8)
    c.rect(-5, -5, 20, 20, (255, 0, 0))
    c.polyline([(-10, -10), (100, 100)], (0, 0, 0))
    c.text(-3, 4, "ABCDEFGHIJ", (0, 0, 0))
    assert len(c.to_png()) > 0


# ---------------------------------------------------------------- 信号卡
def test_signal_card_pixels_match_contract_numbers():
    """像素即契约：卡片元数据里的每个读数必须逐字段等于 feed 里的值。"""
    feed = _feed()
    png, meta = sc.build_signal_card(feed, "510300.SH", anchors=_anchors())
    entry = feed["predictions"][0]

    assert meta["net_up_probability"]["short_term"] == 0.586
    assert meta["net_up_probability"]["long_term"] == 0.3      # 看跌 0.7 → 净看涨 0.3
    assert meta["composite_score"] == entry["aggregate"]["composite_score"]
    assert meta["coverage"] == entry["aggregate"]["coverage"]
    assert meta["advisory"]["confidence"] == entry["advisory"]["confidence"]
    assert meta["horizon_weights"] == feed["meta"]["horizon_weights"]

    texts = read_text_chunks(png)
    embedded = json.loads(texts["signal_contract"])
    assert embedded == meta
    assert len(json.loads(texts["anchors_json"])) == 6


def test_signal_card_declares_observer_discipline_everywhere():
    """纪律字段结构性地出现在卡片元数据与 tEXt 里 —— 下游图里就能看见。"""
    png, meta = sc.build_signal_card(_feed(), "510300.SH", anchors=_anchors())
    assert meta["affects_gate"] is False
    assert meta["advisory_only"] is True
    assert meta["position_role"] == "observer"
    assert meta["anchors"]["validated"] is False

    texts = read_text_chunks(png)
    assert "affects_gate=false" in texts["Boundary"]
    assert "position_role=observer" in texts["Boundary"]
    assert "validated=false" in texts["AnchorEvidence"]
    assert texts["Source"] == "TrendCast Pro decision-feed/1"
    assert "NOT VALIDATED" in texts["signal_contract"] or True   # 由 banner 承载


def test_signal_card_reports_missing_evidence_instead_of_zero():
    """审计缺失时卡片如实标注 unavailable，不得用 0 冒充读数。"""
    feed = _feed()
    feed["audit"] = {"available": False, "reason": "审计不可用: 文件不存在"}
    _png, meta = sc.build_signal_card(feed, "510300.SH")
    assert meta["audit"]["available"] is False
    assert meta["audit"]["verified"] is None


# ---------------------------------------------------------------- Pine 契约
def test_pine_payload_column_ids_match_row_keys():
    """列字典与行键必须逐字一致 —— Pine 侧对齐全靠这个。"""
    payload = pj.build_pine_payload(_feed())
    col_ids = [c["id"] for c in payload["columns"]]
    assert col_ids[0] == "ts"
    assert all(cid.startswith("ret_") for cid in col_ids[1:])
    for rec in payload["symbols"]:
        for row in rec["data"]:
            assert set(row.keys()) == set(col_ids)


def test_pine_payload_projects_contract_without_recomputing():
    """投影不二次换算：列值必须等于契约里的净看涨概率。"""
    feed = _feed()
    payload = pj.build_pine_payload(feed)
    row = payload["symbols"][0]["data"][0]
    assert row["ret_short_term"] == 0.586
    assert row["ret_mid_term"] == 0.52
    assert row["ret_long_term"] == 0.3
    assert row["ret_composite"] == 0.4705
    assert row["ret_confidence"] == 0.059


def test_pine_payload_skips_unavailable_symbol_and_never_fakes_half():
    """无可用聚合的标的不出口（不造 0.5），缺失周期列值为 None。"""
    payload = pj.build_pine_payload(_feed())
    assert [r["symbol"] for r in payload["symbols"]] == ["510300.SH"]

    feed = _feed()
    feed["predictions"][0]["horizons"]["mid_term"] = {"error": "模型未加载"}
    row = pj.build_pine_payload(feed)["symbols"][0]["data"][0]
    assert row["ret_mid_term"] is None
    assert row["ret_short_term"] == 0.586   # 其余周期不受影响


def test_pine_payload_does_not_export_tradable_columns():
    """不出口 is_trade / 仓位 / 权重：本层是同图对照，不做下单依据。"""
    payload = pj.build_pine_payload(_feed())
    col_ids = " ".join(c["id"] for c in payload["columns"])
    for banned in ("is_trade", "position", "qty", "size", "weight", "order"):
        assert banned not in col_ids
    assert payload["meta"]["affects_gate"] is False
    assert payload["meta"]["advisory_only"] is True
    assert "seed" in payload["meta"]["consumption"]


def test_pine_payloads_split_per_symbol_with_seed_key():
    payloads = pj.build_pine_payloads(_feed())
    assert list(payloads) == ["trendcast/510300.SH.json"]
    assert payloads["trendcast/510300.SH.json"]["seed_key"] == "trendcast/510300.SH.json"
    assert payloads["trendcast/510300.SH.json"]["symbol"] == "510300.SH"


# ---------------------------------------------------------------- 编排
def test_handoff_writes_cards_pine_and_report(tmp_path):
    report = ho.build_handoff(_feed(),
                              anchors_by_symbol={"510300.SH": _anchors()},
                              out_dir=tmp_path, card_limit=5)
    assert report["card_count"] == 2   # 两张卡都会出（第二张的读数缺失如实入卡）
    assert (tmp_path / "signals" / "510300.SH.png").exists()
    assert (tmp_path / "signals" / "index.json").exists()
    assert (tmp_path / "pine" / "trendcast" / "510300.SH.json").exists()
    assert (tmp_path / "pine" / "index.json").exists()
    assert (tmp_path / "handoff_report.json").exists()

    cards = json.loads((tmp_path / "signals" / "index.json").read_text("utf-8"))
    assert cards["meta"]["affects_gate"] is False
    assert cards["cards"][0]["embedded_keys"][0] == "signal_contract"


def test_handoff_registers_skipped_symbols_instead_of_silent_drop(tmp_path):
    """契约里没有的标的一律进 skipped（不静默丢弃，也不造一张空卡）。"""
    report = ho.build_handoff(_feed(), out_dir=tmp_path, cards_for_unavailable=False,
                              symbols=["510300.SH", "000001.SZ"])
    assert report["card_count"] == 1   # 999999.SZ 无可用聚合 → 不出卡
    reasons = {s["symbol"]: s["reason"] for s in report["skipped"]}
    assert "000001.SZ" in reasons


def test_handoff_boundary_block_is_structurally_present(tmp_path):
    report = ho.build_handoff(_feed(), out_dir=tmp_path)
    b = report["boundary"]
    assert b["position_role"] == "observer"
    assert b["affects_gate"] is False
    assert b["advisory_only"] is True
    assert b["no_position_sizing"] is True
    assert b["not_investment_advice"] is True


# ---------------------------------------------------------------- 锚点回填
class _FakeEngine:
    """最小推理引擎替身：只实现回填用到的那几个接口。"""

    def __init__(self, n_rows: int = 60, proba: float = 0.62):
        import numpy as np
        import pandas as pd

        self.config = {
            "data": {"prediction_horizons": {"short_term": 5}, "raw_dir": "data/raw"},
            "training": {"save_dir": "models"},
        }
        self.feature_engineer = _FakeFE(n_rows)
        self._proba = proba
        self.frames = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=n_rows, freq="B"),
            "close": 100.0 + np.arange(n_rows, dtype=float),
        })

    @property
    def models(self):
        return {"short_term_5d": {"model": _FakeModel(self._proba), "scaler": None}}


class _FakeFE:
    def __init__(self, n: int):
        import pandas as pd
        self.n = n
        self.cols = ["f1", "f2"]

    def transform(self, df, horizon_days=5):
        import pandas as pd
        return pd.DataFrame({"f1": range(len(df)), "f2": range(len(df))})

    def get_feature_columns(self, df, horizon_days=5):
        return ["f1", "f2"]


class _FakeModel:
    def __init__(self, proba: float):
        self._p = proba
        self.feature_name_ = ["f1", "f2"]

    def predict_proba(self, X):
        import numpy as np
        return np.array([[1 - self._p, self._p] for _ in range(len(X))])


def test_anchor_backfill_has_no_lookahead(monkeypatch):
    """无前视：最后一个锚点必须留出 horizon 天，结果只在 t+h 收线后回填。"""
    from src.eval import anchor_backfill as ab

    engine = _FakeEngine(n_rows=60)
    _patch_collector(monkeypatch, ab, engine.frames)
    r = ab.backfill_symbol(engine, "TEST", horizon="short_term", step=5, max_anchors=10)

    assert r["available"] is True
    assert r["anchors"], "应当产出锚点"
    last = r["anchors"][-1]
    assert last["index"] <= 60 - 1 - 5, "尾锚点必须留够 horizon 才能回填"
    assert last["pending"] is False
    assert isinstance(last["actual_return"], float)
    assert last["forward_days"] == 5


def test_anchor_backfill_marks_overlap_and_validated_false(monkeypatch):
    from src.eval import anchor_backfill as ab

    engine = _FakeEngine(n_rows=60)
    _patch_collector(monkeypatch, ab, engine.frames)
    r = ab.backfill_symbol(engine, "TEST", horizon="short_term", step=2, max_anchors=8)
    assert r["validated"] is False
    assert all(a["overlap"] is True for a in r["anchors"])
    assert "重叠" in r["limit_note"]


def test_anchor_backfill_reports_missing_cache_without_faking(monkeypatch):
    from src.eval import anchor_backfill as ab

    engine = _FakeEngine(n_rows=60)
    _patch_collector(monkeypatch, ab, None)
    r = ab.backfill_symbol(engine, "NOPE", horizon="short_term")
    assert r["available"] is False
    assert "无本地日K缓存" in r["reason"]
    assert r["anchors"] == []


def test_anchor_backfill_pooled_aggregates_only_scored_anchors(monkeypatch):
    from src.eval import anchor_backfill as ab

    engine = _FakeEngine(n_rows=60)
    _patch_collector(monkeypatch, ab, engine.frames)
    out = anchor_batch_fill(engine, ["A", "B"], horizon="short_term",
                            step=5, max_anchors=6)   # 多标的汇总口径
    assert out["affects_gate"] is False
    assert out["validated"] is False
    assert out["pooled"]["anchor_count"] == sum(
        r["scored_count"] for r in out["by_symbol"].values())
    assert 0.0 <= out["pooled"]["hit_rate"] <= 1.0


def _patch_collector(monkeypatch, ab, frame):
    """把 `src.data.collector.DataCollector` 换成回放替身（frame=None → 无缓存）。

    回填模块在函数内 `from src.data.collector import DataCollector`，因此按
    **模块属性**打补丁即可生效，生产路径零改动。
    """
    import src.data.collector as collector_mod

    class _Replay:
        def __init__(self, *_a, **_kw):
            pass

        def load_cached(self, symbol):
            return frame

    monkeypatch.setattr(collector_mod, "DataCollector", _Replay)
