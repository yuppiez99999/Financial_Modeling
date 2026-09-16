"""审计共享记账接口守卫（PredictionAudit.record_prediction_v2）。

下游（tradingview / 28 的 daily_runner 步骤 2.5）把预测送进来时用的是
**逐字段 dict**，与既有 `record_prediction(prediction_dict)` 的口径不同，
且无法标注来源。一旦字段对不上，就会被静默写成 ``symbol=None`` 的坏记录：
不报错、也无法在统计中分辨来源 —— 属于"检测器存在≠生效"的同族缺陷。

本接口的纪律：关键字段缺失**直接报错**（fail-loud），来源显式入库。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.prediction_audit import PredictionAudit  # noqa: E402


@pytest.fixture()
def audit(tmp_path):
    return PredictionAudit(audit_dir=str(tmp_path / "audit"))


def test_writes_record_with_source(audit):
    audit.record_prediction_v2({
        "symbol": "300308.SZ", "horizon": "short_term", "horizon_days": 5,
        "prediction": 1, "direction": "看涨", "probability": 0.63,
        "confidence": 0.26, "source": "trendcast_pro",
    })
    records = audit.load_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["symbol"] == "300308.SZ"
    assert rec["source"] == "trendcast_pro"
    assert rec["verified"] is False


def test_default_source_is_internal(audit):
    audit.record_prediction_v2({"symbol": "X.SH", "horizon": "mid_term"})
    assert audit.load_records()[0]["source"] == "internal"


def test_refuses_records_without_symbol_or_horizon(audit):
    with pytest.raises(ValueError, match="symbol"):
        audit.record_prediction_v2({"horizon": "short_term"})
    with pytest.raises(ValueError, match="horizon"):
        audit.record_prediction_v2({"symbol": "X.SH"})
    assert audit.load_records() == []      # 拒绝即不落盘，不留半条坏记录


def test_append_only_keeps_history(audit):
    for i in range(3):
        audit.record_prediction_v2({"symbol": f"S{i}.SH", "horizon": "short_term"})
    assert len(audit.load_records()) == 3


def test_roundtrip_is_valid_jsonl(audit):
    audit.record_prediction_v2({"symbol": "X.SH", "horizon": "short_term"})
    lines = audit.record_file.read_text(encoding="utf-8").strip().splitlines()
    assert all(json.loads(ln) for ln in lines)
