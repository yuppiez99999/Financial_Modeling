"""标签泄漏守卫（Issue #55 步骤②暴露的真缺陷）。

## 缺陷现场（2026-09-16 实测）

`get_feature_columns` 的排除规则此前**只认 `target_` 前缀**。
三重障碍法标签列名是 `label_tb_{h}d[_bin]` —— 一旦监督集里带该列
（`label_ab` / `model_improvement` 都会带），该函数会把**标签本身当特征返回**：

  - LightGBM 直接学到标签 → **AUC = 1.0、IC = 0.44、命中率 0.75**；
  - 假读数看起来"很漂亮"，且**不报错、不告警** —— 是最危险的那类缺陷。

守卫点：
1. `label_*` 列**绝不**进入特征集；
2. 疑似标签列（含 label/target/fwd_ret/future 语义）出现时**必须告警**；
3. 正常特征列**不受**影响（零回归）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import FeatureEngineer


@pytest.fixture()
def fe():
    return FeatureEngineer({"features": {"technical": {"ma_windows": [5, 10]}}})


def _frame() -> pd.DataFrame:
    n = 40
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="B"),
        "open": np.linspace(100, 120, n),
        "high": np.linspace(101, 121, n),
        "low": np.linspace(99, 119, n),
        "close": np.linspace(100, 120, n),
        "volume": np.full(n, 1000.0),
        "ma_5": np.linspace(100, 120, n),
        "rsi": np.full(n, 50.0),
        "target_5d": np.tile([0.0, 1.0], n // 2),
        "label_tb_5d": np.tile([1, -1], n // 2),
        "label_tb_5d_bin": np.tile([1.0, 0.0], n // 2),
    })


class TestLabelLeakGuard:
    def test_label_columns_excluded(self, fe):
        cols = fe.get_feature_columns(_frame(), 5)
        assert "label_tb_5d" not in cols
        assert "label_tb_5d_bin" not in cols

    def test_target_still_excluded(self, fe):
        cols = fe.get_feature_columns(_frame(), 5)
        assert "target_5d" not in cols

    def test_real_features_survive(self, fe):
        cols = fe.get_feature_columns(_frame(), 5)
        assert "ma_5" in cols
        assert "rsi" in cols

    def test_raw_ohlcv_excluded(self, fe):
        cols = fe.get_feature_columns(_frame(), 5)
        for c in ("date", "open", "high", "low", "close", "volume"):
            assert c not in cols

    def test_registry_declares_both_prefixes(self):
        # 新增标签列名必须挂在族里，否则静默泄漏会复发
        assert "target_" in FeatureEngineer.LABEL_COLUMN_PREFIXES
        assert "label_" in FeatureEngineer.LABEL_COLUMN_PREFIXES

    def test_suspicious_column_warns(self, fe, caplog):
        df = _frame().rename(columns={"label_tb_5d_bin": "tb_label_bin"})
        with caplog.at_level("WARNING"):
            fe.get_feature_columns(df, 5)
        assert any("疑似标签列" in r.message for r in caplog.records) or \
               all("label" not in c for c in fe.get_feature_columns(df, 5))

    def test_all_label_columns_stripped_from_matrix(self, fe):
        df = _frame()
        cols = fe.get_feature_columns(df, 5)
        assert not any(str(c).startswith(("label_", "target_")) for c in cols)
