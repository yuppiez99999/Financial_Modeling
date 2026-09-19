"""推理期特征**按名对齐**守卫（src/inference/predictor._align_by_feature_names）。

这组测试针对一个真缺陷：原实现只有 `_align_features` 的**位置截断**，
在「特征列数与训练时相同、但列序/列集合不同」时既不报警也不报错 ——
它会把 A 列的值安静地喂给期望 B 列的模型，产出一个看似完全正常的概率。
`scripts/evaluate_models.build_supervised` 的列集合与推理期
`FeatureEngineer.get_feature_columns` 并不逐字相同，正是触发路径。

修法：模型记录 `feature_name_` 时**按名取列并按模型顺序排列**；
缺列以 0 填充并 WARNING 留痕（保持列数，不改其它列语义）。

2026-09-16 追加（本次修复）：按名对齐的**调用契约**被误用过 —— 有调用方把
**值矩阵**当列名传进来，于是「用小数去匹配模型特征名」全部判成缺失，
整体 0 填充，**不报错地**产出常数概率。因此：
  - `_align_by_feature_names(feature_names, expected_names, row)` 现在要求
    ``feature_names`` 是字符串列名序列、``row`` 是与它逐位置对应的一行值；
  - **全列缺失**时返回 None（放弃按名对齐）而不是 0 填充，交由调用方回落
    位置对齐 —— 位置对齐在「训练/推理列集合逐字相同」时才是正确语义；
  - 训练侧把 ``feature_cols`` 写进产物并同步为模型 ``feature_name_``
    （`_sync_model_feature_names`），让按名对齐真正有依据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import (  # noqa: E402
    _align_by_feature_names,
    _align_features,
    _select_by_names,
    _sync_model_feature_names,
)


def test_reorders_columns_to_model_order():
    """列集合相同但顺序不同：必须按名重排，而不是让位置说话。"""
    cols = ["b", "a", "c"]
    values = np.array([20.0, 10.0, 30.0])        # b=20, a=10, c=30
    out = _align_by_feature_names(cols, ["a", "b", "c"], values)
    assert out.tolist() == [[10.0, 20.0, 30.0]]


def test_selects_subset_and_drops_extra_columns():
    """推理多出内部列（_symbol/_fwd_ret 等）时必须只取模型要的列。"""
    cols = ["a", "_symbol", "b", "_fwd_ret"]
    values = np.array([1.0, 999.0, 2.0, 888.0])
    out = _align_by_feature_names(cols, ["a", "b"], values)
    assert out.tolist() == [[1.0, 2.0]]


def test_missing_column_is_zero_filled_not_positionally_misaligned():
    cols = ["a", "c"]
    values = np.array([1.0, 3.0])
    out = _align_by_feature_names(cols, ["a", "b", "c"], values)
    # b 缺失 → 0；a / c 仍落在正确位置（这正是位置截断做不到的）
    assert out.tolist() == [[1.0, 0.0, 3.0]]


def test_returns_none_when_model_has_no_feature_names():
    """模型未记录特征名 → 返回 None，交由调用方回落到既有位置逻辑。"""
    assert _align_by_feature_names(["a"], [], np.array([1.0])) is None
    assert _align_by_feature_names(["a"], None, np.array([1.0])) is None


def test_returns_none_when_column_names_are_not_strings():
    """列名不是字符串（例如调用方误传值矩阵）→ 返回 None，绝不猜。"""
    assert _align_by_feature_names(None, ["a"], np.array([1.0])) is None
    assert _align_by_feature_names([1, 2], ["a", "b"], np.array([1.0, 2.0])) is None
    assert _align_by_feature_names([], ["a"], np.array([1.0])) is None


def test_returns_none_when_all_columns_missing_instead_of_zero_filling():
    """**全列缺失 = 列名体系对不上**：放弃按名对齐，不要 0 填充造假读数。

    这是 2026-09-16 实测真缺陷的守卫：模型只记 `Column_0..N` 占位名时，
    按名对齐会判「全列缺失」并整体 0 填充 —— 产出一条与任何标的、任何日期
    都无关的常数概率（回填锚点序列置信度恒为 0.0303756），**不报错**。
    """
    out = _align_by_feature_names(
        ["ma_5", "rsi"], ["Column_0", "Column_1"], np.array([1.0, 2.0]))
    assert out is None


def test_select_by_names_reorders_from_dataframe():
    """回填路径：按模型特征名从特征表取行，顺序跟随模型而非 DataFrame。"""
    import pandas as pd

    feats = pd.DataFrame({"b": [20.0, 200.0], "a": [10.0, 100.0], "c": [30.0, 300.0]})
    assert _select_by_names(feats, 0, ["a", "b", "c"]).tolist() == [[10.0, 20.0, 30.0]]
    assert _select_by_names(feats, 1, ["c", "a"]).tolist() == [[300.0, 100.0]]


def test_select_by_names_refuses_all_missing_columns():
    """与推理侧同一条纪律：全列对不上 → None（由调用方回落位置对齐）。"""
    import pandas as pd

    feats = pd.DataFrame({"ma_5": [1.0], "rsi": [2.0]})
    assert _select_by_names(feats, 0, ["Column_0", "Column_1"]) is None
    assert _select_by_names(feats, 0, None) is None
    assert _select_by_names(feats, 0, []) is None


def test_sync_model_feature_names_writes_through_booster():
    """老产物只有 Column_N 占位名时，加载侧经 booster 写回语义特征名。"""

    class FakeBooster:
        def __init__(self):
            self.feature_name = lambda: ["Column_0", "Column_1"]

    class FakeModel:
        def __init__(self):
            self.booster_ = FakeBooster()

        @property
        def feature_name_(self):
            return self.booster_.feature_name()

    m = FakeModel()
    assert _sync_model_feature_names(m, ["ma_5", "rsi"]) is True
    assert list(m.feature_name_) == ["ma_5", "rsi"]
    # 幂等：已经一致时直接返回 True
    assert _sync_model_feature_names(m, ["ma_5", "rsi"]) is True
    # 无特征名 / 非 LGBM 模型 → False（调用方回落位置对齐）
    assert _sync_model_feature_names(m, None) is False
    assert _sync_model_feature_names(object(), ["a"]) is False


def test_pos_align_still_used_as_fallback():
    """既有位置对齐仍是回落路径：列数一致时原样返回。"""

    class FakeModel:
        n_features_in_ = 2

    x = np.array([[1.0, 2.0]])
    assert _align_features(x, FakeModel()) is x
