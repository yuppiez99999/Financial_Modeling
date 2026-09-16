"""推理期特征**按名对齐**守卫（src/inference/predictor._align_by_feature_names）。

这组测试针对一个真缺陷：原实现只有 `_align_features` 的**位置截断**，
在「特征列数与训练时相同、但列序/列集合不同」时既不报警也不报错 ——
它会把 A 列的值安静地喂给期望 B 列的模型，产出一个看似完全正常的概率。
`scripts/evaluate_models.build_supervised` 的列集合与推理期
`FeatureEngineer.get_feature_columns` 并不逐字相同，正是触发路径。

修法：模型记录 `feature_name_` 时**按名取列并按模型顺序排列**；
缺列以 0 填充并 WARNING 留痕（保持列数，不改其它列语义）。
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
)


def test_reorders_columns_to_model_order():
    """列集合相同但顺序不同：必须按名重排，而不是让位置说话。"""
    cols = ["b", "a", "c"]
    values = np.array([[20.0, 10.0, 30.0]])      # b=20, a=10, c=30
    out = _align_by_feature_names(cols, ["a", "b", "c"], values)
    assert out.tolist() == [[10.0, 20.0, 30.0]]


def test_selects_subset_and_drops_extra_columns():
    """推理多出内部列（_symbol/_fwd_ret 等）时必须只取模型要的列。"""
    cols = ["a", "_symbol", "b", "_fwd_ret"]
    values = np.array([[1.0, 999.0, 2.0, 888.0]])
    out = _align_by_feature_names(cols, ["a", "b"], values)
    assert out.tolist() == [[1.0, 2.0]]


def test_missing_column_is_zero_filled_not_positionally_misaligned():
    cols = ["a", "c"]
    values = np.array([[1.0, 3.0]])
    out = _align_by_feature_names(cols, ["a", "b", "c"], values)
    # b 缺失 → 0；a / c 仍落在正确位置（这正是位置截断做不到的）
    assert out.tolist() == [[1.0, 0.0, 3.0]]


def test_returns_none_when_model_has_no_feature_names():
    """模型未记录特征名 → 返回 None，交由调用方回落到既有位置逻辑。"""
    assert _align_by_feature_names(["a"], [], np.array([[1.0]])) is None
    assert _align_by_feature_names(
        ["a"], None, np.array([[1.0]])) is None


def test_returns_none_for_multirow_input():
    """只处理单行推理输入；多行一律交回调用方（不做沉默假设）。"""
    assert _align_by_feature_names(
        ["a", "b"], ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]])) is None


def test_pos_align_still_used_as_fallback():
    """既有位置对齐仍是回落路径：列数一致时原样返回。"""

    class FakeModel:
        n_features_in_ = 2

    x = np.array([[1.0, 2.0]])
    assert _align_features(x, FakeModel()) is x
