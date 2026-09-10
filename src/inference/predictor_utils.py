"""辅助：模型加载与占位包装函数"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import logging

logger = logging.getLogger(__name__)


def _wrap_pickle_model(obj: Any) -> dict[str, Any]:
    """确保 pickle 加载结果统一为 dict 包含 'model' 和 'scaler' 键。"""
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "scaler": None}


def load_placeholder(model_path: Path) -> dict[str, Any] | None:
    """尝试从本地 pickle 加载占位模型，返回包装后的 dict 或 None。"""
    if not model_path.exists():
        return None
    try:
        data = joblib.load(model_path)
        return _wrap_pickle_model(data)
    except Exception as e:
        logger.warning(f"加载占位模型失败 {model_path}: {e}")
        return None


def try_load_timesfm_or_instance(save_dir: Path, model_key: str, instantiate_fn):
    """优先从 save_dir 加载 timesfm_{model_key}.pkl 占位文件，否则使用 instantiate_fn() 创建实例。

    instantiate_fn 应返回一个 TimesFMFinancePredictor 实例或抛出异常。
    返回值：如果加载到占位 dict 则返回 dict，否则返回实例对象。
    """
    tfm_file = save_dir / f"timesfm_{model_key}.pkl"
    placeholder = load_placeholder(tfm_file)
    if placeholder is not None:
        logger.info(f"使用占位 TimesFM: {tfm_file}")
        return placeholder

    try:
        inst = instantiate_fn()
        logger.info(f"实例化 TimesFM predictor for {model_key}")
        return inst
    except Exception as e:
        logger.error(f"实例化 TimesFM predictor 失败: {e}")
        return None


def load_ensemble_entry(save_dir: Path, model_key: str, lightgbm_file: Path, instantiate_timesfm_fn):
    """构建 ensemble 的 entry 字典，包含 lightgbm 与 timesfm 两个分量（可能为 None）。"""
    entry = {"lightgbm": None, "timesfm": None}

    # LightGBM 部分
    if lightgbm_file.exists():
        try:
            lgb = joblib.load(lightgbm_file)
            # 确保包装为 {'model':..., 'scaler':...}
            if isinstance(lgb, dict) and "model" in lgb:
                entry["lightgbm"] = lgb
            else:
                entry["lightgbm"] = _wrap_pickle_model(lgb)
        except Exception as e:
            logger.warning(f"加载 LightGBM ensemble 部分失败: {e}")

    # TimesFM 部分（占位优先）
    tfm = try_load_timesfm_or_instance(save_dir, model_key, instantiate_timesfm_fn)
    if tfm is not None:
        entry["timesfm"] = tfm

    return entry
