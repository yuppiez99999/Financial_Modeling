"""推理期概率校准器（S19 / H4，T19.3）：把校准层接在预测输出之后。

设计要点（与"离线校准"分开，这个模块只做**推理期套用**）：
  - 校准参数以 JSON 落盘（``models/probability_calibration_<horizon>.json``），
    由 ``python main.py calibration`` 生成；推理期只**读**，不重训；
  - 缺文件 = **不校准**（返回原概率并标注 ``applied=false``），绝不猜系数；
  - 只**新增字段**：``calibrated_probability`` / ``uncertainty``，
    原有 ``probability`` / ``confidence`` 逐字段不变（28 侧可渐进消费）；
  - 不确定性口径：``uncertainty = 1 - |2p_cal - 1|``（校准后概率距 0.5 的距离），
    区间宽度的语义与 ``confidence_curve`` 的 ``confidence_from_interval`` 一致。

无前视说明：推理期只用当日及之前数据算特征 + 已固化的校准参数，无未来信息。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DIR = "models"
FILENAME_TEMPLATE = "probability_calibration_{horizon}.json"


def calibration_path(horizon_name: str, directory: str = DEFAULT_DIR) -> Path:
    return Path(directory) / FILENAME_TEMPLATE.format(horizon=horizon_name)


def load_calibration(horizon_name: str,
                     directory: str = DEFAULT_DIR) -> Optional[Dict[str, Any]]:
    """读取校准参数；**缺失或损坏一律返回 None**（调用方据此不校准，不猜）。"""
    path = calibration_path(horizon_name, directory)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[calibrator] {path} 无法解析，跳过校准（原概率不变）: {e}")
        return None
    if not isinstance(data, dict) or data.get("method") not in ("isotonic", "platt"):
        return None
    return data


def calibrate_probability(proba: float,
                          calibration: Optional[Dict[str, Any]]) -> float:
    """套用校准参数（纯函数）。参数不可用 → 原样返回。"""
    if calibration is None:
        return float(proba)
    method = calibration.get("method")
    params = calibration.get("params") or {}
    p = min(max(float(proba), 1e-6), 1 - 1e-6)
    try:
        if method == "platt":
            import math

            coef = float(params.get("coef", 1.0))
            intercept = float(params.get("intercept", 0.0))
            z = math.log(p / (1 - p))
            out = 1.0 / (1.0 + math.exp(-(coef * z + intercept)))
            return min(max(out, 0.0), 1.0)
        if method == "isotonic":
            xs = [float(x) for x in (params.get("x_thresholds") or [])]
            ys = [float(y) for y in (params.get("y_thresholds") or [])]
            if not xs or len(xs) != len(ys):
                return float(proba)
            return min(max(_interp(p, xs, ys), 0.0), 1.0)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[calibrator] 校准套用失败，返回原概率: {e}")
    return float(proba)


def _interp(x: float, xs, ys) -> float:
    """保序回归的线性插值（两端 clip）——与 sklearn out_of_bounds='clip' 一致。"""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def uncertainty_from_probability(proba: float) -> float:
    """不确定性 = 1 − |2p − 1|：p 越接近 0.5 越不确定（与区间口径同族）。"""
    p = min(max(float(proba), 0.0), 1.0)
    return round(1.0 - abs(2.0 * p - 1.0), 6)


def enrich_prediction(prediction: Dict[str, Any],
                      horizon_name: str,
                      directory: str = DEFAULT_DIR) -> Dict[str, Any]:
    """给单条预测结果**追加**校准字段（不覆盖任何既有字段）。

    返回新 dict；``calibrated_probability`` 为 None 时表示未校准（如实标注）。
    """
    if not isinstance(prediction, dict) or "error" in prediction:
        return prediction
    out = dict(prediction)
    cal = load_calibration(horizon_name, directory)
    raw = prediction.get("probability")
    if cal is None or raw is None:
        out["calibrated_probability"] = None
        out["calibration_applied"] = False
        out["calibration_reason"] = ("未找到可用校准参数（先跑 main.py calibration）"
                                     if cal is None else "预测无概率字段")
        out["uncertainty"] = uncertainty_from_probability(raw) if raw is not None else None
        return out
    cp = calibrate_probability(float(raw), cal)
    out["calibrated_probability"] = round(cp, 4)
    out["calibration_applied"] = True
    out["calibration_method"] = cal.get("method")
    out["calibration_reason"] = ""
    out["uncertainty"] = uncertainty_from_probability(cp)
    return out
