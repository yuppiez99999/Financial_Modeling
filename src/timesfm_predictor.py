# -*- coding: utf-8 -*-
"""
TimesFM 2.5 金融市场预测集成模块
=================================
增强 TrendCast Pro 的短期 (5d) 预测能力。
TimesFM 从 LightGBM 分类模型升级为点预测 + 概率区间。

核心能力：
- 零样本价格预测：对任意 A 股/期货/外汇标的直接预测
- 多步预测：1-20 天点预测 + q10~q90 分位数区间
- 信号生成：基于预测趋势自动产出 buy/hold/sell 信号
- 不确定性量化：输出预测不确定性（LightGBM 二分类不具备）
- Ensemble：与现有 LightGBM 模型加权融合

集成方式：
1. 独立的 TimesFMPredictor 类 → 可单独推理
2. 在 predictor.py / trainer.py 中添加 TimesFM 作为额外模型选项
3. 在 evaluator.py 中添加 TimesFM vs LightGBM 对比评估

作者：TrendCast Pro
日期：2026-06-28
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any
from threading import Lock

import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 懒加载 TimesFM
# ---------------------------------------------------------------------------
_TFM_AVAILABLE = False
_TFM_MODEL = None
_TFM_CONFIG_CLS = None
_TFM_LOAD_ERROR: Optional[str] = None
_TFM_LOAD_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


def _ensure_timesfm(horizon: int = 20, context_len: int = 512, model_name: str = "google/timesfm-2.5-200m-pytorch") -> bool:
    """线程安全的懒加载 TimesFM 模型。"""
    global _TFM_AVAILABLE, _TFM_MODEL, _TFM_CONFIG_CLS, _TFM_LOAD_ERROR
    if _TFM_AVAILABLE and _TFM_MODEL is not None:
        return True

    if _TFM_LOAD_ERROR:
        return False

    with _TFM_LOAD_LOCK:
        if _TFM_AVAILABLE and _TFM_MODEL is not None:
            return True
        try:
            import torch

            # 尽量在可能的环境下提高 matmul 精度
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        except ImportError:
            _TFM_LOAD_ERROR = "PyTorch 未安装"
            _LOGGER.debug(_TFM_LOAD_ERROR)
            return False

        try:
            import timesfm

            _TFM_CONFIG_CLS = timesfm.ForecastConfig
        except ImportError:
            _TFM_LOAD_ERROR = "timesfm 未安装: pip install timesfm[torch]"
            _LOGGER.debug(_TFM_LOAD_ERROR)
            return False

        try:
            _LOGGER.info("正在从 HuggingFace 加载 TimesFM 模型，请耐心等待...")
            _TFM_MODEL = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
            cfg = _TFM_CONFIG_CLS(
                max_context=context_len,
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
            _TFM_MODEL.compile(cfg)
            _TFM_AVAILABLE = True
            _LOGGER.info("TimesFM 模型加载完成")
            return True
        except Exception as e:
            _TFM_LOAD_ERROR = f"模型加载失败: {e}"
            _LOGGER.exception(_TFM_LOAD_ERROR)
            return False


# ---------------------------------------------------------------------------
# 核心预测器
# ---------------------------------------------------------------------------

class TimesFMFinancePredictor:
    """
    TimesFM 金融时序预测器。
    对标 LightGBMClassifier (二分类)，提供连续价格预测 + 概率区间。
    """

    def __init__(
        self,
        horizon_days: int = 20,
        context_days: int = 252,
        verbose: bool = False,
    ):
        self.horizon_days = horizon_days
        self.context_days = max(context_days, 64)
        self.verbose = verbose
        # 可选模型名称（HuggingFace 模型 id），便于替换或轻量化部署
        self.model_name = "google/timesfm-2.5-200m-pytorch"

    @property
    def available(self) -> bool:
        return _TFM_AVAILABLE

    @property
    def load_error(self) -> Optional[str]:
        return _TFM_LOAD_ERROR

    def _load(self) -> bool:
        if _TFM_AVAILABLE:
            return True
        success = _ensure_timesfm(
            horizon=self.horizon_days,
            context_len=self.context_days,
            model_name=self.model_name,
        )
        if success and self.verbose:
            _LOGGER.info(f"[TimesFM] 金融预测器就绪 (horizon={self.horizon_days}d)")
        return success

    # ---- 价格预测 --------------------------------------------------------

    def predict_price(
        self,
        close_prices: np.ndarray,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """
        对收盘价序列做零样本预测。

        Args:
            close_prices: 历史收盘价 (N,) float32
            symbol: 标的代码（仅用于输出标签）

        Returns:
            含点预测 + 分位数 + 信号 + 波动率预测的字典
        """
        if not self._load():
            return {"error": f"TimesFM 不可用: {_TFM_LOAD_ERROR}"}

        prices = close_prices.astype(np.float32)[-self.context_days:]
        prices = np.nan_to_num(prices, nan=np.nanmean(prices) if len(prices) > 0 else 0.0)

        if len(prices) < 64:
            pad = np.full(64 - len(prices), prices[0] if len(prices) > 0 else 0.0)
            prices = np.concatenate([pad, prices])

        try:
            point, quants = _TFM_MODEL.forecast(
                horizon=self.horizon_days,
                inputs=[prices],
            )
            point_arr = point[0].astype(np.float64)
            quant_arr = quants[0].astype(np.float64)

        except Exception as e:
            return {"error": f"预测失败: {e}"}

        last_price = float(prices[-1])

        # ---- 多周期收益预测 ----
        horizons = [1, 3, 5, 10, 20]
        horizon_returns = {}
        max_h = min(self.horizon_days, max(horizons))
        for h in horizons:
            if h <= max_h:
                ret = (point_arr[h - 1] - last_price) / last_price
                horizon_returns[f"return_{h}d"] = round(float(ret) * 100, 2)

        # ---- 信号生成 ----
        signals = {}
        for h_key, ret in horizon_returns.items():
            days = int(h_key.replace("return_", "").replace("d", ""))
            if ret > 0.02:
                signals[f"signal_{days}d"] = "buy"
            elif ret < -0.02:
                signals[f"signal_{days}d"] = "sell"
            else:
                signals[f"signal_{days}d"] = "hold"

        # 整体趋势（5日为准）
        ret_5d = horizon_returns.get("return_5d", 0)
        if ret_5d > 2:
            direction = "看涨"
            prob_up = min(0.95, 0.5 + abs(ret_5d) / 20)
        elif ret_5d < -2:
            direction = "看跌"
            prob_up = max(0.05, 0.5 - abs(ret_5d) / 20)
        else:
            direction = "震荡"
            prob_up = 0.5

        # ---- 预测波动率 ----
        vol_series = np.mean(
            (quant_arr[:, 8] - quant_arr[:, 1]) / (np.abs(point_arr) + 1e-6),
            axis=0,
        )
        vol_avg = float(np.mean(vol_series)) if hasattr(vol_series, "__len__") else float(vol_series)

        # ---- 最大回撤估计 ----
        cummax = np.maximum.accumulate(point_arr)
        drawdowns = (point_arr - cummax) / (cummax + 1e-6)
        max_dd_estimate = float(np.min(drawdowns)) * 100

        return {
            "symbol": symbol,
            "horizon": self.horizon_days,
            "last_price": round(last_price, 2),
            "last_date": datetime.now().strftime("%Y-%m-%d"),
            "point_forecast": [round(float(v), 2) for v in point_arr],
            "quantiles": {
                "q10": [round(float(v), 2) for v in quant_arr[:, 1]],
                "q50": [round(float(v), 2) for v in quant_arr[:, 4]],
                "q90": [round(float(v), 2) for v in quant_arr[:, 8]],
            },
            "horizon_returns": horizon_returns,
            "signals": signals,
            "direction": direction,
            "probability_up": round(prob_up, 3),
            "volatility_forecast": round(vol_avg * 100, 2),
            "max_drawdown_estimate": round(max_dd_estimate, 2),
            "ci_80": [
                (
                    round(float(quant_arr[i, 1]), 2),
                    round(float(quant_arr[i, 8]), 2),
                )
                for i in range(min(20, self.horizon_days))
            ],
            "model": "timesfm-2.5-200m",
        }

    # ---- LightGBM 兼容接口 -----------------------------------------------

    def predict_classification(
        self,
        close_prices: np.ndarray,
        horizon: int = 5,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """
        LightGBM 兼容接口：输出二分类 (涨/跌) + 概率。

        与 LightGBM 的 predict() 输出格式完全兼容，
        可直接用于 evaluate() 和 Ensemble。

        Returns:
            {"prediction": 0/1, "direction": "看涨"/"看跌",
             "probability": 0.XX, "horizon_days": 5}
        """
        # 临时调整 horizon
        orig_horizon = self.horizon_days
        self.horizon_days = max(horizon, 20)
        if not self._load():
            self.horizon_days = orig_horizon
            return {
                "prediction": 0,
                "direction": "未知",
                "probability": 0.5,
                "horizon_days": horizon,
                "error": _TFM_LOAD_ERROR,
            }

        result = self.predict_price(close_prices, symbol)
        self.horizon_days = orig_horizon

        if result.get("error"):
            return {
                "prediction": 0,
                "direction": "未知",
                "probability": 0.5,
                "horizon_days": horizon,
            }

        ret_key = f"return_{horizon}d"
        ret = result["horizon_returns"].get(ret_key, 0)

        if ret > 0:
            prediction = 1
            direction = "看涨"
            probability = min(0.95, 0.5 + abs(ret) / 15)
        else:
            prediction = 0
            direction = "看跌"
            probability = min(0.95, 0.5 + abs(ret) / 15)

        return {
            "prediction": prediction,
            "direction": direction,
            "probability": round(probability, 3),
            "horizon_days": horizon,
            "expected_return_pct": round(ret, 2),
            "volatility_forecast": result.get("volatility_forecast", 0),
            "model": "timesfm-2.5-200m",
        }

    # ---- 批量预测 --------------------------------------------------------

    def predict_batch(
        self,
        symbols_data: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        批量预测多个标的。

        Args:
            symbols_data: {代码: 收盘价数组}

        Returns:
            DataFrame (index=symbol, columns=各周期收益/信号)
        """
        if not self._load():
            return pd.DataFrame({"error": [_TFM_LOAD_ERROR]})

        rows = []
        for sym, prices in symbols_data.items():
            try:
                result = self.predict_price(prices, sym)
                if result.get("error"):
                    continue
                row = {
                    "symbol": sym,
                    "last_price": result["last_price"],
                    "direction": result["direction"],
                    "probability_up": result["probability_up"],
                    "volatility_forecast": result["volatility_forecast"],
                }
                row.update(result["horizon_returns"])
                row.update(result["signals"])
                rows.append(row)
            except Exception as e:
                rows.append({"symbol": sym, "error": str(e)})

        return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Ensemble: TimesFM + LightGBM 加权融合
# ---------------------------------------------------------------------------

def ensemble_predict(
    lightgbm_predictions: Dict[int, float],  # {horizon_days: prob_up}
    timesfm_predictions: Dict[int, float],   # {horizon_days: prob_up}
    tfm_weight: float = 0.4,
) -> Dict[int, Dict[str, Any]]:
    """
    TimesFM + LightGBM 加权融合预测。

    Args:
        lightgbm_predictions: LightGBM 各周期的涨概率
        timesfm_predictions: TimesFM 各周期的涨概率
        tfm_weight: TimesFM 权重 (0~1，默认 0.4)

    Returns:
        {horizon_days: {probability, prediction, direction}}
    """
    lgb_weight = 1 - tfm_weight
    results = {}

    for horizon in lightgbm_predictions:
        lgb_prob = lightgbm_predictions.get(horizon, 0.5)
        tfm_prob = timesfm_predictions.get(horizon, lgb_prob)

        ensemble_prob = lgb_weight * lgb_prob + tfm_weight * tfm_prob

        prediction = 1 if ensemble_prob > 0.5 else 0
        direction = "看涨" if prediction == 1 else "看跌"

        results[horizon] = {
            "probability": round(ensemble_prob, 3),
            "prediction": prediction,
            "direction": direction,
            "lightgbm_contribution": round(lgb_weight * lgb_prob, 3),
            "timesfm_contribution": round(tfm_weight * tfm_prob, 3),
        }

    return results


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="TimesFM 金融预测")
    p.add_argument("--test", action="store_true", help="使用随机游走模拟价格")
    p.add_argument("--horizon", type=int, default=20, help="预测天数")
    p.add_argument("--symbol", default="TEST", help="标的名称")

    args = p.parse_args()

    if args.test:
        # 模拟随机游走价格（含漂移 + 波动）
        days = 252
        returns = np.random.randn(days) * 0.02 + 0.0005  # 日收益 N(0.05%, 2%)
        prices = 100 * np.exp(np.cumsum(returns))

        print(f"=== TimesFM 价格预测 [{args.symbol}] 未来{args.horizon}天 ===")
        predictor = TimesFMFinancePredictor(horizon_days=args.horizon)

        if predictor.available:
            result = predictor.predict_price(prices, args.symbol)
            if result.get("error"):
                print(f"错误: {result['error']}")
            else:
                print(f"最新价: {result['last_price']}")
                print(f"\n多周期收益预测:")
                for k, v in result["horizon_returns"].items():
                    signal = result["signals"].get(k.replace("return", "signal"), "")
                    print(f"  {k.replace('_', ' ')}: {v:+.2f}%  [{signal}]")
                print(f"\n方向: {result['direction']} (涨概率: {result['probability_up']})")
                print(f"预测波动率: {result['volatility_forecast']}%")
                print(f"最大回撤估计: {result['max_drawdown_estimate']}%")
                print(f"\n前7天价格预测:")
                for i in range(min(7, args.horizon)):
                    ci = result["ci_80"][i]
                    print(
                        f"  D+{i+1}: {result['point_forecast'][i]}  "
                        f"[80%CI: {ci[0]} - {ci[1]}]"
                    )
        else:
            print(f"TimesFM 不可用: {predictor.load_error}")
    else:
        print("请使用 --test 运行演示")
