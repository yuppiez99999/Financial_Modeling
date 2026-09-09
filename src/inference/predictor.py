"""金融市场预测模型 - 推理引擎"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.collector import DataCollector
from src.data.preprocessor import FeatureEngineer
from src.timesfm_predictor import TimesFMFinancePredictor
from src.inference.predictor_utils import (
    load_placeholder,
    try_load_timesfm_or_instance,
    load_ensemble_entry,
)

logger = logging.getLogger(__name__)


class PredictionEngine:
    """推理引擎 - 加载训练好的模型进行预测"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.save_dir = Path(config["training"]["save_dir"])
        self.horizons = config["data"]["prediction_horizons"]
        self.models: dict[str, Any] = {}
        self.feature_engineer = FeatureEngineer(config)

    def load_models(self, model_type: str = "lightgbm") -> None:
        """加载所有预测周期的模型"""
        for horizon_name, horizon_days in self.horizons.items():
            model_key = f"{horizon_name}_{horizon_days}d"
            model_file = self.save_dir / f"{model_type}_{model_key}.pkl"

            if model_type == "lightgbm":
                if not model_file.exists():
                    logger.warning(f"模型文件不存在: {model_file}")
                    continue
                data = joblib.load(model_file)
                self.models[model_key] = data
                logger.info(f"已加载 {model_key} 模型")
            elif model_type == "timesfm":
                # 优先使用占位 pickle，否则按需实例化 TimesFM predictor
                def _instantiate():
                    tfm_cfg = self.config.get("model", {}).get("timesfm", {})
                    return TimesFMFinancePredictor(
                        horizon_days=horizon_days,
                        context_days=tfm_cfg.get("context_days", 252),
                        verbose=tfm_cfg.get("verbose", False),
                    )

                res = try_load_timesfm_or_instance(self.save_dir, model_key, _instantiate)
                if res is None:
                    logger.error(f"无法为 {model_key} 获取 TimesFM 预测器或占位模型")
                    continue
                self.models[model_key] = res
                logger.info(f"已注册 TimesFM 预测器/占位 {model_key}")
            elif model_type == "ensemble":
                def _instantiate():
                    tfm_cfg = self.config.get("model", {}).get("timesfm", {})
                    return TimesFMFinancePredictor(
                        horizon_days=horizon_days,
                        context_days=tfm_cfg.get("context_days", 252),
                        verbose=tfm_cfg.get("verbose", False),
                    )

                # always attempt to build ensemble entry (timesfm placeholder may exist even if lightgbm file missing)
                entry = load_ensemble_entry(self.save_dir, model_key, model_file, _instantiate)
                self.models[model_key] = {"ensemble": entry}
            elif model_type == "pytorch_lstm":
                import torch
                from src.train.models.lstm_model import LSTMModel
                checkpoint = torch.load(model_file, map_location="cpu")
                model = LSTMModel(
                    input_size=checkpoint["input_size"],
                    hidden_size=checkpoint["config"]["hidden_size"],
                    num_layers=checkpoint["config"]["num_layers"],
                    dropout=checkpoint["config"]["dropout"],
                )
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                self.models[model_key] = model
                logger.info(f"已加载 {model_key} LSTM 模型")

    def predict(self, symbol: str, horizon_name: str = "short_term") -> dict[str, Any]:
        """
        预测指定标的的未来走势

        Args:
            symbol: 标的代码
            horizon_name: 预测周期 (short_term/mid_term/long_term)

        Returns:
            预测结果字典
        """
        horizon_days = self.horizons.get(horizon_name, 5)
        model_key = f"{horizon_name}_{horizon_days}d"

        if model_key not in self.models:
            return {"error": f"模型 {model_key} 未加载"}

        # 获取最新数据
        collector = DataCollector(self.config)
        df = collector.load_cached(symbol)
        if df is None:
            df = collector._fetch_with_fallback(symbol)
        if df is None or len(df) == 0:
            return {"error": f"无法获取 {symbol} 的数据"}

        # 特征工程
        df_features = self.feature_engineer.transform(df, horizon_days)
        feature_cols = self.feature_engineer.get_feature_columns(df_features, horizon_days)

        # 取最新一行作为输入
        latest = df_features[feature_cols].iloc[-1:].values

        # 预测
        model_data = self.models[model_key]
        # ensemble 支持：当 model_data 包含 ensemble 字段时，融合 LightGBM 与 TimesFM
        if isinstance(model_data, dict) and "ensemble" in model_data:
            ensemble_entry = model_data["ensemble"]

            lgb_entry = ensemble_entry.get("lightgbm")
            tfm_entry = ensemble_entry.get("timesfm")

            lgb_prob = None
            if lgb_entry is not None and isinstance(lgb_entry, dict) and "model" in lgb_entry:
                m = lgb_entry["model"]
                s = lgb_entry.get("scaler")
                try:
                    if s is None:
                        X_in = df_features[self.feature_engineer.get_feature_columns(df_features, horizon_days)].iloc[-1:].values
                    else:
                        X_in = s.transform(df_features[self.feature_engineer.get_feature_columns(df_features, horizon_days)].iloc[-1:].values)
                    lgb_prob = float(m.predict_proba(X_in)[0][1])
                except Exception as e:
                    logger.warning(f"LightGBM 分量预测失败，回退为 None: {e}")
                    lgb_prob = None

            tfm_prob = None
            if tfm_entry is not None and isinstance(tfm_entry, TimesFMFinancePredictor):
                close_prices = df["close"].values if "close" in df.columns else df.iloc[:, 3].values
                tfm_res = tfm_entry.predict_classification(close_prices, horizon=horizon_days, symbol=symbol)
                if not tfm_res.get("error"):
                    tfm_prob = float(tfm_res.get("probability", 0.5))
            else:
                if isinstance(tfm_entry, dict) and "model" in tfm_entry:
                    try:
                        s = tfm_entry.get("scaler")
                        m = tfm_entry.get("model")
                        if s is None:
                            X_in = df_features[self.feature_engineer.get_feature_columns(df_features, horizon_days)].iloc[-1:].values
                        else:
                            X_in = s.transform(df_features[self.feature_engineer.get_feature_columns(df_features, horizon_days)].iloc[-1:].values)
                        tfm_prob = float(m.predict_proba(X_in)[0][1])
                    except Exception as e:
                        logger.warning(f"占位 TimesFM 分量预测失败: {e}")
                        tfm_prob = None

            tfm_weight = float(self.config.get("model", {}).get("ensemble", {}).get("tfm_weight", 0.4))
            lgb_weight = 1.0 - tfm_weight

            if lgb_prob is None and tfm_prob is None:
                return {"error": "ensemble 无可用分量"}
            if lgb_prob is None:
                proba = tfm_prob
            elif tfm_prob is None:
                proba = lgb_prob
            else:
                proba = lgb_weight * lgb_prob + tfm_weight * tfm_prob

            pred = 1 if proba > 0.5 else 0
            direction = "看涨" if pred == 1 else "看跌"
            confidence = proba if pred == 1 else (1 - proba)

            result = {
                "symbol": symbol,
                "horizon": horizon_name,
                "horizon_days": horizon_days,
                "prediction": pred,
                "direction": direction,
                "probability": round(proba, 4),
                "confidence": round(confidence, 4),
                "latest_date": str(df_features.iloc[-1].get("date", "N/A")),
                "latest_close": float(df_features.iloc[-1].get("close", 0)),
            }
            return result

        # 如果模型为 TimesFM predictor 实例（零样本预测器）
        if isinstance(model_data, TimesFMFinancePredictor):
            # TimesFM 需要原始收盘价序列
            close_prices = df["close"].values if "close" in df.columns else df.iloc[:, 3].values
            tfm_res = model_data.predict_classification(close_prices, horizon=horizon_days, symbol=symbol)
            if tfm_res.get("error"):
                return {"error": tfm_res["error"]}
            pred = int(tfm_res.get("prediction", 0))
            proba = float(tfm_res.get("probability", 0.5))
        elif isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            scaler = model_data["scaler"]

            # 特征数对齐：训练时和推理时可能因目标列不同导致特征数不一致
            expected_features = getattr(model, "n_features_in_", None)
            if expected_features is not None and latest.shape[1] != expected_features:
                input_features = latest.shape[1]
                if latest.shape[1] > expected_features:
                    latest = latest[:, :expected_features]
                    logger.warning(
                        f"特征数不匹配 (输入={input_features}, "
                        f"模型={expected_features})，已截取前 {expected_features} 个特征"
                    )
                else:
                    # 特征不足：用0填充
                    pad = np.zeros((1, expected_features - latest.shape[1]))
                    latest = np.hstack([latest, pad])
                    logger.warning(f"特征数不足，已用0填充至 {expected_features}")

            X_scaled = scaler.transform(latest)
            pred = int(model.predict(X_scaled)[0])
            proba = float(model.predict_proba(X_scaled)[0][1])
        else:
            # LSTM 模型需要序列输入
            seq_len = self.config["model"]["pytorch_lstm"]["sequence_length"]
            if len(df_features) >= seq_len:
                import torch
                seq = df_features[feature_cols].iloc[-seq_len:].values
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                with torch.no_grad():
                    proba = float(model_data(seq_tensor).item())
                pred = 1 if proba > 0.5 else 0
            else:
                return {"error": f"数据不足，需要至少 {seq_len} 条记录"}

        # ensemble 支持：当 model_data 包含 ensemble 字段时，融合 LightGBM 与 TimesFM
        if isinstance(model_data, dict) and "ensemble" in model_data:
            ensemble_entry = model_data["ensemble"]
            lgb_entry = ensemble_entry.get("lightgbm")
            tfm_entry = ensemble_entry.get("timesfm")

            lgb_prob = None
            if lgb_entry is not None and isinstance(lgb_entry, dict) and "model" in lgb_entry:
                m = lgb_entry["model"]
                s = lgb_entry.get("scaler")
                try:
                    if s is None:
                        # 如果 scaler 缺失，尝试直接使用原始特征（模型可能接受原始特征）
                        X_in = latest
                    else:
                        X_in = s.transform(latest)
                    lgb_prob = float(m.predict_proba(X_in)[0][1])
                except Exception as e:
                    logger.warning(f"LightGBM 分量预测失败，回退为 None: {e}")
                    lgb_prob = None

            tfm_prob = None
            if tfm_entry is not None and isinstance(tfm_entry, TimesFMFinancePredictor):
                close_prices = df["close"].values if "close" in df.columns else df.iloc[:, 3].values
                tfm_res = tfm_entry.predict_classification(close_prices, horizon=horizon_days, symbol=symbol)
                if not tfm_res.get("error"):
                    tfm_prob = float(tfm_res.get("probability", 0.5))
            else:
                # tfm_entry 可能是占位 dict（{'model':..., 'scaler':...}）
                if isinstance(tfm_entry, dict) and "model" in tfm_entry:
                    # 占位 timesfm 使用 lightgbm-like 接口
                    try:
                        s = tfm_entry.get("scaler")
                        m = tfm_entry.get("model")
                        if s is None:
                            X_in = latest
                        else:
                            X_in = s.transform(latest)
                        tfm_prob = float(m.predict_proba(X_in)[0][1])
                    except Exception as e:
                        logger.warning(f"占位 TimesFM 分量预测失败: {e}")
                        tfm_prob = None

            # 权重
            tfm_weight = float(self.config.get("model", {}).get("ensemble", {}).get("tfm_weight", 0.4))
            lgb_weight = 1.0 - tfm_weight

            # 回退策略
            if lgb_prob is None and tfm_prob is None:
                return {"error": "ensemble 无可用分量"}
            if lgb_prob is None:
                proba = tfm_prob
            elif tfm_prob is None:
                proba = lgb_prob
            else:
                proba = lgb_weight * lgb_prob + tfm_weight * tfm_prob

            pred = 1 if proba > 0.5 else 0
            direction = "看涨" if pred == 1 else "看跌"
            confidence = proba if pred == 1 else (1 - proba)
        else:
            direction = "看涨" if pred == 1 else "看跌"
            confidence = proba if pred == 1 else (1 - proba)

        result = {
            "symbol": symbol,
            "horizon": horizon_name,
            "horizon_days": horizon_days,
            "prediction": pred,
            "direction": direction,
            "probability": round(proba, 4),
            "confidence": round(confidence, 4),
            "latest_date": str(df_features.iloc[-1].get("date", "N/A")),
            "latest_close": float(df_features.iloc[-1].get("close", 0)),
        }

        logger.info(
            f"预测 {symbol} [{horizon_name}/{horizon_days}d]: "
            f"{direction} (概率={proba:.4f}, 置信度={confidence:.4f})"
        )

        return result

    def predict_all_horizons(self, symbol: str) -> dict[str, Any]:
        """预测所有周期"""
        results: dict[str, Any] = {"symbol": symbol, "predictions": {}}
        for horizon_name in self.horizons:
            try:
                results["predictions"][horizon_name] = self.predict(symbol, horizon_name)
            except Exception as e:
                logger.error(f"预测 {symbol} [{horizon_name}] 失败: {e}")
                results["predictions"][horizon_name] = {"error": str(e)}
        return results

    def batch_predict(self, symbols: list[str]) -> list[dict[str, Any]]:
        """批量预测多个标的"""
        all_results = []
        for symbol in symbols:
            logger.info(f"批量预测: {symbol}")
            result = self.predict_all_horizons(symbol)
            all_results.append(result)
        return all_results
