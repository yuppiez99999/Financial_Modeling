"""金融市场预测模型 - 推理引擎"""

from __future__ import annotations

import logging
from datetime import datetime
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
from src.train.registry import ModelRegistry
from src.train.registry.model_registry import ModelArtifact as _ArtifactLike

logger = logging.getLogger(__name__)


def _align_features(latest: "np.ndarray", model: Any) -> "np.ndarray":
    """推理特征数对齐：与模型期望不一致时截断/零填充（防御占位或混搭模型场景）。

    训练与推理的特征列集合可能不同（例如 ensemble 混搭占位模型、或特征
    工程版本差异），直接送入 scaler 会抛特征数错误；这里统一做防御性对齐。
    """
    expected = getattr(model, "n_features_in_", None)
    if expected is None or latest.shape[1] == expected:
        return latest
    if latest.shape[1] > expected:
        logger.warning(
            f"特征数不匹配 (输入={latest.shape[1]}, "
            f"模型={expected})，已截取前 {expected} 个特征"
        )
        return latest[:, :expected]
    pad = np.zeros((1, expected - latest.shape[1]))
    logger.warning(f"特征数不足，已用0填充至 {expected}")
    return np.hstack([latest, pad])


class PredictionEngine:
    """推理引擎 - 加载训练好的模型进行预测"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.save_dir = Path(config["training"]["save_dir"])
        self.horizons = config["data"]["prediction_horizons"]
        self.models: dict[str, Any] = {}
        self.feature_engineer = FeatureEngineer(config)
        # 每日刷新去重：同标的每进程至多尝试一次过期刷新（刷新无新数据也标记）
        self._refresh_attempted: set[str] = set()

    def load_models(self, model_type: str = "lightgbm") -> None:
        """加载所有预测周期的模型。

        加载路径统一收敛到 `ModelRegistry`（格式无关契约）：
        - lightgbm / pytorch_lstm / factor_model 走注册表；
        - ensemble / multifactor 走各自的组合加载逻辑；
        - timesfm 保留占位优先 → 实例化的历史行为。
        """
        self.model_type = model_type
        registry = ModelRegistry(self.config)

        for horizon_name, horizon_days in self.horizons.items():
            model_key = f"{horizon_name}_{horizon_days}d"

            if model_type in ("factor_model", "multifactor"):
                entry = self._load_multifactor(model_key)
                if entry is not None:
                    self.models[model_key] = entry
                continue

            if model_type == "ensemble":
                # 历史行为保留：LightGBM × TimesFM 两分量融合
                def _instantiate():
                    tfm_cfg = self.config.get("model", {}).get("timesfm", {})
                    return TimesFMFinancePredictor(
                        horizon_days=horizon_days,
                        context_days=tfm_cfg.get("context_days", 252),
                        verbose=tfm_cfg.get("verbose", False),
                    )

                lightgbm_file = self.save_dir / f"lightgbm_{model_key}.pkl"
                entry = load_ensemble_entry(self.save_dir, model_key, lightgbm_file, _instantiate)
                self.models[model_key] = {"ensemble": entry}
                logger.info(f"已加载 ensemble（lightgbm × timesfm）{model_key}")
                continue

            if model_type == "timesfm":
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
                continue

            artifact = registry.load_horizon(self.save_dir, model_type, model_key)
            if artifact is None:
                logger.warning(f"模型文件不存在或不可加载: {model_type}_{model_key}")
                continue
            if artifact.format == "torch":
                self.models[model_key] = artifact
            elif artifact.format == "factor":
                # 单因子模型：直接按其自身概率输出
                self.models[model_key] = {"factor": artifact}
            else:
                self.models[model_key] = {
                    "model": artifact.model,
                    "scaler": artifact.scaler,
                    "_artifact": artifact,
                }
            logger.info(f"已加载 {model_key} 模型 [{artifact.format}]")

    def _load_multifactor(self, model_key: str) -> dict[str, Any] | None:
        """加载多因子组合预测器：因子模型 + LightGBM + LSTM 分量。"""
        from src.factors import FactorPredictor
        from src.train.registry import ModelRegistry

        registry = ModelRegistry(self.config)
        factor_artifact = registry.load_horizon(self.save_dir, "factor_model", model_key)
        if factor_artifact is None:
            logger.warning(
                "多因子模型文件不存在: factor_model_%s.pkl（请先 `python main.py train --model-type factor_model`）",
                model_key,
            )
            return None

        tree_artifact = registry.load_horizon(self.save_dir, "lightgbm", model_key)
        tree_entry = (
            {"model": tree_artifact.model, "scaler": tree_artifact.scaler}
            if tree_artifact is not None else None
        )
        seq_model = self._load_sequence(model_key)
        predictor = FactorPredictor(
            self.config,
            factor_artifact.model,
            tree_entry=tree_entry,
            sequence_model=seq_model,
        )
        available = [c for c in predictor.enabled_classes if (
            (c == "factor" and predictor.factor_model is not None)
            or (c == "tree" and tree_entry is not None)
            or (c == "sequence" and seq_model is not None)
        )]
        logger.info(f"多因子组合预测器 {model_key} 分量: {available or ['factor']}")
        return {"multifactor": predictor}

    def _load_sequence(self, model_key: str) -> dict[str, Any] | None:
        """加载 LSTM 分量（torch 缺失时返回 None，组合预测自动降级）。"""
        try:
            from src.train.registry import ModelRegistry

            artifact = ModelRegistry(self.config).load_horizon(
                self.save_dir, "pytorch_lstm", model_key
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LSTM 分量加载失败（降级为无 sequence 类）: {e}")
            return None
        if artifact is None:
            return None
        return {
            "model": artifact.model,
            "feature_cols": artifact.feature_cols,
            "seq_len": artifact.seq_len,
        }

    def _ensure_fresh(self, symbol: str, df: Any, collector: Any) -> Any:
        """推理数据新鲜度保障：缓存最后日期早于今天时，经真实源刷新一次。

        纪律：
        - 只走 collector.fetch_realtime（wind/tencent），绝不采用 simulation
          兜底数据——随机游走的日期恰好等于"今天"，会骗过新鲜度检查；
        - 每标的每进程至多尝试一次（刷新无新数据也标记），避免节假日反复空刷；
        - 刷新失败/无新数据时回退旧缓存（fail-open，不阻断推理）。
        """
        old_last = None
        if df is not None and len(df) > 0 and "date" in df.columns:
            try:
                old_last = pd.to_datetime(df["date"]).max().date()
            except Exception:  # noqa: BLE001
                old_last = None
            if old_last is not None and old_last >= datetime.now().date():
                return df  # 缓存已是最新交易日，直接用
        if symbol in self._refresh_attempted:
            return df
        self._refresh_attempted.add(symbol)
        try:
            fresh = collector.fetch_realtime(symbol)
        except Exception:  # noqa: BLE001
            return df
        if fresh is None or len(fresh) == 0:
            return df
        try:
            fresh_last = pd.to_datetime(fresh["date"]).max().date()
            if old_last is None or fresh_last > old_last:
                return fresh  # 有更新的真实数据（fetch_realtime 已写缓存）
        except Exception:  # noqa: BLE001
            return fresh
        return df

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

        # 获取最新数据（缓存过期时自动刷新一次，见 _ensure_fresh）
        collector = DataCollector(self.config)
        df = collector.load_cached(symbol)
        df = self._ensure_fresh(symbol, df, collector)
        if df is None or len(df) == 0:
            return {"error": f"无法获取 {symbol} 的数据"}

        # 特征工程
        df_features = self.feature_engineer.transform(df, horizon_days)
        feature_cols = self.feature_engineer.get_feature_columns(df_features, horizon_days)

        # 取最新一行作为输入
        latest = df_features[feature_cols].iloc[-1:].values

        # 预测
        model_data = self.models[model_key]

        # 多因子组合预测（factor + tree + sequence 类级加权融合）
        if isinstance(model_data, dict) and "multifactor" in model_data:
            predictor = model_data["multifactor"]
            proba = float(predictor.predict_proba(df_features))
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
                "components": dict(getattr(predictor, "last_components", {}) or {}),
                "explain": predictor.factor_model.explain(5) if predictor.factor_model else {},
            }
            logger.info(
                f"多因子预测 {symbol} [{horizon_name}/{horizon_days}d]: "
                f"{direction} (概率={proba:.4f}, 分量={result['components']})"
            )
            return result

        # 单因子模型（factor_model 直接预测）
        if isinstance(model_data, dict) and "factor" in model_data:
            artifact = model_data["factor"]
            X = self._align_factor_features(df_features, artifact.feature_cols)
            proba = float(artifact.model.predict_proba(X)[0][1])
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
                "explain": artifact.model.explain(5),
            }
            logger.info(f"因子模型预测 {symbol} [{horizon_name}/{horizon_days}d]: {direction} (概率={proba:.4f})")
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
            latest = _align_features(latest, model)

            X_scaled = scaler.transform(latest) if scaler is not None else latest
            pred = int(model.predict(X_scaled)[0])
            proba = float(model.predict_proba(X_scaled)[0][1])
        elif isinstance(model_data, dict) and "ensemble" in model_data:
            # ensemble：在下方统一融合 LightGBM 与 TimesFM 分量
            pass
        elif isinstance(model_data, _ArtifactLike):
            # LSTM：按 checkpoint meta 的特征列对齐（不再靠前缀猜测）
            seq_len = model_data.seq_len or int(
                self.config["model"]["pytorch_lstm"].get("sequence_length", 20)
            )
            if len(df_features) < seq_len:
                return {"error": f"数据不足，需要至少 {seq_len} 条记录"}
            seq = self._sequence_matrix(df_features, model_data, feature_cols, seq_len)
            import torch
            with torch.no_grad():
                proba = float(model_data.model(torch.FloatTensor(seq).unsqueeze(0)).item())
            pred = 1 if proba > 0.5 else 0
        else:
            return {"error": f"未知模型格式: {type(model_data).__name__}"}

        # ensemble 支持：当 model_data 包含 ensemble 字段时，融合 LightGBM 与 TimesFM
        if isinstance(model_data, dict) and "ensemble" in model_data:
            ensemble_entry = model_data["ensemble"]
            lgb_entry = ensemble_entry.get("lightgbm")
            tfm_entry = ensemble_entry.get("timesfm")

            lgb_prob = None
            if lgb_entry is not None and isinstance(lgb_entry, dict) and "model" in lgb_entry:
                m = lgb_entry["model"]
                s = lgb_entry.get("scaler")
                # 特征数对齐（与 lightgbm 分支一致，防御占位/混搭模型场景）
                latest = _align_features(latest, m)
                X_scaled = s.transform(latest) if s is not None else latest
                lgb_prob = float(m.predict_proba(X_scaled)[0][1])

            tfm_prob = None
            if tfm_entry is not None and isinstance(tfm_entry, TimesFMFinancePredictor):
                close_prices = df["close"].values if "close" in df.columns else df.iloc[:, 3].values
                tfm_res = tfm_entry.predict_classification(close_prices, horizon=horizon_days, symbol=symbol)
                if not tfm_res.get("error"):
                    tfm_prob = float(tfm_res.get("probability", 0.5))

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

    def _align_factor_features(self, df_features: pd.DataFrame, factor_cols: list[str]) -> np.ndarray:
        """因子模型推理特征对齐：缺失因子列按中性 0 补齐，列序与训练一致。"""
        if not factor_cols:
            factor_cols = [c for c in df_features.columns if str(c).startswith("factor_")]
        missing = [c for c in factor_cols if c not in df_features.columns]
        if missing:
            logger.warning(f"缺失因子列 {len(missing)} 个，按中性 0 补齐")
        row = {}
        for col in factor_cols:
            row[col] = float(df_features[col].iloc[-1]) if col in df_features.columns else 0.0
        return pd.DataFrame([row])[factor_cols].values

    def _sequence_matrix(self, df_features: pd.DataFrame, artifact: Any,
                         fallback_cols: list[str], seq_len: int) -> np.ndarray:
        """构造 LSTM 序列输入：优先用 meta.feature_cols，并处理特征数不一致。"""
        cols = list(getattr(artifact, "feature_cols", []) or []) or list(fallback_cols)
        missing = [c for c in cols if c not in df_features.columns]
        working = df_features.copy()
        if missing:
            logger.warning(f"LSTM 特征列缺失 {len(missing)} 个，按中性 0 补齐")
            for col in missing:
                working[col] = 0.0
        seq = working[cols].iloc[-seq_len:].to_numpy(dtype=float)
        expected = int(getattr(artifact.model, "input_size", 0) or 0)
        if expected <= 0:
            expected = int(getattr(artifact.model, "lstm", None).input_size) if getattr(
                artifact.model, "lstm", None) is not None else 0
        if expected and seq.shape[1] != expected:
            logger.warning(f"LSTM 特征数不匹配 (输入={seq.shape[1]}, 模型={expected})，已对齐")
            if seq.shape[1] > expected:
                seq = seq[:, :expected]
            else:
                seq = np.hstack([seq, np.zeros((seq.shape[0], expected - seq.shape[1]))])
        return seq

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
