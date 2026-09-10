"""金融市场预测模型 - 模型导出模块"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelExporter:
    """模型导出器 - 支持 ONNX / TorchScript 格式"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.export_format = config.get("export", {}).get("format", "onnx")
        self.output_dir = Path(config.get("export", {}).get("output_dir", "models/exported"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(config["training"]["save_dir"])

    def export_lightgbm_to_onnx(self, model_key: str) -> Path | None:
        """导出 LightGBM 模型为 ONNX 格式"""
        model_file = self.save_dir / f"lightgbm_{model_key}.pkl"
        if not model_file.exists():
            logger.error(f"模型文件不存在: {model_file}")
            return None

        data = joblib.load(model_file)
        model = data["model"]

        # LightGBM 原生支持导出为文本格式模型文件（ONNX 转换兼容性最佳）
        # 同时导出原生 model.txt 和 joblib 格式作为备选
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 方案1：LightGBM 原生 save_model（通用格式）
            native_path = output_dir / f"lightgbm_{model_key}.txt"
            model.booster_.save_model(str(native_path))
            logger.info(f"LightGBM 模型已导出（原生格式）: {native_path}")

            # 方案2：尝试 ONNX 导出（如果环境支持）
            onnx_path = output_dir / f"lightgbm_{model_key}.onnx"
            try:
                import onnxmltools
                from onnxmltools.convert.lightgbm.convert import convert as convert_lgb
                from onnxmltools.convert.common.data_types import FloatTensorType

                n_features = model.n_features_in_
                initial_type = [("input", FloatTensorType([None, n_features]))]
                onnx_model = convert_lgb(model, initial_types=initial_type, target_opset=15)
                onnxmltools.utils.save_model(onnx_model, str(onnx_path))
                logger.info(f"LightGBM 模型已导出（ONNX）: {onnx_path}")
                return onnx_path
            except Exception as e:
                logger.warning(f"ONNX 转换不可用（版本兼容问题），已使用原生格式导出: {e}")
                return native_path

        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            return None

    def export_lstm_to_torchscript(self, model_key: str) -> Path | None:
        """导出 LSTM 模型为 TorchScript 格式"""
        try:
            import torch
            from src.train.models.lstm_model import LSTMModel
        except ImportError:
            logger.error("PyTorch 未安装，无法导出 LSTM 模型")
            return None

        model_file = self.save_dir / f"lstm_{model_key}.pt"
        if not model_file.exists():
            logger.error(f"模型文件不存在: {model_file}")
            return None

        checkpoint = torch.load(model_file, map_location="cpu")
        model = LSTMModel(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["config"]["hidden_size"],
            num_layers=checkpoint["config"]["num_layers"],
            dropout=checkpoint["config"]["dropout"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # TorchScript 导出
        scripted = torch.jit.script(model)
        output_path = self.output_dir / f"lstm_{model_key}.torchscript"
        scripted.save(str(output_path))
        logger.info(f"LSTM 模型已导出为 TorchScript: {output_path}")
        return output_path

    def export_all(self, model_type: str = "lightgbm") -> list[Path]:
        """导出所有模型"""
        horizons = self.config["data"]["prediction_horizons"]
        exported: list[Path] = []

        for horizon_name, horizon_days in horizons.items():
            model_key = f"{horizon_name}_{horizon_days}d"
            logger.info(f"导出 {model_key} 模型...")

            if model_type == "lightgbm":
                path = self.export_lightgbm_to_onnx(model_key)
            elif model_type == "pytorch_lstm":
                path = self.export_lstm_to_torchscript(model_key)
            else:
                logger.error(f"不支持的模型类型: {model_type}")
                continue

            if path:
                exported.append(path)

        logger.info(f"导出完成: {len(exported)} 个模型")
        return exported

    def verify_onnx(self, model_path: Path, test_input: np.ndarray) -> bool:
        """验证 ONNX 模型可正常推理"""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: test_input.astype("float64")})
            logger.info(f"ONNX 验证通过: {model_path}, 输出形状={result[0].shape}")
            return True
        except Exception as e:
            logger.error(f"ONNX 验证失败: {e}")
            return False
