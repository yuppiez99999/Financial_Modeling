"""模型注册表子包：模型格式无关的统一加载入口。

对外契约：
  - ModelRegistry      : 按契约名加载任意支持格式的模型
  - ModelArtifact      : 加载结果（模型 + 特征列 + 元数据）
  - SUPPORTED_FORMATS  : 支持的格式清单
"""

from src.train.registry.model_registry import (
    SUPPORTED_FORMATS,
    ModelArtifact,
    ModelRegistry,
    save_artifact,
)

__all__ = ["ModelRegistry", "ModelArtifact", "SUPPORTED_FORMATS", "save_artifact"]
