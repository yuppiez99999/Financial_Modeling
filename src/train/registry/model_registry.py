"""模型注册表：把"格式差异"收敛到**单一加载入口**。

背景（Q2 问题）：仓库里同时存在三种落盘格式，各自有自己的加载方式：

| 格式 | 产物 | 训练方 | 特点 |
|------|------|--------|------|
| joblib   | `lightgbm_*.pkl` / `factor_model.pkl` | ModelTrainer | `{model, scaler}` dict |
| torch    | `pytorch_lstm_*.pt` | ModelTrainer（LSTM 分支） | checkpoint dict |
| timesfm  | `timesfm_*.pkl` | 占位脚本 | 降级占位（无 torch 时） |

此前 `PredictionEngine.load_models` 用一串 if/elif 逐一判断格式，每加一种格式
就要改推理链路；本模块把它变成注册表 + 契约返回（ModelArtifact），
推理链路只依赖契约，新增格式只需注册一个 loader。

同时统一解决 Q1 遗留的 LSTM 接入缺口：
- LSTM 训练时保存 `feature_cols` 与 `seq_len` 到 checkpoint 的 `meta`，
  推理时按 meta 对齐特征列（此前推理按"非 factor_ 前缀"猜列，极易错位）；
- `input_size` 与实际特征数不一致时给出显式 warning 并按需裁剪 / 补零。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("joblib", "torch", "timesfm", "factor")


@dataclass
class ModelArtifact:
    """格式无关的模型加载结果。"""

    name: str
    format: str
    model: Any
    scaler: Any = None
    feature_cols: list[str] = field(default_factory=list)
    seq_len: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def is_stateful(self) -> bool:
        """是否为需要序列输入的时序模型（LSTM）。"""
        return self.format == "torch"


# ----------------------------------------------------------------------
# 各格式 loader
# ----------------------------------------------------------------------
def _load_joblib(path: Path) -> ModelArtifact | None:
    import joblib

    data = joblib.load(path)
    if isinstance(data, dict) and "model" in data:
        return ModelArtifact(
            name=path.stem, format="joblib", model=data["model"], scaler=data.get("scaler"),
            feature_cols=list(data.get("feature_cols", []) or []),
            meta=dict(data.get("meta", {}) or {}), path=path,
        )
    return ModelArtifact(name=path.stem, format="joblib", model=data, path=path)


def _load_torch(path: Path) -> ModelArtifact | None:
    import torch

    from src.train.models.lstm_model import LSTMModel

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(checkpoint.get("meta", {}) or {})
    cfg = dict(checkpoint.get("config", {}) or {})
    input_size = int(checkpoint.get("input_size", cfg.get("input_size", 0)) or 0)
    if input_size <= 0:
        logger.warning("[registry] %s 缺少 input_size，跳过", path.name)
        return None

    model = LSTMModel(
        input_size=input_size,
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return ModelArtifact(
        name=path.stem,
        format="torch",
        model=model,
        feature_cols=list(meta.get("feature_cols", []) or []),
        seq_len=int(cfg.get("sequence_length", meta.get("seq_len", 20)) or 20),
        meta=meta,
        path=path,
    )


def _load_factor(path: Path) -> ModelArtifact | None:
    from src.factors.factor_model import FactorModel

    model = FactorModel({"training": {"save_dir": str(path.parent)}})
    model.load(str(path))
    return ModelArtifact(
        name=path.stem,
        format="factor",
        model=model,
        feature_cols=list(model.feature_cols_),
        meta={"explain": model.explain(5)},
        path=path,
    )


def _load_timesfm(path: Path, config: dict[str, Any] | None = None) -> ModelArtifact | None:
    """TimesFM 占位：以 joblib 加载占位 dict（真实 TimesFM 由 predictor 实例化）。"""
    import joblib

    data = joblib.load(path)
    payload = data if isinstance(data, dict) else {"model": data}
    return ModelArtifact(
        name=path.stem, format="timesfm", model=payload.get("model"),
        scaler=payload.get("scaler"), meta={"payload_keys": sorted(payload.keys())}, path=path,
    )


class ModelRegistry:
    """模型注册表：按契约名加载任意支持格式的模型。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._loaders: dict[str, Callable[..., ModelArtifact | None]] = {
            "joblib": _load_joblib,
            "torch": _load_torch,
            "factor": _load_factor,
            "timesfm": lambda path: _load_timesfm(path, self.config),
        }

    # ------------------------------------------------------------------
    def register(self, fmt: str, loader: Callable[..., ModelArtifact | None]) -> None:
        """注册自定义格式（扩展点，便于外部系统接入自有模型）。"""
        self._loaders[fmt] = loader
        logger.info("[registry] 已注册模型格式: %s", fmt)

    @property
    def formats(self) -> list[str]:
        return sorted(self._loaders)

    # ------------------------------------------------------------------
    def load(self, path: str | Path, fmt: str | None = None) -> ModelArtifact | None:
        """加载模型；`fmt` 缺省时按扩展名推断（.pkl→joblib, .pt/.pth→torch）。"""
        path = Path(path)
        if not path.exists():
            logger.warning("[registry] 模型文件不存在: %s", path)
            return None
        fmt = fmt or self.infer_format(path)
        loader = self._loaders.get(fmt)
        if loader is None:
            logger.warning("[registry] 未注册的模型格式: %s（支持 %s）", fmt, self.formats)
            return None
        try:
            artifact = loader(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("[registry] 加载 %s (%s) 失败: %s", path.name, fmt, e)
            return None
        if artifact is not None:
            logger.info(
                "[registry] 已加载 %s [%s] 特征 %d 列",
                path.name, artifact.format, len(artifact.feature_cols),
            )
        return artifact

    @staticmethod
    def infer_format(path: Path) -> str:
        name = path.name.lower()
        if name.startswith("factor_model"):
            return "factor"
        if name.startswith("timesfm_"):
            return "timesfm"
        if path.suffix in (".pt", ".pth"):
            return "torch"
        return "joblib"

    # ------------------------------------------------------------------
    def load_horizon(self, save_dir: str | Path, model_type: str, model_key: str) -> ModelArtifact | None:
        """按 `{model_type}_{model_key}` 契约名加载某周期的模型。

        契约（与 ModelTrainer.save 产物一致）：
          lightgbm      → lightgbm_<key>.pkl          (joblib)
          pytorch_lstm  → pytorch_lstm_<key>.pt       (torch)
          factor_model  → factor_model_<key>.pkl      (factor)
          timesfm       → timesfm_<key>.pkl           (timesfm 占位)
        """
        save_dir = Path(save_dir)
        mapping = {
            "lightgbm": (f"lightgbm_{model_key}.pkl", "joblib"),
            "pytorch_lstm": (f"pytorch_lstm_{model_key}.pt", "torch"),
            "lstm": (f"pytorch_lstm_{model_key}.pt", "torch"),
            "factor_model": (f"factor_model_{model_key}.pkl", "factor"),
            "factors": (f"factor_model_{model_key}.pkl", "factor"),
            "timesfm": (f"timesfm_{model_key}.pkl", "timesfm"),
        }
        entry = mapping.get(model_type)
        if entry is None:
            logger.warning("[registry] 未知模型类型: %s", model_type)
            return None
        filename, fmt = entry
        path = save_dir / filename
        return self.load(path, fmt=fmt)


def save_artifact(path: str | Path, artifact: ModelArtifact) -> Path:
    """按 artifact.format 落盘（供训练侧统一保存）。"""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact.format == "joblib":
        joblib.dump(
            {"model": artifact.model, "scaler": artifact.scaler,
             "feature_cols": artifact.feature_cols, "meta": artifact.meta},
            path,
        )
    elif artifact.format == "factor":
        artifact.model.save(str(path))
    else:
        raise ValueError(f"save_artifact 不支持格式: {artifact.format}（torch 请用训练器自身的 save）")
    return path
