"""金融市场预测模型 - PyTorch LSTM 时序模型"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM 时序预测网络"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1)


class PyTorchLSTMTrainer:
    """PyTorch LSTM 训练器"""

    def __init__(self, config: dict[str, Any]):
        self.config = config["model"]["pytorch_lstm"]
        self.training_cfg = config["training"]
        self.save_dir = Path(config["training"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if self.training_cfg.get("use_gpu", False) and torch.cuda.is_available() else "cpu"
        )
        self.model: LSTMModel | None = None
        self.best_val_loss = float("inf")

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """创建时序序列"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return (
            torch.FloatTensor(np.array(X_seq)),
            torch.FloatTensor(np.array(y_seq)),
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict[str, list[float]]:
        """训练 LSTM 模型"""
        logger.info(f"开始训练 PyTorch LSTM 模型 (device={self.device})...")

        seq_len = self.config["sequence_length"]
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, seq_len)

        train_ds = TensorDataset(X_train_seq, y_train_seq)
        val_ds = TensorDataset(X_val_seq, y_val_seq)
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"],
            shuffle=True, num_workers=self.training_cfg.get("num_workers", 0),
        )
        val_loader = DataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False)

        input_size = X_train.shape[1]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            output_size=self.config["output_size"],
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        train_losses: list[float] = []
        val_losses: list[float] = []
        epochs = self.config["epochs"]
        log_interval = self.training_cfg.get("log_interval", 10)

        for epoch in range(epochs):
            # 训练
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.save_dir / "lstm_best.pt")

            if (epoch + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                )

        logger.info(f"LSTM 训练完成，最佳验证损失: {self.best_val_loss:.4f}")
        return {"train_loss": train_losses, "val_loss": val_losses}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        seq_len = self.config["sequence_length"]
        X_seq, _ = self._create_sequences(X, np.zeros(len(X)), seq_len)
        self.model.eval()
        with torch.no_grad():
            X_seq = X_seq.to(self.device)
            probs = self.model(X_seq).cpu().numpy()
        return (probs > 0.5).astype(int)

    def save(self, path: str | None = None) -> Path:
        """保存模型"""
        save_path = Path(path) if path else self.save_dir / "lstm_model.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "input_size": self.model.lstm.input_size,
        }, save_path)
        logger.info(f"模型已保存到 {save_path}")
        return save_path

    def load(self, path: str | None = None) -> None:
        """加载模型"""
        load_path = Path(path) if path else self.save_dir / "lstm_model.pt"
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model = LSTMModel(
            input_size=checkpoint["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        logger.info(f"模型已从 {load_path} 加载")
