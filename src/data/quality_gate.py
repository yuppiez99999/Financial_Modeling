"""数据质量评估模块 - 基于 Token A级数据训练质量分类器

数据来源：北数所备案 Token A级数据产品（finance 域）
功能：
  1. 训练 A/B 级数据质量分类器（LightGBM）
  2. 训练质量评分回归模型（预测 data_quality_score）
  3. 对外部数据源数据做质量打分，输出可信度权重
  4. 供 DataCollector 多源回退链使用：优选高质量数据源
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Token 数据默认路径（小样本优先，全量太大）
TOKEN_DATA_DIR = Path(__file__).parent.parent.parent.parent / "01_数据源与数据处理" / "20260619Token A级数据"


class TokenQualityClassifier:
    """Token 数据质量分类器 - 预测数据属于 A 级还是 B 级"""

    FEATURE_COLS = ["completeness", "accuracy", "timeliness", "compliance_score"]
    LABEL_COL = "token_level"

    def __init__(self, model_dir: str = "models/quality"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = None
        self.regressor = None
        self.encoder_map: dict[str, int] = {}
        self._loaded = False

    def load_token_data(self, domain: str = "finance", use_full: bool = False) -> pd.DataFrame:
        """加载 Token 数据

        Args:
            domain: 数据领域（finance/energy/healthcare/manufacturing/transport）
            use_full: True 用全量数据（500万+），False 用小样本（2-6万条）
        """
        suffix = "192139" if use_full else "191830"
        csv_path = TOKEN_DATA_DIR / f"{domain}_token_A_B_20260619_{suffix}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Token 数据文件不存在: {csv_path}")

        logger.info(f"加载 Token 数据: {csv_path} ({'全量' if use_full else '小样本'})")
        df = pd.read_csv(csv_path)

        # 标签编码：A=1, B=0
        self.encoder_map = {"A": 1, "B": 0}
        df["label"] = df[self.LABEL_COL].map(self.encoder_map)

        logger.info(f"加载完成: {len(df)} 条, A级={df['label'].sum()} ({df['label'].mean():.1%})")
        return df

    def train(self, domain: str = "finance", use_full: bool = False) -> dict[str, Any]:
        """训练质量分类器和回归模型

        Returns:
            训练结果（准确率、AUC 等）
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        from sklearn.model_selection import train_test_split
        import lightgbm as lgb

        df = self.load_token_data(domain, use_full)

        X = df[self.FEATURE_COLS].values
        y_cls = df["label"].values  # A/B 分类
        y_reg = df["data_quality_score"].values  # 质量分回归

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )

        # 分类器：预测 A/B 级
        self.classifier = lgb.LGBMClassifier(
            objective="binary", n_estimators=200, learning_rate=0.05,
            max_depth=6, num_leaves=31, subsample=0.8, verbose=-1,
        )
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.0

        # 回归器：预测质量分
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        self.regressor = lgb.LGBMRegressor(
            objective="regression", n_estimators=200, learning_rate=0.05,
            max_depth=6, num_leaves=31, subsample=0.8, verbose=-1,
        )
        self.regressor.fit(X_train_r, y_train_r)
        y_pred_r = self.regressor.predict(X_test_r)
        mae = float(np.mean(np.abs(y_test_r - y_pred_r)))
        rmse = float(np.sqrt(np.mean((y_test_r - y_pred_r) ** 2)))

        # 保存模型
        joblib.dump(
            {"classifier": self.classifier, "regressor": self.regressor, "features": self.FEATURE_COLS},
            self.model_dir / "quality_model.pkl",
        )

        self._loaded = True
        result = {
            "domain": domain,
            "samples": len(df),
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "quality_mae": round(mae, 4),
            "quality_rmse": round(rmse, 4),
            "feature_importance": dict(zip(
                self.FEATURE_COLS,
                self.classifier.feature_importances_.tolist(),
            )),
        }
        logger.info(f"质量模型训练完成: acc={acc:.4f}, auc={auc:.4f}, mae={mae:.4f}")
        return result

    def load(self) -> bool:
        """加载已训练的质量模型"""
        model_path = self.model_dir / "quality_model.pkl"
        if not model_path.exists():
            return False
        data = joblib.load(model_path)
        self.classifier = data["classifier"]
        self.regressor = data["regressor"]
        self._loaded = True
        logger.info("已加载质量模型")
        return True

    def score_data(self, completeness: float, accuracy: float,
                   timeliness: float, compliance: float = 100.0) -> dict[str, Any]:
        """对数据做质量打分

        Returns:
            {"quality_score": float, "is_a_level": bool, "a_probability": float, "trust_weight": float}
        """
        if not self._loaded:
            self.load()
        if not self._loaded:
            # 模型未训练，返回默认值
            avg_q = np.mean([completeness, accuracy, timeliness, compliance])
            return {
                "quality_score": round(avg_q, 2),
                "is_a_level": avg_q >= 96.0,
                "a_probability": 0.5,
                "trust_weight": round(avg_q / 100.0, 4),
            }

        X = np.array([[completeness, accuracy, timeliness, compliance]])
        a_proba = float(self.classifier.predict_proba(X)[0][1])
        quality = float(self.regressor.predict(X)[0])

        return {
            "quality_score": round(quality, 2),
            "is_a_level": a_proba > 0.5,
            "a_probability": round(a_proba, 4),
            "trust_weight": round(quality / 100.0, 4),
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """对 DataFrame 批量质量打分

        要求 df 含 completeness/accuracy/timeliness/compliance_score 列，
        若无则根据数据特征估算
        """
        if not self._loaded:
            self.load()

        # 如果数据没有质量维度列，用启发式估算
        for col, default in [("completeness", 97.0), ("accuracy", 97.0),
                             ("timeliness", 96.0), ("compliance_score", 100.0)]:
            if col not in df.columns:
                df[col] = default

        X = df[self.FEATURE_COLS].values
        if self._loaded and self.classifier:
            df["quality_score"] = self.regressor.predict(X)
            df["a_probability"] = self.classifier.predict_proba(X)[:, 1]
            df["token_level_pred"] = np.where(df["a_probability"] > 0.5, "A", "B")
            df["trust_weight"] = df["quality_score"] / 100.0
        else:
            df["quality_score"] = df[self.FEATURE_COLS].mean(axis=1)
            df["a_probability"] = 0.5
            df["token_level_pred"] = "A"
            df["trust_weight"] = df["quality_score"] / 100.0

        return df


class DataQualityGate:
    """数据质量门控 - 在多源数据采集中做质量筛选

    用法：
      gate = DataQualityGate()
      gate.train()  # 首次训练
      # 采集数据后
      scored = gate.score_market_data(df)
      # 只用 A 级数据训练
      df_a = scored[scored["token_level_pred"] == "A"]
    """

    def __init__(self):
        self.classifier = TokenQualityClassifier()

    def train(self, domain: str = "finance", use_full: bool = False) -> dict:
        """训练质量模型"""
        return self.classifier.train(domain, use_full)

    def score_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """对行情数据做质量评估

        行情数据没有显式的质量维度，用数据特征估算：
          - completeness: 非空率
          - accuracy: 价格合理性（无异常值）
          - timeliness: 数据新鲜度
          - compliance_score: 100（合规默认满分）
        """
        df = df.copy()

        # 估算完整性：各列非空率
        if len(df) > 0:
            completeness = df.notna().mean().mean() * 100
        else:
            completeness = 0.0

        # 估算准确性：收盘价是否在合理范围（无负值、无极端值）
        if "close" in df.columns and len(df) > 0:
            close = df["close"].dropna()
            if len(close) > 0:
                neg_ratio = (close < 0).mean()
                zeros_ratio = (close == 0).mean()
                accuracy = (1 - neg_ratio - zeros_ratio * 0.5) * 100
                accuracy = max(0, min(100, accuracy))
            else:
                accuracy = 0.0
        else:
            accuracy = 90.0

        # 估算时效性：最新日期距今天数
        if "date" in df.columns and len(df) > 0:
            try:
                latest_date = pd.to_datetime(df["date"].iloc[-1], errors="coerce", utc=True)
                if pd.isna(latest_date):
                    latest_date = pd.to_datetime(df["date"].iloc[-1], errors="coerce")
                days_old = (pd.Timestamp.now(tz="UTC") - latest_date).days if not pd.isna(latest_date) else 999
                timeliness = max(0, 100 - days_old * 0.5)
            except Exception:
                timeliness = 95.0
        else:
            timeliness = 95.0

        # 为每行附加质量分
        df["completeness"] = completeness
        df["accuracy"] = accuracy
        df["timeliness"] = timeliness
        df["compliance_score"] = 100.0

        df = self.classifier.score_dataframe(df)
        logger.info(
            f"数据质量评估: {len(df)}条, 完整性={completeness:.1f}, "
            f"准确性={accuracy:.1f}, 时效性={timeliness:.1f}, "
            f"质量分={df['quality_score'].mean():.1f}, "
            f"A级占比={df['token_level_pred'].eq('A').mean():.1%}"
        )
        return df

    def filter_by_quality(self, df: pd.DataFrame, min_quality: float = 95.0,
                          a_level_only: bool = False) -> pd.DataFrame:
        """按质量筛选数据"""
        scored = self.score_market_data(df)
        mask = scored["quality_score"] >= min_quality
        if a_level_only:
            mask &= scored["token_level_pred"] == "A"
        filtered = scored[mask]
        logger.info(f"质量筛选: {len(df)} -> {len(filtered)} 条 (min_quality={min_quality}, a_only={a_level_only})")
        return filtered
