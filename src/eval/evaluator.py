"""金融市场预测模型 - 模型评估器"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器 - 标准 ML 指标 + 金融专用指标"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.metrics_cfg = config.get("evaluation", {})
        self.results: dict[str, Any] = {}

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                 horizon_name: str = "short_term", horizon_days: int = 5,
                 fee: float = 0.0) -> dict[str, Any]:
        """
        评估模型性能

        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            horizon_name: 预测周期名称
            horizon_days: 预测天数

        Returns:
            评估结果字典
        """
        logger.info(f"开始评估 [{horizon_name}/{horizon_days}d]...")
        y_pred = np.asarray(model.predict(X_test))
        # 序列模型（LSTM）会因滑窗丢弃前若干样本：按实际预测长度反向对齐标签，
        # 保证 y_true 与 y_pred 等长（否则 accuracy_score 直接抛长度不一致）。
        y_test, X_test = self._align_labels_like_pred(X_test, y_test, len(y_pred))

        proba = None
        try:
            proba = model.predict_proba(X_test)
        except (AttributeError, IndexError, ValueError):
            proba = None
        if proba is None:
            y_proba = np.asarray(y_pred, dtype=float)
        else:
            proba = np.asarray(proba)
            # LSTM 的 predict_proba 返回一维概率（非 (n,2) 矩阵）
            y_proba = proba[:, 1] if proba.ndim == 2 else proba
        y_proba = np.asarray(y_proba, dtype=float)

        # 标准 ML 指标
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report,
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        try:
            metrics["auc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            metrics["auc"] = 0.0

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

        # 金融专用指标（含扣费口径，便于与"毛收益"对照）
        financial_metrics = self._compute_financial_metrics(
            y_test, y_pred, y_proba, horizon_days=horizon_days, fee=fee
        )
        metrics["financial"] = financial_metrics

        # 打印结果
        logger.info(f"--- 评估结果 [{horizon_name}/{horizon_days}d] ---")
        logger.info(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        logger.info(f"  精确率 (Precision): {metrics['precision']:.4f}")
        logger.info(f"  召回率 (Recall):    {metrics['recall']:.4f}")
        logger.info(f"  F1 分数:            {metrics['f1']:.4f}")
        logger.info(f"  AUC:                {metrics['auc']:.4f}")
        logger.info(f"  --- 金融指标 ---")
        logger.info(f"  胜率 (Win Rate):    {financial_metrics['win_rate']:.4f}")
        logger.info(f"  盈亏比 (P/F):       {financial_metrics['profit_factor']:.4f}")
        logger.info(f"  夏普比率(近似年化): {financial_metrics['sharpe_ratio']:.4f}")
        logger.info(f"  最大回撤(笔):       {financial_metrics['max_drawdown']:.1f}")
        logger.info(f"  混淆矩阵: TP={cm[1][1]}, FP={cm[0][1]}, TN={cm[0][0]}, FN={cm[1][0]}")

        result = {
            "horizon_name": horizon_name,
            "horizon_days": horizon_days,
            "metrics": metrics,
            "test_samples": int(len(y_test)),
        }

        self.results[f"{horizon_name}_{horizon_days}d"] = result
        return result

    @staticmethod
    def _align_labels_like_pred(X_test: Any, y_test: Any,
                                n_pred: int) -> tuple[np.ndarray, Any]:
        """把标签对齐到预测长度：截掉前 (n - n_pred) 个样本（序列模型滑窗丢弃）。

        通用做法不依赖具体 seq_len 配置，对任何"预测少于输入"的模型都成立，
        因此 LSTM 换结构 / 改窗口都不会再触发长度不一致。
        """
        y_arr = np.asarray(y_test)
        n = len(y_arr)
        if n_pred <= 0 or n_pred > n:
            raise ValueError(f"预测长度 {n_pred} 与标签长度 {n} 不匹配")
        if n_pred == n:
            return y_test, X_test
        drop = n - n_pred
        logger.info(f"序列模型标签对齐：样本 {n} → {n_pred}（丢弃前 {drop} 个滑窗样本）")
        return y_arr.drop if False else y_arr[drop:], np.asarray(X_test)[drop:]

    def _compute_financial_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: np.ndarray, horizon_days: int = 5,
                                   fee: float = 0.0) -> dict[str, float]:
        """计算金融专用指标（符号博弈口径，可选扣费）。

        口径说明（重要，防误读）：
        - 交易模拟为 ±1 **等权符号博弈**（预测涨做多 / 预测跌做空），
          收益单位是"1 次对赌"，**不是收益率、不是资金曲线**；
        - 因此 sharpe_ratio 是"每 horizon_days 个交易日一次对赌"的近似年化
          （mean/std * sqrt(252/horizon_days)），量级会明显大于真实资金夏普；
          **它用于横向比较不同模型/周期的相对优劣，不能当作实盘夏普看**；
        - profit_factor 为"毛盈亏比"（总盈利笔数 / 总亏损笔数，等权 1:1 下等于
          胜笔数/败笔数），**该值有解析上限**：等权 ±1 口径下最大就是
          `winning_trades / losing_trades`，全对时退化为 999 哨兵值，
          切勿解读为"盈亏比 999"；
        - max_drawdown 单位为"笔"（±1 累计净胜局相对峰值的最深回落），
          恒为非正值，**不是百分比**，不可与资金回撤率混用；
        - `fee` 给定时额外输出 `*_net` 扣费口径：按**双边**成本从每笔对赌收益中扣除
          `2 × fee`（与 scripts/evaluate_models.py 的 --fee 口径一致）。
        """
        # 胜率：预测正确的比例
        win_rate = float(np.mean(y_pred == y_true))

        # 模拟交易收益（预测涨则做多，预测跌则做空）
        returns = np.where(y_pred == 1, 1.0, -1.0)  # 简化为等权收益
        actual_direction = np.where(y_true == 1, 1.0, -1.0)
        trade_returns = returns * actual_direction  # 对则+1，错则-1

        gross = self._summarize_trade_returns(trade_returns, horizon_days, label="gross")
        out: dict[str, Any] = dict(gross)
        out["win_rate"] = win_rate
        out["fee_per_trade"] = float(fee) * 2.0
        out["fee_note"] = (
            "fee 为单边费率，扣费口径按双边 2×fee 从每笔对赌收益中扣除"
        )

        if fee and fee > 0:
            net = self._summarize_trade_returns(
                trade_returns - float(fee) * 2.0, horizon_days, label="net"
            )
            for key, value in net.items():
                out[f"{key}_net"] = value
            out["breakeven_win_rate"] = round(
                min(max((1.0 + 2.0 * float(fee)) / 2.0, 0.0), 1.0), 4
            )
        return out

    @staticmethod
    def _summarize_trade_returns(trade_returns: np.ndarray, horizon_days: int,
                                 label: str = "gross") -> dict[str, float]:
        """把逐笔对赌收益汇总为盈亏比 / 近似夏普 / 最大回撤（笔）。

        `label` 仅用于日志可读性；返回键名与既有契约保持一致（不带后缀），
        以便调用方与历史报告无缝对接。
        """
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        total_win = float(wins.sum()) if len(wins) > 0 else 0.0
        total_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

        # 夏普比率：按不重叠周期近似年化（每年约 252/horizon_days 笔）
        ann_factor = float(np.sqrt(252.0 / max(int(horizon_days), 1)))
        std = float(trade_returns.std()) if len(trade_returns) else 0.0
        sharpe_ratio = float(trade_returns.mean() / std * ann_factor) if std > 0 else 0.0

        # 最大回撤：±1 累计净胜局口径（单位：笔，恒为非正值）
        cumulative = np.cumsum(trade_returns) if len(trade_returns) else np.array([0.0])
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = float((cumulative - running_max).min())

        return {
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
            "profit_factor_is_capped": bool(profit_factor == float("inf")),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "sharpe_is_notional": True,
            "max_drawdown": round(max_drawdown, 4),
            "max_drawdown_unit": "trades",
            "total_trades": int(len(trade_returns)),
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
        }

    def generate_report(self, output_path: str | None = None) -> str:
        """生成评估报告"""
        report_lines = [
            "=" * 60,
            "金融市场预测模型 - 评估报告",
            "=" * 60,
            "",
        ]

        for key, result in self.results.items():
            m = result["metrics"]
            fm = m["financial"]
            report_lines.extend([
                f"[{result['horizon_name']}] 预测周期: {result['horizon_days']}天",
                f"  测试样本数: {result['test_samples']}",
                f"  特征数量: {result.get('feature_count', 'N/A')}",
                "",
                "  --- 分类指标 ---",
                f"  准确率:   {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)",
                f"  精确率:   {m['precision']:.4f}",
                f"  召回率:   {m['recall']:.4f}",
                f"  F1 分数:  {m['f1']:.4f}",
                f"  AUC:      {m['auc']:.4f}",
                "",
                "  --- 金融指标（等权符号博弈口径，非资金曲线）---",
                f"  胜率:     {fm['win_rate']:.4f} ({fm['win_rate']*100:.2f}%)",
                f"  盈亏比(毛): {fm['profit_factor']:.4f}"
                + ("（已达解析上限，请勿读作真实盈亏比）" if fm.get("profit_factor_is_capped") else ""),
                f"  夏普比率(对赌口径近似年化): {fm['sharpe_ratio']:.4f}"
                "（横向比较用，非资金夏普）",
                f"  最大回撤(笔):       {fm['max_drawdown']:.1f}",
                f"  总交易数: {fm['total_trades']}",
                f"  盈利交易: {fm['winning_trades']}",
                f"  亏损交易: {fm['losing_trades']}",
                "",
                "-" * 60,
                "",
            ])

        report = "\n".join(report_lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"评估报告已保存到 {output_path}")

        return report
