"""金融市场预测模型 - 定期预测报告生成器

生成日常或每周的市场走向预测报告，帮助投资者做出更精准的投资决策。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """每日预测报告生成器"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.report_dir = Path(config.get("report", {}).get("output_dir", "reports"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.sentiment_analyzer = None
        if config.get("features", {}).get("sentiment_enabled", False):
            try:
                from src.data.sentiment_analyzer import SentimentFeatureGenerator
                self.sentiment_analyzer = SentimentFeatureGenerator(config)
            except Exception as e:
                logger.warning(f"情感分析模块初始化失败: {e}")

    def generate_report(self, predictions: list[dict]) -> str:
        """生成预测报告"""
        now = datetime.now()
        report_lines = [
            "# TrendCast Pro - 市场预测报告",
            "",
            f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**报告周期**: {now.strftime('%Y年%m月%d日')}",
            "",
            "---",
            "",
        ]

        if predictions:
            report_lines.extend(self._generate_market_overview(predictions))
            report_lines.extend(self._generate_detailed_predictions(predictions))
            report_lines.extend(self._generate_sentiment_section())
            report_lines.extend(self._generate_investment_advice(predictions))
        else:
            report_lines.append("**暂无预测数据**")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*本报告仅供参考，不构成投资建议*")

        report = "\n".join(report_lines)
        self._save_report(report, now)
        return report

    def _generate_market_overview(self, predictions: list[dict]) -> list[str]:
        """生成市场概览"""
        bullish = 0
        bearish = 0
        total = 0

        for p in predictions:
            for h, pred in p.get("predictions", {}).items():
                if "error" not in pred:
                    total += 1
                    if pred["direction"] == "看涨":
                        bullish += 1
                    else:
                        bearish += 1

        bullish_ratio = bullish / total if total > 0 else 0
        bearish_ratio = bearish / total if total > 0 else 0

        lines = [
            "## 📊 市场概览",
            "",
            f"- **预测标的数**: {len(predictions)}",
            f"- **总预测次数**: {total}",
            f"- **看涨比例**: {bullish}/{total} ({bullish_ratio:.1%})",
            f"- **看跌比例**: {bearish}/{total} ({bearish_ratio:.1%})",
            "",
            "### 市场情绪指标",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 市场偏多度 | {'🟢 偏多' if bullish_ratio > 0.6 else '🔴 偏空' if bearish_ratio > 0.6 else '🟡 中性'} |",
            f"| 一致性 | {'高' if abs(bullish_ratio - bearish_ratio) > 0.3 else '中'} |",
            "",
        ]
        return lines

    def _generate_detailed_predictions(self, predictions: list[dict]) -> list[str]:
        """生成详细预测列表"""
        lines = ["## 📈 详细预测"]

        for p in predictions:
            symbol = p.get("symbol", "?")
            preds = p.get("predictions", {})
            if not preds:
                continue

            lines.append("")
            lines.append(f"### {symbol}")
            lines.append("")
            lines.append("| 周期 | 方向 | 概率 | 置信度 |")
            lines.append("|------|------|------|--------|")

            for h, pred in preds.items():
                if "error" in pred:
                    continue
                icon = "📈" if pred["direction"] == "看涨" else "📉"
                lines.append(
                    f"| {h} ({pred['horizon_days']}日) | {icon} {pred['direction']} | "
                    f"{pred['probability']:.0%} | {pred['confidence']:.0%} |"
                )

        lines.append("")
        return lines

    def _generate_sentiment_section(self) -> list[str]:
        """生成情感分析部分"""
        lines = ["## 📰 新闻情感分析"]

        if not self.sentiment_analyzer:
            lines.append("- 情感分析模块未启用")
            lines.append("")
            return lines

        try:
            today = datetime.now().strftime("%Y-%m-%d")
            sentiment_df = self.sentiment_analyzer.get_daily_sentiment(today)

            if sentiment_df.empty:
                lines.append("- 今日无新闻数据")
            else:
                avg_sentiment = sentiment_df["sentiment"].mean()
                news_count = len(sentiment_df)
                pos_ratio = sentiment_df["positive_score"].mean()
                neg_ratio = sentiment_df["negative_score"].mean()

                lines.extend([
                    "",
                    f"- **新闻数量**: {news_count}",
                    f"- **平均情感分**: {avg_sentiment:.2f} (范围 -1 到 1)",
                    f"- **正面倾向**: {pos_ratio:.1%}",
                    f"- **负面倾向**: {neg_ratio:.1%}",
                    "",
                    f"**情感判断**: {'🟢 偏多' if avg_sentiment > 0.1 else '🔴 偏空' if avg_sentiment < -0.1 else '🟡 中性'}",
                    "",
                ])

                top_news = sentiment_df.sort_values("sentiment", ascending=False).head(3)
                if not top_news.empty:
                    lines.append("### 重点新闻")
                    for _, row in top_news.iterrows():
                        icon = "📈" if row["sentiment"] > 0 else "📉"
                        lines.append(f"- {icon} {row['title'][:60]}...")
        except Exception as e:
            lines.append(f"- 情感分析失败: {e}")

        lines.append("")
        return lines

    def _generate_investment_advice(self, predictions: list[dict]) -> list[str]:
        """生成投资建议"""
        lines = ["## 💡 投资建议"]

        high_conf_bullish = []
        high_conf_bearish = []

        for p in predictions:
            for h, pred in p.get("predictions", {}).items():
                if "error" in pred:
                    continue
                if pred["confidence"] >= 0.6:
                    if pred["direction"] == "看涨":
                        high_conf_bullish.append((p["symbol"], h, pred["confidence"]))
                    else:
                        high_conf_bearish.append((p["symbol"], h, pred["confidence"]))

        if high_conf_bullish:
            lines.append("")
            lines.append("### 值得关注（高置信看涨）")
            for symbol, horizon, conf in sorted(high_conf_bullish, key=lambda x: -x[2]):
                lines.append(f"- **{symbol}**: {horizon} 看涨 (置信度 {conf:.0%})")

        if high_conf_bearish:
            lines.append("")
            lines.append("### 需要警惕（高置信看跌）")
            for symbol, horizon, conf in sorted(high_conf_bearish, key=lambda x: -x[2]):
                lines.append(f"- **{symbol}**: {horizon} 看跌 (置信度 {conf:.0%})")

        lines.append("")
        lines.append("### 风险提示")
        lines.append("- 市场有风险，投资需谨慎")
        lines.append("- 模型预测仅供参考，不构成投资建议")
        lines.append("- 建议结合多维度分析做出决策")
        lines.append("")

        return lines

    def _save_report(self, report: str, timestamp: datetime):
        """保存报告"""
        filename = f"report_{timestamp.strftime('%Y-%m-%d_%H%M%S')}.md"
        path = self.report_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"报告已保存到 {path}")


class WeeklyReportGenerator(DailyReportGenerator):
    """每周预测报告生成器"""

    def generate_report(self, predictions: list[dict]) -> str:
        """生成周度报告"""
        now = datetime.now()
        week_start = now - timedelta(days=now.weekday())
        week_end = week_start + timedelta(days=6)

        report_lines = [
            "# TrendCast Pro - 周度市场预测报告",
            "",
            f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**报告周期**: {week_start.strftime('%Y年%m月%d日')} ~ {week_end.strftime('%Y年%m月%d日')}",
            "",
            "---",
            "",
        ]

        if predictions:
            report_lines.extend(self._generate_market_overview(predictions))
            report_lines.extend(self._generate_detailed_predictions(predictions))
            report_lines.extend(self._generate_weekly_summary())
        else:
            report_lines.append("**暂无预测数据**")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*本报告仅供参考，不构成投资建议*")

        report = "\n".join(report_lines)
        filename = f"weekly_report_{week_start.strftime('%Y-%m-%d')}.md"
        path = self.report_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"周度报告已保存到 {path}")
        return report

    def _generate_weekly_summary(self) -> list[str]:
        """生成周度总结"""
        lines = ["## 📅 周度总结"]
        lines.append("")
        lines.append("### 本周重点关注")
        lines.append("- 宏观经济数据发布")
        lines.append("- 重要政策公告")
        lines.append("- 行业事件跟踪")
        lines.append("")
        lines.append("### 下周展望")
        lines.append("- 建议关注长期预测信号")
        lines.append("- 结合技术面分析综合判断")
        lines.append("- 控制仓位，做好风险管理")
        lines.append("")
        return lines


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = {"report": {"output_dir": "reports"}, "features": {"sentiment_enabled": True}}
    generator = DailyReportGenerator(config)

    test_predictions = [
        {
            "symbol": "600519.SH",
            "predictions": {
                "short_term": {"direction": "看涨", "probability": 0.65, "confidence": 0.65, "horizon_days": 5},
                "mid_term": {"direction": "看跌", "probability": 0.58, "confidence": 0.58, "horizon_days": 10},
            },
        }
    ]

    report = generator.generate_report(test_predictions)
    print(report)