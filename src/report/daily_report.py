"""金融市场预测模型 - 定期预测报告生成器

生成日常或每周的市场走向预测报告，帮助投资者做出更精准的投资决策。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _fmt_pct(value) -> str:
    """百分比格式化（None / 非法值显示 N/A，绝不显示 0% 冒充真实数据）。"""
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


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
            report_lines.extend(self._generate_gate_section())
            report_lines.extend(self._generate_ic_trend_section())
            report_lines.extend(self._generate_investment_advice(predictions))
        else:
            report_lines.append("**暂无预测数据**")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*本报告仅供参考，不构成投资建议*")

        report = "\n".join(report_lines)
        self._save_report(report, now)
        return report

    def generate(self, predictions: list[dict], period: str = "daily") -> Path:
        """生成报告并返回落盘路径（供 CLI / 测试的稳定接口）。

        Args:
            predictions: 预测结果列表（PredictionEngine.batch_predict 输出）。
            period: "daily" | "weekly"，weekly 时委托 WeeklyReportGenerator。
        Returns:
            生成的报告文件 Path。
        """
        if period == "weekly":
            report = WeeklyReportGenerator(self.config).generate_report(predictions)
            week_start = datetime.now() - timedelta(days=datetime.now().weekday())
            return self.report_dir / f"weekly_report_{week_start.strftime('%Y-%m-%d')}.md"

        self.generate_report(predictions)
        candidates = sorted(self.report_dir.glob("report_*.md"), key=lambda p: p.stat().st_mtime)
        if not candidates:  # 兜底：目录被外部清理时重新落盘
            path = self.report_dir / f"report_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
            path.write_text(self.generate_report(predictions), encoding="utf-8")
            return path
        return candidates[-1]

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

    def _generate_gate_section(self) -> list[str]:
        """生成策略门禁状态章节（Q2）。

        读取 ``reports/strategy_gate.json``（由 ``python main.py gate`` 产出）。
        文件缺失时**不自造结论**：明确写「未评估」，并说明默认只读。
        """
        lines = ["## 🚦 策略门禁（Q2）"]
        gate_dir = self.config.get("strategy_gate", {}).get("report_dir", self.report_dir)
        path = Path(gate_dir) / "strategy_gate.json"
        if not path.exists():
            lines.extend([
                "",
                "- 状态：**未评估**（未找到 `strategy_gate.json`，执行 `python main.py gate` 生成）",
                "- 默认口径：信号**仅作只读观测**，不进入决策路径（fail-close）",
                "",
            ])
            return lines

        try:
            import json as _json

            payload = _json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            lines.extend(["", f"- 门禁结果读取失败：{e}", ""])
            return lines

        state = payload.get("state", "readonly")
        icon = "🟢" if state == "gated" else "🔒"
        lines.extend([
            "",
            f"- 状态：{icon} **{state}**",
            f"- 判定范围：{payload.get('scope', 'all')}",
            f"- 结论：{payload.get('reason', '')}",
        ])
        horizons = payload.get("horizons") or {}
        rows = [
            (h, v) for h, v in horizons.items()
            if isinstance(v, dict) and not str(h).startswith("_")
        ]
        audit = horizons.get("_audit") if isinstance(horizons, dict) else None
        if isinstance(audit, dict):
            lines.append(
                f"- 审计交叉验证：已验证 {audit.get('verified', 0)} 条，"
                f"命中率 {_fmt_pct(audit.get('hit_rate'))}"
            )
        if rows:
            lines.extend([
                "",
                "| 周期 | IC | ICIR | 命中率 | 样本 | 是否达标 |",
                "|------|-----|------|--------|------|----------|",
            ])
            for h, v in sorted(rows):
                lines.append(
                    f"| {h} | {v.get('ic', 0):.4f} | {v.get('icir', 0):.3f} | "
                    f"{v.get('hit_rate', 0):.2%} | {v.get('samples', 0)} | "
                    f"{'✅' if v.get('passed') else '❌'} |"
                )
        blocked = payload.get("blocked_by") or []
        if blocked:
            lines.append("")
            lines.append("**未达标原因**")
            for item in blocked:
                lines.append(f"- {item}")
        if state != "gated":
            lines.append("")
            lines.append("> 未过门禁 → 信号保持**只读观测**，不作为调仓打分因子（fail-close）。")
        lines.append("")
        return lines

    def _generate_ic_trend_section(self) -> list[str]:
        """生成 Q5 信号衰减趋势章节。

        读取 ``reports/ic_trend.json``（由 ``python main.py ic-trend`` 产出）。
        文件缺失时不臆测趋势，明确写「未评估」并给出生成命令。

        为什么日报要放这个：门禁章节只说「现在只读」，
        客户/研究员真正会追问的是「是不是在变差、要多久掉出去」——
        趋势章节回答这个问题，且**不改变门禁判定**（两者口径同源、职责分离）。
        """
        lines = ["## 📉 信号衰减趋势（Q5）"]
        report_dir = (self.config.get("ic_trend", {}) or {}).get("report_dir", self.report_dir)
        path = Path(report_dir) / "ic_trend.json"
        if not path.exists():
            lines.extend([
                "",
                "- 状态：**未评估**（未找到 `ic_trend.json`，执行 `python main.py ic-trend` 生成）",
                "- 说明：趋势监控用于提前安排重训练，不参与门禁放行判定",
                "",
            ])
            return lines

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            lines.extend(["", f"- 趋势结果读取失败：{e}", ""])
            return lines

        horizons = payload.get("horizons") or {}
        rows = [
            (h, v) for h, v in sorted(horizons.items())
            if isinstance(v, dict) and not str(h).startswith("_")
        ]
        lines.extend([
            "",
            f"- 判定口径：窗口 {payload.get('window')} 样本 / 步长 {payload.get('step')}"
            f"（判定时间 {payload.get('generated_at') or 'N/A'}）",
        ])
        if any(v.get("available") for _h, v in rows):
            lines.extend([
                "",
                "| 周期 | 当前 IC | 每步变化 | 命中率 | 趋势 | 距跌破门禁 |",
                "|------|---------|----------|--------|------|------------|",
            ])
            for h, v in rows:
                if not v.get("available"):
                    lines.append(f"| {h} | N/A | N/A | N/A | ❔ unknown | N/A |")
                    continue
                slope = v.get("slope_per_step")
                breach = v.get("steps_to_breach")
                breach_txt = "N/A" if breach is None else ("已跌破" if breach == 0 else f"{breach} 步")
                icon = {"decaying": "🔻", "improving": "🔺", "stable": "➡️"}.get(
                    v.get("status", ""), "❔"
                )
                lines.append(
                    f"| {h} | {v.get('latest_ic', 0):+.4f} | {slope:+.6f} | "
                    f"{_fmt_pct(v.get('latest_hit_rate'))} | {icon} {v.get('status')} | {breach_txt} |"
                )
        decaying = payload.get("decaying") or []
        near = payload.get("near_breach") or []
        if decaying:
            lines.append("")
            lines.append(f"- 🔻 **衰减预警**：{'、'.join(decaying)} —— 建议提前安排重训练")
        if near:
            lines.append("")
            lines.append(
                f"- ⏳ **临近跌破**：{'、'.join(near)}"
                f"（{payload.get('near_breach_steps', 3)} 步内）"
            )
        lines.extend([
            "",
            "> 趋势为**预警信号**，不改变门禁判定：放行与否仍只看当前 IC / 命中率"
            "（见上方「策略门禁」）。两者**同源序列**，不会出现口径打架。",
            "",
        ])
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
            report_lines.extend(self._generate_gate_section())
            report_lines.extend(self._generate_ic_trend_section())
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