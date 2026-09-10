"""专业版 - 预测审计系统

记录每次预测结果，在预测周期到期后回溯验证准确率，生成审计报告。
支持按标的/周期/时间段统计命中率，识别模型漂移。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PredictionAudit:
    """预测审计器 - 记录、回溯、统计"""

    def __init__(self, audit_dir: str = "logs/audit"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.record_file = self.audit_dir / "predictions.jsonl"
        self.report_file = self.audit_dir / "audit_report.md"

    def record_prediction(self, prediction: dict[str, Any]) -> None:
        """记录一条预测（供 PredictionEngine 调用）"""
        entry = {
            "id": f"{prediction.get('symbol','')}_{prediction.get('horizon','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "symbol": prediction.get("symbol"),
            "horizon": prediction.get("horizon"),
            "horizon_days": prediction.get("horizon_days"),
            "prediction": prediction.get("prediction"),
            "direction": prediction.get("direction"),
            "probability": prediction.get("probability"),
            "confidence": prediction.get("confidence"),
            "latest_date": prediction.get("latest_date"),
            "latest_close": prediction.get("latest_close"),
            "verified": False,
            "actual_direction": None,
            "hit": None,
            "actual_return": None,
        }
        with open(self.record_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug(f"审计记录: {entry['id']}")

    def load_records(self) -> list[dict]:
        """加载所有预测记录（公开接口）"""
        return self._load_records()

    def _load_records(self) -> list[dict]:
        """加载所有预测记录"""
        if not self.record_file.exists():
            return []
        records = []
        with open(self.record_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return records

    def verify_predictions(self, data_fetcher) -> int:
        """回溯验证已到期的预测

        Args:
            data_fetcher: 可调用的数据获取函数 (symbol, start, end) -> DataFrame
        Returns:
            新验证的预测数量
        """
        records = self._load_records()
        verified_count = 0
        now = datetime.now()

        updated = []
        for rec in records:
            if rec.get("verified"):
                updated.append(rec)
                continue

            pred_time = datetime.fromisoformat(rec["timestamp"])
            horizon_days = rec.get("horizon_days", 5)
            verify_date = pred_time + timedelta(days=horizon_days)

            if now < verify_date:
                updated.append(rec)
                continue

            # 已到期，获取实际数据验证
            try:
                symbol = rec["symbol"]
                latest_date = rec.get("latest_date")
                start = latest_date or pred_time.strftime("%Y-%m-%d")
                end = now.strftime("%Y-%m-%d")
                df = data_fetcher(symbol, start, end)
                if df is None or len(df) < 2:
                    updated.append(rec)
                    continue

                start_close = float(df.iloc[0]["close"])
                end_close = float(df.iloc[-1]["close"])
                actual_return = (end_close - start_close) / start_close
                actual_direction = 1 if actual_return > 0 else 0

                rec["verified"] = True
                rec["actual_direction"] = actual_direction
                rec["actual_return"] = round(actual_return, 6)
                rec["hit"] = (rec["prediction"] == actual_direction)
                verified_count += 1
            except Exception as e:
                logger.warning(f"验证 {rec.get('id')} 失败: {e}")

            updated.append(rec)

        # 回写
        with open(self.record_file, "w", encoding="utf-8") as f:
            for rec in updated:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"审计验证完成: 新验证 {verified_count} 条")
        return verified_count

    def generate_report(self) -> str:
        """生成审计报告"""
        now = datetime.now()
        records = self._load_records()
        verified = [r for r in records if r.get("verified")]

        if not verified:
            # 尚无已验证记录时也输出完整报告骨架：报告始终可读、字段稳定，
            # 便于上游（28 系统 / 日报）无差别解析。
            pending = len(records) - len(verified)
            lines = [
                "# 预测审计报告",
                "",
                f"生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 总览",
                "",
                "| 指标 | 值 |",
                "|------|-----|",
                f"| 总记录数 | {len(records)} |",
                f"| 总预测数 | {len(records)} |",
                "| 已验证数 | 0 |",
                "| 命中数 | 0 |",
                "| 整体命中率 | 0.00% |",
                f"| 待验证数 | {pending} |",
                "",
                "> 暂无已验证的预测记录（预测尚未到期，或缺少可用于回溯的真实行情）。",
            ]
            report = "\n".join(lines)
            with open(self.report_file, "w", encoding="utf-8") as f:
                f.write(report)
            return report

        total = len(verified)
        hits = sum(1 for r in verified if r.get("hit"))
        hit_rate = hits / total if total > 0 else 0

        # 按周期统计
        by_horizon: dict[str, dict] = {}
        for r in verified:
            h = r.get("horizon", "unknown")
            if h not in by_horizon:
                by_horizon[h] = {"total": 0, "hits": 0}
            by_horizon[h]["total"] += 1
            if r.get("hit"):
                by_horizon[h]["hits"] += 1

        # 按标的统计
        by_symbol: dict[str, dict] = {}
        for r in verified:
            s = r.get("symbol", "unknown")
            if s not in by_symbol:
                by_symbol[s] = {"total": 0, "hits": 0}
            by_symbol[s]["total"] += 1
            if r.get("hit"):
                by_symbol[s]["hits"] += 1

        lines = [
            "# 预测审计报告",
            f"",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 总览",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总记录数 | {len(records)} |",
            f"| 总预测数 | {len(records)} |",
            f"| 已验证数 | {total} |",
            f"| 命中数 | {hits} |",
            f"| 整体命中率 | {hit_rate:.2%} |",
            f"",
            f"## 按预测周期",
            f"",
            f"| 周期 | 验证数 | 命中数 | 命中率 |",
            f"|------|--------|--------|--------|",
        ]
        for h, stats in sorted(by_horizon.items()):
            hr = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"| {h} | {stats['total']} | {stats['hits']} | {hr:.2%} |")

        lines.extend([f"", f"## 按标的", f"", f"| 标的 | 验证数 | 命中数 | 命中率 |", f"|------|--------|--------|--------|"])
        for s, stats in sorted(by_symbol.items(), key=lambda x: x[1]["total"], reverse=True):
            hr = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"| {s} | {stats['total']} | {stats['hits']} | {hr:.2%} |")

        # 模型漂移检测
        recent = [r for r in verified if
                  (now - datetime.fromisoformat(r["timestamp"])).days <= 30]
        recent_hits = sum(1 for r in recent if r.get("hit"))
        recent_rate = recent_hits / len(recent) if recent else 0

        lines.extend([
            f"",
            f"## 模型漂移检测",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 近30天验证数 | {len(recent)} |",
            f"| 近30天命中率 | {recent_rate:.2%} |",
            f"| 整体命中率 | {hit_rate:.2%} |",
            f"| 漂移信号 | {'⚠ 命中率显著下降，建议重训练' if recent_rate < hit_rate - 0.1 and len(recent) > 10 else '正常'} |",
        ])

        report = "\n".join(lines)
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"审计报告已保存到 {self.report_file}")
        return report
