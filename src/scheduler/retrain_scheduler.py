"""专业版 - 自动重训练调度器

支持按 周/月 频率自动重训练模型，可配合系统定时任务或独立运行。
包含模型漂移检测：命中率下降超阈值时自动触发重训练。
"""

from __future__ import annotations

import logging
import schedule
import threading
import time
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """模型自动重训练调度器"""

    def __init__(self, config: dict[str, Any], train_fn: Callable[[], Any] | None = None):
        self.config = config
        scheduler_cfg = config.get("scheduler", {})
        self.freq = scheduler_cfg.get("frequency", "weekly")  # weekly/monthly/daily
        self.time = scheduler_cfg.get("time", "23:00")
        self.day = scheduler_cfg.get("day", "sun")  # mon/tue/.../sun
        self.drift_threshold = scheduler_cfg.get("drift_threshold", 0.1)
        self.auto_retrain_on_drift = scheduler_cfg.get("auto_retrain_on_drift", True)

        self._train_fn = train_fn
        self._thread: threading.Thread | None = None
        self._running = False
        self.last_run: datetime | None = None
        self.run_count = 0

    def _do_retrain(self, reason: str = "scheduled") -> bool:
        """执行重训练"""
        logger.info(f"触发重训练 (原因: {reason})")
        try:
            if self._train_fn:
                self._train_fn()
            else:
                from src.train.trainer import ModelTrainer
                trainer = ModelTrainer(self.config)
                trainer.train_pipeline()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.info(f"重训练完成 (第 {self.run_count} 次)")
            return True
        except Exception as e:
            logger.error(f"重训练失败: {e}")
            return False

    def _check_drift(self) -> bool:
        """检查模型漂移（直接调用审计系统 API 而非解析报告文本）"""
        try:
            from src.audit.prediction_audit import PredictionAudit
            audit = PredictionAudit()
            audit.verify_predictions(self._fetch_data_for_verify)

            # 直接计算漂移，不依赖报告文本解析
            records = audit._load_records()
            verified = [r for r in records if r.get("verified")]
            if not verified:
                return False

            from datetime import datetime, timedelta
            now = datetime.now()
            cutoff = now - timedelta(days=30)
            recent = [r for r in verified
                      if datetime.fromisoformat(r["timestamp"]) >= cutoff]
            recent_hits = sum(1 for r in recent if r.get("hit"))
            overall_hits = sum(1 for r in verified if r.get("hit"))
            recent_rate = recent_hits / len(recent) if recent else 0
            overall_rate = overall_hits / len(verified) if verified else 0

            if len(recent) > 10 and overall_rate - recent_rate > self.drift_threshold:
                logger.warning(
                    f"检测到模型漂移: 近期命中率={recent_rate:.2%}, "
                    f"整体命中率={overall_rate:.2%}, "
                    f"下降={overall_rate - recent_rate:.2%} > 阈值={self.drift_threshold:.2%}"
                )
                return True
        except Exception as e:
            logger.warning(f"漂移检测失败: {e}")
        return False

    def _fetch_data_for_verify(self, symbol: str, start: str, end: str):
        """审计验证用的数据获取回调"""
        from src.data.collector import DataCollector
        collector = DataCollector(self.config)
        return collector.load_cached(symbol)

    def _scheduled_job(self):
        """定时任务入口"""
        self._do_retrain("scheduled")
        if self.auto_retrain_on_drift and self._check_drift():
            self._do_retrain("drift_detected")

    def start(self):
        """启动调度器（后台线程）"""
        if self._running:
            logger.warning("调度器已在运行")
            return

        if self.freq == "daily":
            schedule.every().day.at(self.time).do(self._scheduled_job)
        elif self.freq == "weekly":
            day_map = {
                "mon": schedule.every().monday, "tue": schedule.every().tuesday,
                "wed": schedule.every().wednesday, "thu": schedule.every().thursday,
                "fri": schedule.every().friday, "sat": schedule.every().saturday,
                "sun": schedule.every().sunday,
            }
            job = day_map.get(self.day.lower(), schedule.every().sunday)
            job.at(self.time).do(self._scheduled_job)
        elif self.freq == "monthly":
            schedule.every().day.at(self.time).do(self._monthly_check)
        logger.info(f"调度器已启动: 频率={self.freq}, 时间={self.time}, 星期={self.day}")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _monthly_check(self):
        """月度检查（每月1日执行）"""
        if datetime.now().day == 1:
            self._scheduled_job()

    def _loop(self):
        """调度循环"""
        while self._running:
            schedule.run_pending()
            time.sleep(60)

    def stop(self):
        """停止调度器"""
        self._running = False
        schedule.clear()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("调度器已停止")

    def run_now(self) -> bool:
        """手动触发一次重训练"""
        return self._do_retrain("manual")
