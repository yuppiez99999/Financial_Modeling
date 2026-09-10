"""模型监控报表：统一汇总审计命中率、自适应漂移、数据源可用性。

面向 Q1 排期「定期报告」增强：把分散在审计 / 自适应学习 / 宏观 / 新闻
各模块的运行态信息汇总为一份可读的监控报表，便于日常巡检与对外汇报。
"""
from src.monitor.health_report import HealthReport, ModelMonitor

__all__ = ["ModelMonitor", "HealthReport"]
