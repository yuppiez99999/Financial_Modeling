"""统一适配器：predict → signal → risk → order 一键编排，产出标准化交易数据流。

TradingAdapter 是外部量化交易程序接入 TrendCast Pro 的入口：
  - run(symbols)          ：对一组标的执行完整流程，产出 Order / Signal / RiskBudget
  - export_feed(path)     ：把结果序列化为 JSON / CSV，供外部程序定时拉取
  - push_webhook()        ：复用通知器把订单流推送到外部交易程序的回调地址

不改变已有推理引擎的任何接口，仅在之上叠加"交易适配"语义。
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.trading.signal import SignalEngine
from src.trading.risk import RiskManager
from src.trading.orders import OrderGenerator

logger = logging.getLogger(__name__)


class TradingAdapter:
    """量化交易程序适配器。"""

    def __init__(self, config: dict[str, Any], predictor=None):
        """
        Args:
            config   : TrendCast Pro 配置 dict
            predictor: 已 load_models 的 PredictionEngine 实例；
                       未传则延迟创建。
        """
        self.config = config
        self._predictor = predictor
        self.signal_engine = SignalEngine(config)
        self.risk_manager = RiskManager(config)
        self.order_generator = OrderGenerator(config)
        self.allow_short = self.signal_engine.allow_short
        # 信号准入闸门（Q2）：未达 score 等级时信号只读、不下单（防"未验证信号"进实盘）
        self._gate_decision = self._load_gate()
        self.export_blocked = False
        # 最新行情价：RiskManager 需要当前价算 qty/SL/TP
        # 若调用方不提供，则从预测结果中的 latest_close 读取
        self._prices: dict[str, float] = {}

    def _load_gate(self) -> dict[str, Any] | None:
        """读取策略门禁判定（缺省 None = 未启用/无判定，保持向后兼容）。

        复用 `src/trading/gate.py` 的 `StrategyGate`（Q2 已交付的门禁实现），
        不重复造判定逻辑；只负责"是否强制"与"读取结果"两件事。
        """
        cfg = (self.config.get("strategy_gate", {}) or {})
        # 必须显式开启 enforce_trading：门禁判定不应默认改变下单行为，
        # 否则历史的判定结果会静默地"关掉"交易适配层（难以排查的隐式耦合）。
        if not cfg.get("enforce_trading", False):
            return None
        try:
            from src.trading.gate import StrategyGate

            gate_dir = Path(cfg.get("report_dir", "reports"))
            path = gate_dir / "strategy_gate.json"
            if not path.exists():
                # 无判定结果：不阻塞下单（未评估 ≠ 未通过），但留日志便于排查
                logger.info("[gate] 未找到 %s，跳过门禁强制", path)
                return None
            decision = json.loads(path.read_text(encoding="utf-8"))
            # 用当前配置重新判定（阈值调整必须立即生效，不得沿用旧结论）
            if isinstance(decision, dict) and decision.get("state") is not None:
                return decision
            return StrategyGate(self.config).decide().to_dict()
        except Exception as e:  # noqa: BLE001
            logger.warning("[gate] 读取策略门禁失败，按未判定处理: %s", e)
            return None

    @property
    def signal_readonly(self) -> bool:
        """信号是否仅允许观测（未过门禁）。无判定时不限制（向后兼容）。"""
        if self._gate_decision is None:
            return False
        # StrategyGate: state = gated/readonly/disabled；仅 gated 才放行订单
        if self._gate_decision.get("state") == "disabled":
            return False
        return not bool(self._gate_decision.get("passed"))

    @property
    def predictor(self):
        if self._predictor is None:
            from src.inference.predictor import PredictionEngine
            eng = PredictionEngine(self.config)
            eng.load_models(self.config["model"].get("type", "lightgbm"))
            self._predictor = eng
        return self._predictor

    def _latest_price(self, symbol: str, prediction: dict[str, Any]) -> float | None:
        """从预测结果 / 内部缓存取最新收盘价。"""
        p = self._prices.get(symbol)
        if p is not None:
            return p
        for h in prediction.get("predictions", {}).values():
            if isinstance(h, dict) and h.get("latest_close"):
                return float(h["latest_close"])
        # 顶层
        if prediction.get("latest_close"):
            return float(prediction["latest_close"])
        return None

    def process_symbol(self, symbol: str) -> dict[str, Any]:
        """对单个标执行完整适配流程，返回可序列化结果。"""
        pred = self.predictor.predict_all_horizons(symbol)
        if not pred or "predictions" not in pred:
            return {"symbol": symbol, "error": "预测失败"}

        price = self._latest_price(symbol, pred)
        sig = self.signal_engine.build_signal(symbol, pred)

        # 闸门未过：本次信号只读（保留信号与风控预算，但不产出可执行订单）
        if self.signal_readonly:
            return {
                "symbol": symbol,
                "signal": sig.to_dict(),
                "risk_budget": self.risk_manager.budget(sig, price=price).to_dict(),
                "orders": [],
                "gate_blocked": True,
                "gate_level": (self._gate_decision or {}).get("state", "readonly"),
                "gate_reason": "信号未过策略门禁（IC/命中率），本轮只读观测",
                "latest_price": price,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            }

        budget = self.risk_manager.budget(sig, price=price)
        orders = self.order_generator.generate(budget)

        return {
            "symbol": symbol,
            "signal": sig.to_dict(),
            "risk_budget": budget.to_dict(),
            "orders": [o.to_dict() for o in orders],
            "gate_blocked": False,
            "gate_level": (self._gate_decision or {}).get("state", "ungated"),
            "latest_price": price,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

    def run(self, symbols: list[str]) -> list[dict[str, Any]]:
        """批量执行全部标的。"""
        return [self.process_symbol(s) for s in symbols]

    # ------------------------------------------------------------------
    # 数据导出：供外部交易程序消费
    # ------------------------------------------------------------------
    def export_feed(self, results: list[dict[str, Any]], path: str | Path) -> Path:
        """把结果导出为标准 JSON feed 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "schema_version": "1.0",
            "items": results,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("交易数据流已导出: %s (%d 条)", path, len(results))
        return path

    def export_csv(self, results: list[dict[str, Any]], path: str | Path) -> Path:
        """把订单展平导出为 CSV（便于人工/程序二次处理）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in results:
            if "orders" not in r:
                continue
            for o in r["orders"]:
                rows.append({
                    "symbol": r.get("symbol"),
                    "side": o.get("side"),
                    "order_type": o.get("order_type"),
                    "size": o.get("size"),
                    "price": o.get("price"),
                    "reduce_only": o.get("reduce_only"),
                    "action": r.get("signal", {}).get("action"),
                    "score": r.get("signal", {}).get("score"),
                    "confidence": r.get("signal", {}).get("confidence"),
                })
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else
                                ["symbol", "side", "order_type", "size", "price",
                                 "reduce_only", "action", "score", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
        # write_bytes：csv 终止符 \r\n 原样落盘（等价于 open(newline="")）
        path.write_bytes(buf.getvalue().encode("utf-8"))
        return path

    def push_webhook(self, results: list[dict[str, Any]], subject: str = "TrendCast 交易信号") -> dict[str, bool]:
        """把订单流推送到外部量化程序 Webhook（复用 SignalNotifier）。"""
        from src.notification.notifier import SignalNotifier
        notifier = SignalNotifier(self.config)
        webhook_payload = {
            "event": "trading_signal",
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "items": results,
        }
        ok = notifier.send_webhook(webhook_payload)
        return {"webhook": ok}
