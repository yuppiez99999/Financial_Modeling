"""订单生成：将风控预算输出为标准化下单载荷（可被外部交易程序消费）。

OrderGenerator 把 RiskBudget 转为结构统一、可直接下发给券商 / 交易所 / 量化
程序的订单对象。载荷字段刻意保持"执行引擎无关"：仅携带 side/size/price 与
可选附加的止损止盈条件单，兼容市价单、限价单、条件单等多种执行方式。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

MARKET = "market"
LIMIT = "limit"
STOP = "stop"
TAKE = "take_profit"


@dataclass
class Order:
    """一条标准化订单。"""

    symbol: str
    side: str            # buy / sell
    order_type: str      # market / limit / stop
    size: int | float
    price: float | None
    reduce_only: bool
    meta: dict[str, Any] = field(default_factory=dict)
    client_order_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "size": self.size,
            "price": self.price,
            "reduce_only": self.reduce_only,
            "meta": self.meta,
            "client_order_id": self.client_order_id,
        }


class OrderGenerator:
    """根据风控预算与最新行情生成订单列表。"""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("trading", {}).get("order", {})
        self.order_type = cfg.get("order_type", MARKET)  # market / limit
        self.attach_sl_tp = bool(cfg.get("attach_sl_tp", True))

    def _order_id(self, symbol: str) -> str:
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"TC-{symbol}-{ts}"

    def generate(self, budget: "RiskBudget") -> list[Order]:
        """由 RiskBudget 生成主单 + 可选风控条件单。"""
        if budget.action == "HOLD":
            return []

        side = "buy" if budget.action == "BUY" else "sell"
        if budget.suggested_qty is None or budget.suggested_qty <= 0:
            return []

        orders: list[Order] = []
        # 主单
        price = budget.take_price  # 不适用；主单价格取决于订单类型与外部行情，这里填 None 由执行端决定
        price = None
        if self.order_type == LIMIT:
            # 限价单：需要用户提供期望价格；未提供时回退为市价
            price = None
        main = Order(
            symbol=budget.symbol,
            side=side,
            order_type=self.order_type,
            size=budget.suggested_qty,
            price=price,
            reduce_only=False,
            client_order_id=self._order_id(budget.symbol),
        )
        orders.append(main)

        # 风控条件单：止损 / 止盈（由执行引擎处理为条件单）
        if self.attach_sl_tp and budget.stop_price:
            sl_side = "sell" if side == "buy" else "buy"
            orders.append(Order(
                symbol=budget.symbol,
                side=sl_side,
                order_type=STOP,
                size=budget.suggested_qty,
                price=budget.stop_price,
                reduce_only=True,
                meta={"kind": "stop_loss"},
                client_order_id=self._order_id(budget.symbol),
            ))
        if self.attach_sl_tp and budget.take_price:
            tp_side = "sell" if side == "buy" else "buy"
            orders.append(Order(
                symbol=budget.symbol,
                side=tp_side,
                order_type=TAKE,
                size=budget.suggested_qty,
                price=budget.take_price,
                reduce_only=True,
                meta={"kind": "take_profit"},
                client_order_id=self._order_id(budget.symbol),
            ))
        return orders
