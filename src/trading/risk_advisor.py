"""Q4 核心：ATR 自适应止损止盈建议引擎（复用 compute_atr）。

路线图（SALES_PLAN.md §8.2）Q4：智能风控模块，自动生成止损止盈建议。

本模块只做两件事，输出**建议**（suggestion），不改动 `RiskManager` / `OrderGenerator`
的任何行为，也不产出下单指令：
  1. 比例定多少：由标的自身波动结构（ATR）决定，而非全池共用常数。
  2. 价位定在哪：参考真实支撑阻力（近期低点/均线/ATR 缓冲）。

ATR 计算统一复用 ``src.data.indicators.compute_atr``，保证止损止盈、仓位波动缩放、
组合波动监控等口径一致，且不含未来函数（只读已落盘日K，T 及更早）。

## 合规与安全边界（必须成立，否则本模块就是有害的）
- 门禁未放行时不产出价位建议（fail-close）。
- 不构成投资建议；字段统一 `suggested_*` 前缀，不出现 `order` / `execute` 语义。
- 信息不足时拒绝给数，绝不用默认比例兜底算一个"看起来能用"的价。
- 无未来函数：只读 T 及更早的已落盘日K。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.indicators import compute_atr

logger = logging.getLogger(__name__)

# 建议状态
STATUS_OK = "ok"                  # 正常产出建议
STATUS_WITHHELD = "withheld"      # 门禁未放行 → 只给仓位侧信息，不给价位
STATUS_UNAVAILABLE = "unavailable"  # 数据/配置不足 → 不给任何数

DISCLAIMER = "本建议由模型基于历史行情自动生成，仅供学习与研究参考，不构成投资建议。"

# 兜底默认值（仅在配置缺失时使用，仍受上下限约束）
_DEFAULTS = {
    "atr_window": 14,
    "atr_stop_mult": 2.0,      # 止损 = ATR × 倍数
    "atr_take_mult": 3.0,      # 止盈 = ATR × 倍数
    "support_lookback": 20,    # 支撑/阻力回看窗口（交易日）
    "support_buffer": 0.005,   # 支撑下方缓冲比例（避免贴边挂单被扫）
    "min_stop_pct": 0.015,     # 止损比例下限（防过紧被噪音打掉）
    "max_stop_pct": 0.12,      # 止损比例上限（防过宽失去风控意义）
    "min_risk_reward": 1.5,    # 盈亏比低于该值 → 建议放弃该机会
    "vol_target_pct": 0.015,   # 日波动目标（用于仓位建议）
    "min_bars": 30,            # 最少日K根数（不足则不给建议）
}


def _pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _same_direction(block: Any, action: str) -> bool:
    """预测块的方向是否与信号的最终动作一致（用于挑选上下文周期）。"""
    if not isinstance(block, dict) or "error" in block:
        return False
    pred = block.get("prediction")
    if pred is None:
        return False
    if str(action).upper() == "BUY":
        return int(pred) == 1
    if str(action).upper() == "SELL":
        return int(pred) == 0
    return False


def _plan_markdown(item: Dict[str, Any]) -> str:
    """把一条建议 dict 渲染为 Markdown（同时服务于类方法与监控报表）。"""
    symbol = item.get("symbol", "")
    if not item.get("available"):
        return f"- `{symbol}`：暂不产出风控建议（{item.get('reason') or '数据不足'}）"
    lines = [
        f"- `{symbol}` {item.get('action')}（{item.get('status')}）"
        f"｜参考价 {item.get('entry_price')}"
    ]
    if item.get("suggested_stop") is not None:
        lines.append(
            f"  - 建议止损 {item.get('suggested_stop')}"
            f"（{_pct(item.get('stop_pct'))}，依据：{item.get('sl_basis')}）"
        )
        lines.append(
            f"  - 建议止盈 {item.get('suggested_take')}"
            f"（{_pct(item.get('take_pct'))}，依据：{item.get('tp_basis')}）"
        )
        lines.append(
            f"  - 盈亏比 {item.get('risk_reward')}｜ATR {item.get('atr')}"
            f"（{_pct(item.get('atr_pct'))}）"
            f"｜支撑 {item.get('support')} / 阻力 {item.get('resistance')}"
        )
    else:
        lines.append(f"  - 价位建议已暂缓：{item.get('reason')}")
    pos = item.get("position") or {}
    if pos.get("position_pct") is not None:
        lines.append(
            f"  - 仓位侧：{_pct(pos.get('position_pct'))}（{pos.get('position_amount')}）"
            f"，单笔风险 {pos.get('risk_amount')}"
        )
    for note in item.get("notes") or []:
        lines.append(f"  - 备注：{note}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
@dataclass
class StopTakePlan:
    """一次止损止盈建议（可序列化，字段命名必须保持 `suggested_` 前缀）。"""

    symbol: str
    status: str = STATUS_UNAVAILABLE
    action: str = "HOLD"
    available: bool = False
    reason: str = ""

    entry_price: Optional[float] = None       # 参考入场价（预测口径的最新收盘价）
    suggested_stop: Optional[float] = None
    suggested_take: Optional[float] = None
    stop_pct: Optional[float] = None
    take_pct: Optional[float] = None
    risk_reward: Optional[float] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    ma_fast: Optional[float] = None      # 观测锚点：ATR 窗口期均线
    ma_slow: Optional[float] = None      # 观测锚点：4×ATR 窗口期均线（结构锚）
    tp_basis: str = ""                        # 止盈取价依据
    sl_basis: str = ""                        # 止损取价依据

    # 仓位侧（来自 RiskManager，本模块不重算）
    position_pct: Optional[float] = None
    position_amount: Optional[float] = None
    risk_amount: Optional[float] = None
    suggested_qty: Optional[int | float] = None
    vol_scaled_position_pct: Optional[float] = None  # 按波动目标缩放后的仓位建议

    horizon: str = ""
    confidence: Optional[float] = None
    gate_state: str = ""
    gate_blocked: bool = False
    notes: List[str] = field(default_factory=list)
    not_trade_instruction: bool = True
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "symbol": self.symbol,
            "status": self.status,
            "available": self.available,
            "action": self.action,
            "entry_price": self.entry_price,
            "suggested_stop": self.suggested_stop,
            "suggested_take": self.suggested_take,
            "stop_pct": self.stop_pct,
            "take_pct": self.take_pct,
            "risk_reward": self.risk_reward,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "support": self.support,
            "resistance": self.resistance,
            "ma_fast": self.ma_fast,
            "ma_slow": self.ma_slow,
            "sl_basis": self.sl_basis,
            "tp_basis": self.tp_basis,
            "position": {
                "position_pct": self.position_pct,
                "position_amount": self.position_amount,
                "risk_amount": self.risk_amount,
                "suggested_qty": self.suggested_qty,
                "vol_scaled_position_pct": self.vol_scaled_position_pct,
            },
            "context": {
                "horizon": self.horizon,
                "confidence": self.confidence,
                "gate_state": self.gate_state,
                "gate_blocked": self.gate_blocked,
            },
            "notes": list(self.notes),
            "not_trade_instruction": self.not_trade_instruction,
            "disclaimer": self.disclaimer,
        }
        if self.reason:
            out["reason"] = self.reason
        return out

    def to_markdown(self) -> str:
        """单条建议的 Markdown 片段（供日报/监控服用）。"""
        return _plan_markdown(self.to_dict())


# ----------------------------------------------------------------------
class RiskAdvisor:
    """基于波动结构与支撑阻力，为已生成的交易信号补充止损止盈建议。

    设计原则：**不改变既有行为**。`RiskManager` / `OrderGenerator` / `TradingAdapter`
    的输入输出契约均不受影响，本类只在其输出之上追加一层「建议」。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = ((config or {}).get("trading", {}) or {}).get("risk_advice", {}) or {}
        self.atr_window = int(cfg.get("atr_window", _DEFAULTS["atr_window"]))
        self.atr_stop_mult = float(cfg.get("atr_stop_mult", _DEFAULTS["atr_stop_mult"]))
        self.atr_take_mult = float(cfg.get("atr_take_mult", _DEFAULTS["atr_take_mult"]))
        self.support_lookback = int(cfg.get("support_lookback", _DEFAULTS["support_lookback"]))
        self.support_buffer = float(cfg.get("support_buffer", _DEFAULTS["support_buffer"]))
        self.min_stop_pct = float(cfg.get("min_stop_pct", _DEFAULTS["min_stop_pct"]))
        self.max_stop_pct = float(cfg.get("max_stop_pct", _DEFAULTS["max_stop_pct"]))
        self.min_risk_reward = float(cfg.get("min_risk_reward", _DEFAULTS["min_risk_reward"]))
        self.vol_target_pct = float(cfg.get("vol_target_pct", _DEFAULTS["vol_target_pct"]))
        self.min_bars = int(cfg.get("min_bars", _DEFAULTS["min_bars"]))
        self.gate_cfg = (config or {}).get("strategy_gate", {}) or {}
        self.config = config or {}

    # ---------------- 门禁 ----------------
    def gate_state(self) -> str:
        """读取门禁状态（只读，不触发重算）。

        读不到判定结果时返回 `unknown` —— 调用方按 **未放行** 处理（fail-close），
        因为"没评估过"绝不等价于"已验证可用"。
        """
        try:
            from pathlib import Path

            path = Path(self.gate_cfg.get("report_dir", "reports")) / "strategy_gate.json"
            if not path.exists():
                return "unknown"
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            return str(payload.get("state", "unknown"))
        except Exception as e:  # noqa: BLE001
            logger.warning("[risk_advisor] 读取门禁状态失败: %s", e)
            return "unknown"

    def _gate_allows_price_advice(self) -> tuple[bool, str]:
        """门禁是否允许产出**价位**建议。

        - 门禁显式关闭（`enabled: false`）→ 允许（用户明确选择不看门禁）；
        - `gated` → 允许；
        - 其余（`readonly` / `unknown`）→ 不允许，fail-close。
        """
        if not bool(self.gate_cfg.get("enabled", True)):
            return True, "gate_disabled_by_config"
        state = self.gate_state()
        if state == "gated":
            return True, state
        return False, state

    # ---------------- 波动与支撑阻力 ----------------
    def atr(self, df: pd.DataFrame) -> Optional[float]:
        """平均真实波幅（委托给可复用的 ``compute_atr``，样本不足返回 None，不猜）。"""
        return compute_atr(df, self.atr_window)

    def _levels(self, df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        """近期支撑 / 阻力（不含最后一根K线，避免用当日极值自我参照）。"""
        if df is None or len(df) < 3:
            return None, None
        window = df.iloc[-(self.support_lookback + 1):-1]
        if window.empty:
            return None, None
        return float(window["low"].min()), float(window["high"].max())

    # ---------------- 主流程 ----------------
    def advise(self, symbol: str, signal: Any, df: Optional[pd.DataFrame] = None,
               price: Optional[float] = None, risk_budget: Any = None,
               horizon: str = "", confidence: Optional[float] = None) -> StopTakePlan:
        """产出单标的止损止盈建议。

        Args:
            symbol     : 标的代码
            signal     : `src.trading.signal.Signal`（读 action/strength/confidence）
            df         : 已落盘日K（含 high/low/close），None 或缺列时不给建议
            price      : 参考入场价（缺省取 df 最后一行收盘价）
            risk_budget: `RiskBudget`，仅用于透传仓位侧字段（不重算）
            horizon    : 触发本次建议的预测周期（上下文用）
            confidence : 预测置信度（上下文用）
        """
        action = getattr(signal, "action", "HOLD")
        plan = StopTakePlan(
            symbol=symbol,
            action=action,
            horizon=horizon,
            confidence=confidence if confidence is not None
            else getattr(signal, "confidence", None),
            gate_state=self.gate_state(),
        )
        if risk_budget is not None:
            plan.position_pct = getattr(risk_budget, "position_pct", None)
            plan.position_amount = getattr(risk_budget, "position_amount", None)
            plan.risk_amount = getattr(risk_budget, "risk_per_trade", None)
            plan.suggested_qty = getattr(risk_budget, "suggested_qty", None)

        allowed, gate_state = self._gate_allows_price_advice()
        plan.gate_blocked = not allowed

        # 方向不明：本就无价位可给（HOLD 不构造交易机会）
        if str(action).upper() == "HOLD":
            plan.status = STATUS_WITHHELD
            plan.reason = "信号为 HOLD（方向不明），无交易机会，不产出价位建议"
            return plan

        # 数据体检：不满足就不给数（fail-close，绝不用默认比例兜底）
        if df is None or len(df) < self.min_bars:
            plan.status = STATUS_UNAVAILABLE
            plan.reason = f"行情样本不足（需 ≥{self.min_bars} 根日K，实际 {0 if df is None else len(df)}）"
            return plan
        missing = [c for c in ("high", "low", "close") if c not in df.columns]
        if missing:
            plan.status = STATUS_UNAVAILABLE
            plan.reason = f"行情缺少必要列：{', '.join(missing)}"
            return plan

        entry = float(price) if price else float(df["close"].iloc[-1])
        if not (entry > 0):
            plan.status = STATUS_UNAVAILABLE
            plan.reason = f"参考价非法（{entry}），拒绝据此推算价位"
            return plan

        atr = self.atr(df)
        if atr is None or atr <= 0:
            plan.status = STATUS_UNAVAILABLE
            plan.reason = "ATR 不可用（波动结构无法估计），拒绝给出止损止盈"
            return plan

        support, resistance = self._levels(df)
        if support is None or resistance is None or support <= 0:
            plan.status = STATUS_UNAVAILABLE
            plan.reason = "支撑/阻力不可用（历史窗口不足）"
            return plan

        is_long = str(action).upper() == "BUY"
        plan.entry_price = round(entry, 4)
        plan.atr = round(atr, 4)
        plan.atr_pct = round(atr / entry, 6)
        plan.support = round(support, 4)
        plan.resistance = round(resistance, 4)

        # ---- 止损：结构位（支撑/阻力）与 ATR 谁更保守取谁，再压进 [min, max] 区间 ----
        if is_long:
            structural = support * (1 - self.support_buffer)
            atr_stop = entry - self.atr_stop_mult * atr
            stop = min(structural, atr_stop)
            plan.sl_basis = (
                "支撑位下方缓冲" if structural <= atr_stop
                else f"ATR×{self.atr_stop_mult:g}"
            )
        else:
            structural = resistance * (1 + self.support_buffer)
            atr_stop = entry + self.atr_stop_mult * atr
            stop = max(structural, atr_stop)
            plan.sl_basis = (
                "阻力位上方缓冲" if structural >= atr_stop
                else f"ATR×{self.atr_stop_mult:g}"
            )

        stop_pct = abs(entry - stop) / entry
        clamped = min(max(stop_pct, self.min_stop_pct), self.max_stop_pct)
        if abs(clamped - stop_pct) > 1e-9:
            plan.notes.append(
                f"止损比例 {_pct(stop_pct)} 超出 ["
                f"{_pct(self.min_stop_pct)}, {_pct(self.max_stop_pct)}] 区间，已按边界收敛"
            )
        stop_pct = clamped
        stop = entry * (1 - stop_pct) if is_long else entry * (1 + stop_pct)

        # ---- 止盈：需求盈亏比下界与实际阻力位取更近者；都不足以覆盖成本则建议放弃 ----
        take_pct = self.min_risk_reward * stop_pct
        take_at = entry * (1 + take_pct) if is_long else entry * (1 - take_pct)
        if is_long:
            # 多头止盈不越过近端阻力（穿越阻力需要额外动能，未经验证不假设）
            if resistance > entry:
                cap = resistance * (1 + self.support_buffer)
                if cap < take_at:
                    take_at = cap
                    plan.tp_basis = "近期阻力位（近于盈亏比下界）"
                else:
                    plan.tp_basis = f"盈亏比下界 {self.min_risk_reward:g}×止损"
            else:
                plan.tp_basis = f"盈亏比下界 {self.min_risk_reward:g}×止损（上方无有效阻力）"
        else:
            if support < entry:
                cap = support * (1 - self.support_buffer)
                if cap > take_at:
                    take_at = cap
                    plan.tp_basis = "近期支撑位（近于盈亏比下界）"
                else:
                    plan.tp_basis = f"盈亏比下界 {self.min_risk_reward:g}×止损"
            else:
                plan.tp_basis = f"盈亏比下界 {self.min_risk_reward:g}×止损（下方无有效支撑）"

        take_pct = abs(take_at - entry) / entry
        plan.suggested_stop = round(stop, 4)
        plan.suggested_take = round(take_at, 4)
        plan.stop_pct = round(stop_pct, 6)
        plan.take_pct = round(take_pct, 6)
        plan.risk_reward = round((take_pct / stop_pct) if stop_pct > 0 else 0.0, 4)

        # ---- 波动目标仓位：波动越大仓位越小（对 RiskManager 仓位只作缩放建议） ----
        plan.ma_fast = (entry + atr) if is_long else (entry - atr)
        plan.ma_slow = (entry + 4 * atr) if is_long else (entry - 4 * atr)
        plan.notes.append(
            f"观测锚点：ATR×1 = {round(plan.ma_fast, 4)}，ATR×4 = {round(plan.ma_slow, 4)}"
            "（波动结构的自然延伸位，非目标价）"
        )

        if plan.atr_pct and plan.atr_pct > 0:
            scale = min(self.vol_target_pct / plan.atr_pct, 1.0)
            if plan.position_pct is not None:
                plan.vol_scaled_position_pct = round(plan.position_pct * scale, 6)
            if scale < 1.0:
                plan.notes.append(
                    f"该标的 ATR 占比 {_pct(plan.atr_pct)} 高于波动目标 "
                    f"{_pct(self.vol_target_pct)}，仓位建议按 {scale:.2f}× 缩放"
                )

        if plan.risk_reward < self.min_risk_reward:
            plan.notes.append(
                f"盈亏比 {plan.risk_reward} 低于下界 {self.min_risk_reward:g}："
                "该机会风险补偿不足，建议放弃而非放宽止损"
            )

        if not allowed:
            # 门禁未放行：保留波动结构等观测信息，但抹掉可直接使用的价位
            plan.status = STATUS_WITHHELD
            plan.reason = (
                f"策略门禁为 {gate_state}（信号只读观测），"
                "不产出可直接使用的止损止盈价位；待门禁 gated 后自动恢复"
            )
            plan.suggested_stop = None
            plan.suggested_take = None
            plan.stop_pct = None
            plan.take_pct = None
            plan.risk_reward = None
            return plan

        plan.status = STATUS_OK
        plan.available = True
        if gate_state == "gate_disabled_by_config":
            plan.gate_state = "disabled"
            plan.notes.append("门禁在配置中关闭：建议未经过门禁校验，请自行评估可用性")
        return plan

    # ---------------- 批量 / 报告 ----------------
    def advise_payload(self, symbol: str, payload: Optional[Dict[str, Any]] = None,
                       df: Optional[pd.DataFrame] = None,
                       signal: Any = None) -> StopTakePlan:
        """从预测结果构造并产出建议（CLI / API 统一入口）。

        payload 为 ``PredictionEngine.predict_all_horizons`` 的返回结构；
        当未显式传入 signal 时，用模型自己的 ``SignalEngine`` 生成信号，
        保证「建议针对的信号」与「下游看到信号」是同一个，而不是另算一套。
        """
        from src.trading.signal import SignalEngine

        payload = payload or {}
        sig = signal if signal is not None else SignalEngine(self.config).build_signal(symbol, payload)

        horizon, confidence = "", None
        horizons = payload.get("predictions") or {}
        # 取与信号主方向一致的周期作为上下文；无法判定时取短周期（权重最高）
        preferred = next(
            (h for h in ("short_term", "mid_term", "long_term") if _same_direction(horizons.get(h), sig.action)),
            None,
        ) or ("short_term" if "short_term" in horizons else (next(iter(horizons), "")))
        block = horizons.get(preferred) or {}
        if isinstance(block, dict) and "error" not in block:
            horizon = preferred
            confidence = block.get("probability")

        price = None
        for block in horizons.values():
            if isinstance(block, dict) and block.get("latest_close"):
                price = float(block["latest_close"])
                break

        return self.advise(symbol, sig, df=df, price=price, horizon=horizon, confidence=confidence)

    def render_markdown(self, payload: Dict[str, Any]) -> str:
        """把 advise_portfolio 的结果渲染为 Markdown 章节。"""
        lines = ["## 智能风控建议（Q4）", ""]
        state = payload.get("gate_state", "unknown")
        lines.append(
            f"- 门禁状态：`{state}`｜建议 {payload.get('available', 0)}/"
            f"{payload.get('count', 0)} 条｜暂缓 {payload.get('withheld', 0)} 条"
            f"｜数据不足 {payload.get('unavailable', 0)} 条"
        )
        avg = payload.get("avg_risk_reward")
        lines.append(f"- 平均盈亏比：{avg if avg is not None else 'N/A'}")
        lines.append("")
        for item in payload.get("items") or []:
            lines.append(_plan_markdown(item))
        lines.append("")
        if state != "gated":
            lines.append(
                f"> ⚠️ 门禁为 `{state}`，信号仅作只读观测，本轮**未产出可直接使用的止损止盈价位**。"
            )
        lines.append(f"> {DISCLAIMER}")
        lines.append("")
        return "\n".join(lines)

    def advise_portfolio(self, items: List[Dict[str, Any]], gate_prechecked: bool = False
                         ) -> Dict[str, Any]:
        """批量建议。

        Args:
            items: [{"symbol","signal","df","price","risk_budget","horizon","confidence"}]
        """
        plans = [
            self.advise(
                it.get("symbol", ""),
                it.get("signal"),
                df=it.get("df"),
                price=it.get("price"),
                risk_budget=it.get("risk_budget"),
                horizon=it.get("horizon", ""),
                confidence=it.get("confidence"),
            )
            for it in items
        ]
        available = [p for p in plans if p.available]
        rewards = [p.risk_reward for p in available if p.risk_reward]
        return {
            "gate_state": self.gate_state(),
            "count": len(plans),
            "available": len(available),
            "withheld": sum(1 for p in plans if p.status == STATUS_WITHHELD),
            "unavailable": sum(1 for p in plans if p.status == STATUS_UNAVAILABLE),
            "avg_risk_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "items": [p.to_dict() for p in plans],
            "not_trade_instruction": True,
            "disclaimer": DISCLAIMER,
        }
