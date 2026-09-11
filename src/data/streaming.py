"""实时数据流接入：盘中快照轮询 → 分钟级滚动特征 → 信号缓存（Q3 路线）。

路线图（`SALES_PLAN.md` §8.2）Q3：
  > 实时数据流接入，支持分钟级预测更新

## 为什么是"轮询快照 + 防未来函数聚合"，而不是 WebSocket 行情源

1. **当前数据源是免费公开日K源**（Wind MCP 需 Key，腾讯源仅日K），没有分钟级推送权限；
2. 更关键的是**评估口径一致性**：模型的判定基准是"收盘价算出的日频方向"，如果盘中用
   未收盘的价格反复重算，等于给模型喂了一个训练时从未见过的分布 —— 既不是它学的东西，
   也无法用同一套 IC / 命中率门禁去验证；
3. 因此本模块的定位是**观测与预警**（盘中信号漂移提醒），不是"盘中下单决策"。
   盘中快照只用来回答一个问题：**今天开盘后的走势，是否让昨收模型的观点发生了变化？**

## 防未来函数（本模块的核心纪律）

分钟预测必须使用"**T 日收盘后已落盘的日K**"作为历史窗口，当日盘中价格只能作为**待评估的
新观测**。绝不允许把盘中价格并入日K后重算特征 —— 那会让当日特征含有当日收盘信息，
在回测中构成典型未来函数，指标必然虚高。

## 数据纪律（与既有链路一致）

- 只使用**真实数据源**，绝不使用 simulation 兜底数据（随机游走数据会伪造"今天"）；
- 实时快照不写入 `data/raw/*.csv`（分钟级数据污染日K缓存是严重事故），
  只写独立的 `data/realtime/*.json`；
- 任何网络/解析异常 fail-open：返回 `available=False`，不抛异常、不阻断主链路。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# 实时抓取腾讯分钟线的上游地址（qt.gtimg.cn 的实时快照接口，含最新价/今开/昨收）
_QUOTE_URL = "https://qt.gtimg.cn/q="

# 快照默认保留天数（磁盘上的分钟历史，用于事后复盘）
DEFAULT_RETENTION_DAYS = 30


def _to_tencent_code(symbol: str) -> Optional[str]:
    """复用日K源同一套代码映射（600519.SH → sh600519）。"""
    from src.data.tencent_client import to_tencent_code

    return to_tencent_code(symbol)


# ----------------------------------------------------------------------
@dataclass
class IntradaySnapshot:
    """单只标的的一次盘中快照（最小可用字段）。"""

    symbol: str
    ts: str
    price: float
    prev_close: float
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    source: str = "tencent_quote"

    @property
    def change_pct(self) -> float:
        """相对昨收的涨跌幅（昨收缺失/为 0 时返回 0，不猜）。"""
        if not self.prev_close:
            return 0.0
        return (self.price - self.prev_close) / self.prev_close

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts,
            "price": round(self.price, 4),
            "prev_close": round(self.prev_close, 4),
            "open": round(self.open, 4),
            "high": round(self.high, 4),
            "low": round(self.low, 4),
            "volume": self.volume,
            "change_pct": round(self.change_pct, 6),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IntradaySnapshot":
        return cls(
            symbol=str(payload.get("symbol", "")),
            ts=str(payload.get("ts", "")),
            price=float(payload.get("price", 0.0) or 0.0),
            prev_close=float(payload.get("prev_close", 0.0) or 0.0),
            open=float(payload.get("open", 0.0) or 0.0),
            high=float(payload.get("high", 0.0) or 0.0),
            low=float(payload.get("low", 0.0) or 0.0),
            volume=float(payload.get("volume", 0.0) or 0.0),
            source=str(payload.get("source", "tencent_quote")),
        )


class MarketClock:
    """A 股交易时段判定（纯本地计算，不触网）。

    只做"是否可能处于交易时段"的粗判（工作日 + 上午/下午时段），
    不含法定节假日日历（节假日接口不可用时会把休市日也判为盘中，
    因此快照一律带 `is_trading_hours` 标记，下游自行决定是否采信）。
    """

    MORNING = (9 * 60 + 30, 11 * 60 + 30)
    AFTERNOON = (13 * 60, 15 * 60)

    @staticmethod
    def is_trading_hours(now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        if now.weekday() >= 5:  # 周末休市
            return False
        minutes = now.hour * 60 + now.minute
        return (
            MarketClock.MORNING[0] <= minutes <= MarketClock.MORNING[1]
            or MarketClock.AFTERNOON[0] <= minutes <= MarketClock.AFTERNOON[1]
        )


# ----------------------------------------------------------------------
class RealtimeQuoteClient:
    """腾讯实时快照客户端（盘中最新价 / 昨收 / 今开）。

    - 免费、无 Key；解析失败返回 None（fail-open）；
    - 兼容 `.SH` / `.SZ`，其它市场代码直接返回 None（不支持的绝不编造）；
    - 请求前把腾讯域名并入 NO_PROXY（与日K源同一坑位）。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = ((config or {}).get("streaming", {}) or {})
        self.config = config or {}
        self.timeout = float(cfg.get("quote_timeout", 5))
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests  # 延迟导入：未安装 requests 时本模块仍可被导入

            from src.data.tencent_client import ensure_no_proxy

            ensure_no_proxy()
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Mozilla/5.0 (TrendCastPro)"})
        return self._session

    def fetch_quotes(self, symbols: Sequence[str]) -> Dict[str, IntradaySnapshot]:
        """批量拉取快照；单只失败只影响该只（逐只容错）。"""
        codes: Dict[str, str] = {}
        for s in symbols:
            code = _to_tencent_code(s)
            if code:
                codes[code] = s
            else:
                logger.warning(f"[streaming] 不支持该标的的实时快照: {s}")
        if not codes:
            return {}

        quotes: Dict[str, IntradaySnapshot] = {}
        try:
            session = self._get_session()
            resp = session.get(
                _QUOTE_URL + ",".join(sorted(codes)), timeout=self.timeout
            )
            resp.raise_for_status()
            # 腾讯返回 GBK 编码的 v_sh600519="1~贵州茅台~600519~1700.00~...";
            text = resp.content.decode("gbk", errors="replace")
            for line in text.split(";"):
                line = line.strip()
                if not line or "=" not in line:
                    continue
                code, _, raw = line.partition("=")
                code = code.strip().removeprefix("v_")
                symbol = codes.get(code)
                if not symbol:
                    continue
                fields = raw.strip().strip('"').split("~")
                snap = self._parse_fields(symbol, fields)
                if snap is not None:
                    quotes[symbol] = snap
        except Exception as e:  # noqa: BLE001  fail-open：观测路径绝不阻断主链路
            logger.warning(f"[streaming] 实时快照拉取失败: {e}")
            return {}
        return quotes

    @staticmethod
    def _parse_fields(symbol: str, fields: List[str]) -> Optional[IntradaySnapshot]:
        """腾讯快照字段（`~` 分隔）→ IntradaySnapshot。

        索引约定：3=最新价，4=昨收，5=今开，33=最高，34=最低，6=成交量(手)。
        字段不足或价格非正 → 返回 None（宁缺勿造）。
        """

        def _num(idx: int) -> float:
            if idx >= len(fields):
                return 0.0
            try:
                return float(fields[idx])
            except (TypeError, ValueError):
                return 0.0

        price = _num(3)
        prev_close = _num(4)
        if price <= 0 or prev_close <= 0:
            return None
        return IntradaySnapshot(
            symbol=symbol,
            ts=datetime.now().isoformat(timespec="seconds"),
            price=price,
            prev_close=prev_close,
            open=_num(5),
            high=_num(33),
            low=_num(34),
            volume=_num(6),
        )


# ----------------------------------------------------------------------
class IntradayStore:
    """盘中快照的滚动存储（JSONL，按交易日分文件）。

    独立于 `data/raw/`（日K缓存）：分钟级数据混入日K缓存会污染训练集，
    这是本项目明令禁止的事故类型。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 base_dir: Optional[str] = None) -> None:
        cfg = ((config or {}).get("streaming", {}) or {})
        self.dir = Path(base_dir or cfg.get("realtime_dir", "data/realtime"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = int(cfg.get("retention_days", DEFAULT_RETENTION_DAYS))

    def path_for(self, day: Optional[str] = None) -> Path:
        day = day or datetime.now().strftime("%Y-%m-%d")
        return self.dir / f"snapshots_{day}.jsonl"

    def append(self, snapshots: Sequence[IntradaySnapshot]) -> Optional[Path]:
        """追加快照；写失败仅告警（观测路径 fail-open）。"""
        if not snapshots:
            return None
        path = self.path_for()
        try:
            with open(path, "a", encoding="utf-8") as f:
                for snap in snapshots:
                    f.write(json.dumps(snap.to_dict(), ensure_ascii=False) + "\n")
            return path
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[streaming] 快照落盘失败: {e}")
            return None

    def load(self, symbol: Optional[str] = None,
             day: Optional[str] = None) -> List[IntradaySnapshot]:
        """读取某日快照（可按标的过滤）；文件不存在返回空列表。"""
        path = self.path_for(day)
        if not path.exists():
            return []
        out: List[IntradaySnapshot] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 单行损坏不影响其余记录
                    if symbol and payload.get("symbol") != symbol:
                        continue
                    out.append(IntradaySnapshot.from_dict(payload))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[streaming] 快照读取失败: {e}")
            return []
        return out

    def cleanup(self, now: Optional[datetime] = None) -> List[str]:
        """清理超过保留期的快照文件，返回被删除的文件名。"""
        removed: List[str] = []
        now = now or datetime.now()
        try:
            for path in sorted(self.dir.glob("snapshots_*.jsonl")):
                day = path.stem.replace("snapshots_", "")
                try:
                    ts = datetime.strptime(day, "%Y-%m-%d")
                except ValueError:
                    continue
                if (now - ts).days > self.retention_days:
                    path.unlink()
                    removed.append(path.name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[streaming] 快照清理失败: {e}")
        return removed


# ----------------------------------------------------------------------
@dataclass
class RollingFeatures:
    """分钟级滚动特征（严格基于"截至 T-1 收盘的已落盘数据"计算）。"""

    symbol: str
    asof: str = ""
    close_last: float = 0.0
    ma_fast: float = 0.0
    ma_slow: float = 0.0
    volatility: float = 0.0
    momentum: float = 0.0
    rsi: float = 0.0
    bars: int = 0
    available: bool = False
    reason: str = ""
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asof": self.asof,
            "close_last": round(self.close_last, 4),
            "ma_fast": round(self.ma_fast, 4),
            "ma_slow": round(self.ma_slow, 4),
            "volatility": round(self.volatility, 6),
            "momentum": round(self.momentum, 6),
            "rsi": round(self.rsi, 4),
            "bars": self.bars,
            "available": self.available,
            "reason": self.reason,
            "features": {k: round(float(v), 6) for k, v in self.features.items()},
        }


class RollingFeatureBuilder:
    """把日K窗口转成一组便于盘中比对的滚动特征。

    **只用截至 asof 的已收盘数据**：调用方传入的 `daily_df` 必须是缓存中的日K
    （最后一行为 T-1 或更早的已收盘交易日），绝不接受含当日盘中价格的 K 线。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = ((config or {}).get("streaming", {}) or {})
        self.fast = int(cfg.get("ma_fast", 5))
        self.slow = int(cfg.get("ma_slow", 20))
        self.vol_window = int(cfg.get("vol_window", 20))
        self.rsi_window = int(cfg.get("rsi_window", 14))
        self.min_bars = int(cfg.get("min_bars", 30))

    def build(self, symbol: str, daily_df: Optional[pd.DataFrame]) -> RollingFeatures:
        """计算滚动特征；数据不足返回 `available=False`（不猜、不补）。"""
        if daily_df is None or len(daily_df) == 0:
            return RollingFeatures(symbol=symbol, reason="无日K数据")
        if "close" not in daily_df.columns:
            return RollingFeatures(symbol=symbol, reason="日K缺少 close 列")

        df = daily_df.sort_values("date") if "date" in daily_df.columns else daily_df
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < self.min_bars:
            return RollingFeatures(
                symbol=symbol, bars=int(len(close)),
                reason=f"历史 K 线不足（{len(close)} < {self.min_bars}）",
            )

        ret = close.pct_change()
        ma_fast = float(close.rolling(self.fast).mean().iloc[-1])
        ma_slow = float(close.rolling(self.slow).mean().iloc[-1])
        vol = float(ret.rolling(self.vol_window).std().iloc[-1])
        momentum = float(close.iloc[-1] / close.iloc[-self.slow - 1] - 1) if len(close) > self.slow else 0.0
        rsi = self._rsi(close, self.rsi_window)

        asof = str(df["date"].iloc[-1]) if "date" in df.columns else ""
        return RollingFeatures(
            symbol=symbol,
            asof=asof,
            close_last=float(close.iloc[-1]),
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            volatility=vol,
            momentum=momentum,
            rsi=rsi,
            bars=int(len(close)),
            available=True,
            features={
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "ma_gap": (ma_fast - ma_slow) / ma_slow if ma_slow else 0.0,
                "volatility": vol,
                "momentum": momentum,
                "rsi": rsi,
            },
        )

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> float:
        """Wilder 口径 RSI（0~100）；窗口内无波动时返回 50（中性）。"""
        delta = close.diff().dropna()
        if len(delta) < window:
            return 50.0
        gain = delta.clip(lower=0).rolling(window).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(window).mean().iloc[-1]
        if not loss or loss <= 0:
            return 100.0 if gain and gain > 0 else 50.0
        rs = float(gain) / float(loss)
        return float(100 - 100 / (1 + rs))
