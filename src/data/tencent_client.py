"""腾讯财经免费行情数据源（A股/ETF 日K，前复权）。

定位（数据源优先级链中的免费真实数据档）：
  wind (P0) → tencent (P1) → simulation (P6 兜底)
本客户端使 config.data.source 中的 "tencent" 真实生效（早期仅为预留接口，
无实现，导致真实数据永远降级到 simulation 兜底、预测概率恒 0.5）。

接口: GET https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
  param=<sh|sz><code>,day,<end_date>,<count>,qfq
返回: data.<code>.qfqday = [[date, open, close, high, low, volume, ...], ...]
腾讯原始列序为 date/open/close/high/low，本客户端统一重排为标准
date/open/high/low/close/volume（与 DataCollector 缓存口径一致）。

设计约束:
- 免费无 Key；任何网络/解析异常均返回 None（fail-open），由 DataCollector
  继续向 simulation 降级，绝不向上抛出。
- Windows 系统代理会拦截国内金融 API 域名 → 请求前把腾讯域名并入 NO_PROXY。
- 单页上限 800 条，按 end_date 向历史翻页直至覆盖 start_date。
- 期货(RB.SHF)/外汇(USDCNH.FXCM) 代码不支持 → 返回 None。
- 兼容 Python 3.8（本机运行环境）。
"""
from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TENCENT_HOSTS = ("web.ifzq.gtimg.cn", "ifzq.gtimg.cn", "gtimg.cn", "qt.gtimg.cn")
_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_PAGE_SIZE = 800
_MAX_PAGES = 6          # 800×6 ≈ 覆盖近 19 年日K，足够 start_date=2020-01-01
_REQUEST_TIMEOUT = 10


def ensure_no_proxy() -> None:
    """把腾讯行情域名并入 NO_PROXY 环境变量（幂等）。

    系统代理(如 127.0.0.1:7897)会拒绝转发国内金融 API 域名，必须在
    请求前设置 NO_PROXY 放行（见工作区 AGENTS.md 数据源强制规则）。
    """
    merged = {h.strip() for h in os.environ.get("NO_PROXY", "").split(",") if h.strip()}
    merged.update(_TENCENT_HOSTS)
    value = ",".join(sorted(merged))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def to_tencent_code(symbol: str) -> Optional[str]:
    """项目标的代码 → 腾讯代码。600519.SH→sh600519；000858.SZ→sz000858。

    期货/外汇等非 .SH/.SZ 代码不支持，返回 None。
    """
    s = (symbol or "").strip().upper()
    if s.endswith(".SH"):
        return "sh" + s[: -len(".SH")]
    if s.endswith(".SZ"):
        return "sz" + s[: -len(".SZ")]
    return None


class TencentClient:
    """腾讯财经日K数据客户端（免费，前复权）。"""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.timeout = float(
            (self.config.get("data", {}) or {}).get("tencent_timeout", _REQUEST_TIMEOUT)
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (TrendCastPro)"})

    # ------------------------------------------------------------------
    def fetch(
        self, symbol: str, start_date: str = "", end_date: str = ""
    ) -> Optional[pd.DataFrame]:
        """拉取日K并返回 date/open/high/low/close/volume（date 升序）。

        失败（网络/解析/不支持代码/空数据）一律返回 None。
        """
        code = to_tencent_code(symbol)
        if code is None:
            logger.warning(f"[tencent] 不支持该标的代码: {symbol}（仅 .SH/.SZ）")
            return None
        ensure_no_proxy()
        try:
            rows: List[list] = []
            end = (end_date or "").strip()
            for _page in range(_MAX_PAGES):
                part = self._request_page(code, end)
                if not part:
                    break
                rows = part + rows  # 向历史方向前拼
                earliest = str(part[0][0])
                if not start_date or earliest <= str(start_date):
                    break
                end = (pd.Timestamp(earliest) - timedelta(days=1)).strftime("%Y-%m-%d")
            if not rows:
                logger.warning(f"[tencent] {symbol} 返回空数据")
                return None
            return self._parse_rows(rows)
        except Exception as e:  # noqa: BLE001  fail-open：任何异常都降级
            logger.warning(f"[tencent] 拉取 {symbol} 失败: {e}")
            return None

    # ------------------------------------------------------------------
    def _request_page(self, code: str, end: str) -> List[list]:
        """请求单页日K，返回清洗后的原始行列表（空列表=无数据）。

        接口参数为 6 字段: param=<code>,day,<start>,<end>,<count>,qfq
        start 留空取该 end 之前 count 条；end 留空取最新。
        """
        param = f"{code},day,,{end},{_PAGE_SIZE},qfq"
        resp = self.session.get(_KLINE_URL, params={"param": param}, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        node = (payload.get("data") or {}).get(code) or {}
        klines = node.get("qfqday") or node.get("day") or []
        out: List[list] = []
        for row in klines:
            if isinstance(row, (list, tuple)) and len(row) >= 6:
                out.append(list(row[:6]))
        return out

    @staticmethod
    def _parse_rows(rows: List[list]) -> pd.DataFrame:
        """腾讯原始行 [date, open, close, high, low, volume] → 标准 OHLCV 列序。"""
        df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        for col in ("open", "close", "high", "low", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        df["date"] = df["date"].astype(str)
        return df[["date", "open", "high", "low", "close", "volume"]]