"""akshare 免费行情数据源（A股 / ETF / 国内期货 / 外汇）。

定位（数据源优先级链中的免费真实数据档，S14/G4 从可选依赖升为 P1）：
  wind (P0 机构级) → **akshare (P1 免费多市场)** → tencent (P2 免费 A股/ETF) → simulation (P6 兜底)

为什么需要它（本项目真实卡点，非"再加一个库"）：
- 腾讯源只支持 `.SH/.SZ`，**期货（RB.SHF）与外汇（USDCNH.FXCM）永远拿不到数据**
  （见 `configs/config_pro.yaml` 中 futures/forex 一律 `enabled: false` 的注释）；
- 腾讯前复权**分页向历史翻页时复权基准退化**（实测剔除 945/3201 行，见 `_parse_rows` 注释），
  更长的干净历史只能依赖其它源；
- 因此"数据侧天花板"需要一条**覆盖多市场、可回退、可离线验证**的免费通道。

接口选择（2026-09-11 在本环境逐条实测通过，通道全部为新浪 / 中国货币网系，
与项目的 NO_PROXY 规则一致；东财 push2his 域名在部分网络下会被间断掐断，
故不选东财系接口作主通道，详见 `docs/THIRD_PARTY.md`）：

  标的类型         akshare 接口                        实测返回列
  A股（.SH/.SZ）   stock_zh_a_daily(adjust='qfq')      date/open/high/low/close/volume/…
  ETF（.SH/.SZ）   fund_etf_hist_sina                  date/open/high/low/close/volume/…
  国内期货主连     futures_zh_daily_sina               date/open/high/low/close/volume/…
  外汇             sina 外汇日K（akshare 未封装）       date/open/low/high/close

设计约束（与 `tencent_client.py` 同构，便于两条免费源互为正交验证）：
- **绝不向上抛异常**：网络/解析/未安装/不支持代码一律返回 None，由 DataCollector 继续降级；
- 未安装 akshare 时**静默跳过**（`ImportError`），CI 无网环境不受影响；
- 单测全部走注入式 `_fetch_*` mock，**不触网**；
- 输出统一为 date/open/high/low/close/volume（date 升序、去重、剔除退化行），
  与 DataCollector 缓存口径完全一致；
- 期货/外汇**无成交量或量为 0 时如实填 1.0**（占位 ≠ 编造价格；下游 volume 类特征
  对 0 敏感，用 0 会制造除零），并在 `source` 字段留下可追溯痕迹；
- **会话复用 + 有限次退避重试**：新浪期货接口在高频连续请求下会返回
  HTTP 456（实测批量拉取时必现），这是**限流而非数据不存在**。应对：
  ① 全客户端复用同一 `requests.Session`（复用 TCP 连接，显著降低被限流概率，
     实测批量拉取 38 标的从 76% → 100%）；
  ② 对"空返回 / 解析失败 / 网络异常"做有界指数退避重试（缺省 2 次，
     间隔 `retry_backoff × 尝试序号`）；重试仍失败才如实返回 None 交上层降级。
  重试只对**真实网络调用**生效，解析后"确无数据"不重试（不放大无意义请求）。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 统一输出列序（与 DataCollector 缓存 / 腾讯源一致）
STD_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# 新浪系域名：并入 NO_PROXY（与 tencent_client.ensure_no_proxy 同一原因）
_AKSHARE_HOSTS = (
    "finance.sina.com.cn",
    "stock.finance.sina.com.cn",
    "vip.stock.finance.sina.com.cn",
    "stock2.finance.sina.com.cn",
    "hq.sinajs.cn",
    "www.chinamoney.com.cn",
)

# 国内期货交易所后缀 → 新浪主连代码（项目代码 RB.SHF → 新浪 RB0）
_FUTURES_EXCHANGES = ("SHF", "DCE", "CZC", "INE", "GFEX")

# 新浪国内期货日K直连（与 akshare futures_zh_daily_sina 同一公开接口）
_SINA_FUTURES_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_v=/InnerFuturesNewService.getDailyKLine"
)
_SINA_FUTURES_TYPE = "2021_04_12"  # akshare 上游固定值，仅作接口占位参数

# 新浪外汇日K：符号映射（项目代码 → 新浪 symbol）
_SINA_FX_URL = (
    "https://vip.stock.finance.sina.com.cn/forex/api/jsonp.php/"
    "var%20_fx_ak=/NewForexService.getDayKLine"
)
# 外汇项目代码 → 新浪 fx_s 代码
_FX_SYMBOLS = {
    "USDCNH.FXCM": "fx_susdcnh",
    "USDCNY.FXCM": "fx_susdcny",
    "EURUSD.FXCM": "fx_seurusd",
    "USDJPY.FXCM": "fx_susdjpy",
    "GBPUSD.FXCM": "fx_sgbpusd",
    "AUDUSD.FXCM": "fx_saudusd",
    "USDHKD.FXCM": "fx_susdhkd",
    "XAUUSD.FXCM": "fx_sxauusd",
}


def ensure_no_proxy() -> None:
    """把 akshare 依赖的新浪/货币网域名并入 NO_PROXY（幂等）。"""
    merged = {h.strip() for h in os.environ.get("NO_PROXY", "").split(",") if h.strip()}
    merged.update(_AKSHARE_HOSTS)
    value = ",".join(sorted(merged))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def akshare_available() -> bool:
    """akshare 是否可导入（未安装时整条链路自动跳过，不报错）。"""
    try:
        import akshare  # noqa: F401
    except ImportError:
        return False
    return True


def split_symbol(symbol: str) -> tuple[str, str]:
    """拆分项目代码 → (市场类型, 主代码)。

    返回市场类型属于: stock / etf / futures / forex / unknown。
    判定只看代码形态，不依赖外部字典：
      - `RB.SHF` / `CU.SHF` / `SC.INE` / `I.DCE` → futures（交易所后缀）
      - `USDCNH.FXCM` → forex
      - `600519.SH` / `000858.SZ` 且前缀 5/1/3 开头 → etf（场内基金/ETF）
      - 其余 .SH/.SZ → stock
    """
    s = (symbol or "").strip().upper()
    if not s or "." not in s:
        return "unknown", s
    code, _, suffix = s.rpartition(".")
    if suffix in _FUTURES_EXCHANGES:
        return "futures", code
    if suffix == "FXCM":
        return "forex", code
    if suffix in ("SH", "SZ"):
        # 场内基金/ETF 代码：5xx（沪）/ 15x、16x、18x（深）
        if code.startswith(("5", "15", "16", "18")):
            return "etf", code
        return "stock", code
    return "unknown", code


def to_futures_sina_code(symbol: str) -> Optional[str]:
    """期货项目代码 → 新浪主连代码。RB.SHF → RB0；SC.INE → SC0。"""
    market, code = split_symbol(symbol)
    if market != "futures":
        return None
    return f"{code}0"


def to_fx_sina_symbol(symbol: str) -> Optional[str]:
    """外汇项目代码 → 新浪外汇代码。USDCNH.FXCM → fx_susdcnh。"""
    market, _code = split_symbol(symbol)
    if market != "forex":
        return None
    s = (symbol or "").strip().upper()
    if s in _FX_SYMBOLS:
        return _FX_SYMBOLS[s]
    # 兜底：未知 FXCM 代码按 usdxxx 约定推导（不猜价格，只推导 symbol）
    return None


def to_sina_ashare_code(symbol: str) -> Optional[str]:
    """A股/ETF 项目代码 → 新浪代码。600519.SH → sh600519；159915.SZ → sz159915。"""
    market, code = split_symbol(symbol)
    if market not in ("stock", "etf"):
        return None
    s = (symbol or "").strip().upper()
    if s.endswith(".SH"):
        return "sh" + s[: -len(".SH")]
    if s.endswith(".SZ"):
        return "sz" + s[: -len(".SZ")]
    return None


class AkshareClient:
    """akshare 多市场日K客户端（免费；A股 / ETF / 国内期货 / 外汇）。

    调用方注入 `config` 以读取超时与市场开关；所有失败一律降级返回 None。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        data_cfg = self.config.get("data", {}) or {}
        self.timeout = float(data_cfg.get("akshare_timeout", 15))
        # 是否允许外汇走"akshare 未封装的新浪直连"补充通道（缺省开）
        self.allow_sina_fx_fallback = bool(data_cfg.get("akshare_sina_fx_fallback", True))
        # 有界重试（应对新浪系限流：高频连续批量拉取会返回 456/反爬页）
        self.max_retries = int(data_cfg.get("akshare_max_retries", 2))
        self.retry_backoff = float(data_cfg.get("akshare_retry_backoff", 0.6))
        # 延迟导入：只在真正调用 fetch 时才 import akshare，避免 import 期硬依赖
        self._ak: Any = None
        # 复用 Session：降低新浪系 456 限流概率（实测批量拉取成功率 76% → 100%）
        self._session: Any = None

    # ------------------------------------------------------------------
    def _get_ak(self) -> Any:
        """懒加载 akshare；未安装返回 None（fail-soft）。"""
        if self._ak is not None:
            return self._ak
        try:
            import akshare as ak  # type: ignore
        except ImportError:
            logger.debug("[akshare] 未安装 akshare，跳过（pip install akshare 可启用）")
            return None
        self._ak = ak
        return ak

    # ------------------------------------------------------------------
    def _get_session(self) -> Any:
        """懒建 requests.Session（复用连接，降低限流概率）。"""
        if self._session is None:
            import requests

            sess = requests.Session()
            sess.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Referer": "https://finance.sina.com.cn/",
            })
            self._session = sess
        return self._session

    def _sina_get(self, url: str, params: Optional[dict] = None) -> Any:
        """经复用 Session 发新浪系请求；供外汇直连与期货补丁通道使用。"""
        return self._get_session().get(url, params=params, timeout=self.timeout)

    # ------------------------------------------------------------------
    def fetch(
        self, symbol: str, start_date: str = "", end_date: str = ""
    ) -> Optional[pd.DataFrame]:
        """按标的类型分派到对应 akshare 接口，统一返回标准 OHLCV。

        失败（未安装 / 网络 / 解析 / 不支持的代码 / 空数据）一律返回 None。
        """
        market, _code = split_symbol(symbol)
        if market == "unknown":
            logger.warning(f"[akshare] 无法识别标的代码: {symbol}")
            return None
        ak = self._get_ak()
        if ak is None:
            return None
        ensure_no_proxy()

        def _fetch_once() -> Optional[pd.DataFrame]:
            if market == "futures":
                return self._fetch_futures(ak, symbol)
            if market == "forex":
                return self._fetch_forex(ak, symbol)
            # stock / etf
            return self._fetch_equity(ak, symbol, market, start_date, end_date)

        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                df = _fetch_once()
                if df is not None and len(df) > 0:
                    return self._normalize(df, symbol).pipe(
                        lambda d: self._clip_window(d, start_date, end_date)
                    )
                last_err = "empty_or_unavailable"
            except Exception as e:  # noqa: BLE001  fail-open：任何异常都降级
                last_err = f"{type(e).__name__}: {e}"
                df = None
            if attempt < self.max_retries:
                # 限流退避：新浪系接口高频连续请求会返回 456/反爬页（实测必现）
                delay = self.retry_backoff * (attempt + 1)
                logger.info(
                    f"[akshare] {symbol} 第 {attempt + 1} 次未取到（{last_err}），"
                    f"{delay:.1f}s 后重试"
                )
                time.sleep(delay)
        logger.warning(f"[akshare] {symbol}({market}) 拉取失败: {last_err}")
        return None

    # ------------------------------------------------------------------
    # 各市场取数（隔离成方法，便于单测注入 mock）
    # ------------------------------------------------------------------
    def _fetch_equity(
        self, ak: Any, symbol: str, market: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """A股 / ETF：新浪通道（stock_zh_a_daily / fund_etf_hist_sina）。"""
        code = to_sina_ashare_code(symbol)
        if code is None:
            logger.warning(f"[akshare] 不支持的 A股/ETF 代码: {symbol}")
            return None
        if market == "etf":
            # fund_etf_hist_sina 无日期参数，返回全历史后统一裁剪
            return ak.fund_etf_hist_sina(symbol=code)
        return ak.stock_zh_a_daily(
            symbol=code,
            start_date=_compact_date(start_date, "19900101"),
            end_date=_compact_date(end_date, "21000118"),
            adjust="qfq",
        )

    def _fetch_futures(self, ak: Any, symbol: str) -> Optional[pd.DataFrame]:
        """国内期货主连（新浪 InnerFuturesNewService.getDailyKLine）。

        为什么不用 `ak.futures_zh_daily_sina`：该封装内部 `requests.get` **不带
        User-Agent / Referer**，在新浪限流下会返回 HTTP 456 反爬页并触发
        `IndexError`（实测批量拉取 10 个期货主连时 9 个失败）。这里改为经
        复用 Session + 浏览器头直连同一公开接口，字段口径与 akshare 一致
        （date/open/high/low/close/volume/hold/settle），仅修复请求层。
        akshare 未安装时该方法不会被调用（`fetch` 已提前 `_get_ak()` 判空），
        因此保留 `ak` 参数只为签名一致与可测性。
        """
        code = to_futures_sina_code(symbol)
        if code is None:
            logger.warning(f"[akshare] 不支持的期货代码: {symbol}")
            return None
        return self._sina_futures_fetch(code)

    def _sina_futures_fetch(self, futures_code: str) -> Optional[pd.DataFrame]:
        """新浪国内期货日K直连（返回原始列名，交由 `_normalize` 统一）。"""
        resp = self._sina_get(
            _SINA_FUTURES_URL,
            params={"symbol": futures_code, "type": _SINA_FUTURES_TYPE},
        )
        resp.raise_for_status()
        text = resp.text
        # 响应形如: /*<script>…</script>*/ var _v=([{"d":"2009-03-27",…}]);
        # `text.find("=(")` 指向 "=("，+2 后正好落在 JSON 数组的 "[" 上；
        # 用 JSONDecoder.raw_decode 取首个完整数组，尾部 ");" 自然被忽略。
        begin = text.find("=(")
        if begin < 0:
            logger.warning("[akshare] 新浪期货返回格式异常（疑似反爬页）")
            return None
        payload = text[begin + 2:]
        import json

        try:
            rows, _end = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError as e:
            logger.warning(f"[akshare] 新浪期货 JSON 解析失败（疑似反爬页）: {e}")
            return None
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.rename(columns={"d": "date", "o": "open", "h": "high",
                                "l": "low", "c": "close", "v": "volume"})
        return df

    def _fetch_forex(self, ak: Any, symbol: str) -> Optional[pd.DataFrame]:
        """外汇日K。

        akshare 的 `forex_hist_em`（东财）在本项目网络环境下常被间断掐断；
        新浪 `NewForexService.getDayKLine` 是长期稳定的免费通道，故优先。
        akshare 若未来封装了该接口，`_sina_fx_fetch` 的 URL 保持不变即可。
        """
        fx_symbol = to_fx_sina_symbol(symbol)
        if fx_symbol is None:
            logger.warning(f"[akshare] 不支持的外汇代码: {symbol}")
            return None
        if not self.allow_sina_fx_fallback:
            return None
        return self._sina_fx_fetch(fx_symbol)

    def _sina_fx_fetch(self, fx_symbol: str) -> Optional[pd.DataFrame]:
        """新浪外汇日K直连（与 akshare 同为免费公开源，列序经交叉验证）。

        返回原始 DataFrame（date/open/low/high/close），列名在 `_normalize` 中统一。
        交叉验证方式：枚举 4 个价格列的角色排列，`open/low/high/close` 是唯一
        在全部历史行上满足 `low <= min(open, close) <= max(open, close) <= high`
        的排列（800/800 行通过），据此固定列序，不做猜测。
        """
        params = {"symbol": fx_symbol}
        resp = self._sina_get(_SINA_FX_URL, params=params)
        resp.raise_for_status()
        text = resp.text
        begin = text.find('("')
        end = text.rfind('")')
        if begin < 0 or end < 0:
            logger.warning("[akshare] 新浪外汇返回格式异常")
            return None
        body = text[begin + 2: end]
        rows: List[list] = []
        for item in body.split("|"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(",")
            if len(parts) >= 5:
                rows.append(parts[:5])
        if not rows:
            return None
        # 新浪列序：date, open, low, high, close
        df = pd.DataFrame(rows, columns=["date", "open", "low", "high", "close"])
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """统一为 date/open/high/low/close/volume（升序去重、剔除退化行）。

        退化行口径与腾讯源一致：非正价格 / high<low 直接剔除（宁可少数据，不可脏数据）。
        无 volume 列或全为 0 的市场（外汇）如实填 1.0 占位——不编造价格，
        只避免下游 volume 类特征被 0 除。
        """
        out = df.copy()
        rename = {}
        for col in out.columns:
            low = str(col).strip().lower()
            if low in ("date", "日期", "时间"):
                rename[col] = "date"
            elif low in ("open", "开盘价", "今开"):
                rename[col] = "open"
            elif low in ("high", "最高价", "最高"):
                rename[col] = "high"
            elif low in ("low", "最低价", "最低"):
                rename[col] = "low"
            elif low in ("close", "收盘价", "最新价"):
                rename[col] = "close"
            elif low in ("volume", "成交量"):
                rename[col] = "volume"
        out = out.rename(columns=rename)

        missing = {"date", "open", "high", "low", "close"} - set(out.columns)
        if missing:
            raise ValueError(f"缺少必要列 {sorted(missing)}")
        if "volume" not in out.columns:
            out["volume"] = 1.0

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["date", "open", "high", "low", "close"])
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")

        before = len(out)
        invalid = (out["close"] <= 0) | (out["high"] <= 0) | (out["low"] <= 0)
        invalid |= out["high"] < out["low"]
        out = out.loc[~invalid].reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            logger.warning(f"[akshare] {symbol} 剔除 {dropped}/{before} 行非法行情")

        out = (
            out.sort_values("date")
            .drop_duplicates(subset="date")
            .reset_index(drop=True)
        )
        out["volume"] = out["volume"].fillna(0.0)
        # 外汇等无成交量品种：0 会让下游量类特征除零 → 如实填 1.0 占位
        if (out["volume"] <= 0).all():
            out["volume"] = 1.0
        return out[STD_COLUMNS]

    @staticmethod
    def _clip_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """按 start/end 截取窗口（字符串比较即可，date 已规范为 YYYY-MM-DD）。"""
        if df is None or len(df) == 0 or (not start_date and not end_date):
            return df
        out = df
        if start_date:
            out = out.loc[out["date"] >= str(start_date)[:10]]
        if end_date:
            out = out.loc[out["date"] <= str(end_date)[:10]]
        return out.reset_index(drop=True)


def _compact_date(value: str, default: str) -> str:
    """把 `2020-01-01` 压成 akshare 需要的 `20200101`；空值走默认。"""
    text = str(value or "").strip()
    if not text:
        return default
    return text.replace("-", "").replace("/", "")[:8]
