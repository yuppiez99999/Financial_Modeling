"""数据采集器：Wind MCP 优先 → akshare → 腾讯数据 → 模拟数据兜底。

面向推理引擎的最小化实现，支持：
- load_cached(symbol): 从本地缓存加载历史行情。
- _fetch_with_fallback(symbol): 按优先级拉取数据并写缓存。
在无外部数据源（Wind / akshare / 腾讯）的环境下，自动生成模拟行情数据兜底。

数据源优先级（S14/G4：akshare 由可选依赖升为 P1）：
  wind (P0) → akshare (P1 免费多市场) → tencent (P2 免费 A股/ETF) → simulation (P6 兜底)
akshare 是本链路中**唯一覆盖期货（RB.SHF 等）与外汇（USDCNH.FXCM）的免费档**，
补齐 README「已知限制」中的「期货/外汇未开」与「腾讯前复权历史退化」两个天花板。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCollector:
    """数据采集器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.save_dir = Path(config.get("training", {}).get("save_dir", "models"))
        # raw 数据缓存目录
        self.raw_dir = Path(config.get("data", {}).get("raw_dir", "data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sources = list(config.get("data", {}).get("source", ["simulation"]))
        # 数据质量门控（config.data.quality_gate）：对真实源数据做质量体检，
        # 不达标时**不删除数据**（避免训练集体量骤降），只记录告警与质量分，
        # 供监控报表呈现"当前训练数据质量"。
        self.quality_gate_enabled = bool(config.get("data", {}).get("quality_gate", False))
        self._quality_gate = None
        self.last_quality: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _get_quality_gate(self):
        """懒加载数据质量门控（仅 quality_gate 开启时需要）。"""
        if self._quality_gate is None:
            from src.data.quality_gate import DataQualityGate

            self._quality_gate = DataQualityGate()
        return self._quality_gate

    def assess_quality(self, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
        """对行情数据做质量体检（fail-soft：任何异常都不影响主链路）。"""
        if not self.quality_gate_enabled or df is None or len(df) == 0:
            return {}
        try:
            scored = self._get_quality_gate().score_market_data(df)
            info = {
                "symbol": symbol,
                "rows": int(len(scored)),
                "quality_score": round(float(scored["quality_score"].mean()), 2),
                "a_level_ratio": round(float(scored["token_level_pred"].eq("A").mean()), 4),
                "min_quality": round(float(scored["quality_score"].min()), 2),
            }
            self.last_quality[symbol] = info
            return info
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[quality] {symbol} 质量体检失败，跳过: {e}")
            return {}

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("\\", "_")
        return self.raw_dir / f"{safe}.csv"

    def load_cached(self, symbol: str) -> pd.DataFrame | None:
        """从本地缓存加载行情数据，无缓存或列不完整时返回 None。"""
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取缓存失败 {path}: {e}")
            return None
        missing = {"date", "open", "high", "low", "close", "volume"} - set(df.columns)
        if missing:
            logger.warning(f"缓存缺列 {sorted(missing)}，忽略: {path}")
            return None
        df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
        return df

    def _save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol)
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    def _fetch_simulation(self, symbol: str, n: int = 300) -> pd.DataFrame:
        """生成模拟行情（随机游走）作为数据兜底。"""
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
        # 对数随机游走
        rets = rng.normal(0, 0.015, n)
        close = 100 * np.exp(np.cumsum(rets))
        # 构造 OHLCV
        high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
        open_ = np.r_[close[0], close[:-1]]
        volume = rng.integers(100_000, 5_000_000, n)
        return pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def _fetch_with_fallback(
        self, symbol: str, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame | None:
        """按数据源优先级拉取数据；无外部源时用模拟数据兜底。

        仅真实数据源(wind/akshare/tencent)成功时才写缓存；simulation 兜底不落盘，
        避免随机游走数据污染 data/raw/<symbol>.csv 被后续推理误用。
        """
        for src in self.sources:
            try:
                if src == "wind":
                    df = self._fetch_wind(symbol)
                elif src == "akshare":
                    df = self._fetch_akshare(symbol, start_date=start_date, end_date=end_date)
                elif src == "tencent":
                    df = self._fetch_tencent(symbol, start_date=start_date, end_date=end_date)
                elif src == "simulation":
                    df = self._fetch_simulation(symbol)
                    if df is not None and len(df) > 0:
                        return self._clip_window(df, start_date, end_date)
                    continue
                else:
                    continue
                if df is not None and len(df) > 0:
                    self._save_cache(symbol, df)
                    return df
            except Exception as e:  # noqa: BLE001
                logger.warning(f"数据源 {src} 获取 {symbol} 失败: {e}")
        logger.warning(f"所有数据源均无法获取 {symbol} 的数据")
        return None

    def _fetch_wind(self, symbol: str) -> pd.DataFrame | None:
        """Wind MCP 数据源（需 WIND_API_KEY）。未配置时返回 None。"""
        if not os.environ.get("WIND_API_KEY"):
            logger.warning("未配置 WIND_API_KEY，跳过 Wind 数据源")
            return None
        # 预留接入 Wind MCP 的接口；缺少 SDK 时抛错走 fallback
        try:
            from src.data.wind_client import WindClient  # type: ignore

            return WindClient(self.config).fetch(symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Wind MCP 数据源不可用: {e}")
            return None

    def _fetch_akshare(
        self, symbol: str, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame | None:
        """akshare 数据源（免费多市场：A股/ETF/国内期货/外汇）。

        定位：链路上唯一覆盖期货与外汇的免费档，同时提供**独立于腾讯**的
        A股/ETF 数据通道（两条源互为交叉验证，可识别单源复权退化）。

        未安装 akshare 或接口异常时返回 None（fail-open），继续向 tencent 降级。
        """
        try:
            from src.data.akshare_client import AkshareClient

            df = AkshareClient(self.config).fetch(
                symbol, start_date=start_date, end_date=end_date
            )
        except Exception as e:  # noqa: BLE001  fail-open
            logger.warning(f"akshare 数据源不可用: {e}")
            return None
        if df is not None and len(df) > 0:
            self._save_cache(symbol, df)
            self.assess_quality(symbol, df)
        return df

    def _fetch_tencent(
        self, symbol: str, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame | None:
        """腾讯数据源（免费 A股/ETF 日K，前复权）。失败返回 None 走降级。"""
        try:
            from src.data.tencent_client import TencentClient

            df = TencentClient(self.config).fetch(
                symbol, start_date=start_date, end_date=end_date
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"腾讯数据源不可用: {e}")
            return None
        if df is not None and len(df) > 0:
            self._save_cache(symbol, df)
            self.assess_quality(symbol, df)
        return df

    def quality_summary(self) -> dict[str, Any]:
        """汇总本次进程内已体检标的质量（供监控报表 / CLI 输出）。"""
        if not self.last_quality:
            return {"available": False, "reason": "no_assessment"}
        scores = [v["quality_score"] for v in self.last_quality.values()]
        return {
            "available": True,
            "assessed_symbols": len(scores),
            "avg_quality_score": round(sum(scores) / len(scores), 2),
            "min_quality_score": round(min(scores), 2),
            "details": dict(self.last_quality),
        }

    def fetch_realtime(
        self, symbol: str, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame | None:
        """只走真实数据源(wind/akshare/tencent)拉取并写缓存；全部失败返回 None。

        与 _fetch_with_fallback 的区别：绝不落 simulation 兜底数据——
        供推理链路的"缓存过期自动刷新"使用，防止随机游走假数据
        （日期恰好等于今天，可骗过新鲜度检查）替换真实历史缓存。
        """
        for src in self.sources:
            if src == "simulation":
                continue
            try:
                if src == "wind":
                    df = self._fetch_wind(symbol)
                elif src == "akshare":
                    df = self._fetch_akshare(symbol, start_date=start_date, end_date=end_date)
                elif src == "tencent":
                    df = self._fetch_tencent(symbol, start_date=start_date, end_date=end_date)
                else:
                    continue
                if df is not None and len(df) > 0:
                    self._save_cache(symbol, df)
                    return df
            except Exception as e:  # noqa: BLE001
                logger.warning(f"真实数据源 {src} 获取 {symbol} 失败: {e}")
        logger.warning(f"所有真实数据源均无法获取 {symbol} 的数据")
        return None

    # ------------------------------------------------------------------
    def collect_all(self) -> dict[str, pd.DataFrame]:
        """按 config.data.markets 采集全部启用标的（ModelTrainer Step1 契约）。

        训练数据始终重新拉取（不优先读缓存），防止历史污染缓存参与训练；
        按 start_date/end_date 截取窗口，行数不足 min_train_rows 的标的跳过。
        """
        data_cfg = self.config.get("data", {}) or {}
        markets = data_cfg.get("markets", {}) or {}
        start = str(data_cfg.get("start_date", "") or "")
        end = str(data_cfg.get("end_date", "") or "")
        min_rows = int(data_cfg.get("min_train_rows", 60))

        wanted: list[str] = []
        for _name, mcfg in markets.items():
            if (mcfg or {}).get("enabled"):
                for symbol in (mcfg or {}).get("symbols", []) or []:
                    if symbol not in wanted:
                        wanted.append(symbol)

        all_data: dict[str, pd.DataFrame] = {}
        for symbol in wanted:
            df = self._fetch_with_fallback(symbol, start_date=start, end_date=end)
            if df is None or df.empty:
                logger.warning(f"[collect_all] {symbol} 采集失败，跳过")
                continue
            df = self._clip_window(df, start, end)
            if len(df) < min_rows:
                logger.warning(f"[collect_all] {symbol} 数据不足({len(df)}<{min_rows})，跳过")
                continue
            all_data[symbol] = df.reset_index(drop=True)
            logger.info(f"[collect_all] {symbol}: {len(df)} 行")
        logger.info(f"[collect_all] 完成: {len(all_data)}/{len(wanted)} 标的")
        return all_data

    @staticmethod
    def _clip_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """按 start/end 日期截取窗口（date 列存在且窗口非空时）。"""
        if "date" not in df.columns or (not start_date and not end_date):
            return df
        d = pd.to_datetime(df["date"])
        mask = pd.Series(True, index=df.index)
        if start_date:
            mask &= d >= pd.Timestamp(start_date)
        if end_date:
            mask &= d <= pd.Timestamp(end_date)
        return df.loc[mask]
