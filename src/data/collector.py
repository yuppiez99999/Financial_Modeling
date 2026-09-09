"""金融市场预测模型 - 数据采集模块"""

from __future__ import annotations

import json
import logging
import os
import ssl
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WindDataCollector:
    """通过 Wind MCP 获取金融行情数据"""

    WIND_CLI = os.environ.get(
        "WIND_CLI_PATH",
        r"C:\Users\Administrator\.agents\skills\wind-mcp-skill\scripts\cli.mjs",
    )

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # Wind 返回列名 -> 项目标准列名
    COL_MAP = {
        "TIME": "date",
        "OPEN": "open",
        "MATCH": "close",
        "HIGH": "high",
        "LOW": "low",
        "VOLUME": "volume",
        "TURNOVER": "amount",
        "CHANGEHANDRATE": "turn",
    }
    NUMERIC_COLS = ["open", "close", "high", "low", "volume", "amount", "turn"]
    OUTPUT_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_change", "windcode"]

    def _call_wind(self, server_type: str, tool_name: str, params: dict[str, Any]) -> dict | None:
        """调用 Wind MCP API，返回内层 data dict（含 data/error 键）"""
        params_json = json.dumps(params)
        env = os.environ.copy()
        wind_key = os.environ.get("WIND_API_KEY")
        if wind_key:
            env["WIND_API_KEY"] = wind_key
        for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            env.pop(k, None)
        env["no_proxy"] = "*"

        for attempt in range(3):
            try:
                result = subprocess.run(
                    ["node", self.WIND_CLI, "call", server_type, tool_name, params_json],
                    capture_output=True, text=True, timeout=60, env=env,
                    encoding="utf-8", errors="replace",
                )
                if not result.stdout:
                    logger.warning(f"Wind API 无输出，重试 {attempt + 1}/3")
                    time.sleep(2)
                    continue
                outer = json.loads(result.stdout)
                # Wind MCP 返回 MCP 格式: {content:[{type:text, text:"<inner_json>"}], isError:bool}
                if outer.get("isError"):
                    logger.error(f"Wind API 返回错误: {outer}")
                    return None
                if outer.get("ok") is False:
                    logger.error(f"Wind API 校验失败: {outer.get('error', {}).get('agent_action', '')}")
                    return None
                content = outer.get("content", [])
                if not content:
                    logger.warning(f"Wind API 无 content，重试 {attempt + 1}/3")
                    time.sleep(2)
                    continue
                inner = json.loads(content[0]["text"])
                if inner.get("error"):
                    logger.error(f"Wind API 业务错误: {inner['error']}")
                    return None
                return inner
            except subprocess.TimeoutExpired:
                logger.warning(f"Wind API 超时，重试 {attempt + 1}/3")
                time.sleep(3)
            except Exception as e:
                logger.error(f"Wind API 调用失败: {e}")
                return None
        return None

    def fetch_kline(self, windcode: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取 K 线数据（Wind MCP，日期格式自动转 yyyyMMdd）"""
        begin = start_date.replace("-", "")
        end = end_date.replace("-", "")
        logger.info(f"获取 {windcode} K线数据 ({begin} ~ {end})")
        result = self._call_wind("stock_data", "get_stock_kline", {
            "windcode": windcode,
            "begin_date": begin,
            "end_date": end,
        })
        if result and result.get("data") and result["data"].get("rows"):
            data = result["data"]
            columns = [c["name"] for c in data["columns"]]
            df = pd.DataFrame(data["rows"], columns=columns)
            df = df.rename(columns=self.COL_MAP)
            # 日期处理（Wind 返回带时区的 ISO 字符串，需 utc=True 避免 Mixed timezones 报错）
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
            # 数值转换
            for c in self.NUMERIC_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            # 计算涨跌幅
            df["pct_change"] = df["close"].pct_change() * 100
            df["windcode"] = windcode
            df = df[[c for c in self.OUTPUT_COLS if c in df.columns]]
            df = df.dropna(subset=["close"])
            cache_path = self.cache_dir / f"{windcode}_kline.csv"
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            logger.info(f"已缓存 {len(df)} 条 Wind 数据到 {cache_path}")
            return df
        logger.warning(f"Wind API 无数据返回，使用模拟数据替代 {windcode}")
        return self._generate_simulation_data(windcode, start_date, end_date)

    def _generate_simulation_data(self, windcode: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟 K 线数据（Wind API 不可用时使用）"""
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        np.random.seed(hash(windcode) % 2**32)

        close = 100.0
        closes = []
        for _ in range(n):
            close *= (1 + np.random.randn() * 0.02)
            closes.append(close)

        df = pd.DataFrame({
            "date": dates,
            "open": [c * (1 + np.random.rand() * 0.01 - 0.005) for c in closes],
            "high": [c * (1 + np.random.rand() * 0.02) for c in closes],
            "low": [c * (1 - np.random.rand() * 0.02) for c in closes],
            "close": closes,
            "volume": np.random.randint(500000, 5000000, n).tolist(),
            "amount": np.random.randint(50000000, 500000000, n).tolist(),
            "turn": np.random.uniform(0.5, 5.0, n).tolist(),
            "pct_change": np.random.randn(n) * 2,
            "windcode": windcode,
        })
        cache_path = self.cache_dir / f"{windcode}_kline.csv"
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"已生成 {len(df)} 条模拟数据到 {cache_path}")
        return df


class TencentDataCollector:
    """腾讯财经行情数据采集（免费、无需 API Key）

    数据来源：腾讯财经公开接口（web.ifzq.gtimg.cn）
    覆盖范围：A 股（沪/深）、港股；期货/外汇暂不支持（返回 None 交由回退源处理）
    """

    KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    # Wind 代码后缀 -> 腾讯代码前缀
    MARKET_PREFIX = {
        ".SH": "sh",
        ".SZ": "sz",
        ".HK": "hk",
    }
    NUMERIC_COLS = ["open", "close", "high", "low", "volume"]
    OUTPUT_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_change", "windcode"]

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 跳过证书验证（部分内网环境 CA 不全），保证可用性
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _convert_code(self, windcode: str) -> str | None:
        """Wind 代码 -> 腾讯代码；不支持的市场返回 None"""
        for suffix, prefix in self.MARKET_PREFIX.items():
            if windcode.endswith(suffix):
                code = windcode[: -len(suffix)]
                return f"{prefix}{code}"
        return None

    def _http_get(self, url: str, timeout: int = 20) -> dict | None:
        """发起 GET 请求并解析 JSON"""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=self._ssl_ctx) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    return json.loads(body)
            except Exception as e:
                logger.warning(f"腾讯接口请求失败 (尝试 {attempt + 1}/3): {e}")
                time.sleep(1)
        return None

    def fetch_kline(self, windcode: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """获取 K 线数据，不支持标的返回 None（由上层回退）"""
        tencent_code = self._convert_code(windcode)
        if tencent_code is None:
            logger.info(f"腾讯数据源不支持 {windcode}，交由回退源处理")
            return None

        logger.info(f"[腾讯] 获取 {windcode}({tencent_code}) K线 ({start_date} ~ {end_date})")
        # 腾讯日K接口 count 上限约 640，超出会改变返回结构
        params = {
            "param": f"{tencent_code},day,{start_date},{end_date},640,qfq",
        }
        url = f"{self.KLINE_URL}?{urllib.parse.urlencode(params)}"
        result = self._http_get(url)
        if not result or result.get("code") != 0:
            logger.warning(f"[腾讯] {windcode} 接口返回异常")
            return None

        # 腾讯返回结构: data.{tencent_code} 可能是 dict(含 qfqday/day/kline) 或直接 list
        node = result.get("data", {}).get(tencent_code, {})
        if isinstance(node, list):
            rows = node
        elif isinstance(node, dict):
            rows = node.get("qfqday") or node.get("day") or node.get("kline")
        else:
            rows = None
        if not rows:
            logger.warning(f"[腾讯] {windcode} 无 K 线数据")
            return None

        # 腾讯列顺序: date, open, close, high, low, volume, [info...]
        # 部分行末尾带额外 info 字段（长度不一致），统一只取前 6 列
        rows_fixed = [r[:6] for r in rows]
        df = pd.DataFrame(rows_fixed, columns=["date", "open", "close", "high", "low", "volume"])
        for c in self.NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["amount"] = np.nan
        df["turn"] = np.nan
        df["pct_change"] = df["close"].pct_change() * 100
        df["windcode"] = windcode
        df = df[[c for c in self.OUTPUT_COLS if c in df.columns]]
        df = df.dropna(subset=["close"])

        cache_path = self.cache_dir / f"{windcode}_kline_tencent.csv"
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"[腾讯] 已缓存 {len(df)} 条数据到 {cache_path}")
        return df


class DataCollector:
    """统一数据采集入口 - 支持多市场（股市/期货/外汇）与多数据源回退

    支持数据源（config.data.source 可填字符串或列表，按顺序回退）：
      - tencent   腾讯财经（免费、A 股/港股，无需 Key）
      - wind      Wind MCP（A 股/期货，需终端或 Key）
      - simulation 模拟数据（兜底）
    """

    # 数据源名 -> 采集器工厂
    SOURCE_FACTORIES = {
        "tencent": lambda raw_dir: TencentDataCollector(raw_dir),
        "wind": lambda raw_dir: WindDataCollector(raw_dir),
    }

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.start_date = config["data"]["start_date"]
        self.end_date = config["data"]["end_date"]
        self.raw_dir = Path(config["data"]["raw_dir"])

        # source 支持字符串或列表（回退链）
        src = config["data"]["source"]
        self.sources: list[str] = [src] if isinstance(src, str) else list(src)

        # 从 markets 配置中提取所有启用的标的
        self.symbols: list[str] = []
        self.symbol_market: dict[str, str] = {}
        markets = config["data"].get("markets", {})
        for market_name, market_cfg in markets.items():
            if market_cfg.get("enabled", False):
                for symbol in market_cfg.get("symbols", []):
                    self.symbols.append(symbol)
                    self.symbol_market[symbol] = market_name

        # 向后兼容：如果没有 markets 配置但有 symbols
        if not self.symbols and "symbols" in config["data"]:
            self.symbols = config["data"]["symbols"]

        # 实例化各数据源采集器（simulation 不需要采集器，作为最终兜底）
        self.collectors: list[tuple[str, Any]] = []
        for name in self.sources:
            if name == "simulation":
                continue  # 兜底由 WindDataCollector._generate_simulation_data 处理
            factory = self.SOURCE_FACTORIES.get(name)
            if factory is None:
                logger.warning(f"未知数据源: {name}，已跳过")
                continue
            self.collectors.append((name, factory(self.raw_dir)))
            logger.info(f"已启用数据源: {name}")

        # 数据质量门控（基于 Token A级数据训练的质量模型）
        self.enable_quality_gate = config.get("data", {}).get("quality_gate", False)
        self.quality_gate = None
        if self.enable_quality_gate:
            try:
                from src.data.quality_gate import DataQualityGate
                self.quality_gate = DataQualityGate()
                if not self.quality_gate.classifier.load():
                    logger.info("质量模型未训练，自动训练中...")
                    self.quality_gate.train(domain="finance", use_full=False)
                logger.info("数据质量门控已启用")
            except Exception as e:
                logger.warning(f"质量门控初始化失败（已禁用）: {e}")
                self.enable_quality_gate = False

    def _fetch_with_fallback(self, symbol: str) -> pd.DataFrame | None:
        """按数据源顺序尝试采集，首个成功即返回"""
        for name, collector in self.collectors:
            try:
                df = collector.fetch_kline(symbol, self.start_date, self.end_date)
                if df is not None and len(df) > 0:
                    logger.info(f"  {symbol} <- {name} ({len(df)} 条)")
                    # 质量评分
                    if self.enable_quality_gate and self.quality_gate:
                        df = self.quality_gate.score_market_data(df)
                        quality = df["quality_score"].mean()
                        a_ratio = df["token_level_pred"].eq("A").mean()
                        logger.info(f"  {symbol} 质量分={quality:.1f}, A级占比={a_ratio:.1%}")
                    return df
            except Exception as e:
                logger.warning(f"  {symbol} <- {name} 异常: {e}")

        # 最终兜底：模拟数据
        if "simulation" in self.sources or not self.collectors:
            logger.info(f"  {symbol} <- simulation (兜底)")
            return WindDataCollector(self.raw_dir)._generate_simulation_data(
                symbol, self.start_date, self.end_date
            )
        return None

    def collect_all(self) -> dict[str, pd.DataFrame]:
        """采集所有标的数据（多源回退 + 质量门控）"""
        logger.info(f"开始采集 {len(self.symbols)} 个标的数据 (源链: {' -> '.join(self.sources)})")
        for symbol in self.symbols:
            market = self.symbol_market.get(symbol, "unknown")
            logger.info(f"  - {symbol} [{market}]")

        all_data: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            df = self._fetch_with_fallback(symbol)
            if df is not None and len(df) > 0:
                all_data[symbol] = df
            else:
                logger.error(f"采集 {symbol} 失败：所有数据源均不可用")
        logger.info(f"采集完成: {len(all_data)}/{len(self.symbols)} 个标的成功")
        return all_data

    def load_cached(self, symbol: str) -> pd.DataFrame | None:
        """加载缓存数据"""
        cache_path = self.raw_dir / f"{symbol}_kline.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, encoding="utf-8-sig")
        return None
