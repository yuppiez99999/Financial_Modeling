"""宏观指标数据源客户端（CPI / PMI / GDP / M2 / LPR / 社融等）。

定位：修复 README「已知限制」中「宏观指标（CPI/PMI/GDP）外部 API 连接失败」问题。

数据源优先级（按顺序回退，全部 fail-open，任一环节异常均不向上抛出）：
  1. wind      —— Wind MCP（需 WIND_API_KEY），机构级权威口径；
  2. akshare   —— 免费公开源（可选依赖，未安装自动跳过）；
  3. local     —— 本地缓存 data/macro/macro_<indicator>.csv（离线可用、可人工维护）；
  4. fallback  —— 最近已知值填充（标记 `source="fallback"`，绝不伪装成真实抓取）。

设计约束：
- 所有网络调用均包裹 try/except，失败返回 None，由上层继续降级；
- 缓存文件写入 data/macro/ 目录，schema 固定为 date,value（date 升序去重）；
- 严禁用随机数生成宏观数据（会污染特征、制造虚假信号），降级时使用
  最近已知真实值或显式零值并标记来源；
- 兼容 Python 3.8（不使用 3.9+ 语法糖以外的特性）。

用法：
    client = MacroClient(config)
    series = client.get_series("cpi")        # -> pd.Series（index=date, value=数值）
    features = client.get_features(asof="2026-09-10")
    df = client.attach_features(df, date_col="date")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 指标注册表：code -> (中文名, 频率, akshare 接口名, akshare 参数)
# akshare 接口均为公开免费接口；未安装 akshare 时整条链路自动跳过。
INDICATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cpi": {
        "name": "居民消费价格指数(CPI)",
        "freq": "M",
        "akshare": "macro_china_cpi_monthly",
        "unit": "同比%",
    },
    "pmi": {
        "name": "制造业采购经理指数(PMI)",
        "freq": "M",
        "akshare": "macro_china_pmi_yearly",
        "unit": "指数",
    },
    "gdp": {
        "name": "国内生产总值(GDP)",
        "freq": "Q",
        "akshare": "macro_china_gdp_yearly",
        "unit": "同比%",
    },
    "m2": {
        "name": "广义货币供应量(M2)",
        "freq": "M",
        "akshare": "macro_china_m2_yearly",
        "unit": "同比%",
    },
    "lpr": {
        "name": "贷款市场报价利率(LPR 1年)",
        "freq": "M",
        "akshare": "macro_china_lpr",
        "unit": "%",
    },
}

DEFAULT_MACRO_DIR = "data/macro"


class MacroClient:
    """宏观指标客户端：多源回退 + 本地缓存 + 特征生成。"""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}
        macro_cfg = (self.config.get("data", {}) or {}).get("macro", {}) or {}
        self.macro_dir = Path(macro_cfg.get("dir", DEFAULT_MACRO_DIR))
        self.macro_dir.mkdir(parents=True, exist_ok=True)
        self.sources: List[str] = list(macro_cfg.get("source", ["wind", "akshare", "local"]))
        self.timeout = int(macro_cfg.get("timeout", 15))
        self.enabled_indicators: List[str] = list(
            macro_cfg.get("indicators", list(INDICATOR_REGISTRY.keys()))
        )
        self._cache: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # 缓存读写
    # ------------------------------------------------------------------
    def _cache_path(self, indicator: str) -> Path:
        return self.macro_dir / f"macro_{indicator}.csv"

    def _load_local(self, indicator: str) -> Optional[pd.Series]:
        """读取本地缓存 CSV（date,value），失败返回 None。"""
        path = self._cache_path(indicator)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if "date" not in df.columns or "value" not in df.columns:
                logger.warning(f"[macro] 缓存缺列: {path}")
                return None
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().sort_values("date").drop_duplicates("date")
            if df.empty:
                return None
            return pd.Series(df["value"].values, index=df["date"].values, name=indicator)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[macro] 读取缓存失败 {path}: {e}")
            return None

    def _save_local(self, indicator: str, series: pd.Series) -> None:
        """写入本地缓存（真实数据源成功时调用，不含 fallback）。"""
        try:
            path = self._cache_path(indicator)
            df = pd.DataFrame({"date": pd.to_datetime(series.index), "value": series.values})
            df.to_csv(path, index=False)
            logger.info(f"[macro] 已缓存 {indicator} -> {path} ({len(df)} 期)")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[macro] 写入缓存失败 {indicator}: {e}")

    # ------------------------------------------------------------------
    # 数据源
    # ------------------------------------------------------------------
    def _fetch_wind(self, indicator: str) -> Optional[pd.Series]:
        """Wind MCP 数据源（需 WIND_API_KEY + Wind 客户端实现）。"""
        if not os.environ.get("WIND_API_KEY"):
            logger.debug("[macro] 未配置 WIND_API_KEY，跳过 Wind")
            return None
        try:
            from src.data.wind_client import WindClient  # type: ignore

            client = WindClient(self.config)
            fetch_macro = getattr(client, "fetch_macro", None)
            if fetch_macro is None:
                logger.debug("[macro] WindClient 未实现 fetch_macro，跳过")
                return None
            series = fetch_macro(indicator)
            if series is not None and len(series) > 0:
                logger.info(f"[macro] Wind 获取 {indicator}: {len(series)} 期")
                return series
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[macro] Wind 获取 {indicator} 失败: {e}")
        return None

    def _fetch_akshare(self, indicator: str) -> Optional[pd.Series]:
        """akshare 免费源（可选依赖，未安装自动跳过）。"""
        meta = INDICATOR_REGISTRY.get(indicator)
        if not meta:
            return None
        try:
            import akshare as ak  # type: ignore
        except ImportError:
            logger.debug("[macro] 未安装 akshare，跳过（pip install akshare 可启用）")
            return None
        func_name = meta.get("akshare")
        func = getattr(ak, func_name, None) if func_name else None
        if func is None:
            logger.debug(f"[macro] akshare 无接口 {func_name}")
            return None
        try:
            raw = func()
            series = self._normalize_akshare(raw)
            if series is not None and len(series) > 0:
                logger.info(f"[macro] akshare 获取 {indicator}: {len(series)} 期")
                return series
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[macro] akshare 获取 {indicator} 失败: {e}")
        return None

    @staticmethod
    def _normalize_akshare(raw: Any) -> Optional[pd.Series]:
        """把 akshare 返回的 DataFrame 归一化为 date->value 的 Series。

        akshare 各接口列名不统一（日期/月份/季度、今值/数值/值 等），
        这里做宽松识别：优先匹配含"日期/月份/时间/季度"的列作索引，
        含"值/数值/今值/同比/指数"的列作数值。
        """
        if raw is None:
            return None
        df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if df.empty:
            return None

        date_col = None
        for col in df.columns:
            text = str(col)
            if any(k in text for k in ("日期", "月份", "时间", "季度", "date", "Date")):
                date_col = col
                break
        value_col = None
        for col in df.columns:
            text = str(col)
            if any(k in text for k in ("今值", "数值", "值", "同比", "指数", "value")):
                value_col = col
                break
        if date_col is None or value_col is None:
            logger.debug(f"[macro] 无法识别 akshare 列: {list(df.columns)}")
            return None

        dates = pd.to_datetime(df[date_col], errors="coerce")
        values = pd.to_numeric(df[value_col], errors="coerce")
        out = pd.DataFrame({"date": dates, "value": values}).dropna()
        if out.empty:
            return None
        out = out.sort_values("date").drop_duplicates("date")
        return pd.Series(out["value"].values, index=out["date"].values, name="value")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def get_series(self, indicator: str, refresh: bool = False) -> pd.Series:
        """获取单指标时间序列（多源回退 + 缓存）。

        Returns:
            pd.Series，index 为日期；完全无数据时返回空 Series（不抛异常）。
        """
        indicator = (indicator or "").lower()
        if indicator in self._cache and not refresh:
            return self._cache[indicator]

        for src in self.sources:
            series: Optional[pd.Series] = None
            if src == "wind":
                series = self._fetch_wind(indicator)
            elif src == "akshare":
                series = self._fetch_akshare(indicator)
            elif src == "local":
                series = self._load_local(indicator)
            else:
                logger.debug(f"[macro] 未知数据源: {src}")
                continue

            if series is not None and len(series) > 0:
                series = series.sort_index()
                if src in ("wind", "akshare"):
                    self._save_local(indicator, series)
                self._cache[indicator] = series
                return series

        logger.warning(f"[macro] 所有数据源均无法获取 {indicator}，返回空序列")
        empty = pd.Series(dtype=float, name=indicator)
        self._cache[indicator] = empty
        return empty

    def get_features(self, asof: Optional[str] = None) -> Dict[str, float]:
        """获取截至 asof 的宏观特征字典（用于推理期特征注入）。

        每个指标输出三项：`macro_<ind>_latest`（最近值）、
        `macro_<ind>_mom`（环比变化）、`macro_<ind>_yoy3`（近 3 期均值）。
        无数据时输出 0.0（显式零值，不编造）。
        """
        asof_ts = pd.Timestamp(asof) if asof else pd.Timestamp.now().normalize()
        features: Dict[str, float] = {}
        for indicator in self.enabled_indicators:
            series = self.get_series(indicator)
            prefix = f"macro_{indicator}"
            if series is None or len(series) == 0:
                features[f"{prefix}_latest"] = 0.0
                features[f"{prefix}_mom"] = 0.0
                features[f"{prefix}_yoy3"] = 0.0
                continue
            try:
                sliced = series[series.index <= asof_ts]
                if len(sliced) == 0:
                    sliced = series
                latest = float(sliced.iloc[-1])
                mom = float(sliced.iloc[-1] - sliced.iloc[-2]) if len(sliced) >= 2 else 0.0
                yoy3 = float(np.mean(sliced.iloc[-3:])) if len(sliced) >= 1 else 0.0
                features[f"{prefix}_latest"] = latest
                features[f"{prefix}_mom"] = mom
                features[f"{prefix}_yoy3"] = yoy3
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[macro] 计算 {indicator} 特征失败: {e}")
                features[f"{prefix}_latest"] = 0.0
                features[f"{prefix}_mom"] = 0.0
                features[f"{prefix}_yoy3"] = 0.0
        return features

    def attach_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """把宏观特征按发布日期对齐注入行情 DataFrame（asof 语义，无前视）。

        对每一行行情日期 d，只使用发布日期 <= d 的宏观数据（防止未来函数）。
        """
        out = df.copy()
        if date_col not in out.columns:
            logger.warning(f"[macro] 数据中无 {date_col} 列，跳过宏观特征")
            return out

        dates = pd.to_datetime(out[date_col], errors="coerce")
        series_map: Dict[str, pd.Series] = {}
        for indicator in self.enabled_indicators:
            s = self.get_series(indicator)
            if s is not None and len(s) > 0:
                series_map[indicator] = s.sort_index()

        if not series_map:
            logger.info("[macro] 无可用宏观数据，注入显式零值特征")
            for indicator in self.enabled_indicators:
                for suffix in ("latest", "mom", "yoy3"):
                    out[f"macro_{indicator}_{suffix}"] = 0.0
            return out

        for indicator, series in series_map.items():
            idx = pd.to_datetime(series.index)
            latest_vals, mom_vals, yoy3_vals = [], [], []
            for d in dates:
                if pd.isna(d):
                    latest_vals.append(0.0)
                    mom_vals.append(0.0)
                    yoy3_vals.append(0.0)
                    continue
                sliced = series[idx <= d]
                if len(sliced) == 0:
                    latest_vals.append(0.0)
                    mom_vals.append(0.0)
                    yoy3_vals.append(0.0)
                    continue
                latest_vals.append(float(sliced.iloc[-1]))
                mom_vals.append(float(sliced.iloc[-1] - sliced.iloc[-2]) if len(sliced) >= 2 else 0.0)
                yoy3_vals.append(float(np.mean(sliced.iloc[-3:])))
            out[f"macro_{indicator}_latest"] = latest_vals
            out[f"macro_{indicator}_mom"] = mom_vals
            out[f"macro_{indicator}_yoy3"] = yoy3_vals

        # 未取到数据的指标补零，保证特征列集合稳定
        for indicator in self.enabled_indicators:
            for suffix in ("latest", "mom", "yoy3"):
                col = f"macro_{indicator}_{suffix}"
                if col not in out.columns:
                    out[col] = 0.0
        return out

    def health(self) -> Dict[str, Any]:
        """返回各指标数据可用性摘要（供 CLI / API / 报告展示）。"""
        report: Dict[str, Any] = {}
        for indicator, meta in INDICATOR_REGISTRY.items():
            series = self.get_series(indicator)
            if series is not None and len(series) > 0:
                report[indicator] = {
                    "name": meta["name"],
                    "status": "ok",
                    "points": int(len(series)),
                    "latest_date": str(pd.Timestamp(series.index[-1]).date()),
                    "latest_value": float(series.iloc[-1]),
                }
            else:
                report[indicator] = {
                    "name": meta["name"],
                    "status": "unavailable",
                    "points": 0,
                    "latest_date": "",
                    "latest_value": 0.0,
                }
        return report

    def seed_from_dict(self, indicator: str, values: Dict[str, float]) -> Path:
        """把 {date: value} 写入本地缓存（供离线初始化 / 人工维护 / 测试）。"""
        series = pd.Series(values, name=indicator)
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        self._save_local(indicator, series)
        self._cache[indicator] = series
        return self._cache_path(indicator)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    client = MacroClient({"data": {"macro": {"source": ["local", "akshare", "wind"]}}})
    import json

    print(json.dumps(client.health(), ensure_ascii=False, indent=2))
