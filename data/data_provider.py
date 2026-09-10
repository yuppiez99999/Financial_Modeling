# data/data_provider.py
"""数据提供器：AKShare 为主，本地缓存为辅。"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# 避开国内金融域名走代理
for _dom in ["push2his.eastmoney.com", "push2.eastmoney.com", "eastmoney.com",
             "sinajs.cn", "sina.com.cn"]:
    os.environ.setdefault("NO_PROXY", _dom)

# 尝试导入 akshare
try:
    import akshare as ak
    _HAS_AKSHARE = True
except Exception:
    _HAS_AKSHARE = False


class DataProvider:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = str(Path(__file__).resolve().parent / "cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}.parquet"

    def _meta_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}.meta.json"

    def _load_cache(self, key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        p = self._cache_path(key)
        m = self._meta_path(key)
        if not p.exists() or not m.exists():
            return None
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
            fetched_at = meta.get("fetched_at", 0)
            if time.time() - fetched_at > max_age_hours * 3600:
                return None  # 缓存过期
            return pd.read_parquet(p)
        except Exception:
            return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        p = self._cache_path(key)
        m = self._meta_path(key)
        try:
            df.to_parquet(p, index=False)
            meta = {"fetched_at": time.time(), "rows": len(df), "key": key}
            m.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def get_index_daily(self, symbol: str = "sh000300", max_cache_hours: int = 24) -> pd.DataFrame:
        """
        获取指数日线数据。symbol 示例：sh000300（沪深300）。
        """
        cache_key = f"index_daily_{symbol}"
        cached = self._load_cache(cache_key, max_age_hours=max_cache_hours)
        if cached is not None and len(cached) > 0:
            return cached

        if not _HAS_AKSHARE:
            raise RuntimeError("akshare 不可用，且无可用缓存")

        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or len(df) == 0:
            raise RuntimeError(f"获取指数 {symbol} 数据失败")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._save_cache(cache_key, df)
        return df

    def normalize_to_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化为统一行情字段：date, open, high, low, close, volume。"""
        cols = {}
        for src, dst in [("date", "date"), ("open", "open"), ("high", "high"),
                         ("low", "low"), ("close", "close"), ("volume", "volume")]:
            if src in df.columns:
                cols[dst] = df[src]
        out = pd.DataFrame(cols)
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
            out = out.sort_values("date").reset_index(drop=True)
        return out
