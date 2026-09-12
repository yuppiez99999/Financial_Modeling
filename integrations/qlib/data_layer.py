"""qlib 数据层（S13 / G3，T13.1）：本项目 CSV 行情 → qlib .bin 列存。

为什么需要（Issue #29 集成方案 G3）：
  本项目行情缓存在 `data/raw/<symbol>.csv`（逐标的一文件，pandas 逐行读），
  qlib 的 `DataHandler` 需要 `.bin` 列存（每列一文件 + instrument 日历）才能
  发挥其高频因子表达引擎的性能。本模块把现有缓存**单向转换**成 qlib 格式，
  不改任何现有数据路径（现有 `load_market_data` 读取逻辑一字不动）。

与 qlib 运行时解耦：
  - `.bin` 写出格式为 qlib 标准（`<day_index>\t<value>` 浮点文本 + calendar），
    qlib 安装与否**不影响本模块运行**（CI 离线可跑）；
  - qlib 官方 dump 工具（`qlib.dump_bin.py`）面向 CSV 批量目录，且要求
    `pandas + qlib` 版本耦合，这里按项目「fail-soft、零重型依赖」原则
    用纯 numpy 重实现（写入格式逐字节对齐 qlib 读取器语义）。

无前视保证：
  - 只做格式转换，不重采样、不改价格：`date/open/high/low/close/volume`
    原样搬运，行序保持升序（与 `load_cached` 的排序口径一致）；
  - factor（复权因子）缺省为 1.0：本项目缓存已是**前复权**价格
    （腾讯 P1 源口径），qlib 侧不再二次复权，避免重复调整扭曲价格。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# qlib .bin 存储的必备字段（qlib DataHandler LP 格式）
REQUIRED_FIELDS = ("open", "high", "low", "close", "volume")

# 每标的最少行数：低于此不值得转（walk-forward 连一折都切不出来）
MIN_ROWS = 60


def to_qlib_dataframe(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """把本项目行情缓存 DataFrame 整理成 qlib 标准列（不写盘）。

    - 列：date/open/high/low/close/volume + factor=1.0（前复权已含）；
    - 行序按 date 升序、去重（与 `DataCollector.load_cached` 同口径）；
    - 列缺失 / 行数不足 / 全 NaN 时返回 None（不猜、不凑）。
    """
    required = {"date", *REQUIRED_FIELDS}
    missing = required - set(df.columns)
    if missing:
        logger.warning("[qlib-dump] %s 缺列 %s，跳过", symbol, sorted(missing))
        return None
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce"),
        **{col: pd.to_numeric(df[col], errors="coerce") for col in REQUIRED_FIELDS},
    })
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset="date")
    out["factor"] = 1.0
    # 价格/成交量无效（<=0 或 NaN）的行丢弃：退化值进 .bin 会污染整个因子列
    for col in REQUIRED_FIELDS:
        out = out[out[col].notna() & (out[col] > 0)]
    if len(out) < MIN_ROWS:
        logger.warning("[qlib-dump] %s 有效行数 %d < %d，跳过", symbol, len(out), MIN_ROWS)
        return None
    return out.reset_index(drop=True)


def dump_bin(dump_dir: str | Path, df: pd.DataFrame, symbol: str,
             fields: tuple = REQUIRED_FIELDS) -> int:
    """把单标的行情写成 qlib .bin 列存（返回写出的行数）。

    目录结构（qlib 标准）::
        dump_dir/
          calendars/day.txt          # 全局交易日历（调用方负责汇总，见 dump_all）
          instruments/all.txt        # 标的 → 起止日期索引
          features/<symbol>/<field>.day.bin

    .bin 单文件格式：每行 ``<交易日序号>\t<浮点值>``（qlib 读取器按 float32 解析）。
    """
    dump_dir = Path(dump_dir)
    feat_dir = dump_dir / "features" / str(symbol)
    feat_dir.mkdir(parents=True, exist_ok=True)
    rows = len(df)
    for field in fields:
        values = df[field].to_numpy(dtype=np.float32)
        payload = "".join(f"{i}\t{float(v)!r}\n" for i, v in enumerate(values))
        (feat_dir / f"{field}.day.bin").write_bytes(payload.encode())
    return rows


def dump_all(dump_dir: str | Path, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """批量转换：{symbol: raw_df} → qlib 数据目录 + 汇总元数据（report_only）。

    返回 {written, skipped, total_rows, calendar_days, dump_dir}。
    任何单标的失败只跳过该标的（fail-soft），不影响其余标的。
    """
    dump_dir = Path(dump_dir)
    written: List[str] = []
    skipped: Dict[str, str] = {}
    total_rows = 0
    per_symbol_range: Dict[str, List[str]] = {}

    for symbol, raw in (data or {}).items():
        try:
            std = to_qlib_dataframe(raw, symbol)
            if std is None:
                skipped[str(symbol)] = "invalid_or_insufficient_rows"
                continue
            rows = dump_bin(dump_dir, std, symbol)
            written.append(str(symbol))
            total_rows += rows
            per_symbol_range[str(symbol)] = [
                std["date"].iloc[0].strftime("%Y-%m-%d"),
                std["date"].iloc[-1].strftime("%Y-%m-%d"),
            ]
        except Exception as e:  # noqa: BLE001 - 单标的失败不得中断批量转换
            skipped[str(symbol)] = f"error: {e}"
            logger.warning("[qlib-dump] %s 转换失败: %s", symbol, e)

    # 全局交易日历：所有标的日期的并集（升序去重）
    all_dates: set = set()
    for raw in (data or {}).values():
        try:
            dates = pd.to_datetime(raw["date"], errors="coerce").dropna()
            all_dates.update(dates.dt.normalize().tolist())
        except Exception:  # noqa: BLE001
            continue
    calendar = sorted(all_dates)
    cal_dir = dump_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in calendar) + "\n", encoding="utf-8")

    # instruments 索引：symbol \t start \t end（qlib all.txt 格式）
    inst_dir = dump_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for sym, rng in per_symbol_range.items():
        lines.append(f"{sym}\t{rng[0]}\t{rng[1]}")
    (inst_dir / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "written": written,
        "skipped": skipped,
        "symbols_written": len(written),
        "total_rows": total_rows,
        "calendar_days": len(calendar),
        "dump_dir": str(dump_dir),
        "affects_gate": False,
    }
