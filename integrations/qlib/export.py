"""qlib 因子导出层（S13 / G3，T13.1）：Alpha158 → 横截面因子 parquet。

为什么走 parquet：
  Issue #29 集成方案 G3 的验收链路是「qlib 数据层 → 因子导出 → 增量验证」。
  parquet 是 qlib 与本项目之间**最窄的接口面**：
  - qlib 侧（若安装）可以直接 `pd.read_parquet` 进 Alpha158 消费链路；
  - 本项目侧 `src/factors/qlib_factor_provider.py` 只依赖 pandas，
    qlib 安装与否不影响（CI 离线可跑）；
  - 落盘即快照：因子版本可追溯（文件名含生成时间与标的数）。

边界：导出是**一次性离线动作**，不接推理链路、不触网、`affects_gate=false`。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from integrations.qlib.alpha158 import compute_alpha158

logger = logging.getLogger(__name__)

MIN_ROWS = 60


def export_factors(data: Dict[str, pd.DataFrame], out_dir: str | Path = "data/qlib_factors",
                   symbol_col: str = "_symbol", date_col: str = "date") -> Dict[str, Any]:
    """对全部标的计算 Alpha158 并合并为一张横截面因子表，落盘 parquet。

    Args:
        data      : {symbol: 行情 DataFrame}（升序；含 open/high/low/close/volume）；
        out_dir   : 落盘目录（缺省 data/qlib_factors/）；
        symbol_col / date_col: 合并表中的标的与日期列名。

    Returns:
        {path, symbols, rows, columns, affects_gate}；无可用标的时 rows=0、
        path=None（不生成空文件，避免下游误消费空表）。
    """
    frames = []
    written = 0
    for symbol, raw in (data or {}).items():
        try:
            df = raw.sort_values(date_col).drop_duplicates(subset=[date_col])
            if len(df) < MIN_ROWS:
                logger.warning("[qlib-export] %s 行数 %d < %d，跳过", symbol, len(df), MIN_ROWS)
                continue
            factors = compute_alpha158(df)
            if factors.empty:
                continue
            merged = factors.copy()
            merged[symbol_col] = str(symbol)
            merged[date_col] = df[date_col].values
            frames.append(merged)
            written += 1
        except Exception as e:  # noqa: BLE001 - 单标的失败不拖垮整批导出
            logger.warning("[qlib-export] %s 因子计算失败: %s", symbol, e)

    if not frames:
        return {"path": None, "symbols": 0, "rows": 0, "columns": 0, "affects_gate": False}

    combined = pd.concat(frames, ignore_index=True)
    # 丢弃 warmup 期（任一因子 NaN 的行）：横截面因子表只保留「完整可比较」的行
    factor_cols = [c for c in combined.columns if c not in (symbol_col, date_col)]
    combined = combined.dropna(subset=factor_cols).reset_index(drop=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = out_path / f"alpha158_{len(written if isinstance(written, int) else []) or ''}"
    file_path = out_path / f"alpha158_{stamp}_n{combined[symbol_col].nunique()}.parquet"
    combined.to_parquet(file_path, index=False)
    logger.info("[qlib-export] Alpha158 因子表已落盘: %s（%d 行 × %d 列）",
                file_path, len(combined), len(factor_cols))
    return {
        "path": str(file_path),
        "symbols": int(combined[symbol_col].nunique()),
        "rows": int(len(combined)),
        "columns": int(len(factor_cols)),
        "affects_gate": False,
    }
