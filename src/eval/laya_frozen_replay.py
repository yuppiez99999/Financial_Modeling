"""Laya 接入前的**冻结快照回放**（S26 / J1 · T26.6，Issue #66）。

## 问题从哪来（为什么这一步必须在 T26.1 之前做）

Issue #66 的 J 轮把 Laya 定为**只读第二决策源**，但留下一个**顺序风险**：

    T26.1（是否接受 ~1.7GB 重型依赖）要是先批了，
    T26.3 的对照很可能是**不可复现的**。

原因不是"数据会变"这么含糊，而是一条**实测可查**的结构事实：
对照用的行情是 `data/raw/<symbol>.csv`，那是一份**增量缓存**——

- 它只保留最近 N 行（实测 `600519.SH` = 400 行，窗口随每次追加而**滑动**）；
- 它按**日期标签**对齐，而各行日期与真实交易日**全程错位**（实测同一价格序列的
  日期标签整整偏移 5 个交易日）。

两条叠加的后果：**同一份模型输出，在"采集日 A"与"采集日 B"上做对照，
样本会落在不同的价格序列上** —— 读数变了，却查不出原因。
这正是本项目反复吃过的亏（Issue #55 各轮读数被拆在不同子集上，事后要专门做一轮
「统一切片 + 冻结」才敢写结论）。

⇒ 本模块把"冻结快照对账"从**事后补救**提前成**接入前的准入条件**：
在 T26.1 审批材料里直接回答「现在批了，对照能不能离线复现」。

## 判据（写死，防事后找补）

1. **增量不是快照**：把缓存与冻结快照按**位置**比（同一价格序列）——
   金额列必须逐行相等；不等即缓存已重算/被污染，整条对照链**不可采信**；
2. **行数即窗口**：缓存只保留最近 N 行（滑动窗口），冻结快照是**全窗口**——
   `logical_window_ok = rows_frozen >= rows_cache`，否则冻结快照本身有截断；
3. **日期标签必须逐值核**，不能只比区间：区间相同而内部错位是**最隐蔽**的一种；
4. **只读**：`affects_gate` / `affects_signal` 恒 False，不改数据、不改配置、不联网；
5. **不编造**：缺缓存 / 缺快照 / 长度不足 → 如实 `unverifiable`，不给结论。
"""
from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 与 `full-pool-baseline`（Issue #55 第九轮）同一批冻结快照：28 只 A股/ETF。
FROZEN_GLOB = "data/raw/*.frozen.20260917.csv"
DEFAULT_RAW_DIR = "data/raw"
MIN_COMMON_ROWS = 100      # 对账样本下限：低于此不判"可复现"
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as e:  # noqa: BLE001 - 读取失败如实降级，不抛
        logger.warning(f"[laya-replay] 读取失败 {path}: {e}")
        return None
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def reconcile_pair(cache_path: str, frozen_path: str) -> Dict[str, Any]:
    """按**位置**对账一只标的的缓存与冻结快照。

    为什么按位置而不是按日期合并：缓存日期标签**错了**（错位），
    按日期 join 会把错误当成"数据不一致"报出来，掩盖真正的结论
    ——「同一价格序列、日期标签错位」。按位置比才能同时回答两件事：
    ① 价格是不是同一份；② 日期标签有没有错位。
    """
    cache = _read(cache_path)
    frozen = _read(frozen_path)
    out: Dict[str, Any] = {
        "kind": "laya_frozen_pair",
        "cache_path": cache_path,
        "frozen_path": frozen_path,
        "affects_gate": False,
        "affects_signal": False,
    }
    if cache is None or frozen is None:
        out.update({"verdict": "unverifiable",
                    "reason": "缓存或冻结快照缺失（本模块只对账，不生成数据）"})
        return out

    n_cache, n_frozen = len(cache), len(frozen)
    n = int(min(n_cache, n_frozen))
    out["rows_cache"] = n_cache
    out["rows_frozen"] = n_frozen
    out["natural_window_ok"] = bool(n_frozen >= n_cache)
    if n < MIN_COMMON_ROWS:
        out.update({"verdict": "unverifiable",
                    "reason": f"重叠行数不足（{n} < {MIN_COMMON_ROWS}），不判可复现"})
        return out

    # ① 价格序列：按位置逐行比（同一窗口前缀）
    identical = True
    for col in PRICE_COLUMNS:
        if col not in cache.columns or col not in frozen.columns:
            continue
        a = cache[col].to_numpy(dtype=float)[:n]
        b = frozen[col].to_numpy(dtype=float)[:n]
        if not np.allclose(a, b, rtol=0.0, atol=1e-6, equal_nan=True):
            identical = False
            break
    out["identical_window"] = bool(identical)

    # ② 日期标签：逐值核（不是只比区间）
    d_cache = cache["date"].iloc[:n].reset_index(drop=True)
    d_frozen = frozen["date"].iloc[:n].reset_index(drop=True)
    eq = (d_cache == d_frozen)
    n_label_mismatch = int((~eq).sum())
    out["n_label_mismatch"] = n_label_mismatch
    out["label_aligned"] = bool(n_label_mismatch == 0)
    out["date_label_shift_days"] = (
        float((d_cache - d_frozen).dt.days.median()) if n_label_mismatch else 0.0)
    out["date_range_cache"] = [str(cache["date"].iloc[0])[:10], str(cache["date"].iloc[-1])[:10]]
    out["date_range_frozen"] = [str(frozen["date"].iloc[0])[:10], str(frozen["date"].iloc[-1])[:10]]

    if not identical:
        out.update({"verdict": "snapshot_diverged",
                    "reason": ("缓存与冻结快照在重叠窗口内**不是同一价格序列**"
                               "（增量已被重算或被污染）⇒ 任何基于缓存的对照读数都不可复现")})
        return out

    out["verdict"] = "replayable" if out["natural_window_ok"] else "window_truncated"
    out["reason"] = (
        "同一价格序列（金额列逐行相等）；日期标签错位不影响可复现性 —— "
        "冻结快照自带全窗口日期，回放口径应以**快照**为准"
        if n_label_mismatch else
        "同一价格序列且日期标签对齐，可离线复现")
    return out


def discover_pairs(raw_dir: str = "data/raw",
                   frozen_glob: str = FROZEN_GLOB
                   ) -> List[tuple[str, str]]:
    """列出「缓存 ↔ 冻结快照」成对存在的标的（缺任一侧不算一对）。

    `frozen_glob` 是**相对仓库根**的 glob（与 `FROZEN_GLOB` 一致，可被守卫直接
    断言"快照已入库"）；调用方传了别的 `raw_dir`（如测试的临时目录）时，
    相应地在该目录下找快照。
    """
    pattern = frozen_glob
    if raw_dir != DEFAULT_RAW_DIR:
        # 只换目录、**保留原 glob 模式**（如 `*.frozen.20240101.csv`）
        pattern = os.path.join(raw_dir, os.path.basename(frozen_glob))
    pairs: List[tuple[str, str]] = []
    for fpath in sorted(glob.glob(pattern)):
        base = os.path.basename(fpath)
        symbol = base.split(".frozen.")[0]
        cpath = os.path.join(raw_dir, f"{symbol}.csv")
        if os.path.exists(cpath):
            pairs.append((cpath, fpath))
    return pairs


def build_replay_report(raw_dir: str = DEFAULT_RAW_DIR,
                        frozen_glob: str = FROZEN_GLOB,
                        symbols: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """冻结快照回放对账报告：逐标的 + 汇总 + 结论。

    结论词表（fail-close，不把"没证明"写成"证明"）：
      - `replayable`             : 全部成对标的对账通过（对照可离线复现）
      - `replayable_with_label_drift` : 价格同一、仅日期标签错位（可复现，但口径须认快照）
      - `snapshot_diverged`      : 存在价格序列不一致的标的（对照不可复现）
      - `unverifiable`           : 成对标的不足（缺缓存 / 缺快照）
    """
    pairs = discover_pairs(raw_dir, frozen_glob)
    if symbols:
        want = {str(s) for s in symbols}
        pairs = [(c, f) for c, f in pairs
                 if os.path.basename(f).split(".frozen.")[0] in want]

    rows: List[Dict[str, Any]] = []
    for cpath, fpath in pairs:
        row = reconcile_pair(cpath, fpath)
        row["symbol"] = os.path.basename(fpath).split(".frozen.")[0]
        rows.append(row)

    ok = [r for r in rows if r.get("verdict") in ("replayable", "replayable_with_label_drift")]
    diverged = [r for r in rows if r.get("verdict") == "snapshot_diverged"]
    truncated = [r for r in rows if r.get("verdict") == "window_truncated"]
    label_drift = [r for r in rows if not r.get("label_aligned", True)]

    if not rows:
        conclusion = "unverifiable"
        reason = "无「缓存 ↔ 冻结快照」成对标的：无法对账（本模块不生成数据）"
    elif diverged:
        conclusion = "snapshot_diverged"
        reason = (f"{len(diverged)} 只标的在重叠窗口内与冻结快照**不是同一价格序列** ⇒"
                  " 对照读数不可复现，接入前必须先重建冻结快照")
    elif truncated:
        conclusion = "snapshot_diverged"
        reason = (f"{len(truncated)} 只标的的冻结快照短于当前缓存窗口（快照被截断）⇒"
                  " 回放窗口与生产窗口不一致，须先补全快照")
    elif label_drift:
        conclusion = "replayable_with_label_drift"
        reason = (f"{len(ok)}/{len(rows)} 只标的对账通过（价格序列逐行相等）；"
                  f"{len(label_drift)} 只标的**日期标签错位** ⇒ 可离线复现，"
                  "但对照口径必须以冻结快照的日期为准，不得混用缓存的错位标签")
    else:
        conclusion = "replayable"
        reason = f"{len(ok)}/{len(rows)} 只标的缓存与冻结快照逐行一致 ⇒ 对照可离线复现"

    return {
        "kind": "laya_frozen_replay",
        "generated_at": _now(),
        "frozen_glob": frozen_glob,
        "n_pairs": len(rows),
        "n_replayable": len(ok),
        "n_diverged": len(diverged),
        "n_window_truncated": len(truncated),
        "n_label_drift": len(label_drift),
        "pairs": rows,
        "conclusion": conclusion,
        "reason": reason,
        "admission_implication": (
            "T26.1 的准入材料：本报告 `replayable*` 才说明"
            "「现在批准重型依赖后，T26.3 的对照能离线复算」；"
            "`snapshot_diverged` / `unverifiable` 时应**先修数据链再批依赖**"),
        "affects_gate": False,
        "affects_signal": False,
        "note": ("本模块只读对账：不写数据、不改配置、不联网；"
                 "Laya 输出仍不进信号路径（结构性保证）。"),
    }
