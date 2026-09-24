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

## 更正与补强（T26.8，2026-09-25，跨机实测后追加）

T26.6 的对账在**交付机**上按位置前缀比对成立，但在任何其他机器上会遇到第三条
结构事实（跨机实测 2026-09-25）：**滑窗缓存会在更晚的复权锚点下整体重算** ——
分红/拆股后，前复权全序列被重排：与冻结快照按日期合并后，**最近若干个共同日期的
OHLC 完全相等、越往历史逐日漂移**（实测 000408.SZ：最近 10 个共同日期相等、
2020 年代差 116 倍；volume 两边口径不同、全窗 0 相等）。旧对账有两个后果：

- **按位置前缀比对隐含"缓存与快照窗口形状一致"**：窗口起点不同的机器上，
  位置错位即整段判 `snapshot_diverged`，把"窗口形状不同"与"价格序列分叉"混为一谈；
- **diverged 的处置写成"重建冻结快照"**：重建会使 Issue #55 第九轮
  `reports/*.frozen.*.json` 基线失效 —— 方向反了。缓存分叉不该动快照，
  该动的是**对照的数据基**。

补强（判据不放松，口径变诚实）：

1. **对齐搜索替代单一前缀**：候选 = 头部锚（shift 0）∪ 尾部锚（shift = Δ ± 5），
   取尾部窗口价格相等数最多的对齐 —— 头部锚保住 T26.6 原语义（窗口一致时行为不变），
   尾部锚覆盖"缓存窗口 ≠ 快照窗口"的真实几何；
2. **价格同一性只看 OHLC**：volume 两边口径不同（股/手）不能作为"同一价格序列"的
   否决项，改为单独报告 `volume_identical_rate`；
3. **分歧必须带诊断**：对齐后仍分叉时，给出 `divergence_pattern` ——
   `qfq_reanchor`（尾部窗口价格相等、更早历史漂移 = 复权锚点重算）/ `hard_divergence`
   （尾部也不相等）/ `undetermined`（共同可比行不足，无法定位），
   **不给无解释的 diverged**；
4. **对照数据基切换**（`load_frozen_frames`）：对照/回放需要机器无关数据基时，
   直接读**已入 git 的冻结快照**（离线、不漂移、任何机器一致），
   而不是修缓存、更不是重建快照。
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
PRICE_COLUMNS = ("open", "high", "low", "close")
VOLUME_COLUMN = "volume"   # 单独报告，不参与"同一价格序列"判定（跨机口径不同）
OFFSET_SEARCH = 5          # 尾部锚邻域（±5 行）：缓存可比快照多/少最近几行
TAIL_WINDOW = 20           # 尾部窗口：qfq 锚点重算的观察区
TAIL_MATCH_MIN = 0.5       # 尾部窗口价格相等占比 ≥ 此值 → 锚点重算形态
MIN_DIAG_COMMON = 30       # 分歧诊断的最少对齐可比行数

DIVERGENCE_PATTERNS = ("qfq_reanchor", "recent_revision", "hard_divergence", "undetermined")
OLDER_MATCH_MAX = 0.05     # 尾部窗口之外相等占比 ≤ 此值（且尾部达标）→ qfq_reanchor
OLDER_MATCH_MIN = 0.95     # 尾部窗口之外相等占比 ≥ 此值（且尾部达标）→ recent_revision


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


def _aligned_rows(cache: pd.DataFrame, frozen: pd.DataFrame, shift: int) -> int:
    """给定行位移（cache[i] ↔ frozen[i+shift]）下的可比行数。"""
    lo = max(0, -shift)
    hi = min(len(cache), len(frozen) - shift)
    return max(0, hi - lo)


def _row_equal(cache: pd.DataFrame, frozen: pd.DataFrame, shift: int) -> np.ndarray:
    """对齐后的逐行价格相等布尔向量（OHLC 全列相等才记相等）。"""
    lo = max(0, -shift)
    hi = min(len(cache), len(frozen) - shift)
    a = cache[list(PRICE_COLUMNS)].to_numpy(dtype=float)[lo:hi]
    b = frozen[list(PRICE_COLUMNS)].to_numpy(dtype=float)[lo + shift:hi + shift]
    return np.isclose(a, b, rtol=0.0, atol=1e-6, equal_nan=True).all(axis=1)


def _best_alignment(cache: pd.DataFrame, frozen: pd.DataFrame) -> int:
    """头部锚 ∪ 尾部锚邻域内，选**尾部窗口价格相等数**最多的行位移。

    - 头部锚（shift=0）：T26.6 原语义 —— 窗口形状一致（同机同批采集）时正确；
    - 尾部锚（shift = n_frozen - n_cache ± OFFSET_SEARCH）：缓存与快照窗口
      起点/终点不同但重叠于冻结日附近时正确 —— 两序列都终止在冻结日附近，
      而缓存可能比快照多/少最近几行（实测本机 000408.SZ 多 1 个交易日）。
    平手规则：相等数 → 可比行数 → 位移绝对值小者优先。
    """
    end_shift = len(frozen) - len(cache)
    candidates = [0] + [end_shift + o for o in range(-OFFSET_SEARCH, OFFSET_SEARCH + 1)]
    best: Optional[tuple[int, int, int]] = None
    for shift in sorted(set(candidates)):
        m = _aligned_rows(cache, frozen, shift)
        if m <= 0:
            continue
        eq = _row_equal(cache, frozen, shift)
        tail = eq[-min(TAIL_WINDOW, m):]
        key = (int(tail.sum()), m, -abs(shift))
        if best is None or key > best:
            best = (key[0], key[1], -abs(shift))
            best_shift = shift
    return int(best_shift) if best is not None else 0


def reconcile_pair(cache_path: str, frozen_path: str) -> Dict[str, Any]:
    """对账一只标的的缓存与冻结快照（对齐搜索 + 分歧诊断）。

    为什么先做对齐再比价：缓存日期标签**可能错位**、窗口形状**可能不同**
    （跨机实测：交付机 400 行滑窗 vs 其他机器全长窗口），按单一前缀比会把
    "窗口形状不同"误判成"价格序列分叉"。两序列都终止在冻结日附近 ⇒
    用尾部锚定 + 小邻域搜索找到正确行位移后再逐行比，才能同时回答：
    ① 价格是不是同一份；② 日期标签有没有错位；③ 分叉的话是哪一种分叉。
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
    if int(min(n_cache, n_frozen)) < MIN_COMMON_ROWS:
        out.update({"verdict": "unverifiable",
                    "reason": f"重叠行数不足（{min(n_cache, n_frozen)} < {MIN_COMMON_ROWS}），不判可复现"})
        return out

    shift = _best_alignment(cache, frozen)
    m = _aligned_rows(cache, frozen, shift)
    if m < MIN_COMMON_ROWS:
        out.update({"verdict": "unverifiable",
                    "reason": f"对齐后可比行数不足（{m} < {MIN_COMMON_ROWS}），不判可复现"})
        return out

    out["rows_cache"] = n_cache
    out["rows_frozen"] = n_frozen
    out["aligned_rows"] = m
    out["align_shift"] = int(shift)
    out["natural_window_ok"] = bool(n_frozen >= n_cache)

    eq = _row_equal(cache, frozen, shift)
    identical = bool(eq.all())
    out["identical_window"] = identical

    # volume 口径单独报告：股/手等口径差不该否决"同一价格序列"
    lo = max(0, -shift)
    hi = min(n_cache, n_frozen - shift)
    if VOLUME_COLUMN in cache.columns and VOLUME_COLUMN in frozen.columns:
        vol_eq = np.isclose(
            cache[VOLUME_COLUMN].to_numpy(dtype=float)[lo:hi],
            frozen[VOLUME_COLUMN].to_numpy(dtype=float)[lo + shift:hi + shift],
            rtol=0.0, atol=1e-6, equal_nan=True)
        out["volume_identical_rate"] = round(float(vol_eq.mean()), 6)
    else:
        out["volume_identical_rate"] = None

    # 日期标签：只对**价格相等的对齐行**逐值核 —— 价格不等的行本来就不是同一行，
    # 拿它们算标签错位会把"窗口起点不同"混进"标签漂移"
    d_cache = cache["date"].iloc[lo:hi].reset_index(drop=True)
    d_frozen = frozen["date"].iloc[lo + shift:hi + shift].reset_index(drop=True)
    eq_idx = pd.Series(eq)
    n_compared = int(eq_idx.sum())
    label_mismatch = int((d_cache[eq_idx].reset_index(drop=True)
                          != d_frozen[eq_idx].reset_index(drop=True)).sum()) if n_compared else 0
    out["n_label_mismatch"] = label_mismatch
    out["label_aligned"] = bool(label_mismatch == 0)
    if n_compared:
        diffs = (d_cache[eq_idx].reset_index(drop=True)
                 - d_frozen[eq_idx].reset_index(drop=True)).dt.days
        out["date_label_shift_days"] = float(diffs.median()) if label_mismatch else 0.0
    else:
        out["date_label_shift_days"] = 0.0
    out["date_range_cache"] = [str(cache["date"].iloc[0])[:10], str(cache["date"].iloc[-1])[:10]]
    out["date_range_frozen"] = [str(frozen["date"].iloc[0])[:10], str(frozen["date"].iloc[-1])[:10]]

    if not identical:
        out.update(_diagnose_divergence(cache, frozen, shift, eq))
        return out

    out["verdict"] = "replayable" if out["natural_window_ok"] else "window_truncated"
    out["reason"] = (
        "同一价格序列（价格列逐行相等）；日期标签错位不影响可复现性 —— "
        "冻结快照自带全窗口日期，回放口径应以**快照**为准"
        if label_mismatch else
        "同一价格序列且日期标签对齐，可离线复现")
    return out


def _diagnose_divergence(cache: pd.DataFrame, frozen: pd.DataFrame,
                         shift: int, eq: np.ndarray) -> Dict[str, Any]:
    """分歧诊断：qfq 锚点重算 / 硬分叉 / 无法定位 —— diverged 必须带解释。

    实测形态（2026-09-25，000408.SZ）：分红/拆股后缓存按**更晚的复权锚点**重算，
    与冻结快照按日期合并后最近若干共同日期 OHLC 完全相等、越往历史逐日漂移
    （尾窗 10/10 相等、2020 年代差 116 倍）。qfq_reanchor 的处置是**换数据基**
    （对照改读冻结快照，T26.8 `load_frozen_frames`），不是重建快照 ——
    重建会使 Issue #55 第九轮 frozen 基线失效。
    """
    tail_n = min(TAIL_WINDOW, eq.size)
    tail = eq[-tail_n:]
    older = eq[:-tail_n] if eq.size > tail_n else np.empty(0, dtype=bool)
    tail_rate = float(tail.mean()) if tail.size else 0.0
    older_rate = float(older.mean()) if older.size else 0.0
    base = {
        "tail_price_match_rate": round(tail_rate, 6),
        "older_price_match_rate": round(older_rate, 6),
    }
    if eq.size >= MIN_DIAG_COMMON and tail_rate >= TAIL_MATCH_MIN and older_rate <= OLDER_MATCH_MAX:
        pattern = "qfq_reanchor"
        reason = ("缓存与冻结快照在更晚的**复权锚点**下重算（分红/拆股后前复权全序列重排："
                  f"尾部 {tail_n} 行价格相等、更早历史漂移）⇒ 缓存基对照不可复现；"
                  "处置是**换数据基**（对照改读冻结快照 load_frozen_frames），不是重建快照")
    elif (eq.size >= MIN_DIAG_COMMON and tail_rate >= TAIL_MATCH_MIN
          and older_rate >= OLDER_MATCH_MIN):
        pattern = "recent_revision"
        reason = ("缓存与冻结快照**历史完全一致**、仅最近个别行不同（快照日盘中采集"
                  "或数据源修订）⇒ 该差异不构成序列分叉，但对照口径仍应以冻结快照为准"
                  "（数据基 load_frozen_frames），缓存最近行不采信")
    elif eq.size >= MIN_DIAG_COMMON and tail_rate < TAIL_MATCH_MIN and older_rate <= OLDER_MATCH_MAX:
        pattern = "hard_divergence"
        reason = ("缓存与冻结快照在重叠窗口内**不是同一价格序列**（尾部也不相等："
                  "增量已被重算或被污染）⇒ 缓存基对照不可复现")
    else:
        pattern = "undetermined"
        reason = ("缓存与冻结快照分叉，但对齐可比行不足以定位分歧来源"
                  "（可能日期标签错位或窗口几乎不重叠）⇒ 缓存基对照不可复现，须人工核查")
    return {"verdict": "snapshot_diverged", "divergence_pattern": pattern,
            "reason": reason, **base}


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
      - `snapshot_diverged`      : 存在价格序列不一致的标的（缓存基对照不可复现）
      - `unverifiable`           : 成对标的不足（缺缓存 / 缺快照）

    `snapshot_diverged` **不阻断一切**：带 `divergence_pattern` 诊断 ——
    `qfq_reanchor`（复权锚点重算，跨机常态）说明坏的是**缓存基**对照，
    对照可改用冻结快照数据基（`load_frozen_frames`）；只有 `hard_divergence`
    （尾部也不相等）才指向数据污染、须人工核查。
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
    qfq = [r for r in diverged if r.get("divergence_pattern") == "qfq_reanchor"]
    hard = [r for r in diverged if r.get("divergence_pattern") == "hard_divergence"]
    recent = [r for r in diverged if r.get("divergence_pattern") == "recent_revision"]
    undetermined = [r for r in diverged
                    if r.get("divergence_pattern") not in
                    ("qfq_reanchor", "hard_divergence", "recent_revision")]

    if not rows:
        conclusion = "unverifiable"
        reason = "无「缓存 ↔ 冻结快照」成对标的：无法对账（本模块不生成数据）"
    elif diverged:
        conclusion = "snapshot_diverged"
        if hard or undetermined:
            reason = (f"{len(diverged)} 只标的缓存基对照不可复现（qfq 锚点重算 {len(qfq)} / "
                      f"仅最近行修订 {len(recent)} / 硬分叉 {len(hard)} / "
                      f"无法定位 {len(undetermined)}）⇒ 缓存基对照不可采信；"
                      "硬分叉与无法定位项须先修数据链或人工核查")
        else:
            reason = (f"{len(diverged)} 只标的缓存与快照在更晚复权锚点下重算"
                      "（分红/拆股后前复权全序列重排，跨机常态）⇒ **缓存基**对照不可复现，"
                      "但对照数据基可切换为冻结快照（load_frozen_frames，机器无关）；"
                      "**不必**重建快照 —— 重建会使 reports/*.frozen.*.json 第九轮基线失效")
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
        "n_qfq_reanchor": len(qfq),
        "n_recent_revision": len(recent),
        "n_hard_divergence": len(hard),
        "n_undetermined_divergence": len(undetermined),
        "pairs": rows,
        "conclusion": conclusion,
        "reason": reason,
        "admission_implication": (
            "T26.1 的准入材料：缓存基对照以本报告 `replayable*` 为准；"
            "diverged 且全部为 `qfq_reanchor` 时，**缓存基**不可复现但**快照基**对照"
            "（laya-decision 的 data_base=frozen.* 路径）不受影响、仍可离线复算；"
            "存在 `hard_divergence` / `unverifiable` 时应**先修数据链再批依赖**"),
        "affects_gate": False,
        "affects_signal": False,
        "note": ("本模块只读对账：不写数据、不改配置、不联网；"
                 "Laya 输出仍不进信号路径（结构性保证）。"),
    }


def load_frozen_frames(raw_dir: str = DEFAULT_RAW_DIR,
                       frozen_glob: str = FROZEN_GLOB,
                       symbols: Optional[Sequence[str]] = None
                       ) -> Dict[str, pd.DataFrame]:
    """读取**已入库的冻结快照**为逐标的行情帧 —— 机器无关的对照数据基（T26.8）。

    背景（T26.8 跨机实测）：滑窗缓存在跨机/跨日场景会因复权锚点重算而与冻结
    基线分叉（最近若干共同日期 OHLC 相等、历史逐日漂移），且它本就是 gitignore
    的机器本地文件 ⇒ 缓存不能作为可复现对照的数据基。冻结快照已入 git、
    不随重算漂移，是仓库内唯一机器无关的行情源 —— 对照/评估需要离线可复现时
    直接读它（`laya-decision` 的 `data_base=frozen.*` 路径），不联网、不写数据。

    列契约：date/open/high/low/close/volume（与滑窗缓存同构，`FeatureEngineer`
    可直接消费）；逐帧按日期排序去重。缺目录/缺文件如实返回空或跳过。
    """
    pattern = frozen_glob
    if raw_dir != DEFAULT_RAW_DIR:
        pattern = os.path.join(raw_dir, os.path.basename(frozen_glob))
    want = {str(s) for s in symbols} if symbols else None
    frames: Dict[str, pd.DataFrame] = {}
    for fpath in sorted(glob.glob(pattern)):
        symbol = os.path.basename(fpath).split(".frozen.")[0]
        if want is not None and symbol not in want:
            continue
        df = _read(fpath)
        if df is None or df.empty:
            continue
        frames[symbol] = df
    return frames
