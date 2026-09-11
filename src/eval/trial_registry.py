"""评估试验登记（S13）：把「试了多少次」变成不可篡改的事实。

问题从哪来（S11 + S12 的共同结论）：
  S11 用多重比较校正查实了「换预测周期」不成立，S12 用正交对照查实了
  「加横截面/宏观/情感特征」也没有可验证的增量。两次都撞上同一堵墙：

  > 只要反复试，总会试出某个"看起来达标"的组合。
  > 而校正只能惩罚**本次**比较过的次数 —— 它不知道上周还试过什么。

  这就是「研究者自由度」：口径、特征集、折数、标的池、中性带……
  每调一次都是在增加一次未登记的试验。校正公式拿不到这个次数，
  于是 p 值看似有效，实际早已失真。

本模块做什么：
  把每次评估（`ic` / `ic-pool` / `horizon-scan` / `feature-experiment` …）
  登记成一条**追加写入、不可改写**的事实记录：

  - 记什么：时间、命令、口径指纹（周期/特征开关/折数/阈值/标的池）、结果摘要；
  - 算什么：**累计试验次数** → 校正时用它，而不是只数本次；
  - 给什么：`--asof` 查询某时点的累计次数，供 S11/S12 的校正接口消费；
  - 守什么：追加写（append-only）；文件无法解析时**不覆盖**，如实报错。

本模块**不做什么**：
  - **不改写历史**：不删除、不修改既有记录（改了就失去了"不可篡改"的意义）；
  - **不自动校正**：只提供次数，怎么用由调用方决定（S11/S12 已有各自口径）；
  - **不猜**：文件缺失 = 0 次；文件损坏 = `available=false` + 原因，绝不当作 0。

无前视说明：
  登记只记录**已经跑完**的评估事实，不含任何未来信息；查询按 `asof` 时间过滤，
  保证"当时的校正只知道当时试过多少次"。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "trials.jsonl"

# 参与"口径指纹"的配置路径：这些是最容易被反复调整、从而放大研究者自由度的旋钮
FINGERPRINT_KEYS: Sequence[str] = (
    "data.prediction_horizons",
    "data.start_date",
    "data.end_date",
    "features.extended_indicators",
    "features.sentiment_enabled",
    "features.macro_enabled",
    "model.type",
    "model.factors.enabled",
    "strategy_gate.min_ic",
    "strategy_gate.min_hit_rate",
    "strategy_gate.scope",
    "strategy_gate.neutral_band.enabled",
    "horizon_scan.candidates",
)


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = (config or {}).get("trial_registry", {}) or {}
    return data if isinstance(data, dict) else {}


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def registry_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = _cfg(config)
    directory = cfg.get("dir", "logs")
    name = cfg.get("filename", DEFAULT_FILENAME)
    return Path(directory) / name


def fingerprint(config: Optional[Dict[str, Any]],
                keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """抽取「口径指纹」：反复调这些旋钮就是在增加未登记的试验。"""
    use = list(keys or FINGERPRINT_KEYS)
    out: Dict[str, Any] = {}
    for key in use:
        value = _dig(config or {}, key)
        if value is not None:
            out[key] = value
    return out


def fingerprint_hash(fp: Dict[str, Any]) -> str:
    """指纹的稳定摘要（键排序后再序列化，保证同口径得到同一 hash）。"""
    raw = json.dumps(fp, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_entry(command: str, config: Optional[Dict[str, Any]],
               summary: Optional[Dict[str, Any]] = None,
               note: str = "",
               at: Optional[str] = None) -> Dict[str, Any]:
    """构造一条试验登记记录（纯函数，便于单测）。"""
    fp = fingerprint(config)
    return {
        "at": at or datetime.now().isoformat(timespec="seconds"),
        "command": str(command),
        "fingerprint": fp,
        "fingerprint_hash": fingerprint_hash(fp),
        "summary": dict(summary or {}),
        "note": str(note or ""),
    }


def load_entries(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """读取全部登记记录（只读）。

    文件缺失 = 空列表（尚未登记过）；文件损坏 = 抛出 `ValueError`，
    由调用方如实标注 `available=false` —— **绝不把损坏当成 0 次**。
    """
    path = registry_path(config)
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    bad = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            bad += 1
            logger.warning(f"[trial-registry] 第 {lineno} 行无法解析，已跳过")
            continue
        if isinstance(row, dict):
            entries.append(row)
        else:
            bad += 1
    if bad and not entries:
        raise ValueError(f"试验登记文件损坏（{bad} 行无法解析）: {path}")
    if bad:
        logger.warning(f"[trial-registry] 跳过 {bad} 行损坏记录（保留其余 {len(entries)} 条）")
    return entries


def record(command: str, config: Optional[Dict[str, Any]],
           summary: Optional[Dict[str, Any]] = None,
           note: str = "",
           at: Optional[str] = None) -> Dict[str, Any]:
    """追加登记一条试验记录并返回它（append-only，绝不改写既有记录）。

    ``at`` 缺省为当前时间；显式传入仅用于回填/测试（真实登记不传）。
    """
    entry = make_entry(command, config, summary=summary, note=note, at=at)
    path = registry_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        # 登记失败不得让评估本身失败：如实告警，返回记录但不落盘。
        # 目录创建失败也走这条路径（例如登记路径被同名文件占用）。
        logger.warning(f"[trial-registry] 登记失败（评估结果仍然有效）: {e}")
        entry["_persisted"] = False
        return entry
    entry["_persisted"] = True
    return entry


def count_trials(config: Optional[Dict[str, Any]] = None,
                 command: Optional[str] = None,
                 asof: Optional[str] = None,
                 fingerprint_hash_value: Optional[str] = None) -> Dict[str, Any]:
    """统计累计试验次数（供多重比较校正消费）。

    Args:
        command : 只统计该命令（缺省 = 全部命令）
        asof    : 只统计该时点之前（ISO 字符串，字典序比较）—— 保证"当时不知道未来"
        fingerprint_hash_value: 只统计同一口径的试验

    Returns:
        ``{"available", "count", "total", "by_command", "reason", "file"}``；
        文件损坏 → ``available=false``（**不当作 0 次**）。
    """
    path = registry_path(config)
    result: Dict[str, Any] = {
        "available": True, "count": 0, "total": 0, "by_command": {},
        "reason": "", "file": str(path), "asof": str(asof or ""),
    }
    try:
        entries = load_entries(config)
    except Exception as e:  # noqa: BLE001
        result["available"] = False
        result["reason"] = str(e)
        return result

    filtered: List[Dict[str, Any]] = []
    for row in entries:
        if asof and str(row.get("at", "")) > str(asof):
            continue
        if command and str(row.get("command")) != str(command):
            continue
        if fingerprint_hash_value and str(row.get("fingerprint_hash")) != str(fingerprint_hash_value):
            continue
        filtered.append(row)

    by_command: Dict[str, int] = {}
    for row in filtered:
        key = str(row.get("command") or "unknown")
        by_command[key] = by_command.get(key, 0) + 1

    result["count"] = len(filtered)
    result["total"] = len(entries)
    result["by_command"] = by_command
    result["trial_index"] = len(filtered) + 1  # 本次是第几次（供校正使用）
    return result


def summary(config: Optional[Dict[str, Any]] = None,
            asof: Optional[str] = None) -> Dict[str, Any]:
    """给监控报表/决策单用的汇总视图。"""
    res = count_trials(config, asof=asof)
    res["fingerprint"] = fingerprint(config)
    res["fingerprint_hash"] = fingerprint_hash(res["fingerprint"])
    if res.get("available") and res.get("total"):
        res["latest_at"] = max(str(e.get("at", "")) for e in load_entries(config))
    else:
        res["latest_at"] = ""
    return res
