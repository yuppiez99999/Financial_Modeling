"""Laya 对照判据的**接入前预注册**（S26 / J1 · T26.7，Issue #66）。

## 问题从哪来（为什么这一步也必须在 T26.1 之前做）

T26.6 已经回答了一个顺序风险：「现在批了 1.7GB 依赖，对照**能不能离线复现**」。
但还剩下**更靠后**的一个洞：

    T26.1 批的是依赖代价，可 T26.3 的对照**没有预先写死的判定规则**。

现状（读代码可见）：`offline_contrast()` 会算出 `delta_hit_rate`、
`spearman_vs_baseline`、`confidence_return_bands`（分档×收益）——
**一堆读数，但没有任何一条「达到什么算通过」的线**。
后果很具体：

    等真实权重接进来、跑出数字，再回头定「多少算好」——
    那就是**事后挑规则**（post-hoc），读数再漂亮也**不可证伪**。

本项目已反复吃过这类亏，并已建了对应纪律：
S13 试验登记（append-only 记「试了多少次」）、S17 统一试验预算、
S25 的 DSR > 0.5 采信标准。**对照判定规则必须先冻结、再跑**。

⇒ 本模块把「T26.3 的判定规则」从事后解释提前成**接入前的预注册**：
在 T26.1 审批材料里直接给出「批了之后，什么算通过、什么算不通过」。

## 判据（写死，防事后找补）

1. **规则先冻结、再跑数**：规则内容进 `criterion_fingerprint`（sha256），
   规则被改动 → 指纹变 → 判定标记 `criterion_amended=True`，
   该次读数**不得**再按旧口径当「预注册结果」引用；
2. **fail-close**：没有预注册规则（`criterion=None`）时，任何读数都**不能**判 `pass`，
   只能 `no_preregistered_criterion`（材料不足，不给结论）；
3. **不编造**：真实权重未接入 / 读数缺失 → `unverifiable`，不给通过与否；
4. **多重比较不吃掉**：同一口径重复跑 → 计入试验预算（复用 S13 登记口径），
   达标线随试验次数**不放松**（本模块只做门，不做择优）；
5. **只读**：`affects_gate` / `affects_signal` 恒 False，不改配置、不联网。

## 与 T26.3 的关系

本模块**不改** `offline_contrast()` 的任何读数，只提供：
  - `default_criterion()`：接入前预注册的默认判定规则（本文件即冻结来源）；
  - `apply_criterion(contrast, criterion=...)`：把冻结规则套到 T26.3 的读数上，
    给出 `pass` / `fail` / `unverifiable` / `no_preregistered_criterion`。
读数为 `measured` 且过规则 → `pass`；过不了 → `fail`（不通过则整阶段取消）。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 判定结论词表（fail-close）
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_UNVERIFIABLE = "unverifiable"
VERDICT_NO_CRITERION = "no_preregistered_criterion"

# 预注册规则的冻结版本号：规则一旦改动，此处必须递增（指纹随之变化）
CRITERION_VERSION = "1.0.0"

# 预注册时间：早于 T26.1 人工签字（记录在案的「规则先于跑数」证据）
PREREGISTERED_AT = "2026-09-24T00:00:00+00:00"


def default_criterion() -> Dict[str, Any]:
    """接入前预注册的 T26.3 判定规则（**冻结来源**，改规则须递版本号）。

    规则取了三条**可独立证伪**的门（都基于 T26.3 已有的读数，不新增计算）：

    1. `min_delta_hit_rate`：高置信子集命中率相对基线的增量下限。
       取 **0.02**（2pp）—— 与项目既有的「效应量 ≥ 0.10 偏秩相关」同量级思路一致，
       但对**命中率**这种粗粒度读数放宽到 2pp（命中率分辨率本就低于 IC）。
       理由：`model_degeneration_conclusion` 已认定现有概率 spread ≈ 1pp，
       如果 Laya 连 2pp 命中率增量都给不出，它作第二决策源就没有边际。
    2. `max_spearman_vs_baseline`：两源 Spearman 相关**上限**。
       取 **0.85** —— 超上限说明两源高度重复，边际为零（本模块的「不重复」门）。
       与 T26.3 原注释一致：「Spearman 高即说明两源重复，边际为零」。
    3. `require_band_monotonicity`：置信度分档 × 已实现收益必须**单调非递减**
       （复用契约层「高置信 ≠ 可采信」判据的反面：高置信要至少不跑输）。
       单调性用**分档均值收益**序列的 Spearman(档位, 均值收益) ≥ 0 判定，
       且样本量不足的分档不出结论（`unverifiable`）。

    三条**全过**才判 `pass`；任一条明确不过 → `fail`；样本不足 → `unverifiable`。
    """
    return {
        "version": CRITERION_VERSION,
        "preregistered_at": PREREGISTERED_AT,
        "applies_to": "T26.3 离线对照（Laya 置信度 vs 现有 LightGBM 概率）",
        "frozen_before": "T26.1 人工准入（重型依赖 ~1.7GB）—— 规则先于跑数",
        "rules": {
            "min_delta_hit_rate": 0.02,
            "max_spearman_vs_baseline": 0.85,
            "require_band_monotonicity": True,
            "min_high_conf_samples": 10,
            "min_bands_for_monotonicity": 3,
        },
        "rationale": {
            "min_delta_hit_rate": (
                "现有概率 spread ≈ 1pp（model_degeneration）、置信度贴地；"
                "Laya 若给不出 2pp 命中率增量，作第二决策源无边际"
            ),
            "max_spearman_vs_baseline": (
                "两源高度重复（高 Spearman）⇒ 无独立信息，边际为零；"
                "0.85 为「仍算独立」的上限"
            ),
            "require_band_monotonicity": (
                "复用契约层「高置信 ≠ 可采信」判据：高置信分档的已实现收益"
                "至少不能随置信度下降（单调非递减）"
            ),
        },
        "affects_gate": False,
        "affects_signal": False,
    }


def criterion_fingerprint(criterion: Optional[Dict[str, Any]] = None) -> str:
    """规则指纹（sha256）：规则一改，指纹即变——用于识别「事后改规则」。"""
    c = criterion if criterion is not None else default_criterion()
    payload = json.dumps(c.get("rules", {}), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _band_monotonicity(bands: Any, min_bands: int) -> Optional[bool]:
    """分档×收益的单调性：Spearman(档位序号, 分档均值收益) ≥ 0 视为单调非递减。

    返回 None 表示样本不足（分档太少 / 缺 mean_return），**不猜**。
    """
    if not isinstance(bands, list) or len(bands) < min_bands:
        return None
    idx: list = []
    ret: list = []
    for i, b in enumerate(bands):
        if not isinstance(b, dict):
            return None
        mr = b.get("mean_return")
        if mr is None:
            return None
        idx.append(float(i))
        ret.append(float(mr))
    n = len(idx)
    if n < min_bands:
        return None
    # Spearman（秩相关）——数据已经很小，直接手算，避免额外依赖接口
    def _rank(vals: list) -> list:
        order = sorted(range(n), key=lambda k: vals[k])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ri, rr = _rank(idx), _rank(ret)
    mi, mr_ = sum(ri) / n, sum(rr) / n
    num = sum((ri[k] - mi) * (rr[k] - mr_) for k in range(n))
    di = sum((ri[k] - mi) ** 2 for k in range(n)) ** 0.5
    dr = sum((rr[k] - mr_) ** 2 for k in range(n)) ** 0.5
    if di == 0 or dr == 0:
        return None  # 全同值：无信息，不出结论
    return (num / (di * dr)) >= 0.0


def apply_criterion(contrast: Optional[Dict[str, Any]],
                    criterion: Optional[Dict[str, Any]] = None,
                    *, n_trials: int = 1) -> Dict[str, Any]:
    """把**冻结的**预注册规则套到 T26.3 的对照读数上，给 fail-close 判定。

    - `criterion=None` → `no_preregistered_criterion`（材料不足，不给结论）；
    - 读数缺失 / `unverifiable` / 真实权重未接入 → `unverifiable`；
    - 三条规则全过 → `pass`；任一条明确不过 → `fail`。
    """
    fprint = criterion_fingerprint(criterion if criterion is not None else default_criterion())
    base: Dict[str, Any] = {
        "kind": "laya_contrast_prereg_verdict",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criterion_version": (criterion or {}).get("version"),
        "criterion_fingerprint": fprint,
        "n_trials": int(n_trials),
        "affects_gate": False,
        "affects_signal": False,
    }

    if criterion is None:
        base.update({
            "verdict": VERDICT_NO_CRITERION,
            "conclusion": "未提供预注册判定规则 ⇒ 无法判定通过与否（fail-close，不给结论）",
            "checks": {},
        })
        return base

    if not isinstance(contrast, dict) or contrast.get("verdict") == "unverifiable" \
            or not contrast.get("available", False) or contrast.get("stop_loss"):
        base.update({
            "verdict": VERDICT_UNVERIFIABLE,
            "conclusion": ("对照读数不可量化（真实权重未接入或样本不足）⇒ 不能判定通过与否；"
                           "规则已预注册，待真实权重接入后原样套用即可"),
            "checks": {},
        })
        return base

    rules = dict((criterion or {}).get("rules") or {})
    checks: Dict[str, Any] = {}
    failures: list = []
    unknowns: list = []

    # 门 ①：高置信子集命中率增量下限
    delta = contrast.get("delta_hit_rate")
    hi_n = int(contrast.get("laya_high_conf_samples") or 0)
    min_n = int(rules.get("min_high_conf_samples", 10))
    lo = float(rules.get("min_delta_hit_rate", 0.02))
    if hi_n < min_n or delta is None:
        unknowns.append("delta_hit_rate（高置信子集样本不足或缺失）")
        checks["delta_hit_rate"] = {"pass": None, "delta": delta, "threshold": lo,
                                    "high_conf_samples": hi_n, "required": min_n}
    else:
        ok = float(delta) >= lo
        checks["delta_hit_rate"] = {"pass": ok, "delta": float(delta), "threshold": lo}
        if not ok:
            failures.append(f"命中率增量 {float(delta):.4f} < {lo}（第二决策源无边际）")

    # 门 ②：两源 Spearman 相关上限（不重复）
    rho = contrast.get("spearman_vs_baseline")
    hi_rho = float(rules.get("max_spearman_vs_baseline", 0.85))
    if rho is None:
        unknowns.append("spearman_vs_baseline（缺失或不可算）")
        checks["spearman"] = {"pass": None, "rho": None, "threshold": hi_rho}
    else:
        ok = abs(float(rho)) <= hi_rho
        checks["spearman"] = {"pass": ok, "rho": float(rho), "threshold": hi_rho}
        if not ok:
            failures.append(f"两源 Spearman |{float(rho):.4f}| > {hi_rho}（两源重复，边际为零）")

    # 门 ③：置信度分档 × 收益单调非递减
    if rules.get("require_band_monotonicity"):
        min_bands = int(rules.get("min_bands_for_monotonicity", 3))
        mono = _band_monotonicity(contrast.get("confidence_return_bands"), min_bands)
        if mono is None:
            unknowns.append("band_monotonicity（分档不足或收益缺失）")
            checks["band_monotonicity"] = {"pass": None, "required_bands": min_bands}
        else:
            checks["band_monotonicity"] = {"pass": bool(mono)}
            if not mono:
                failures.append("置信度分档×收益非单调（高置信跑输低置信）")

    if failures:
        verdict = VERDICT_FAIL
        conclusion = ("未过预注册门：" + "；".join(failures) + "。按排期纪律，"
                      "对照不通过**不得**以「Laya 有提升」为由放行")
    elif unknowns:
        verdict = VERDICT_UNVERIFIABLE
        conclusion = ("部分门无法判定（" + "；".join(unknowns) + "）⇒ 不判 pass，"
                      "补齐读数后按同一冻结规则重判")
    else:
        verdict = VERDICT_PASS
        conclusion = ("三条预注册门全过（命中率增量达标 + 两源不重复 + 分档单调）⇒ "
                      "Laya 可作只读第二决策源；**仍不进信号路径**（只读纪律）")

    base.update({
        "verdict": verdict,
        "conclusion": conclusion,
        "checks": checks,
        "failures": failures,
        "unknowns": unknowns,
        "note": ("判定规则在 T26.1 之前冻结（criterion_fingerprint 钉死）；"
                 "规则一经改动，指纹变、本次读数不得再当预注册结果引用。"),
    })
    return base


def build_prereg_report(contrast: Optional[Dict[str, Any]] = None,
                        criterion: Optional[Dict[str, Any]] = None,
                        *, n_trials: int = 1) -> Dict[str, Any]:
    """一页报告：预注册规则 + 指纹 + 当前读数下的判定（接入前即可审阅）。"""
    crit = criterion if criterion is not None else default_criterion()
    verdict = apply_criterion(contrast, crit, n_trials=n_trials)
    return {
        "kind": "laya_contrast_prereg",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criterion": crit,
        "criterion_fingerprint": criterion_fingerprint(crit),
        "verdict": verdict,
        "boundary": [
            "只读：affects_gate / affects_signal 恒 False",
            "规则先于跑数（frozen_before=T26.1）；改规则须递版本号并重跑",
            "无预注册规则 ⇒ 一律不判 pass（fail-close）",
        ],
        "affects_gate": False,
        "affects_signal": False,
    }
