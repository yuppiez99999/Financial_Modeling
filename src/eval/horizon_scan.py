"""多周期口径探索评估（S10）：把「门禁卡在周期选择上」变成可复算、可审计的证据。

问题从哪来（S9 实测结论）：
  §16.5 用真实 26 标的池跑出了一条**反直觉但对决策最关键**的结论 ——
  现行门禁口径是 short/mid/long = 5/10/20 日，其中 short/mid 长期卡在门槛线；
  但把预测周期拉长到 40/60 日，整池与分池**全部过线**：

    | 预测周期 | 整池 IC | 命中率 | 结论 |
    |---------|---------|--------|------|
    | 5 日    | +0.014  | 50.31% | ❌ |
    | 20 日   | +0.073  | 50.84% | ❌ |
    | 40 日   | +0.098  | 53.87% | ✅ |
    | 60 日   | +0.150  | 56.82% | ✅ |

  也就是说：**信号是真实存在的，只是 5/10/20 日这个尺度上模型没有优势。**

S9 把这条结论写成「属于产品口径变更，需要人工决策，本轮不做」。
本模块（S10）落地的**不是**这次变更，而是**支撑这次决策所需要的能力**：
让「换个周期会怎样」这件事变成一条命令，且**结果可复算、口径可审计、默认不改门禁**。

本模块做什么：
  - 给一组候选周期（如 5/10/20/40/60 日）各自独立跑**同一套** walk-forward 口径；
  - 输出每周期 / 每分池的 IC、命中率、样本、是否达标；
  - 给出门禁口径切换的收益/代价**对照**（不改结论，只给证据）；
  - 落盘 ``reports/horizon_scan.json``，供监控报表 / API / 日报消费。

本模块**不做什么**（边界比功能重要）：
  - **不改现行门禁口径**：`data.prediction_horizons` 一个字不动，
    `strategy_gate` 放行结论逐字段不变（`affects_gate=false`）；
  - **不自动选周期**：不"挑一条能过线的周期"就宣布解锁 ——
    那是典型的**多重比较**（在多个候选里挑最好看的），会让指标失去统计意义；
  - **不做长期外推**：候选周期受样本长度约束，样本不足以覆盖一个完整周期时
    **如实拒答**（``available=false``），绝不用半截窗口凑数；
  - **不下单、不给仓位**：只报告口径敏感性，不产出交易信号。

无前视保证：
  每个候选周期都用 ``build_supervised`` 重新构造目标（前视窗口随周期变化），
  walk-forward 折**在每个周期内部**独立切分，测试折严格在训练折之后；
  本模块只做汇总与判定，不接触价格原始序列。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.inference.ic import ICCalculator

logger = logging.getLogger(__name__)

REPORT_NAME = "horizon_scan.json"

# 候选周期（交易日）。5/10/20 = 现行门禁口径；40/60 = S9 实测发现的强势区间。
DEFAULT_CANDIDATES = (5, 10, 20, 40, 60)

# 单周期最少有效样本：低于此只记指标不判定（与 ICCalculator.min_samples 含义不同，
# 后者管单周期内样本数，这里管"这个周期值不值得给结论"）。
DEFAULT_MIN_SAMPLES = 30

# 候选周期上限占数据长度的比例：一个 60 日周期至少要有这么长的历史，
# 否则 forward return 覆盖不住，指标没有含义。
DEFAULT_MIN_COVERAGE = 1.0


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (config or {}).get("horizon_scan", {}) or {}


def _current_horizon_days(config: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """现行门禁口径：{horizon 名: 天数}（用于标注"哪个是现行口径"）。"""
    cfg = ((config or {}).get("data", {}) or {}).get("prediction_horizons", {}) or {}
    out: Dict[str, int] = {}
    for name, days in cfg.items():
        try:
            out[str(name)] = int(days)
        except (TypeError, ValueError):  # pragma: no cover - 配置异常时的兜底
            continue
    return out


class HorizonScanner:
    """多周期口径扫描（纯读、无网络、无副作用）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = _cfg(config)
        raw = cfg.get("candidates", None)
        self.candidates: List[int] = self._normalize_candidates(
            raw if raw is not None else DEFAULT_CANDIDATES
        )
        self.min_samples = int(cfg.get("min_samples", DEFAULT_MIN_SAMPLES))
        self.ic_calc = ICCalculator(self.config)
        gate_cfg = (self.config.get("strategy_gate", {}) or {})
        self.report_dir = Path(cfg.get("report_dir", gate_cfg.get("report_dir", "reports")))
        # 现行门禁口径（只用于"标注"，不参与判定）
        self.current_days = _current_horizon_days(self.config)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_candidates(raw: Any) -> List[int]:
        """候选周期归一化：去重、升序、剔除非法值。

        非法值**丢弃而不是抛错**：门禁的诊断工具不能因为一个配置笔误整条挂掉。
        但也不静默修正（例如把 0 变成 1）—— 那会让"扫了哪些周期"变得不可解释。
        """
        out: List[int] = []
        for item in (raw or []):
            try:
                days = int(item)
            except (TypeError, ValueError):
                continue
            if days > 0 and days not in out:
                out.append(days)
        return sorted(out)

    # ------------------------------------------------------------------
    def assess_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估每个候选周期的样本覆盖度（够不够跑一个完整周期）。

        这是**能否给结论**的前置条件：60 日周期需要至少 60 根 K 线才有一个
        已到期的 forward return。样本不足时如实标注，不硬凑。
        """
        lengths: Dict[str, int] = {}
        for symbol, df in (data or {}).items():
            try:
                lengths[str(symbol)] = int(len(df))
            except Exception:  # noqa: BLE001 - 非 DataFrame 输入不得中断扫描
                continue
        max_len = max(lengths.values()) if lengths else 0

        per_horizon: Dict[str, Any] = {}
        for days in self.candidates:
            # 需要的行数远不止 days：一个 walk-forward 折还要留出训练窗口。
            # 这里用「至少 days 根 K 线」作为**最低**门槛，只做 fail-close 判断。
            required = days
            enough = max_len >= required
            per_horizon[str(days)] = {
                "horizon_days": days,
                "max_symbol_length": max_len,
                "required_rows": required,
                "available": bool(enough),
                "reason": "" if enough else (
                    f"最长标的历史 {max_len} 行 < 该周期所需 {required} 行，"
                    "forward return 覆盖不住，不给结论"
                ),
            }
        return {
            "symbol_lengths": lengths,
            "max_symbol_length": max_len,
            "per_horizon": per_horizon,
        }

    # ------------------------------------------------------------------
    def evaluate_horizon(
        self,
        horizon_days: int,
        sequences: Dict[str, Any],
        asset_class: Optional[str] = None,
        symbol_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """对单个候选周期给出门禁判定（与现行门禁完全同一套阈值）。

        Args:
            sequences   : ``{"scores": [...], "returns": [...], "window_size": int|None,
                            "neutral_band": float}`` —— 由调用方用既有 walk-forward
                            口径构造（本模块不训练、不读行情）；
            asset_class : 分池标识（'' / None = 整池口径）；
            symbol_count: 参与该次的标的数（只用于展示，不参与判定）。

        Returns:
            可直接进 JSON 的结果（含是否现行口径、指标、判定、reason）。
        """
        payload = sequences or {}
        res = self.ic_calc.evaluate(
            f"{int(horizon_days)}d",
            int(horizon_days),
            list(payload.get("scores") or []),
            list(payload.get("returns") or []),
            window_size=payload.get("window_size"),
            neutral_band=payload.get("neutral_band", 0.0) or 0.0,
        ).to_dict()

        entry: Dict[str, Any] = {
            "horizon_days": int(horizon_days),
            "asset_class": asset_class or "",
            "symbol_count": int(symbol_count) if symbol_count is not None else None,
            "is_current": int(horizon_days) in set(self.current_days.values()),
            "available": bool(res.get("available")) and res.get("samples", 0) >= self.min_samples,
            "passed": False,
            "ic": res.get("ic"),
            "icir": res.get("icir"),
            "hit_rate": res.get("hit_rate"),
            "hit_rate_raw": res.get("hit_rate_raw"),
            "neutral_band": res.get("neutral_band"),
            "samples": res.get("samples", 0),
            "windows": res.get("windows", 0),
            "reason": "",
        }
        if not entry["available"]:
            entry["reason"] = res.get("reason") or (
                f"有效样本 {res.get('samples', 0)} < {self.min_samples}，不给结论"
            )
            return entry
        entry["passed"] = bool(res.get("passed"))
        if not entry["passed"]:
            entry["reason"] = res.get("reason") or "未达门禁"
        else:
            entry["reason"] = "达标"
        return entry

    # ------------------------------------------------------------------
    def decide(
        self,
        sequences_by_horizon: Dict[str, Dict[str, Any]],
        coverage: Optional[Dict[str, Any]] = None,
        pools: Optional[Dict[str, Dict[str, Any]]] = None,
        symbol_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """汇总多周期扫描结果。

        Args:
            sequences_by_horizon: ``{horizon_days(str): {scores, returns, ...}}``；
            coverage            : ``assess_coverage`` 的输出（可选）；
            pools               : ``{asset_class: {horizon_days(str): {...}}}``（可选，
                                  分池口径的扫描结果）；
            symbol_count        : 整池标的数。

        Returns:
            ``reports/horizon_scan.json`` 的完整 payload。
            ⚠️ 结果**只作证据**：``affects_gate`` 恒为 False，不改变放行结论。
        """
        currents = set(self.current_days.values())
        # ---- 整池口径 ----
        pooled: Dict[str, Any] = {}
        for key, seq in (sequences_by_horizon or {}).items():
            try:
                days = int(key)
            except (TypeError, ValueError):  # pragma: no cover
                continue
            pooled[str(days)] = self.evaluate_horizon(
                days, seq, asset_class=None, symbol_count=symbol_count
            )

        # ---- 分池口径（可选）----
        pools_out: Dict[str, Any] = {}
        for cls, per_h in (pools or {}).items():
            rows: Dict[str, Any] = {}
            for key, seq in (per_h or {}).items():
                try:
                    days = int(key)
                except (TypeError, ValueError):  # pragma: no cover
                    continue
                rows[str(days)] = self.evaluate_horizon(days, seq, asset_class=cls)
            if rows:
                pools_out[str(cls)] = {
                    "asset_class": cls,
                    "horizons": rows,
                    "passed_horizons": sorted(
                        int(d) for d, r in rows.items() if r.get("passed")
                    ),
                    "available_horizons": sorted(
                        int(d) for d, r in rows.items() if r.get("available")
                    ),
                }

        summary = self._summarize(pooled, currents)
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "src/eval/horizon_scan.py",
            "candidates": list(self.candidates),
            "current_horizons": self.current_days,
            "current_horizon_days": sorted(currents),
            "thresholds": {
                "min_ic": self.ic_calc.min_ic,
                "min_hit_rate": self.ic_calc.min_hit_rate,
                "min_samples": self.ic_calc.min_samples,
                "min_windows": self.ic_calc.min_windows,
                "scan_min_samples": self.min_samples,
            },
            "coverage": coverage or {},
            "pooled": pooled,
            "pools": pools_out,
            "summary": summary,
            # ⚠️ 恒为 False：扫描是**证据**，不是解锁手段，不改变 strategy_gate 结论
            "affects_gate": False,
            "gate_note": (
                "多周期扫描只提供口径敏感性证据，不改变现行门禁口径"
                "（data.prediction_horizons 未变），不参与 strategy_gate 判定。"
            ),
        }

    # ------------------------------------------------------------------
    def _summarize(self, pooled: Dict[str, Any], currents: set) -> Dict[str, Any]:
        """把整池扫描压成一句话结论（如实，不做任何"已解锁"暗示）。"""
        passed = sorted(int(d) for d, r in pooled.items() if r.get("passed"))
        available = sorted(int(d) for d, r in pooled.items() if r.get("available"))
        current_passed = sorted(
            int(d) for d, r in pooled.items()
            if r.get("passed") and int(d) in currents
        )
        candidates_passed = [d for d in passed if d not in currents]

        if not available:
            narrative = "所有候选周期样本均不足，无法给出口径敏感性结论。"
        elif not passed:
            narrative = (
                f"候选周期 {available} 全部未达门禁：信号弱于周期选择无关，"
                "换周期解决不了问题。"
            )
        elif current_passed and not candidates_passed:
            narrative = (
                "现行口径已达标，候选周期无额外收益：无需调整周期。"
            )
        elif not current_passed and candidates_passed:
            narrative = (
                f"现行口径未达标，但候选周期 {candidates_passed} 达标 —— "
                "信号在更长周期上更强。⚠️ 这**不是**自动解锁："
                "切换门禁周期属于产品口径变更，须人工决策并重新评估。"
            )
        else:
            narrative = (
                f"现行口径 {current_passed} 与候选周期 {candidates_passed} 均达标，"
                "周期选择对结论不敏感。"
            )
        return {
            "passed_horizons": passed,
            "available_horizons": available,
            "current_passed_horizons": current_passed,
            "candidate_passed_horizons": candidates_passed,
            "narrative": narrative,
        }

    # ------------------------------------------------------------------
    def save(self, payload: Dict[str, Any], path: Optional[str] = None) -> Path:
        """落盘扫描报告（监控报表 / API 只读消费）。"""
        out = Path(path) if path else (self.report_dir / REPORT_NAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """读取既有扫描报告（纯读，缺失返回 available=False，绝不臆测）。"""
        src = Path(path) if path else (self.report_dir / REPORT_NAME)
        if not src.exists():
            return {
                "available": False,
                "reason": "no_horizon_scan_report",
                "hint": "先运行 `python main.py horizon-scan` 生成 reports/horizon_scan.json",
            }
        try:
            payload = json.loads(src.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[horizon-scan] 扫描报告读取失败: {e}")
            return {"available": False, "error": str(e)}
        payload["available"] = True
        payload["report_path"] = str(src)
        return payload

    # ------------------------------------------------------------------
    def summarize_rows(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """把整池扫描压成报表行（不塞原始序列）。"""
        rows: List[Dict[str, Any]] = []
        for key in sorted((payload.get("pooled") or {}).keys(), key=lambda k: int(k)):
            entry = (payload.get("pooled") or {})[key]
            rows.append({
                "horizon_days": int(key),
                "is_current": bool(entry.get("is_current")),
                "available": bool(entry.get("available")),
                "passed": bool(entry.get("passed")),
                "ic": entry.get("ic"),
                "hit_rate": entry.get("hit_rate"),
                "samples": entry.get("samples", 0),
                "reason": entry.get("reason", ""),
            })
        return rows


def compare_with_current(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把候选周期与现行门禁口径对照，回答「换周期到底有没有用」。

    ⚠️ 输出是**决策输入**，不是放行结论：
    即便候选周期全部达标，也不得据此宣称"信号已可用"或调用 `strategy_gate.gated`。
    """
    pooled = payload.get("pooled") or {}
    currents = set(payload.get("current_horizon_days") or [])
    if not pooled:
        return {
            "available": False, "current_passed": False, "candidate_passed": [],
            "narrative": "扫描报告为空，无法对照（先跑 `python main.py horizon-scan`）",
        }

    def _passed(days: int) -> bool:
        return bool((pooled.get(str(days)) or {}).get("passed"))

    current_passed = any(_passed(d) for d in currents)
    candidate_passed = sorted(
        int(d) for d in pooled.keys() if int(d) not in currents and _passed(int(d))
    )
    degraded = sorted(
        int(d) for d in currents if d in {int(k) for k in pooled.keys()} and not _passed(d)
    )

    if current_passed and not candidate_passed:
        narrative = "现行门禁口径已达标，候选周期未带来额外收益 → 不建议切换。"
    elif not current_passed and candidate_passed:
        narrative = (
            f"现行门禁口径未达标，候选周期 {candidate_passed} 达标。"
            "切换属产品口径变更（须人工决策），且需重新做泄漏与偏差审查；"
            "本报告不构成放行依据。"
        )
    elif current_passed and candidate_passed:
        narrative = (
            f"现行口径与候选周期 {candidate_passed} 均达标，周期选择不敏感 → 无切换必要。"
        )
    else:
        narrative = "现行口径与候选周期均未达标：问题不在周期选择，换周期无用。"

    return {
        "available": True,
        "current_horizons": sorted(currents),
        "current_passed": current_passed,
        "current_failed_horizons": degraded,
        "candidate_passed": candidate_passed,
        "affects_gate": False,
        "narrative": narrative,
    }
