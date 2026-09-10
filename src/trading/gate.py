"""策略门禁：把「只读观测信号」升级为「可进入决策路径的打分因子」的准入判定。

Q2 路线图硬要求：
  > 按评估脚本的 **IC / 命中率门禁** 判定是否将信号从「只读观测」升级为「调仓打分因子」

设计原则（与《为28终极量化交易系统提供策略决策依据》§5 一致）：
- **fail-close**：门禁不可用 / 数据缺失 / 未达标 → 一律判定为「未放行」，
  信号保持只读观测，绝不因缺数据而默认放行；
- **可解释**：每条判定给出 blocked_by 原因列表，可直接进日报与监控报表；
- **可复算**：同一份输入无论何时执行都得到同一结论（无随机、无时间依赖）。

状态机：
  readonly  ：未过门禁（默认态，信号只作参考）
  gated     ：已过门禁，可作为打分因子进入决策路径（仍不产出仓位/下单建议）
  disabled  ：显式关闭门禁（`strategy_gate.enabled: false` 或 `force_readonly: true`）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_READONLY = "readonly"
STATE_GATED = "gated"
STATE_DISABLED = "disabled"

# 审计样本的兜底门槛（与 src/monitor/health_report.py 的漂移判定口径保持一致）
DEFAULT_MIN_AUDIT_VERIFIED = 10
DEFAULT_MIN_AUDIT_HIT_RATE = 0.5


@dataclass
class GateDecision:
    """门禁判定结果（可直接序列化进 API / 日报）。"""

    state: str
    passed: bool
    scope: str = "all"
    reason: str = ""
    blocked_by: List[str] = field(default_factory=list)
    horizons: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "passed": self.passed,
            "scope": self.scope,
            "reason": self.reason,
            "blocked_by": list(self.blocked_by),
            "horizons": self.horizons,
            "thresholds": self.thresholds,
            "generated_at": self.generated_at or datetime.now().isoformat(timespec="seconds"),
        }


class StrategyGate:
    """IC / 命中率策略门禁（纯读、无副作用）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = ((config or {}).get("strategy_gate", {}) or {})
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        # 显式锁死只读：即使后续指标达标也不放行（人工复核开关）
        self.force_readonly = bool(cfg.get("force_readonly", False))
        # 判定范围：all（所有周期都达标）/ any（任一周期达标）
        self.scope = str(cfg.get("scope", "all")).lower()
        self.min_ic = float(cfg.get("min_ic", 0.03))
        self.min_hit_rate = float(cfg.get("min_hit_rate", 0.52))
        self.min_samples = int(cfg.get("min_samples", 30))
        self.min_windows = int(cfg.get("min_windows", 3))
        # 审计口径（与 monitor 报表对齐，用于交叉验证；未开则只看 IC 评估）
        self.use_audit = bool(cfg.get("use_audit", False))
        self.min_audit_verified = int(cfg.get("min_audit_verified", DEFAULT_MIN_AUDIT_VERIFIED))
        self.min_audit_hit_rate = float(cfg.get("min_audit_hit_rate", DEFAULT_MIN_AUDIT_HIT_RATE))

    # ------------------------------------------------------------------
    def evaluate_ic(self, ic_payload: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """依据 IC 评估结果判定。

        Args:
            ic_payload: ``ICCalculator.evaluate_all`` 的输出（含 ``horizons`` 字段）。
        """
        horizons = (ic_payload or {}).get("horizons") or {}
        blocked: List[str] = []
        if not horizons:
            return {}, ["无可用的 IC 评估结果"]

        normalized: Dict[str, Any] = {}
        for hname, item in horizons.items():
            passed = bool(item.get("passed"))
            if not passed:
                blocked.append(f"{hname}: {item.get('reason') or '未达门禁'}")
            normalized[hname] = {
                "passed": passed,
                "ic": item.get("ic", 0.0),
                "icir": item.get("icir", 0.0),
                "hit_rate": item.get("hit_rate", 0.0),
                "samples": item.get("samples", 0),
                "windows": item.get("windows", 0),
                "reason": item.get("reason", ""),
            }
        return normalized, blocked

    def evaluate_audit(self, audit_stats: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """依据预测审计命中率交叉验证（可选）。"""
        if not self.use_audit:
            return {}, []
        if not audit_stats or not audit_stats.get("available", True):
            return {}, ["审计命中率不可用"]

        verified = int(audit_stats.get("verified", 0) or 0)
        rate = float(audit_stats.get("hit_rate", 0.0) or 0.0)
        detail = {"verified": verified, "hit_rate": rate,
                  "min_verified": self.min_audit_verified,
                  "min_hit_rate": self.min_audit_hit_rate}
        blocked: List[str] = []
        if verified < self.min_audit_verified:
            blocked.append(f"审计已验证样本 {verified} < {self.min_audit_verified}")
        if rate < self.min_audit_hit_rate:
            blocked.append(f"审计命中率 {rate:.2%} < {self.min_audit_hit_rate:.0%}")
        return detail, blocked

    def decide(self, ic_payload: Optional[Dict[str, Any]] = None,
               audit_stats: Optional[Dict[str, Any]] = None) -> GateDecision:
        """综合判定信号能否升级为打分因子。"""
        thresholds = {
            "min_ic": self.min_ic,
            "min_hit_rate": self.min_hit_rate,
            "min_samples": self.min_samples,
            "min_windows": self.min_windows,
            "scope": self.scope,
            "use_audit": self.use_audit,
            "min_audit_verified": self.min_audit_verified,
            "min_audit_hit_rate": self.min_audit_hit_rate,
        }
        now = datetime.now().isoformat(timespec="seconds")

        if not self.enabled:
            return GateDecision(
                state=STATE_DISABLED, passed=False, scope=self.scope,
                reason="门禁已关闭（strategy_gate.enabled=false），信号保持只读观测",
                blocked_by=["gate_disabled"], thresholds=thresholds, generated_at=now,
            )
        if self.force_readonly:
            return GateDecision(
                state=STATE_READONLY, passed=False, scope=self.scope,
                reason="人工锁定只读（strategy_gate.force_readonly=true）",
                blocked_by=["force_readonly"], thresholds=thresholds, generated_at=now,
            )

        horizon_results, blocked = self.evaluate_ic(ic_payload or {})
        audit_detail, audit_blocked = self.evaluate_audit(audit_stats or {})

        if not horizon_results:
            return GateDecision(
                state=STATE_READONLY, passed=False, scope=self.scope,
                reason="无 IC 评估数据，fail-close 保持只读观测",
                blocked_by=blocked or ["no_ic_data"], horizons=horizon_results,
                thresholds=thresholds, generated_at=now,
            )

        passes = [h for h, v in horizon_results.items() if v["passed"]]
        all_pass = len(passes) == len(horizon_results)
        passed = all_pass if self.scope == "all" else bool(passes)
        # 审计交叉验证是「与门」：任一口径未过则整体不予放行（fail-close）
        blocked_by = list(blocked) + list(audit_blocked)
        if audit_blocked:
            passed = False
        if not passed and not blocked_by:
            blocked_by = [f"{self.scope} 范围内周期未全部达标"]

        state = STATE_GATED if passed else STATE_READONLY
        reason = (
            f"已过门禁（{len(passes)}/{len(horizon_results)} 周期达标，scope={self.scope}）"
            if passed
            else "未过门禁，信号保持只读观测（fail-close）"
        )
        if audit_detail:
            horizon_results["_audit"] = audit_detail

        return GateDecision(
            state=state, passed=passed, scope=self.scope, reason=reason,
            blocked_by=blocked_by, horizons=horizon_results,
            thresholds=thresholds, generated_at=now,
        )

    # ------------------------------------------------------------------
    def evaluate_from_audit_records(self, records: List[Dict[str, Any]],
                                    asof: Optional[str] = None) -> Dict[str, Any]:
        """从审计记录（JSONL 载入的 dict 列表）实时统计命中率。

        只统计 ``verified=True`` 的记录；无记录时返回 available=False（fail-close 由 decide 处理）。
        """
        verified = [r for r in (records or []) if r.get("verified")]
        hits = sum(1 for r in verified if r.get("hit"))
        rate = hits / len(verified) if verified else 0.0
        by_horizon: Dict[str, Dict[str, Any]] = {}
        for r in verified:
            h = str(r.get("horizon", "unknown"))
            b = by_horizon.setdefault(h, {"verified": 0, "hits": 0})
            b["verified"] += 1
            if r.get("hit"):
                b["hits"] += 1
        for h, b in by_horizon.items():
            b["hit_rate"] = b["hits"] / b["verified"] if b["verified"] else 0.0
        return {
            "available": bool(verified),
            "verified": len(verified),
            "hits": hits,
            "hit_rate": round(rate, 4),
            "by_horizon": by_horizon,
            "asof": asof or datetime.now().isoformat(timespec="seconds"),
        }


def recent_audit_stats(audit_dir: str = "logs/audit", days: int = 30) -> Dict[str, Any]:
    """读取审计目录并统计近 N 天命中率（只读，fail-soft）。"""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        from src.audit.prediction_audit import PredictionAudit

        records = PredictionAudit(audit_dir).load_records()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[gate] 读取审计记录失败: {e}")
        return {"available": False, "error": str(e)}

    recent = [r for r in records if str(r.get("timestamp", "")) >= cutoff]
    return StrategyGate().evaluate_from_audit_records(recent)
