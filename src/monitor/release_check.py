"""发布态健康检查（S14）：把散落的运维信号收敛成**一条可执行的判断**。

问题从哪来（S11–S13 的汇总）：
  到这里仓库里已经有了很多东西：门禁（S7/Q2）、IC 趋势（Q5）、分池（S9）、多周期扫描（S10）、
  周期切换决策单（S11）、特征扩充对照（S12）、试验登记（S13）……
  但它们是**一堆各自独立的报告**。运维上真正要回答的只有一个问题：

  > 现在这套东西，能不能继续按当前口径往下跑？需不需要先做某件事？

  过去这个判断靠人翻五份报告。于是最常见的失败模式不是"模型坏了"，
  而是**没人注意到某份报告过期了** —— 门禁判定用的是两周前的 IC，
  校正用的是过期扫描，谁也没发现。

本模块做什么：
  把各产物收敛成三类结论，**只做汇总与排序，不重算任何指标**：

  - ``blocking``：必须处理，否则当前结论不可信（例：扫描报告过期、
    试验登记损坏、模型产物缺失）；
  - ``action``  ：建议动作（例：IC 衰减 → 提前重训练；有显著特征臂待人工确认）；
  - ``info``    ：只是状态（门禁仍 readonly、当前未过线）。

  每条都带 `code` / `severity` / `message` / `next_step`，可直接进告警路由。

本模块**不做什么**：
  - **不自动修**：不触发重训练、不改配置、不删产物 —— 只报告；
  - **不重算指标**：不跑评估、不重训，只读既有产物（报表必须随时可跑）；
  - **不掩盖**：产物缺失 / 损坏一律如实进 `blocking`，**绝不当作健康**。

优先级说明（这是本模块最需要看清的部分）：
  「产物可用性」高于「指标好坏」。一份过期的门禁判定比一个不达标但新鲜的判定更危险
  —— 前者会让人以为"已经评估过了"。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEVERITY_BLOCKING = "blocking"
SEVERITY_ACTION = "action"
SEVERITY_INFO = "info"

# 结论：ready（可继续）/ attention（有建议动作）/ blocked（必须先处理）
STATUS_READY = "ready"
STATUS_ATTENTION = "attention"
STATUS_BLOCKED = "blocked"


def _item(code: str, severity: str, message: str, next_step: str = "",
          source: str = "") -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "next_step": next_step,
        "source": source,
    }


def _age_days(stamp: str) -> Optional[float]:
    raw = str(stamp or "").strip()
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
    return (now - ts).total_seconds() / 86400.0


def check_release_state(config: Optional[Dict[str, Any]] = None,
                        gate: Optional[Dict[str, Any]] = None,
                        ic_trend: Optional[Dict[str, Any]] = None,
                        horizon_scan: Optional[Dict[str, Any]] = None,
                        horizon_decision: Optional[Dict[str, Any]] = None,
                        feature_experiment: Optional[Dict[str, Any]] = None,
                        trial_registry: Optional[Dict[str, Any]] = None,
                        models_missing: Optional[List[str]] = None,
                        max_scan_age_days: float = 7.0,
                        max_trend_age_days: float = 30.0) -> Dict[str, Any]:
    """汇总各产物 → 发布态结论（纯函数，便于单测与复用）。"""
    items: List[Dict[str, Any]] = []

    # --- 1) 产物可用性（最高优先级：过期/损坏的产物比"不达标"更危险）---
    if horizon_scan is None:
        items.append(_item(
            "scan_missing", SEVERITY_BLOCKING,
            "多周期扫描报告缺失，任何口径相关结论都无当前证据",
            "运行 `python main.py horizon-scan`", "horizon_scan"))
    elif not horizon_scan.get("available", True):
        items.append(_item(
            "scan_unavailable", SEVERITY_BLOCKING,
            f"多周期扫描不可用：{horizon_scan.get('reason') or horizon_scan.get('error') or '未知原因'}",
            "先修复数据链路再重跑 `python main.py horizon-scan`", "horizon_scan"))
    else:
        age = _age_days(horizon_scan.get("generated_at", ""))
        if age is None:
            items.append(_item(
                "scan_timestamp_unparsable", SEVERITY_BLOCKING,
                "扫描报告缺少可解析的生成时间，无法判断结论是否仍有效",
                "重跑 `python main.py horizon-scan` 重新生成", "horizon_scan"))
        elif age > float(max_scan_age_days):
            items.append(_item(
                "scan_stale", SEVERITY_BLOCKING,
                f"扫描报告已过期（{age:.1f} 天 > {max_scan_age_days} 天），"
                "决策单的校正结论基于旧数据",
                "重跑 `python main.py horizon-scan && python main.py horizon-decision`",
                "horizon_scan"))

    if horizon_decision is None:
        items.append(_item(
            "decision_missing", SEVERITY_ACTION,
            "周期切换决策单未生成：口径变更这条路没有被评估过",
            "运行 `python main.py horizon-decision`", "horizon_decision"))
    elif horizon_decision.get("status") == "stale":
        items.append(_item(
            "decision_stale", SEVERITY_BLOCKING,
            "决策单标记 stale：其证据不可复现",
            "重跑 `horizon-scan` 后重新生成决策单", "horizon_decision"))

    if feature_experiment is None:
        items.append(_item(
            "feature_experiment_missing", SEVERITY_ACTION,
            "特征扩充对照实验未跑：无法判断加特征这条路的证据",
            "运行 `python main.py feature-experiment`", "feature_experiment"))
    elif feature_experiment.get("verdict") == "adopt":
        items.append(_item(
            "feature_experiment_adopt", SEVERITY_ACTION,
            "特征扩充对照实验有显著臂（adopt）：等人工确认是否接入生产特征集",
            "审阅 `reports/feature_experiment.json` 并决定是否落地", "feature_experiment"))

    if trial_registry is not None and not trial_registry.get("available", True):
        items.append(_item(
            "trials_unavailable", SEVERITY_BLOCKING,
            f"试验登记不可用：{trial_registry.get('reason') or trial_registry.get('error')}；"
            "校正次数可能被低估",
            "检查 `logs/trials.jsonl` 完整性（勿手改）", "trial_registry"))

    if models_missing:
        items.append(_item(
            "models_missing", SEVERITY_BLOCKING,
            f"模型产物缺失：{', '.join(models_missing)}",
            "运行 `python main.py train` 重新训练", "models"))

    # --- 2) 指标状态（只是状态，不是故障）---
    if gate is None:
        items.append(_item(
            "gate_missing", SEVERITY_ACTION,
            "门禁判定缺失：无法确认信号是否仍为只读",
            "运行 `python main.py gate`", "gate"))
    elif str(gate.get("state")) == "readonly":
        items.append(_item(
            "gate_readonly", SEVERITY_INFO,
            "门禁仍为 readonly：信号只读观测，未进入调仓打分路径",
            "无需处理；如需解锁按门禁口径提升 IC/命中率", "gate"))
    elif str(gate.get("state")) == "gated":
        items.append(_item(
            "gate_gated", SEVERITY_INFO,
            "门禁已放行（gated）：信号可作为打分因子，仍需人工确认是否接入",
            "确认下游是否按新状态消费信号", "gate"))

    if ic_trend is not None and ic_trend.get("available", True):
        decaying = ic_trend.get("decaying") or []
        if decaying:
            items.append(_item(
                "ic_decaying", SEVERITY_ACTION,
                f"IC 持续衰减的周期：{', '.join(map(str, decaying))}",
                "评估是否提前重训练（`python main.py schedule` 或手动 `main.py train`）",
                "ic_trend"))
        age = _age_days(ic_trend.get("generated_at", ""))
        if age is not None and age > float(max_trend_age_days):
            items.append(_item(
                "ic_trend_stale", SEVERITY_ACTION,
                f"IC 趋势报告已过期（{age:.1f} 天），衰减告警可能已不反映现状",
                "运行 `python main.py ic-trend`", "ic_trend"))

    # --- 3) 结论 ---
    blocking = [i for i in items if i["severity"] == SEVERITY_BLOCKING]
    actions = [i for i in items if i["severity"] == SEVERITY_ACTION]
    if blocking:
        status = STATUS_BLOCKED
    elif actions:
        status = STATUS_ATTENTION
    else:
        status = STATUS_READY

    blocking_codes = {i["code"] for i in blocking}
    action_codes = {i["code"] for i in actions}
    return {
        "status": status,
        "blocking": blocking,
        "actions": actions,
        "info": [i for i in items if i["severity"] == SEVERITY_INFO],
        "counts": {"blocking": len(blocking), "action": len(actions),
                   "info": len(items) - len(blocking) - len(actions)},
        # 供告警路由做去重/静默：只暴露"稳定标识"，不含时间戳
        "keys": sorted(blocking_codes | action_codes),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "affects_gate": False,
        "note": (
            "发布态检查只做汇总与排序，不重算指标、不自动修复；"
            "产物可用性优先于指标好坏（过期的判定比不达标更危险）。"
        ),
    }


def collect_and_check(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从既有产物读取各状态并做发布态检查（只读、fail-soft）。"""
    cfg = config or {}
    data: Dict[str, Any] = {}

    def _safe(name: str, fn) -> None:
        try:
            data[name] = fn()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[release-check] {name} 采集失败: {e}")
            data[name] = None

    def _gate():
        from src.monitor.health_report import ModelMonitor

        return ModelMonitor(cfg)._collect_gate()

    def _ic_trend():
        from src.monitor.health_report import ModelMonitor

        return ModelMonitor(cfg)._collect_ic_trend()

    def _scan():
        from src.eval.horizon_scan import HorizonScanner

        return HorizonScanner(cfg).load()

    def _decision():
        from src.eval import horizon_decision as hd

        return hd.load(cfg)

    def _experiment():
        from src.eval.feature_experiment import load

        return load(cfg)

    def _trials():
        from src.eval.trial_registry import summary

        return summary(cfg)

    def _models():
        from src.monitor.health_report import ModelMonitor

        return (ModelMonitor(cfg)._collect_models() or {}).get("missing") or []

    _safe("gate", _gate)
    _safe("ic_trend", _ic_trend)
    _safe("horizon_scan", _scan)
    _safe("horizon_decision", _decision)
    _safe("feature_experiment", _experiment)
    _safe("trial_registry", _trials)
    _safe("models_missing", _models)

    scan_cfg = (cfg.get("release_check", {}) or {})
    return check_release_state(
        cfg,
        gate=data.get("gate"),
        ic_trend=data.get("ic_trend"),
        horizon_scan=data.get("horizon_scan"),
        horizon_decision=data.get("horizon_decision"),
        feature_experiment=data.get("feature_experiment"),
        trial_registry=data.get("trial_registry"),
        models_missing=data.get("models_missing") or None,
        max_scan_age_days=float(scan_cfg.get("max_scan_age_days", 7.0)),
        max_trend_age_days=float(scan_cfg.get("max_trend_age_days", 30.0)),
    )


def route_alerts(result: Dict[str, Any], config: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """把发布态结论转成告警路由决定（**不发送**，只给决定）。

    设计取舍：
      - ``blocking`` 每次都发（漏报代价高）；
      - ``action`` 只在 ``keys`` 变化时发 —— 否则每天同一句"IC 在衰减"
        会把人训练成忽略告警；
      - ``info`` 从不单独发。

    状态存在 `logs/alert_state.json`（仅记上一次的 keys，用于去重）。
    """
    cfg = (config or {}).get("release_check", {}) or {}
    state_path = Path(cfg.get("state_file", "logs/alert_state.json"))
    previous: Dict[str, Any] = {}
    try:
        if state_path.exists():
            import json

            previous = json.loads(state_path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[release-check] 告警状态读取失败（按首次处理）: {e}")
        previous = {}

    prev_keys = set(previous.get("keys") or [])
    cur_keys = set(result.get("keys") or [])
    added = sorted(cur_keys - prev_keys)
    resolved = sorted(prev_keys - cur_keys)

    status = result.get("status")
    should_send = bool(result.get("blocking")) or bool(added) or bool(resolved)

    decision = {
        "should_send": should_send,
        "reason": (
            "存在 blocking 项" if result.get("blocking")
            else ("待处理项发生变化" if (added or resolved) else "与上次一致，静默")
        ),
        "status": status,
        "new_keys": added,
        "resolved_keys": resolved,
        "keys": sorted(cur_keys),
        "channel": str(cfg.get("channel", "webhook")),
        "state_file": str(state_path),
    }
    if should_send:
        try:
            import json

            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps({"keys": sorted(cur_keys), "at": result.get("checked_at", "")},
                           ensure_ascii=False),
                encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[release-check] 告警状态落盘失败（不影响本次决定）: {e}")
            decision["state_persisted"] = False
    return decision
