"""模型监控报表生成器（ModelMonitor / HealthReport）。

汇总五类运行态信息为一份报告：
  1. **预测审计**：命中率（整体 / 分周期 / 近 30 天）、待验证数、漂移信号；
  2. **自适应学习**：各周期近期准确率、漂移检测状态；
  3. **数据源健康**：宏观指标可用性、行情缓存覆盖、实时流状态（Q3）；
  4. **模型产物**：已训练模型文件与更新时间；
  5. **实时流（Q3）**：盘中快照覆盖、快照新鲜度、盘中观点失真预警。

设计约束：
- **纯读操作**：不写审计记录、不触发重训练、不触网（宏观为读缓存态）；
- **fail-soft**：任一子模块异常只降级该章节，不影响整份报告；
- **可序列化**：`to_dict()` 输出稳定字段，供 API / 28 系统消费。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 命中率低于该阈值且已验证样本充足时判定为漂移
DEFAULT_DRIFT_THRESHOLD = 0.5
DEFAULT_MIN_VERIFIED = 10


class HealthReport:
    """监控报表数据容器（可直接 to_dict / to_markdown）。"""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    @property
    def generated_at(self) -> str:
        return self.payload.get("generated_at", "")

    @property
    def status(self) -> str:
        return self.payload.get("status", "unknown")

    def to_dict(self) -> Dict[str, Any]:
        return self.payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.payload, ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        return render_markdown(self.payload)

    def save(self, path: str | Path) -> Path:
        """保存为 Markdown（.md）或 JSON（.json），按后缀自动选择。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            out.write_text(self.to_json(), encoding="utf-8")
        else:
            out.write_text(self.to_markdown(), encoding="utf-8")
        logger.info(f"监控报表已保存: {out}")
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"<HealthReport status={self.status} at={self.generated_at}>"


class ModelMonitor:
    """模型监控器：采集各子系统运行态并生成报表。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
                 min_verified: int = DEFAULT_MIN_VERIFIED) -> None:
        self.config = config or {}
        self.drift_threshold = float(drift_threshold)
        self.min_verified = int(min_verified)

    # ------------------------------------------------------------------
    # 各章节采集（均 fail-soft）
    # ------------------------------------------------------------------
    def _collect_audit(self) -> Dict[str, Any]:
        """预测审计命中率（只读）。"""
        audit_cfg = (self.config.get("audit", {}) or {})
        audit_dir = audit_cfg.get("dir", "logs/audit")
        try:
            from src.audit.prediction_audit import PredictionAudit

            audit = PredictionAudit(audit_dir)
            records = audit.load_records()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[monitor] 读取审计记录失败: {e}")
            return {"available": False, "error": str(e)}

        verified = [r for r in records if r.get("verified")]
        hits = sum(1 for r in verified if r.get("hit"))
        hit_rate = hits / len(verified) if verified else 0.0

        by_horizon: Dict[str, Dict[str, Any]] = {}
        for r in verified:
            h = str(r.get("horizon", "unknown"))
            bucket = by_horizon.setdefault(h, {"verified": 0, "hits": 0})
            bucket["verified"] += 1
            if r.get("hit"):
                bucket["hits"] += 1
        for h, b in by_horizon.items():
            b["hit_rate"] = b["hits"] / b["verified"] if b["verified"] else 0.0

        cutoff = datetime.now() - timedelta(days=30)
        recent = []
        for r in verified:
            try:
                if datetime.fromisoformat(r["timestamp"]) >= cutoff:
                    recent.append(r)
            except Exception:  # noqa: BLE001
                continue
        recent_hits = sum(1 for r in recent if r.get("hit"))
        recent_rate = recent_hits / len(recent) if recent else 0.0

        drift = len(verified) >= self.min_verified and hit_rate < self.drift_threshold
        return {
            "available": True,
            "total_records": len(records),
            "verified": len(verified),
            "pending": len(records) - len(verified),
            "hits": hits,
            "hit_rate": round(hit_rate, 4),
            "by_horizon": by_horizon,
            "recent_30d_verified": len(recent),
            "recent_30d_hit_rate": round(recent_rate, 4),
            "drift": drift,
            "drift_threshold": self.drift_threshold,
            "min_verified": self.min_verified,
        }

    def _collect_adaptive(self) -> Dict[str, Any]:
        """自适应学习性能历史（只读，不触发重训练）。"""
        save_dir = (self.config.get("training", {}) or {}).get("save_dir", "models")
        path = Path(save_dir) / "monitor" / "performance_history.json"
        if not path.exists():
            return {"available": False, "reason": "no_history"}

        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[monitor] 读取性能历史失败: {e}")
            return {"available": False, "error": str(e)}

        if not isinstance(history, list) or not history:
            return {"available": False, "reason": "empty_history"}

        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        recent = [r for r in history if str(r.get("timestamp", "")) >= cutoff]

        by_horizon: Dict[str, List[float]] = {}
        for r in recent:
            try:
                acc = float((r.get("metrics") or {}).get("accuracy", 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
            by_horizon.setdefault(str(r.get("horizon", "unknown")), []).append(acc)

        summary = {
            h: {
                "records": len(vals),
                "avg_accuracy": round(sum(vals) / len(vals), 4),
                "min_accuracy": round(min(vals), 4),
                "max_accuracy": round(max(vals), 4),
            }
            for h, vals in by_horizon.items()
        }
        return {
            "available": True,
            "total_records": len(history),
            "recent_30d_records": len(recent),
            "by_horizon": summary,
        }

    def _collect_data_sources(self) -> Dict[str, Any]:
        """数据源健康：宏观指标 + 行情缓存覆盖（只读）。"""
        result: Dict[str, Any] = {}

        try:
            from src.data.macro_client import MacroClient

            health = MacroClient(self.config).health()
            ok = sum(1 for v in health.values() if v.get("status") == "ok")
            result["macro"] = {
                "available_indicators": ok,
                "total_indicators": len(health),
                "details": health,
            }
        except Exception as e:  # noqa: BLE001
            result["macro"] = {"available_indicators": 0, "total_indicators": 0, "error": str(e)}

        try:
            raw_dir = Path((self.config.get("data", {}) or {}).get("raw_dir", "data/raw"))
            files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
            latest = ""
            if files:
                latest_ts = max(f.stat().st_mtime for f in files)
                latest = datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S")
            result["market_cache"] = {
                "raw_dir": str(raw_dir),
                "cached_symbols": len(files),
                "latest_update": latest,
            }
            # 持仓池覆盖度（Q2-4）：配置里的启用标的有多少已落地行情缓存
            markets = ((self.config.get("data", {}) or {}).get("markets", {}) or {})
            wanted: List[str] = []
            for mcfg in markets.values():
                if (mcfg or {}).get("enabled"):
                    wanted.extend((mcfg or {}).get("symbols", []) or [])
            cached = {f.stem for f in files}
            missing = [s for s in wanted if s not in cached]
            result["holdings_pool"] = {
                "configured": len(wanted),
                "cached": len([s for s in wanted if s in cached]),
                "coverage": round(
                    len([s for s in wanted if s in cached]) / len(wanted), 4
                ) if wanted else 0.0,
                "missing": missing[:20],
                "missing_count": len(missing),
            }
            result["quality_gate"] = {
                "enabled": bool((self.config.get("data", {}) or {}).get("quality_gate", False)),
            }
        except Exception as e:  # noqa: BLE001
            result["market_cache"] = {"cached_symbols": 0, "error": str(e)}

        return result

    def _collect_streaming(self) -> Dict[str, Any]:
        """实时数据流状态（Q3，只读）：盘中快照覆盖 / 新鲜度 / 观点失真预警。

        只读本地快照文件，不触网、不主动拉行情 —— 监控报表必须随时可跑且零副作用。
        """
        cfg = (self.config.get("streaming", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return {"available": False, "reason": "streaming_disabled",
                    "hint": "在配置中开启 streaming.enabled 后启用分钟级更新"}

        try:
            from src.data.streaming import IntradayStore, MarketClock

            store = IntradayStore(self.config)
            today = datetime.now().strftime("%Y-%m-%d")
            snaps = store.load(day=today)
            symbols = sorted({s.symbol for s in snaps})
            last_ts = max((s.ts for s in snaps), default="")
            stale_minutes = 0
            if last_ts:
                try:
                    stale_minutes = int(
                        (datetime.now() - datetime.fromisoformat(last_ts)).total_seconds() // 60
                    )
                except ValueError:
                    stale_minutes = 0

            files = sorted(store.dir.glob("snapshots_*.jsonl"))
            return {
                "available": bool(snaps),
                "reason": "" if snaps else "no_snapshot_today",
                "dir": str(store.dir),
                "poll_seconds": int(cfg.get("poll_seconds", 60)),
                "retention_days": store.retention_days,
                "is_trading_hours": MarketClock.is_trading_hours(),
                "symbols_today": len(symbols),
                "snapshots_today": len(snaps),
                "last_snapshot_at": last_ts,
                "snapshot_age_minutes": stale_minutes,
                "history_files": len(files),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[monitor] 实时流状态采集失败: {e}")
            return {"available": False, "error": str(e)}

    def _collect_risk_advice(self) -> Dict[str, Any]:
        """智能风控建议状态（Q4，只读）：门禁是否放行 + 最近一次建议摘要。

        只读 `reports/risk_advice.json`（由 `python main.py risk-advice --json` 落盘），
        不触网、不跑模型 —— 监控报表必须随时可跑且零副作用。
        """
        path = Path((self.config.get("report", {}) or {}).get("output_dir", "reports")) / \
            "risk_advice.json"
        if not path.exists():
            return {
                "available": False,
                "reason": "no_risk_advice_report",
                "hint": "先运行 `python main.py risk-advice --all --json`（或指定标的）生成建议",
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload.get("items") or []
            withheld = [i for i in items if i.get("status") == "withheld"]
            return {
                "available": True,
                "source": str(path),
                "gate_state": payload.get("gate_state", "unknown"),
                "count": payload.get("count", len(items)),
                "advised": payload.get("available", 0),
                "withheld": payload.get("withheld", len(withheld)),
                "unavailable": payload.get("unavailable", 0),
                "avg_risk_reward": payload.get("avg_risk_reward"),
                "not_trade_instruction": True,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[monitor] 读取风控建议失败: {e}")
            return {"available": False, "error": str(e)}

    def _collect_gate(self) -> Dict[str, Any]:
        """信号准入闸门状态（只读）。

        优先读最近一次评估报告（logs/eval_report.json）里的 gate 结果；
        无报告时如实标注 unavailable，绝不用估算值冒充判定。
        """
        try:
            gate_dir = Path(
                (self.config.get("strategy_gate", {}) or {}).get("report_dir", "reports")
            )
            path = gate_dir / "strategy_gate.json"
            if not path.exists():
                return {
                    "available": False,
                    "reason": "no_gate_report",
                    "hint": "先运行 `python main.py gate` 生成 reports/strategy_gate.json",
                }
            payload = json.loads(path.read_text(encoding="utf-8"))
            gate = dict(payload) if isinstance(payload, dict) else {}
            gate.update({"available": True, "source": str(path)})
            return gate
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[monitor] 读取策略门禁状态失败: {e}")
            return {"available": False, "error": str(e)}

    def _collect_factor_model(self) -> Dict[str, Any]:
        """多因子模型状态（只读）：因子权重 / 族权重 / IC 排名。"""
        save_dir = Path((self.config.get("training", {}) or {}).get("save_dir", "models"))
        path = save_dir / "factor_model_short_term_5d.pkl"
        if not path.exists():
            return {
                "available": False,
                "reason": "no_factor_model",
                "hint": "先运行 `python main.py train --model-type factor_model`",
            }
        try:
            from src.factors.factor_model import FactorModel

            model = FactorModel(self.config)
            model.load(str(path))
            explain = model.explain(8)
            explain.update({"available": True, "path": str(path)})
            return explain
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[monitor] 读取多因子模型失败: {e}")
            return {"available": False, "error": str(e)}

    def _collect_models(self) -> Dict[str, Any]:
        """已训练模型产物清单（只读）。"""
        save_dir = Path((self.config.get("training", {}) or {}).get("save_dir", "models"))
        horizons = (self.config.get("data", {}) or {}).get("prediction_horizons", {}) or {}
        expected = [f"lightgbm_{name}_{days}d.pkl" for name, days in horizons.items()]
        model_type = str((self.config.get("model", {}) or {}).get("type", "lightgbm"))
        # 按当前配置的模型类型补充期望产物（避免把"未启用 LSTM"误判为缺失）
        if model_type == "pytorch_lstm":
            expected = [f"pytorch_lstm_{name}_{days}d.pt" for name, days in horizons.items()]
        elif model_type in ("factor_model", "multifactor"):
            expected = [f"factor_model_{name}_{days}d.pkl" for name, days in horizons.items()]

        found: List[Dict[str, Any]] = []
        if save_dir.exists():
            for pattern in ("lightgbm_*.pkl", "timesfm_*.pkl", "pytorch_lstm_*.pkl"):
                for f in sorted(save_dir.glob(pattern)):
                    found.append({
                        "file": f.name,
                        "size_kb": round(f.stat().st_size / 1024, 1),
                        "updated_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
        names = {f["file"] for f in found}
        return {
            "save_dir": str(save_dir),
            "count": len(found),
            "expected": expected,
            "missing": [n for n in expected if n not in names],
            "files": found,
        }

    # ------------------------------------------------------------------
    # 报表
    # ------------------------------------------------------------------
    def collect(self) -> HealthReport:
        """采集全部章节并生成报表。"""
        audit = self._collect_audit()
        adaptive = self._collect_adaptive()
        sources = self._collect_data_sources()
        models = self._collect_models()
        gate = self._collect_gate()
        factors = self._collect_factor_model()
        streaming = self._collect_streaming()
        risk_advice = self._collect_risk_advice()

        issues: List[str] = []
        if audit.get("available") and audit.get("drift"):
            issues.append(
                f"审计命中率 {audit['hit_rate']:.2%} 低于阈值 {self.drift_threshold:.0%}"
                f"（已验证 {audit['verified']} 条）"
            )
        if models.get("missing"):
            issues.append(f"缺少模型文件: {', '.join(models['missing'])}")
        macro = sources.get("macro", {})
        if macro.get("total_indicators") and not macro.get("available_indicators"):
            issues.append("宏观指标全部不可用（CPI/PMI/GDP 等）")
        if not sources.get("market_cache", {}).get("cached_symbols"):
            issues.append("无行情缓存（data/raw 为空）")
        if gate.get("available") and not gate.get("passed"):
            issues.append(
                f"策略门禁为 {gate.get('state', 'readonly')}（信号只读），未达打分因子放行条件"
            )
        # 实时流（Q3）：启用后盘中应有新鲜快照；交易时段内超过 3 个轮询周期即告警
        if streaming.get("available") or streaming.get("reason") == "no_snapshot_today":
            if streaming.get("is_trading_hours"):
                age = int(streaming.get("snapshot_age_minutes", 0) or 0)
                max_age = max(int(streaming.get("poll_seconds", 60) / 60 * 3), 3)
                if not streaming.get("available") or age > max_age:
                    issues.append(f"实时流盘中无新鲜快照（最近快照 {age} 分钟前，阈值 {max_age} 分钟）")
            elif not streaming.get("available"):
                issues.append("实时流启用但今日无快照（休市或实时源不可用）")

        # 智能风控建议（Q4）：门禁未放行时建议必然 withheld，这是预期行为而非故障，
        # 只有「报告存在却一条建议都没产出」才提示（可能全池行情缺失）
        if risk_advice.get("available"):
            if risk_advice.get("advised", 0) == 0 and risk_advice.get("count", 0) > 0:
                issues.append(
                    f"风控建议全部未产出（{risk_advice.get('count')} 只标的）："
                    f"门禁 {risk_advice.get('gate_state')} 或行情缺失"
                )
        status = "ok" if not issues else ("warning" if len(issues) < 3 else "critical")

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "issues": issues,
            "audit": audit,
            "adaptive": adaptive,
            "data_sources": sources,
            "models": models,
            "gate": gate,
            "factors": factors,
            "streaming": streaming,
            "risk_advice": risk_advice,
        }
        return HealthReport(payload)


# ----------------------------------------------------------------------
# Markdown 渲染
# ----------------------------------------------------------------------
def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def render_markdown(payload: Dict[str, Any]) -> str:
    """把报表 payload 渲染为 Markdown。"""
    status_icon = {"ok": "✅", "warning": "⚠️", "critical": "🔴"}.get(payload.get("status"), "❔")
    lines = [
        "# TrendCast Pro · 模型监控报表",
        "",
        f"- 生成时间：{payload.get('generated_at', '')}",
        f"- 整体状态：{status_icon} {payload.get('status', 'unknown')}",
        "",
    ]

    issues = payload.get("issues") or []
    if issues:
        lines.extend(["## 待处理事项", ""])
        for issue in issues:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")
    else:
        lines.extend(["> 未发现异常。", ""])

    audit = payload.get("audit", {}) or {}
    lines.extend(["## 预测审计", ""])
    if audit.get("available"):
        lines.extend([
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总记录数 | {audit.get('total_records', 0)} |",
            f"| 已验证数 | {audit.get('verified', 0)} |",
            f"| 待验证数 | {audit.get('pending', 0)} |",
            f"| 整体命中率 | {_fmt_pct(audit.get('hit_rate'))} |",
            f"| 近 30 天命中率 | {_fmt_pct(audit.get('recent_30d_hit_rate'))}（{audit.get('recent_30d_verified', 0)} 条） |",
            f"| 漂移信号 | {'⚠️ 是' if audit.get('drift') else '✅ 否'} |",
            "",
        ])
        by_horizon = audit.get("by_horizon") or {}
        if by_horizon:
            lines.extend([
                "### 分周期命中率", "",
                "| 周期 | 验证数 | 命中数 | 命中率 |",
                "|------|--------|--------|--------|",
            ])
            for h, b in sorted(by_horizon.items()):
                lines.append(f"| {h} | {b.get('verified', 0)} | {b.get('hits', 0)} | {_fmt_pct(b.get('hit_rate'))} |")
            lines.append("")
    else:
        lines.extend([f"- 不可用：{audit.get('error') or audit.get('reason') or '无审计记录'}", ""])

    adaptive = payload.get("adaptive", {}) or {}
    lines.extend(["## 自适应学习", ""])
    if adaptive.get("available"):
        lines.append(
            f"- 历史记录总数：{adaptive.get('total_records', 0)}"
            f"（近 30 天 {adaptive.get('recent_30d_records', 0)} 条）"
        )
        by_h = adaptive.get("by_horizon") or {}
        if by_h:
            lines.extend([
                "", "| 周期 | 记录数 | 平均准确率 | 最低 | 最高 |",
                "|------|--------|-----------|------|------|",
            ])
            for h, s in sorted(by_h.items()):
                lines.append(
                    f"| {h} | {s['records']} | {_fmt_pct(s['avg_accuracy'])} | "
                    f"{_fmt_pct(s['min_accuracy'])} | {_fmt_pct(s['max_accuracy'])} |"
                )
        lines.append("")
    else:
        lines.extend([f"- 不可用：{adaptive.get('error') or adaptive.get('reason') or '无性能历史'}", ""])

    sources = payload.get("data_sources", {}) or {}
    lines.extend(["## 数据源健康", ""])
    macro = sources.get("macro", {}) or {}
    lines.append(f"- 宏观指标可用：{macro.get('available_indicators', 0)}/{macro.get('total_indicators', 0)}")
    for code, info in (macro.get("details") or {}).items():
        mark = "✅" if info.get("status") == "ok" else "⚠️"
        latest = (
            f"{info.get('latest_value', 0):.2f} @ {info.get('latest_date', '')}"
            if info.get("points") else "无数据"
        )
        lines.append(f"  - {mark} {code}：{info.get('points', 0)} 期 {latest}")
    cache = sources.get("market_cache", {}) or {}
    lines.append(
        f"- 行情缓存：{cache.get('cached_symbols', 0)} 个标的"
        f"（最近更新 {cache.get('latest_update') or 'N/A'}）"
    )
    pool = sources.get("holdings_pool", {}) or {}
    if pool:
        lines.append(
            f"- 持仓池覆盖：{pool.get('cached', 0)}/{pool.get('configured', 0)}"
            f"（{_fmt_pct(pool.get('coverage'))}）"
            + (f"，缺 {pool.get('missing_count')} 个" if pool.get("missing_count") else "")
        )
    qg = sources.get("quality_gate", {}) or {}
    if qg:
        lines.append(f"- 数据质量门控：{'已启用' if qg.get('enabled') else '未启用'}")
    lines.append("")

    gate = payload.get("gate", {}) or {}
    lines.extend(["## 策略门禁（Q2）", ""])
    if gate.get("available"):
        state = str(gate.get("state", "readonly"))
        icon = "🟢" if state == "gated" else "🔒"
        lines.append(f"- 状态：{icon} **{state}**（{gate.get('reason', '')}）")
        lines.append(f"- 来源：`{gate.get('source', '')}`"
                     f"（判定时间 {gate.get('generated_at') or 'N/A'}）")
        blocked = gate.get("blocked_by") or []
        if blocked:
            lines.append(f"- 未放行原因：{'；'.join(str(b) for b in blocked)}")
        horizons = gate.get("horizons") or {}
        if isinstance(horizons, dict) and horizons:
            lines.extend([
                "", "| 周期 | IC | ICIR | 命中率 | 样本 | 通过 |",
                "|------|----|------|--------|------|------|",
            ])
            for name, h in sorted(horizons.items()):
                if not isinstance(h, dict):
                    continue
                try:
                    ic = float(h.get("ic", 0.0) or 0.0)
                    icir = float(h.get("icir", 0.0) or 0.0)
                except (TypeError, ValueError):
                    ic, icir = 0.0, 0.0
                lines.append(
                    f"| {name} | {ic:+.4f} | {icir:+.4f} | "
                    f"{_fmt_pct(h.get('hit_rate'))} | {h.get('samples', 0)} | "
                    f"{'✅' if h.get('passed') else '❌'} |"
                )
        lines.append("")
    else:
        lines.append(
            f"- 不可用：{gate.get('error') or gate.get('reason') or '无门禁报告'}"
            f"（{gate.get('hint', '')}）"
        )
        lines.append("")

    factors = payload.get("factors", {}) or {}
    lines.extend(["## 多因子模型（Q2）", ""])
    if factors.get("available"):
        fw = factors.get("family_weights") or {}
        if fw:
            lines.append("族权重：" + " · ".join(f"{k}={v}" for k, v in fw.items()))
        rows = factors.get("top_factors") or []
        if rows:
            lines.extend([
                "", "| 因子 | 族 | 权重 | IC |", "|------|-----|------|-----|",
            ])
            for r in rows:
                lines.append(
                    f"| {r['factor']} | {r['family']} | {r['weight']:.4f} | {r['ic']:+.4f} |"
                )
        lines.append("")
    else:
        lines.append(
            f"- 不可用：{factors.get('error') or factors.get('reason') or '无因子模型'}"
            f"（{factors.get('hint', '')}）"
        )
        lines.append("")

    streaming = payload.get("streaming", {}) or {}
    lines.extend(["## 实时数据流（Q3）", ""])
    if streaming.get("available"):
        lines.extend([
            f"- 快照目录：`{streaming.get('dir', '')}`（历史文件 {streaming.get('history_files', 0)} 个）",
            f"- 轮询间隔：{streaming.get('poll_seconds', 0)} 秒；保留期 {streaming.get('retention_days', 0)} 天",
            f"- 今日快照：{streaming.get('snapshots_today', 0)} 条 / {streaming.get('symbols_today', 0)} 个标的",
            f"- 最近快照：{streaming.get('last_snapshot_at') or 'N/A'}"
            f"（{streaming.get('snapshot_age_minutes', 0)} 分钟前）",
            f"- 当前{'处于' if streaming.get('is_trading_hours') else '不处于'}交易时段",
        ])
    else:
        lines.append(
            f"- 不可用：{streaming.get('error') or streaming.get('reason') or '未启用'}"
            f"（{streaming.get('hint', '')}）"
        )
    lines.append("")

    risk_advice = payload.get("risk_advice", {}) or {}
    lines.extend(["## 智能风控建议（Q4）", ""])
    if risk_advice.get("available"):
        lines.extend([
            f"- 门禁状态：`{risk_advice.get('gate_state', 'unknown')}`",
            f"- 建议产出：{risk_advice.get('advised', 0)}/{risk_advice.get('count', 0)} 条"
            f"（暂缓 {risk_advice.get('withheld', 0)}｜数据不足 {risk_advice.get('unavailable', 0)}）",
            f"- 平均盈亏比：{risk_advice.get('avg_risk_reward')}",
            "- 定位：**建议而非下单指令**；门禁未放行时不产出可直接使用的止损止盈价位",
        ])
    else:
        lines.append(
            f"- 不可用：{risk_advice.get('error') or risk_advice.get('reason') or '未生成'}"
            f"（{risk_advice.get('hint', '')}）"
        )
    lines.append("")

    models = payload.get("models", {}) or {}
    lines.extend(["## 模型产物", ""])
    lines.append(f"- 目录：`{models.get('save_dir', '')}`，共 {models.get('count', 0)} 个文件")
    if models.get("missing"):
        lines.append(f"- ⚠️ 缺失：{', '.join(models['missing'])}")
    for f in models.get("files") or []:
        lines.append(f"  - `{f['file']}`（{f['size_kb']} KB，{f['updated_at']}）")
    lines.append("")

    lines.extend(["---", "", "*本报表仅供运维巡检参考，不构成投资建议。*"])
    return "\n".join(lines)
